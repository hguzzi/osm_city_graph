from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Literal

import networkx as nx
import numpy as np
import pandas as pd

from .core import symmetrize_matrix, validate_cities

LOGGER = logging.getLogger(__name__)


def _require_osmnx():
    try:
        import geopandas as gpd
        import osmnx as ox
    except ImportError as exc:
        raise RuntimeError(
            "Road-network mode requires OSMnx and GeoPandas. Install the project "
            "dependencies with: pip install -r requirements.txt"
        ) from exc
    return ox, gpd


def geocode_missing_cities(cities: pd.DataFrame) -> pd.DataFrame:
    """Geocode rows with missing coordinates using OSMnx/Nominatim."""
    ox, _ = _require_osmnx()
    out = cities.copy()
    if "latitude" not in out.columns:
        out["latitude"] = np.nan
    if "longitude" not in out.columns:
        out["longitude"] = np.nan

    out["latitude"] = pd.to_numeric(out["latitude"], errors="coerce")
    out["longitude"] = pd.to_numeric(out["longitude"], errors="coerce")
    for index, row in out.iterrows():
        if pd.notna(row["latitude"]) and pd.notna(row["longitude"]):
            continue
        query = str(row.get("query") or row.get("name") or row.get("id"))
        LOGGER.info("Geocoding %s", query)
        latitude, longitude = ox.geocoder.geocode(query)
        out.at[index, "latitude"] = latitude
        out.at[index, "longitude"] = longitude
    return validate_cities(out)


def bbox_for_cities(cities: pd.DataFrame, padding_km: float = 5.0) -> tuple[float, float, float, float]:
    """Create an EPSG:4326 bounding box as (left, bottom, right, top)."""
    cities = validate_cities(cities)
    if padding_km < 0:
        raise ValueError("padding_km must be non-negative")
    mean_latitude = math.radians(float(cities["latitude"].mean()))
    latitude_padding = padding_km / 111.0
    longitude_padding = padding_km / max(111.0 * math.cos(mean_latitude), 1e-6)
    return (
        float(cities["longitude"].min() - longitude_padding),
        float(cities["latitude"].min() - latitude_padding),
        float(cities["longitude"].max() + longitude_padding),
        float(cities["latitude"].max() + latitude_padding),
    )


def load_or_download_road_graph(
    cities: pd.DataFrame,
    *,
    cache_path: Path,
    network_type: str = "drive",
    padding_km: float = 5.0,
    refresh: bool = False,
    fallback_speed_kph: float = 40.0,
) -> nx.MultiDiGraph:
    """Load a cached OSM graph or download it from the cities' bounding box."""
    ox, _ = _require_osmnx()
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    ox.settings.use_cache = True
    ox.settings.log_console = True

    if cache_path.exists() and not refresh:
        LOGGER.info("Loading cached road graph from %s", cache_path)
        graph = ox.io.load_graphml(cache_path)
    else:
        bbox = bbox_for_cities(cities, padding_km=padding_km)
        LOGGER.info("Downloading OSM %s network for bbox=%s", network_type, bbox)
        graph = ox.graph.graph_from_bbox(
            bbox,
            network_type=network_type,
            simplify=True,
            retain_all=False,
            truncate_by_edge=True,
        )
        graph = ox.routing.add_edge_speeds(graph, fallback=fallback_speed_kph)
        graph = ox.routing.add_edge_travel_times(graph)
        ox.io.save_graphml(graph, cache_path)

    # Older caches might not contain speeds/travel times.
    if any("speed_kph" not in data for _, _, _, data in graph.edges(keys=True, data=True)):
        graph = ox.routing.add_edge_speeds(graph, fallback=fallback_speed_kph)
    if any("travel_time" not in data for _, _, _, data in graph.edges(keys=True, data=True)):
        graph = ox.routing.add_edge_travel_times(graph)
    return graph


def snap_cities_to_road_nodes(cities: pd.DataFrame, graph: nx.MultiDiGraph) -> np.ndarray:
    """Map city coordinates to their nearest OSM road-network nodes."""
    ox, gpd = _require_osmnx()
    cities = validate_cities(cities)
    projected_graph = ox.projection.project_graph(graph)
    points = gpd.GeoDataFrame(
        cities[["id", "name"]].copy(),
        geometry=gpd.points_from_xy(cities["longitude"], cities["latitude"]),
        crs="EPSG:4326",
    ).to_crs(projected_graph.graph["crs"])
    nodes = ox.distance.nearest_nodes(
        projected_graph,
        X=points.geometry.x.to_numpy(),
        Y=points.geometry.y.to_numpy(),
    )
    return np.asarray(nodes, dtype=np.int64)


def directed_route_cost_matrix(
    graph: nx.MultiDiGraph,
    city_nodes: np.ndarray,
    *,
    weight: str,
    divisor: float,
) -> np.ndarray:
    """Compute all city-to-city shortest-path costs from a road graph."""
    city_nodes = np.asarray(city_nodes)
    n = len(city_nodes)
    matrix = np.full((n, n), np.inf, dtype=float)
    np.fill_diagonal(matrix, 0.0)

    origins: dict[int, list[int]] = {}
    for city_index, node in enumerate(city_nodes.tolist()):
        origins.setdefault(int(node), []).append(city_index)

    for origin_node, city_indices in origins.items():
        LOGGER.info("Running Dijkstra from OSM node %s", origin_node)
        lengths = nx.single_source_dijkstra_path_length(graph, origin_node, weight=weight)
        for city_index in city_indices:
            for destination_index, destination_node in enumerate(city_nodes.tolist()):
                value = lengths.get(int(destination_node))
                if value is not None:
                    matrix[city_index, destination_index] = float(value) / divisor
    return matrix


def road_matrices(
    cities: pd.DataFrame,
    graph: nx.MultiDiGraph,
    *,
    directed: bool,
    symmetrize: Literal["mean", "min", "max"] = "mean",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return snapped node IDs, road-distance km, and travel-time min matrices."""
    city_nodes = snap_cities_to_road_nodes(cities, graph)
    distance = directed_route_cost_matrix(graph, city_nodes, weight="length", divisor=1000.0)
    travel_time = directed_route_cost_matrix(
        graph, city_nodes, weight="travel_time", divisor=60.0
    )
    if not directed:
        distance = symmetrize_matrix(distance, symmetrize)
        travel_time = symmetrize_matrix(travel_time, symmetrize)
    return city_nodes, distance, travel_time
