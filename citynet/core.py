from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Literal

import networkx as nx
import numpy as np
import pandas as pd

Topology = Literal["complete", "threshold", "knn"]
CouplingModel = Literal["exponential", "inverse", "gravity"]


def validate_cities(cities: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize a city table.

    Required columns are ``id``, ``name``, ``latitude``, and ``longitude``.
    ``population`` is optional and defaults to 1.0.
    """
    required = {"id", "name", "latitude", "longitude"}
    missing = sorted(required.difference(cities.columns))
    if missing:
        raise ValueError(f"Missing required city columns: {', '.join(missing)}")

    out = cities.copy()
    out["id"] = out["id"].astype(str)
    out["name"] = out["name"].astype(str)
    if out["id"].duplicated().any():
        duplicates = out.loc[out["id"].duplicated(), "id"].tolist()
        raise ValueError(f"City IDs must be unique; duplicates: {duplicates}")

    for column in ("latitude", "longitude"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    if out[["latitude", "longitude"]].isna().any().any():
        raise ValueError("Latitude and longitude must be numeric and non-null")
    if not out["latitude"].between(-90, 90).all():
        raise ValueError("Latitude must be between -90 and 90 degrees")
    if not out["longitude"].between(-180, 180).all():
        raise ValueError("Longitude must be between -180 and 180 degrees")

    if "population" not in out.columns:
        out["population"] = 1.0
    out["population"] = pd.to_numeric(out["population"], errors="coerce").fillna(1.0)
    if (out["population"] <= 0).any():
        raise ValueError("Population values must be positive")

    return out.reset_index(drop=True)


def great_circle_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return great-circle distance in kilometres using the haversine formula."""
    radius_km = 6371.009
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    return 2.0 * radius_km * math.asin(math.sqrt(a))


def great_circle_matrix(cities: pd.DataFrame) -> np.ndarray:
    """Compute the symmetric all-pairs great-circle distance matrix in km."""
    cities = validate_cities(cities)
    n = len(cities)
    matrix = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            distance = great_circle_km(
                float(cities.at[i, "latitude"]),
                float(cities.at[i, "longitude"]),
                float(cities.at[j, "latitude"]),
                float(cities.at[j, "longitude"]),
            )
            matrix[i, j] = distance
            matrix[j, i] = distance
    return matrix


def symmetrize_matrix(matrix: np.ndarray, method: Literal["mean", "min", "max"] = "mean") -> np.ndarray:
    """Symmetrize a directed cost matrix, preserving unreachable pairs as infinity."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")

    n = matrix.shape[0]
    result = np.full_like(matrix, np.inf)
    np.fill_diagonal(result, 0.0)
    for i in range(n):
        for j in range(i + 1, n):
            values = [value for value in (matrix[i, j], matrix[j, i]) if np.isfinite(value)]
            if not values:
                value = np.inf
            elif method == "mean":
                value = float(np.mean(values))
            elif method == "min":
                value = float(np.min(values))
            elif method == "max":
                value = float(np.max(values))
            else:
                raise ValueError(f"Unsupported symmetrization method: {method}")
            result[i, j] = result[j, i] = value
    return result


def coupling_from_cost(
    cost: float,
    *,
    model: CouplingModel = "exponential",
    scale: float = 50.0,
    alpha: float = 1.0,
    population_i: float = 1.0,
    population_j: float = 1.0,
    epsilon: float = 1e-9,
) -> float:
    """Convert a distance/time cost into a non-negative epidemic coupling strength."""
    if not np.isfinite(cost) or cost < 0:
        return 0.0
    if scale <= 0:
        raise ValueError("scale must be positive")
    if alpha <= 0:
        raise ValueError("alpha must be positive")

    if model == "exponential":
        return float(math.exp(-cost / scale))
    if model == "inverse":
        return float(1.0 / ((cost + epsilon) ** alpha))
    if model == "gravity":
        return float((population_i * population_j) / ((cost + epsilon) ** alpha))
    raise ValueError(f"Unsupported coupling model: {model}")


def _candidate_pairs(
    cost_matrix: np.ndarray,
    *,
    topology: Topology,
    directed: bool,
    threshold: float | None,
    k: int,
) -> set[tuple[int, int]]:
    n = cost_matrix.shape[0]
    pairs: set[tuple[int, int]] = set()

    if topology == "complete":
        iterator: Iterable[tuple[int, int]]
        if directed:
            iterator = ((i, j) for i in range(n) for j in range(n) if i != j)
        else:
            iterator = ((i, j) for i in range(n) for j in range(i + 1, n))
        return {(i, j) for i, j in iterator if np.isfinite(cost_matrix[i, j])}

    if topology == "threshold":
        if threshold is None or threshold <= 0:
            raise ValueError("A positive threshold is required for threshold topology")
        if directed:
            iterator = ((i, j) for i in range(n) for j in range(n) if i != j)
        else:
            iterator = ((i, j) for i in range(n) for j in range(i + 1, n))
        return {
            (i, j)
            for i, j in iterator
            if np.isfinite(cost_matrix[i, j]) and cost_matrix[i, j] <= threshold
        }

    if topology == "knn":
        if k < 1:
            raise ValueError("k must be at least 1")
        for i in range(n):
            candidates = [
                (float(cost_matrix[i, j]), j)
                for j in range(n)
                if i != j and np.isfinite(cost_matrix[i, j])
            ]
            for _, j in sorted(candidates)[: min(k, len(candidates))]:
                if directed:
                    pairs.add((i, j))
                else:
                    pairs.add((min(i, j), max(i, j)))
        return pairs

    raise ValueError(f"Unsupported topology: {topology}")


def build_city_graph(
    cities: pd.DataFrame,
    *,
    selected_cost_matrix: np.ndarray,
    geodesic_km_matrix: np.ndarray | None = None,
    road_distance_km_matrix: np.ndarray | None = None,
    travel_time_min_matrix: np.ndarray | None = None,
    topology: Topology = "complete",
    directed: bool = False,
    threshold: float | None = None,
    k: int = 3,
    coupling_model: CouplingModel = "exponential",
    coupling_scale: float = 50.0,
    coupling_alpha: float = 1.0,
    normalize_coupling: bool = True,
    cost_name: str = "cost",
    cost_unit: str = "km",
) -> nx.Graph | nx.DiGraph:
    """Construct the macro-level city graph from pairwise cost matrices."""
    cities = validate_cities(cities)
    selected_cost_matrix = np.asarray(selected_cost_matrix, dtype=float)
    n = len(cities)
    if selected_cost_matrix.shape != (n, n):
        raise ValueError("selected_cost_matrix shape does not match number of cities")

    graph: nx.Graph | nx.DiGraph = nx.DiGraph() if directed else nx.Graph()
    graph.graph.update(
        {
            "model": "city_macro_graph",
            "topology": topology,
            "cost_name": cost_name,
            "cost_unit": cost_unit,
            "coupling_model": coupling_model,
            "coupling_scale": float(coupling_scale),
            "coupling_alpha": float(coupling_alpha),
        }
    )

    for row in cities.itertuples(index=False):
        attributes = row._asdict()
        city_id = str(attributes.pop("id"))
        graph.add_node(city_id, **attributes)

    pairs = _candidate_pairs(
        selected_cost_matrix,
        topology=topology,
        directed=directed,
        threshold=threshold,
        k=k,
    )

    raw_couplings: list[float] = []
    staged_edges: list[tuple[str, str, dict[str, float | str]]] = []
    for i, j in sorted(pairs):
        city_i = cities.iloc[i]
        city_j = cities.iloc[j]
        cost = float(selected_cost_matrix[i, j])
        raw = coupling_from_cost(
            cost,
            model=coupling_model,
            scale=coupling_scale,
            alpha=coupling_alpha,
            population_i=float(city_i["population"]),
            population_j=float(city_j["population"]),
        )
        attributes: dict[str, float | str] = {
            "cost": cost,
            "cost_name": cost_name,
            "cost_unit": cost_unit,
            "coupling_raw": raw,
        }
        if geodesic_km_matrix is not None:
            attributes["geodesic_km"] = float(geodesic_km_matrix[i, j])
        if road_distance_km_matrix is not None:
            attributes["road_distance_km"] = float(road_distance_km_matrix[i, j])
        if travel_time_min_matrix is not None:
            attributes["travel_time_min"] = float(travel_time_min_matrix[i, j])

        staged_edges.append((str(city_i["id"]), str(city_j["id"]), attributes))
        raw_couplings.append(raw)

    maximum = max(raw_couplings, default=0.0)
    for source, target, attributes in staged_edges:
        raw = float(attributes["coupling_raw"])
        attributes["coupling"] = raw / maximum if normalize_coupling and maximum > 0 else raw
        graph.add_edge(source, target, **attributes)

    return graph


def graph_matrix(
    graph: nx.Graph | nx.DiGraph,
    *,
    node_order: list[str],
    attribute: str,
    fill_value: float = 0.0,
) -> np.ndarray:
    """Extract an edge-attribute matrix in a specified node order."""
    index = {node: i for i, node in enumerate(node_order)}
    matrix = np.full((len(node_order), len(node_order)), fill_value, dtype=float)
    for source, target, data in graph.edges(data=True):
        value = float(data.get(attribute, fill_value))
        i, j = index[str(source)], index[str(target)]
        matrix[i, j] = value
        if not graph.is_directed():
            matrix[j, i] = value
    return matrix


def row_stochastic_matrix(matrix: np.ndarray) -> np.ndarray:
    """Normalize a non-negative matrix row-wise; isolated rows remain zero."""
    matrix = np.asarray(matrix, dtype=float)
    if np.any(matrix < 0):
        raise ValueError("matrix must be non-negative")
    row_sums = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums > 0)
