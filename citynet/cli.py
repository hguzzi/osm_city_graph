from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .core import build_city_graph, great_circle_matrix, validate_cities
from .io import export_results
from .osm_backend import geocode_missing_cities, load_or_download_road_graph, road_matrices

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a weighted city graph from OpenStreetMap roads or geodesic distances."
    )
    parser.add_argument("--cities", type=Path, required=True, help="Input CSV file")
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--mode", choices=["road", "geodesic"], default="road")
    parser.add_argument("--metric", choices=["distance", "travel_time"], default="distance")
    parser.add_argument("--network-type", default="drive")
    parser.add_argument("--padding-km", type=float, default=5.0)
    parser.add_argument("--fallback-speed-kph", type=float, default=40.0)
    parser.add_argument("--refresh-road-cache", action="store_true")
    parser.add_argument("--directed", action="store_true")
    parser.add_argument("--symmetrize", choices=["mean", "min", "max"], default="mean")
    parser.add_argument("--topology", choices=["complete", "threshold", "knn"], default="knn")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument(
        "--coupling-model", choices=["exponential", "inverse", "gravity"], default="exponential"
    )
    parser.add_argument("--coupling-scale", type=float, default=50.0)
    parser.add_argument("--coupling-alpha", type=float, default=1.0)
    parser.add_argument("--no-normalize-coupling", action="store_true")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING"], default="INFO")
    return parser


def run(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    raw_cities = pd.read_csv(args.cities)
    has_coordinates = (
        {"latitude", "longitude"}.issubset(raw_cities.columns)
        and raw_cities[["latitude", "longitude"]].notna().all().all()
    )
    cities = validate_cities(raw_cities) if has_coordinates else geocode_missing_cities(raw_cities)
    geodesic = great_circle_matrix(cities)

    road_distance: np.ndarray | None = None
    travel_time: np.ndarray | None = None
    city_nodes: np.ndarray | None = None

    if args.mode == "road":
        cache_path = args.output / "road_network.graphml"
        road_graph = load_or_download_road_graph(
            cities,
            cache_path=cache_path,
            network_type=args.network_type,
            padding_km=args.padding_km,
            refresh=args.refresh_road_cache,
            fallback_speed_kph=args.fallback_speed_kph,
        )
        city_nodes, road_distance, travel_time = road_matrices(
            cities,
            road_graph,
            directed=args.directed,
            symmetrize=args.symmetrize,
        )
        cities = cities.copy()
        cities["osm_road_node"] = city_nodes.astype(str)
        if args.metric == "distance":
            selected = road_distance
            cost_name, cost_unit = "road_distance", "km"
        else:
            selected = travel_time
            cost_name, cost_unit = "travel_time", "min"
    else:
        if args.metric != "distance":
            raise ValueError("Geodesic mode supports only --metric distance")
        selected = geodesic
        cost_name, cost_unit = "geodesic_distance", "km"

    graph = build_city_graph(
        cities,
        selected_cost_matrix=selected,
        geodesic_km_matrix=geodesic,
        road_distance_km_matrix=road_distance,
        travel_time_min_matrix=travel_time,
        topology=args.topology,
        directed=args.directed,
        threshold=args.threshold,
        k=args.k,
        coupling_model=args.coupling_model,
        coupling_scale=args.coupling_scale,
        coupling_alpha=args.coupling_alpha,
        normalize_coupling=not args.no_normalize_coupling,
        cost_name=cost_name,
        cost_unit=cost_unit,
    )

    metadata = {
        "mode": args.mode,
        "metric": args.metric,
        "network_type": args.network_type if args.mode == "road" else None,
        "directed": args.directed,
        "symmetrize": args.symmetrize if not args.directed else None,
        "topology": args.topology,
        "threshold": args.threshold,
        "k": args.k,
        "coupling_model": args.coupling_model,
        "coupling_scale": args.coupling_scale,
        "coupling_alpha": args.coupling_alpha,
        "number_of_cities": int(graph.number_of_nodes()),
        "number_of_edges": int(graph.number_of_edges()),
    }
    export_results(
        output_dir=args.output,
        cities=cities,
        graph=graph,
        selected_cost_matrix=selected,
        geodesic_km_matrix=geodesic,
        road_distance_km_matrix=road_distance,
        travel_time_min_matrix=travel_time,
        metadata=metadata,
    )
    LOGGER.info("Created %s nodes and %s edges", graph.number_of_nodes(), graph.number_of_edges())
    LOGGER.info("Results written to %s", args.output.resolve())


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        run(args)
    except Exception as exc:
        parser.exit(status=1, message=f"Error: {exc}\n")


if __name__ == "__main__":
    main()
