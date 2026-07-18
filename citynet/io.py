from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from .core import graph_matrix, row_stochastic_matrix


def matrix_to_frame(matrix: np.ndarray, city_ids: list[str]) -> pd.DataFrame:
    return pd.DataFrame(matrix, index=city_ids, columns=city_ids)


def edge_table(graph: nx.Graph | nx.DiGraph) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for source, target, data in graph.edges(data=True):
        rows.append({"source": source, "target": target, **data})
    return pd.DataFrame(rows)


def plot_city_graph(graph: nx.Graph | nx.DiGraph, output_path: Path) -> None:
    """Plot the macro graph in geographic coordinates."""
    fig, ax = plt.subplots(figsize=(10, 8))
    maximum = max((float(data.get("coupling", 0.0)) for _, _, data in graph.edges(data=True)), default=1.0)
    maximum = maximum or 1.0

    for source, target, data in graph.edges(data=True):
        x = [float(graph.nodes[source]["longitude"]), float(graph.nodes[target]["longitude"])]
        y = [float(graph.nodes[source]["latitude"]), float(graph.nodes[target]["latitude"])]
        width = 0.5 + 4.0 * float(data.get("coupling", 0.0)) / maximum
        ax.plot(x, y, linewidth=width, alpha=0.55)

    x_values = [float(data["longitude"]) for _, data in graph.nodes(data=True)]
    y_values = [float(data["latitude"]) for _, data in graph.nodes(data=True)]
    ax.scatter(x_values, y_values, s=80, zorder=3)
    for node, data in graph.nodes(data=True):
        ax.annotate(
            str(data.get("name", node)),
            (float(data["longitude"]), float(data["latitude"])),
            xytext=(4, 4),
            textcoords="offset points",
        )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("City-level mobility graph")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def export_results(
    *,
    output_dir: Path,
    cities: pd.DataFrame,
    graph: nx.Graph | nx.DiGraph,
    selected_cost_matrix: np.ndarray,
    geodesic_km_matrix: np.ndarray,
    road_distance_km_matrix: np.ndarray | None,
    travel_time_min_matrix: np.ndarray | None,
    metadata: dict[str, object],
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    city_ids = cities["id"].astype(str).tolist()

    cities.to_csv(output_dir / "cities_geocoded.csv", index=False)
    edge_table(graph).to_csv(output_dir / "city_edges.csv", index=False)
    matrix_to_frame(selected_cost_matrix, city_ids).to_csv(output_dir / "selected_cost_matrix.csv")
    matrix_to_frame(geodesic_km_matrix, city_ids).to_csv(output_dir / "geodesic_km_matrix.csv")
    if road_distance_km_matrix is not None:
        matrix_to_frame(road_distance_km_matrix, city_ids).to_csv(
            output_dir / "road_distance_km_matrix.csv"
        )
    if travel_time_min_matrix is not None:
        matrix_to_frame(travel_time_min_matrix, city_ids).to_csv(
            output_dir / "travel_time_min_matrix.csv"
        )

    coupling = graph_matrix(graph, node_order=city_ids, attribute="coupling")
    transition = row_stochastic_matrix(coupling)
    matrix_to_frame(coupling, city_ids).to_csv(output_dir / "coupling_matrix.csv")
    matrix_to_frame(transition, city_ids).to_csv(output_dir / "transition_matrix.csv")

    nx.write_graphml(graph, output_dir / "city_graph.graphml")
    plot_city_graph(graph, output_dir / "city_graph.png")
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
