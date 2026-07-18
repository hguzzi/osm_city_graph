# OpenStreetMap City Graph

This package constructs the **macro-level graph** for a hierarchical epidemic model:

- each node is a city;
- node attributes include name, coordinates, and population;
- edge costs are road distance, free-flow travel time, or geodesic distance;
- edge coupling is a transformed interaction strength for inter-city diffusion;
- outputs include GraphML, edge tables, distance matrices, a coupling matrix, and a row-stochastic transition matrix.

## 1. Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Alternatively, install the local package:

```bash
pip install -e .
```

## 2. Input file

The CSV must contain `id` and `name`. It can either contain coordinates:

```csv
id,name,latitude,longitude,population
city_a,City A,38.90,16.58,25000
```

or a geocoding query:

```csv
id,name,query,population
catanzaro,Catanzaro,"Catanzaro, Calabria, Italy",1
```

When coordinates are absent, the program geocodes each query through OSMnx/Nominatim and saves the resolved coordinates in the output folder.

## 3. Fast test without downloading roads

```bash
python -m citynet.cli \
  --cities examples/cities_coordinates.csv \
  --output outputs_geodesic \
  --mode geodesic \
  --topology knn \
  --k 2 \
  --coupling-model exponential \
  --coupling-scale 30
```

## 4. OpenStreetMap road-distance graph

```bash
python -m citynet.cli \
  --cities examples/cities_names.csv \
  --output outputs_road \
  --mode road \
  --metric distance \
  --network-type drive \
  --topology knn \
  --k 2 \
  --coupling-model exponential \
  --coupling-scale 50
```

The first run downloads and caches `road_network.graphml`. Later runs reuse it unless `--refresh-road-cache` is supplied.

## 5. Travel-time graph

```bash
python -m citynet.cli \
  --cities examples/cities_names.csv \
  --output outputs_time \
  --mode road \
  --metric travel_time \
  --topology threshold \
  --threshold 45 \
  --coupling-scale 30
```

In this example, the threshold and coupling scale are expressed in **minutes** because the selected metric is travel time.

## 6. Directed versus undirected graphs

Road graphs are intrinsically directed because of one-way streets. The default city graph is undirected and averages the two directional route costs. Use:

```bash
--directed
```

to retain asymmetric city-to-city costs. For an undirected graph, `--symmetrize` accepts `mean`, `min`, or `max`.

## 7. Topology choices

- `--topology complete`: connect every reachable city pair.
- `--topology threshold --threshold R`: retain pairs whose selected cost is at most `R`.
- `--topology knn --k K`: connect each city to its `K` closest reachable cities. In the undirected case, the union of neighbourhoods is used.

## 8. Coupling functions

Let `c_ij` denote road distance, travel time, or geodesic distance.

### Exponential kernel

```text
coupling_ij = exp(-c_ij / scale)
```

### Inverse kernel

```text
coupling_ij = 1 / (c_ij + epsilon)^alpha
```

### Gravity kernel

```text
coupling_ij = population_i * population_j / (c_ij + epsilon)^alpha
```

By default, edge couplings are divided by the maximum coupling so they lie in `[0, 1]`. The exported `transition_matrix.csv` additionally normalizes each row to sum to one and can be used as a movement kernel in the city-level SEIRDQ model.

## 9. Outputs

Each run creates:

- `cities_geocoded.csv`: normalized city table and snapped OSM node IDs;
- `city_edges.csv`: macro-edge list and weights;
- `city_graph.graphml`: city graph for NetworkX, Gephi, or Cytoscape;
- `selected_cost_matrix.csv`: matrix used for topology/coupling;
- `geodesic_km_matrix.csv`;
- `road_distance_km_matrix.csv` when road mode is used;
- `travel_time_min_matrix.csv` when road mode is used;
- `coupling_matrix.csv`;
- `transition_matrix.csv`;
- `city_graph.png`;
- `metadata.json`;
- `road_network.graphml`: cached OSM road graph.

## 10. Use in the hierarchical SEIRDQ model

If `P = transition_matrix` and `X_i(t)` is a compartment count or proportion in city `i`, a simple inter-city mobility term is:

```text
movement_i = mobility_rate * sum_j(P[j, i] * X_j - P[i, j] * X_i)
```

The movement term can be applied separately to `S`, `E`, `I`, `R`, or `Q`, with compartment-specific rates. For example, quarantined individuals can be assigned a zero or strongly reduced mobility rate.

## 11. Tests

The tests do not call OpenStreetMap services:

```bash
python -m unittest discover -s tests -v
```

## Practical notes

A single bounding-box OSM request is best for cities in the same local or regional study area. A list spanning a country can create a very large road graph; in that case, process the study area by regions or use a dedicated routing engine. Free-flow travel times depend on OSM speed tags and imputation, so they should be interpreted as model inputs rather than observed traffic times.

## Python-version compatibility

This release supports Python 3.10 and newer.

- Python 3.10 installs OSMnx 2.0.7 automatically.
- Python 3.11 or newer installs OSMnx 2.1 or newer.

Check the interpreter used by pip:

```bash
python --version
python -m pip --version
```

Always install with `python -m pip` so that pip targets the same interpreter:

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

If an older virtual environment contains a partially resolved installation,
remove it and create a fresh environment before installing again.
