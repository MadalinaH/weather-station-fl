# Federated Learning for Finnish Weather Temperature Prediction

> A federated learning system that trains collaborative regression models across 15 Finnish weather stations to predict next-day mean temperature, comparing two graph construction strategies using the GTVMin algorithm.

---

## What This Is

This project implements a **Federated Learning (FL)** pipeline over 15 Finnish Meteorological Institute (FMI) weather stations. Instead of pooling all data into one central model, each station trains its own local model - but stations *collaborate* by sharing model weights with their geographic neighbours each round.

The core algorithm is **GTVMin** (Graph Total Variation Minimisation), solved via closed-form Ridge regression. Two FL systems are compared, differing only in how the collaboration graph is constructed:

- **System A** - edges weighted by geographic distance only
- **System B** - edges weighted by distance × Pearson temperature correlation

Three experiments with different training set sizes test whether FL benefit changes as local data becomes scarce.

**Key finding:** FL provides a small but consistent improvement over the local baseline on reduced training data (Reduced experiment: −0.008 RMSE). With abundant data (Full experiment) the graph regularisation adds nothing - the joint search selects near-zero alpha, effectively switching off collaboration. System A and System B perform identically throughout, since all nearby station pairs have strongly positive Pearson correlation (multiplier ≈ 1.0).

---

## Project Structure

```
fl_project/
│
├── standardise_data.py     Phase 1 - reads downloaded FMI CSVs,
│                           aggregates hourly → daily, saves clean station CSVs
│
├── build_network.py        Phase 2 - builds two 15×15 adjacency matrices
│                           (distance graph and correlation graph) and plots them
│
├── prepare_data.py         Phase 3 - splits each station's data into
│                           train/val/test, normalises features, saves pickle
│
├── fl_algorithm.py         Phase 4 - GTVMin federated learning algorithm
│                           using Ridge regression (the core FL logic)
│
├── evaluate.py             Phase 5 - computes MSE and RMSE for any
│                           set of weights on any data split
│
├── run_experiment.py       Phase 6 - orchestrates everything: hyperparameter
│                           search, three experiments, all figures and CSVs
│
├── graph_sensitivity.py    Analyses how graph connectivity changes as d_max
│                           is varied from 300 km down to 150 km
│
├── hdd_analysis.py         Post-processing - converts predicted temperatures
│                           to Heating Degree Days (HDD) using the FMI standard
│                           and compares against actual HDD on the test set
│
├── data/
│   ├── helsinki.csv            Clean daily CSVs, one per station
│   ├── turku.csv               columns: date, t2m, tmin, tmax, ws_10min
│   ├── oulu.csv
│   ├── tampere.csv
│   ├── jyvaskyla.csv
│   ├── kuopio.csv
│   ├── rovaniemi.csv
│   ├── sodankyla.csv
│   ├── inari.csv
│   ├── hanko.csv
│   ├── kajaani.csv
│   ├── kittila.csv
│   ├── muonio.csv
│   ├── pelkosenniemi.csv
│   ├── raahe.csv
│   ├── missing_summary.csv         NaN counts per station per variable
│   ├── adj_system_a.npy            15×15 distance-based adjacency matrix
│   ├── adj_system_b.npy            15×15 correlation-weighted adjacency matrix
│   └── prepared_data.pkl           Train/val/test splits for all 15 stations
│
├── results/
│   ├── network_system_a.png                Map of System A collaboration graph
│   ├── network_system_b.png                Map of System B collaboration graph
│   ├── graph_sensitivity.png               Connectivity vs d_max for all 4 thresholds
│   ├── convergence_exp1.png                Training MSE vs FL round (Exp 1)
│   ├── test_rmse_by_station_exp3.png       Per-station bar chart, Experiment 3
│   ├── test_rmse_by_experiment.png         RMSE vs training size, all systems
│   ├── sigma_heatmap_exp3.png              Val RMSE over (alpha, sigma) grid, Exp 3
│   ├── experiment1_full_data.csv           Per-station detailed results
│   ├── experiment2_reduced_data.csv
│   ├── experiment3_minimal_data.csv
│   ├── combined_summary.csv               Compact summary of all experiments
│   ├── hdd_analysis.csv                   HDD metrics per station per system
│   └── hdd_per_station.png                Grouped bar chart of mean daily HDD
│
├── requirements.txt        Pinned Python dependencies
└── .venv/                  Virtual environment
```

---

## Getting Started

### Prerequisites

- Python 3.9+
- The 15 station CSV files already downloaded in `data/` (see [Data](#data) below)

### Setup

```bash
# From the fl_project/ directory
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

### Run the full pipeline

```bash
# Activate the venv
source .venv/bin/activate

# Phase 1 - standardise raw CSVs into daily format
python standardise_data.py

# Phase 2 - build collaboration graphs
python build_network.py

# Phase 3 - prepare train/val/test splits
python prepare_data.py

# Phase 4 - quick self-test of the FL algorithm
python fl_algorithm.py

# Phase 6 - run all three experiments and generate all figures
python run_experiment.py

# Optional - graph connectivity sensitivity analysis
python graph_sensitivity.py
```

Each script is self-contained and prints progress to the console.

---

## Data

### Source

Data was downloaded from the **FMI open data portal**:
[https://en.ilmatieteenlaitos.fi/download-observations](https://en.ilmatieteenlaitos.fi/download-observations)

### Download settings used

| Setting | Value |
|---|---|
| Tab | Hourly observations |
| Variables | Average temperature, Maximum temperature, Minimum temperature |
| Time period | 01/01/2022 - 31/12/2024 |
| Time zone | UTC |
| Format | CSV |

One file was downloaded per station and saved to `data/`.

### The 15 stations

| Station | Coordinates | Region |
|---|---|---|
| Helsinki Kaisaniemi | 60.1756°N, 24.9414°E | South |
| Hanko Tulliniemi | 59.8171°N, 22.9083°E | South |
| Turku Artukainen | 60.5149°N, 22.2663°E | South-West |
| Tampere Härmälä | 61.4940°N, 23.7700°E | South-West |
| Jyväskylä Airport | 62.3996°N, 25.6787°E | Central |
| Kuopio Maaninka | 63.1484°N, 27.3084°E | Central-East |
| Raahe Nahkiainen | 64.6736°N, 24.5603°E | West |
| Kajaani Airport | 64.2853°N, 27.6924°E | Central-North |
| Oulu Vihreäsaari | 65.0090°N, 25.3960°E | North-West |
| Rovaniemi Apukka | 66.5600°N, 26.0100°E | Lapland |
| Pelkosenniemi Pyhätunturi | 67.0239°N, 27.2201°E | Lapland |
| Sodankylä | 67.3668°N, 26.6500°E | Lapland |
| Kittilä Airport | 67.7014°N, 24.8467°E | Lapland |
| Muonio Oustajärvi | 67.9624°N, 23.6824°E | Lapland |
| Inari Saariselkä | 68.4200°N, 27.4100°E | Lapland |

### What `standardise_data.py` does to the raw files

The raw CSVs have hourly rows with FMI's column headers. The script:
1. Parses Year/Month/Day/Time into a proper UTC datetime
2. Aggregates hourly → daily:
   - `t2m` = mean(Average temperature) - daily mean [°C]
   - `tmin` = min(Minimum temperature) - daily minimum [°C]
   - `tmax` = max(Maximum temperature) - daily maximum [°C]
   - `ws_10min` = mean(Wind speed) - daily mean wind speed [m/s]
3. Reindexes to cover every calendar day from 2022-01-01 to 2024-12-31
4. Overwrites each raw CSV with the clean daily version

### Known data quality issues

| Station | Issue |
|---|---|
| Tampere Härmälä | `ws_10min` entirely absent - no anemometer at this station |
| Inari Saariselkä | `ws_10min` entirely absent - same reason |
| Hanko Tulliniemi | `ws_10min` entirely absent + 6 missing `t2m` days |
| Kajaani Airport | `ws_10min` entirely absent |
| Kittilä Airport | `ws_10min` entirely absent |
| Muonio Oustajärvi | `ws_10min` entirely absent |
| Pelkosenniemi Pyhätunturi | `ws_10min` entirely absent |
| Raahe Nahkiainen | `ws_10min` entirely absent |
| Turku Artukainen | 22 days of missing `ws_10min` |
| Jyväskylä Airport | 1 day missing across all variables |
| Sodankylä | 2 days missing across all variables |

Because 8 of 15 stations lack wind speed measurements entirely, **`ws_10min` is excluded from the feature set**. All 15 FL clients must have identical feature dimensions for GTVMin to work. The final feature set has 8 dimensions - see `prepare_data.py`.

---

## How the Code Works

### The prediction task

For each station and each day `t`:
- **Features** `X[t]` = `[t2m[t], tmin[t], tmax[t], t2m_lag1[t], t2m_lag2[t], delta_t[t], sin_day[t], cos_day[t]]` - 8 features
- **Label** `y[t]` = `t2m[t+1]` - tomorrow's mean temperature

| Feature | Description |
|---|---|
| `t2m` | Today's daily mean temperature [°C] |
| `tmin` | Today's daily minimum temperature [°C] |
| `tmax` | Today's daily maximum temperature [°C] |
| `t2m_lag1` | Mean temperature 1 day ago [°C] |
| `t2m_lag2` | Mean temperature 2 days ago [°C] |
| `delta_t` | Day-over-day change: `t2m[t] − t2m[t−1]` [°C] |
| `sin_day` | `sin(2π × day_of_year / 365)` - seasonal cycle |
| `cos_day` | `cos(2π × day_of_year / 365)` - seasonal cycle |

The first 2 rows per station are dropped to remove NaNs introduced by `t2m_lag2`. This is a one-step-ahead time series forecasting problem solved with linear regression.

### The FL algorithm (GTVMin)

Each station `i` maintains a weight vector `w[i]` of shape `(8,)`. Prediction is:

```
ŷ[t] = X[t] @ w[i]
```

Each round, station `i`'s model is pulled toward its neighbours' models. One round:

```
1. s_i     = Σ A[i,i']                          (neighbourhood strength)
2. θ_i     = (1/s_i) Σ A[i,i'] * w[i']          (neighbour centroid)
3. ỹ       = y_train - X_train @ θ_i             (modified labels)
4. solve   Ridge(alpha = α * s_i).fit(X_train, ỹ) → u*
5. w[i]    = u* + θ_i                            (recover true weights)
```

All stations compute their new `w[i]` from the **previous round's weights** (synchronous update). The Ridge penalty `α * s_i` scales with neighbourhood strength - stations with more/stronger connections are pulled harder toward the consensus.

### Why Ridge, not gradient descent?

GTVMin has a closed-form solution. Ridge regression finds it exactly in one solve per station per round, with no learning rate to tune and guaranteed convergence.

### The two collaboration graphs

**System A - Distance graph**

```
A[i,j] = exp(-d²  / (2 × sigma²))   if d ≤ d_max
        = 0                           otherwise
```

Pure geography: stations within d_max = 200 km collaborate, weighted by Gaussian decay.

**System B - Correlation graph**

```
A[i,j] = max(0, pearson_corr(T_i, T_j)) × exp(-d² / (2 × sigma²))   if d ≤ d_max
        = 0                                                             otherwise
```

Same distance gate, but edge weight is also scaled by how strongly the two stations' temperatures co-vary. Computed on **training data only** to avoid data leakage - and recomputed for each experiment's training period.

### The three experiments

| Experiment | Training period | Val period | Test period | Train days |
|---|---|---|---|---|
| 1 - Full data | 2022-01-01 → 2023-12-31 | 2024 H1 | 2024 H2 | 728 |
| 2 - Reduced data | 2022-01-01 → 2022-12-31 | 2023 H1 | 2023 H2 | 364 |
| 3 - Minimal data | 2022-01-01 → 2022-06-30 | 2022 Q3 | 2022 Q4 | 181 |

For each experiment, a joint grid search over alpha × sigma (12 × 7 = 84 combinations) is run on the validation set.

Alpha candidates: `[0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500]`  
Sigma candidates: `[50, 75, 100, 125, 150, 175, 200]` km

---

## Results

### Combined summary (mean RMSE across all 15 stations)

| Experiment | System | Train RMSE | Val RMSE | Test RMSE | Best α | Best σ |
|---|---|---|---|---|---|---|
| Full data | Baseline | 4.89 °C | 5.25 °C | 4.96 °C | 50.0 | - |
| Full data | System A | 4.90 °C | 5.25 °C | 4.96 °C | 100.0 | 50 km |
| Full data | System B | 4.90 °C | 5.25 °C | **4.96 °C** | 100.0 | 50 km |
| Reduced data | Baseline | 5.02 °C | 5.27 °C | 4.86 °C | 100.0 | - |
| Reduced data | System A | 5.03 °C | 5.26 °C | **4.85 °C** | 50.0 | 50 km |
| Reduced data | System B | 5.03 °C | 5.26 °C | **4.85 °C** | 50.0 | 50 km |
| Minimal data | Baseline | 3.95 °C | 2.65 °C | 3.24 °C | 0.5 | - |
| Minimal data | System A | 3.98 °C | 2.63 °C | 3.26 °C | 50.0 | 200 km |
| Minimal data | System B | 3.98 °C | 2.63 °C | 3.26 °C | 50.0 | 200 km |

### Interpretation

**Experiment 1 (Full data - 2 years):** FL makes no meaningful difference. With two years of training data each station fits a good local model. The joint search selects a large alpha (100) and narrow sigma (50 km) - near-minimal collaboration.

**Experiment 2 (Reduced data - 1 year):** FL provides a small consistent improvement (−0.008 RMSE on test). The graph regularisation acts as a useful prior when local data is limited.

**Experiment 3 (Minimal data - 6 months):** FL improves validation RMSE (2.63 °C vs 2.65 °C) but does not transfer to the test set. The very short training period makes the models fragile and the selected alpha = 50 / sigma = 200 km pushes toward strong collaboration, which does not generalise.

**Sigma tuning:** The joint (alpha, sigma) search consistently selects sigma = 50 km for Experiments 1 and 2, confirming that collaboration is unhelpful with abundant data. The heatmap (`results/sigma_heatmap_exp3.png`) shows performance is broadly flat across the (alpha, sigma) grid - the task is not highly sensitive to sigma within the tested range.

**System A vs System B:** Identical across all experiments and sigma values. All station pairs within 200 km have strongly positive Pearson correlation (r ≈ 0.9–1.0), so the System B correlation multiplier ≈ 1.0 everywhere. The geographic proximity filter at d_max = 200 km already ensures only climatically similar stations are connected.

### Graph connectivity

The graph sensitivity analysis (`graph_sensitivity.py`) tested d_max ∈ {300, 250, 200, 150} km with sigma = d_max / 2:

| d_max | Edges | Min degree | Isolated? |
|---|---|---|---|
| 300 km | 41/105 | 3 | ✅ None |
| 250 km | 33/105 | 3 | ✅ None |
| **200 km** | **27/105** | **2** | **✅ None** |
| 150 km | 18/105 | 1 | ✅ None |

d_max = 200 km was chosen: every station has at least 2 neighbours (a comfortable connectivity margin) while the graph remains sparse enough to reflect genuine geographic proximity. At 150 km, Helsinki and Raahe would each have only 1 neighbour, making them vulnerable to a single noisy connection.

### Heating Degree Days (HDD)

HDD quantifies daily heating energy demand. Following the FMI standard (indoor baseline 17 °C, heating threshold 12 °C):

```
HDD = max(17 - T, 0)   if T < 12 °C
HDD = 0                 otherwise
```

Results are reported for Experiment 1 (full data) test set (2024 H2). Northern stations (Muonio, Inari, Kittilä) show the highest actual HDD (~12 °C·days/day), while southern stations (Hanko, Helsinki) average ~6 °C·days/day. All systems overestimate HDD at southern stations and are well-calibrated at northern ones. Results are in `results/hdd_analysis.csv` and `results/hdd_per_station.png`.

---

## Output Files Reference

### `data/`

| File | Description |
|---|---|
| `<station>.csv` | Clean daily CSV with columns `[date, t2m, tmin, tmax, ws_10min]` |
| `missing_summary.csv` | NaN counts per station per variable |
| `adj_system_a.npy` | 15×15 numpy array - System A adjacency matrix |
| `adj_system_b.npy` | 15×15 numpy array - System B adjacency matrix |
| `prepared_data.pkl` | Pickle dict: `{station: {X_train, y_train, X_val, y_val, X_test, y_test}}` |

### `results/`

| File | Description |
|---|---|
| `network_system_a.png` | Map of Finland showing System A graph edges |
| `network_system_b.png` | Map of Finland showing System B graph edges |
| `graph_sensitivity.png` | Network maps at d_max = 300/250/200/150 km |
| `convergence_exp1.png` | Training MSE vs FL round for Experiment 1 |
| `test_rmse_by_station_exp3.png` | Per-station test RMSE bar chart, Experiment 3 |
| `test_rmse_by_experiment.png` | Mean test RMSE vs training set size (all systems) |
| `sigma_heatmap_exp3.png` | Heatmap of val RMSE over (alpha, sigma) grid - Experiment 3, System A |
| `experiment{1,2,3}_*.csv` | Per-station train/val/test MSE and RMSE for each system |
| `combined_summary.csv` | Compact summary: all experiments × all systems |
| `hdd_analysis.csv` | Mean daily HDD and HDD MAE per station per system (Experiment 1 test set) |
| `hdd_per_station.png` | Grouped bar chart: Actual vs predicted mean daily HDD per station |

---

## Architecture

```
FMI Portal
        │
        ▼
standardise_data.py  ──► data/<station>.csv  (15 files, daily)
        │
        ├──────────────────────────────────────────────────────►  build_network.py
        │                                                               │
        │                                               adj_system_a.npy
        │                                               adj_system_b.npy
        ▼
prepare_data.py  ──► data/prepared_data.pkl
        │
        ▼
run_experiment.py
  ├── prepare_experiment_data()    re-splits data per experiment
  ├── build_system_b_for_experiment()   recomputes B graph on training data
  ├── hyperparameter_search()
  │       └── fl_algorithm.run_fl()  ◄── fl_algorithm.py
  ├── run_baseline()
  ├── evaluate()  ◄── evaluate.py
  └── save figures + CSVs  ──► results/

graph_sensitivity.py  (standalone - analyses connectivity vs d_max)
hdd_analysis.py       (called from run_experiment.py after Experiment 1)
```

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `numpy` | 2.0.2 | Array operations, adjacency matrices |
| `pandas` | 2.3.3 | CSV loading, date handling, aggregation |
| `scikit-learn` | 1.6.1 | Ridge regression |
| `matplotlib` | 3.9.4 | All figures |

Install all with:
```bash
pip install -r requirements.txt
```

---

## Design Decisions

**Why drop `ws_10min` from features?**
8 of 15 stations have no wind speed measurements. All FL clients must share identical feature dimensionality for the GTVMin weight vectors to be comparable and combinable. Dropping `ws_10min` was the cleanest solution.

**Why not normalise labels?**
Labels (`y = t2m[t+1]`) are kept in raw °C so that MSE and RMSE are directly interpretable in meaningful units.

**Why per-station normalisation?**
In a real federated system, clients never share raw data. Each station's scaler is fit on that station's training data only, mirroring this constraint.

**Why d_max = 200 km?**
The graph sensitivity analysis confirmed that all 15 stations remain connected at d_max = 150 km (the strictest viable threshold). d_max = 200 km was chosen to give each station at least 2 neighbours, providing a more robust collaboration structure while keeping the graph sparse enough to reflect genuine geographic proximity.
