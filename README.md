# Federated Learning for Finnish Weather Temperature Prediction

> A federated learning system that trains collaborative regression models across 9 Finnish weather stations to predict next-day mean temperature, comparing two graph construction strategies using the GTVMin algorithm.

---

## What This Is

This project implements a **Federated Learning (FL)** pipeline over 9 Finnish Meteorological Institute (FMI) weather stations. Instead of pooling all data into one central model, each station trains its own local model - but stations *collaborate* by sharing model weights with their geographic neighbours each round.

The core algorithm is **GTVMin** (Graph Total Variation Minimisation), solved via closed-form Ridge regression. Two FL systems are compared, differing only in how the collaboration graph is constructed:

- **System A** - edges weighted by geographic distance only
- **System B** - edges weighted by distance × Pearson temperature correlation

Three experiments with different training set sizes test whether FL benefit changes as local data becomes scarce.

**Key finding:** With the current graph (d_max = 200 km, 8 edges), FL does not meaningfully outperform the local baseline in any experiment. A graph sensitivity analysis showed that widening d_max to 300 km (15 edges) produced a marginal improvement in the minimal-data setting (~0.02 °C), suggesting collaboration helps only when the graph is dense enough to carry sufficient signal.

---

## Project Structure

```
fl_project/
│
├── standardise_data.py     Phase 1 - reads downloaded FMI CSVs,
│                           aggregates hourly → daily, saves clean station CSVs
│
├── build_network.py        Phase 2 - builds two 9×9 adjacency matrices
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
├── hdd_analysis.py         Post-processing - converts predicted temperatures
│                           to Heating Degree Days (HDD) using the FMI standard
│                           and compares against actual HDD on the test set
│
├── data/
│   ├── helsinki_kaisaniemi.csv     Clean daily CSVs, one per station
│   ├── turku_artukainen.csv        columns: date, t2m, tmin, tmax, ws_10min
│   ├── oulu_vihreasaari.csv
│   ├── tampere_harmala.csv
│   ├── jyvaskyla_airport.csv
│   ├── kuopio_maaninka.csv
│   ├── rovaniemi_apukka.csv
│   ├── sodankyla.csv
│   ├── inari_saariselka.csv
│   ├── missing_summary.csv         NaN counts per station per variable
│   ├── adj_system_a.npy            9×9 distance-based adjacency matrix
│   ├── adj_system_b.npy            9×9 correlation-weighted adjacency matrix
│   └── prepared_data.pkl           Train/val/test splits for all 9 stations
│
├── results/
│   ├── network_system_a.png                Map of System A collaboration graph
│   ├── network_system_b.png                Map of System B collaboration graph
│   ├── convergence_exp1.png                Training MSE vs FL round (Exp 1)
│   ├── test_rmse_by_station_exp1.png       Per-station bar chart, Experiment 1
│   ├── test_rmse_by_station_exp2.png       Per-station bar chart, Experiment 2
│   ├── test_rmse_by_station_exp3.png       Per-station bar chart, Experiment 3
│   ├── test_rmse_by_experiment.png         RMSE vs training size, all systems
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
- The 9 station CSV files already downloaded in `data/` (see [Data](#data) below)

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
| Variables | Average temperature, Maximum temperature, Minimum temperature, Wind speed |
| Time period | 01/01/2022 - 31/12/2024 |
| Time zone | UTC |
| Format | CSV |

One file was downloaded per station and saved to `data/` with the filenames listed in the project structure above.

### The 9 stations

| Station | Coordinates | fmisid |
|---|---|---|
| Helsinki Kaisaniemi | 60.1756°N, 24.9414°E | 100971 |
| Turku Artukainen | 60.5149°N, 22.2663°E | 100949 |
| Oulu Vihreäsaari | 65.0090°N, 25.3960°E | 101794 |
| Tampere Härmälä | 61.4940°N, 23.7700°E | 101124 |
| Jyväskylä Airport | 62.3996°N, 25.6787°E | 137208 |
| Kuopio Maaninka | 63.1484°N, 27.3084°E | 101572 |
| Rovaniemi Apukka | 66.5600°N, 26.0100°E | 101933 |
| Sodankylä | 67.3668°N, 26.6500°E | 101932 |
| Inari Saariselkä | 68.4200°N, 27.4100°E | 102005 |

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
| Turku Artukainen | 22 days of missing `ws_10min` |
| Jyväskylä Airport | 1 day missing across all variables |
| Sodankylä | 2 days missing across all variables |

Because two stations lack wind speed entirely, **`ws_10min` is excluded from the feature set**. All 9 FL clients must have identical feature dimensions for GTVMin to work. The final feature set has 8 dimensions - see `prepare_data.py`.

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
A[i,j] = exp(-d²  / (2 × 100²))   if d ≤ 200 km
        = 0                         otherwise
```

Pure geography: stations within 200 km collaborate, weighted by a Gaussian decay. This is the strictest viable threshold - all 9 stations remain connected with 8 edges.

**System B - Correlation graph**

```
A[i,j] = max(0, pearson_corr(T_i, T_j)) × exp(-d² / (2 × 100²))   if d ≤ 200 km
        = 0                                                           otherwise
```

Same distance gate, but edge weight is also scaled by how strongly the two stations' temperatures co-vary. Computed on **training data only** to avoid data leakage - and recomputed for each experiment's training period.

### The three experiments

| Experiment | Training period | Val period | Test period | Train days |
|---|---|---|---|---|
| 1 - Full data | 2022-01-01 → 2023-12-31 | 2024 H1 | 2024 H2 | 730 |
| 2 - Reduced data | 2022-01-01 → 2022-12-31 | 2023 H1 | 2023 H2 | 365 |
| 3 - Minimal data | 2022-01-01 → 2022-06-30 | 2022 Q3 | 2022 Q4 | 181 |

For each experiment, alpha is tuned independently on the validation set from:
`[0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500]`

---

## Results

### Combined summary (mean RMSE across all 9 stations)

| Experiment | System | Train RMSE | Val RMSE | Test RMSE | Best α |
|---|---|---|---|---|---|
| Full data | Baseline | 5.16 °C | 5.56 °C | 5.30 °C | — |
| Full data | System A | 5.16 °C | 5.56 °C | 5.30 °C | 0.0001 |
| Full data | System B | 5.16 °C | 5.56 °C | 5.30 °C | 0.0001 |
| Reduced data | Baseline | 5.32 °C | 5.51 °C | 5.14 °C | — |
| Reduced data | System A | 5.32 °C | 5.51 °C | 5.14 °C | 0.0001 |
| Reduced data | System B | 5.32 °C | 5.51 °C | 5.14 °C | 0.0001 |
| Minimal data | Baseline | 4.14 °C | 2.64 °C | 3.23 °C | — |
| Minimal data | System A | 4.16 °C | 2.62 °C | 3.23 °C | 100 |
| Minimal data | System B | 4.16 °C | 2.62 °C | 3.23 °C | 100 |

### Interpretation

**Experiment 1 & 2 (Full and Reduced data):** FL makes no meaningful difference. With 1–2 years of training data, each station already has enough observations to fit a good local model. The hyperparameter search selects near-zero alpha (0.0001) - the regularisation toward neighbours is switched off and FL degenerates to local Ridge.

**Experiment 3 (Minimal data - 6 months):** FL provides no improvement with the current graph (d_max = 200 km, 8 edges). System A and System B both match the Baseline at 3.23 °C. The graph sensitivity analysis found that a wider graph (d_max = 300 km, 15 edges) produced a marginal gain (~0.02 °C) in this setting - the longer-range connections removed when tightening to 200 km were the ones contributing collaboration signal. With only 181 training days, the graph needs to be dense enough to compensate for the limited local data.

**System A vs System B:** Identical results across all experiments. All station pairs within 200 km have strongly positive Pearson correlation (Finnish temperatures follow the same seasonal cycle), so the correlation multiplier in System B ≈ 1.0 and the two graphs are effectively the same.

**Graph parameters:** d_max = 200 km, σ = 100 km (strictest viable threshold — all 9 stations connected with 8 edges, as determined by `graph_sensitivity.py`). The production graph previously used d_max = 300 km / σ = 150 km.

### Heating Degree Days (HDD)

HDD (Heating Degree Days) quantifies daily heating energy demand. Following the Finnish Meteorological Institute standard (indoor baseline 17 °C, heating threshold 12 °C), predicted temperatures are converted to HDD as a post-processing step:

```
HDD = max(17 - T, 0)   if T < 12 °C
HDD = 0                 otherwise
```

Results are reported for Experiment 1 (full data) test set only. HDD MAE measures how accurately each system estimates daily heating demand - a practical metric for energy planning applications. Results are saved to `results/hdd_analysis.csv` and `results/hdd_per_station.png`.

---

## Output Files Reference

### `data/`

| File | Description |
|---|---|
| `<station>.csv` | Clean daily CSV with columns `[date, t2m, tmin, tmax, ws_10min]` |
| `missing_summary.csv` | NaN counts per station per variable |
| `adj_system_a.npy` | 9×9 numpy array - System A adjacency matrix |
| `adj_system_b.npy` | 9×9 numpy array - System B adjacency matrix |
| `prepared_data.pkl` | Pickle dict: `{station: {X_train, y_train, X_val, y_val, X_test, y_test}}` |

### `results/`

| File | Description |
|---|---|
| `network_system_a.png` | Map of Finland showing System A graph edges |
| `network_system_b.png` | Map of Finland showing System B graph edges |
| `convergence_exp1.png` | Training MSE vs FL round for Experiment 1 |
| `test_rmse_by_station_exp{1,2,3}.png` | Per-station test RMSE bar charts |
| `test_rmse_by_experiment.png` | Mean test RMSE vs training set size (all systems) |
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
standardise_data.py  ──► data/<station>.csv  (9 files, daily)
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
```

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `numpy` | 2.0.2 | Array operations, adjacency matrices |
| `pandas` | 2.3.3 | CSV loading, date handling, aggregation |
| `scikit-learn` | 1.6.1 | Ridge regression |
| `matplotlib` | 3.9.4 | All figures |
| `fmiopendata` | 0.5.0 | FMI WFS API wrapper (used during exploration; data now loaded from CSV) |

Install all with:
```bash
pip install -r requirements.txt
```

---

## Design Decisions

**Why drop `ws_10min` from features?**
Two stations (Tampere Härmälä, Inari Saariselkä) have no wind speed measurements. All 9 FL clients must share identical feature dimensionality for the GTVMin weight vectors to be comparable and combinable. Dropping `ws_10min` was the cleanest solution.

**Why not normalise labels?**
Labels (`y = t2m[t+1]`) are kept in raw °C so that MSE and RMSE are directly interpretable in meaningful units.

**Why per-station normalisation?**
In a real federated system, clients never share raw data. Each station's scaler is fit on that station's training data only, mirroring this constraint.
