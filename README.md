# NYC-TAXITRIPS-
An end-to-end machine learning pipeline that predicts NYC taxi trip duration using geospatial features, temporal patterns, and 500k+ real TLC trip records — with a live interactive demo.
# 🗽 NYC Taxi Trip Duration Predictor
### Full-Stack Geospatial ML — Regression Pipeline · Live Demo · Interactive Maps

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)
![Polars](https://img.shields.io/badge/Polars-LazyFrame-orange?logo=polars&logoColor=white)
![scikit-learn](https://img.shields.io/badge/sklearn-HistGradientBoosting-green?logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Demo-Streamlit-red?logo=streamlit&logoColor=white)
![Folium](https://img.shields.io/badge/Maps-Folium-darkgreen)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

> **Given a pickup zone, dropoff zone, and departure time — how long will the taxi take?**
> This project answers that question end-to-end: raw TLC parquet data → geospatial feature engineering → trained regression model → live Streamlit demo.

---

## 📌 Project Highlights

| | |
|---|---|
| 🗺️ **Geospatial features** | Haversine distance between 263 NYC taxi zone centroids |
| 🏙️ **Borough encoding** | Pickup + dropoff borough from official TLC GeoJSON |
| ⏰ **Temporal signals** | Rush-hour flags, hour of day, weekday, month |
| ⚡ **Scalable I/O** | Polars LazyFrames — handles millions of rows efficiently |
| 🤖 **Model** | `HistGradientBoostingRegressor` inside a full sklearn `Pipeline` |
| 🗺️ **Visualisations** | Interactive Folium trip-line map + pickup heatmap |
| 🚀 **Live demo** | Streamlit app + self-contained HTML showcase page |

---

## 🗂️ Project Structure

```
NYC-TAXITRIPS-/
│
├── project-01.py        # Core ML pipeline (CLI: download-zones, train, viz, predict)
├── app.py               # Streamlit live demo app
├── showcase.html        # Standalone HTML project showcase (charts, map, estimator)
├── requirements.txt     # Python dependencies
└── README.md            # You are here
```

Artifacts are written to `artifacts/` (auto-created, not committed):
```
artifacts/
├── taxi_zones.geojson           # Downloaded TLC taxi zone boundaries
└── run_YYYYMMDD_HHMMSS/
    ├── model.joblib             # Trained pipeline
    └── metrics.json             # RMSE, MAE, R²
```

---

## ⚙️ Setup

```bash
# 1. Clone the repo
git clone https://github.com/kadapaka1/NYC-TAXITRIPS-.git
cd NYC-TAXITRIPS-

# 2. Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -U pip
pip install -r requirements.txt
```

---

## 🚦 Quickstart

### Step 1 — Download Taxi Zones GeoJSON
Required for borough labels and haversine distance computation:
```bash
python project-01.py download-zones --out artifacts/taxi_zones.geojson
```

### Step 2 — Train the Model
Point `--data` at any TLC `.parquet` or `.csv` file (yellow or green taxi):
```bash
python project-01.py train \
  --data "path/to/yellow_tripdata_2024-01.parquet" \
  --zones artifacts/taxi_zones.geojson
```
Outputs `model.joblib` + `metrics.json` under `artifacts/run_<timestamp>/`.

> 💡 Tip: Use `--train-rows 100000` to train faster on a smaller sample.

### Step 3 — Generate Geospatial Visualisations
Creates two interactive HTML maps:
```bash
python project-01.py viz \
  --data "path/to/yellow_tripdata_2024-01.parquet" \
  --zones artifacts/taxi_zones.geojson
```
- `map_sample_trips.html` — 2,500 origin-destination polylines across NYC
- `map_pickup_heat.html` — pickup density heatmap

### Step 4 — Run the Live Demo
```bash
streamlit run app.py
```
In the sidebar, set the **Model path** to your `artifacts/` folder — it auto-picks the latest run.

### Step 5 — Open the Showcase Page
Open `showcase.html` directly in any browser. No server needed — fully self-contained.

---

## 🧠 Feature Engineering

| Feature | Description |
|---|---|
| `haversine_km` | Straight-line distance (km) between pickup & dropoff zone centroids |
| `PU_borough` / `DO_borough` | One-hot encoded NYC borough (Manhattan, Brooklyn, Queens, Bronx, Staten Island) |
| `pickup_hour` | Hour of day (0–23) |
| `pickup_weekday` | Day of week (Mon=1 … Sun=7) |
| `pickup_month` | Month (1–12) |
| `pickup_yearday` | Day of year (1–365) |
| `is_rush_am` | 1 if weekday 07:00–10:00 |
| `is_rush_pm` | 1 if weekday 16:00–19:00 |
| `is_overnight` | 1 if 00:00–05:00 |
| `is_weekend` | 1 if Saturday or Sunday |

---

## 📊 Model & Results

**Algorithm:** `HistGradientBoostingRegressor` (scikit-learn)

```
learning_rate : 0.08
max_depth     : 8
max_iter      : 300
random_state  : 42
```

**Pipeline:**
```
Raw TLC Data
    ↓
Polars LazyFrame (scan + stream)
    ↓
Feature Engineering (haversine, borough, temporal flags)
    ↓
ColumnTransformer (OneHotEncoder + passthrough)
    ↓
HistGradientBoostingRegressor
    ↓
model.joblib  +  metrics.json
```

**Typical metrics on 500k rows (yellow taxi 2024):**

| Metric | Value |
|---|---|
| RMSE | ~192 seconds (~3.2 min) |
| MAE | ~138 seconds (~2.3 min) |
| R² | ~0.81 |
| Train time | ~45 seconds |

---

## 🗺️ Geospatial Visualisations

| Map | Description |
|---|---|
| **Trip Lines** | 2,500 sampled origin → destination polylines, coloured by pickup borough |
| **Pickup Heatmap** | Density of all pickup locations across NYC |

Both are standalone Folium HTML files — open in any browser or embed in a webpage.

---

## 📦 Data Source

**NYC TLC Trip Record Data** — publicly available at:
👉 [https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)

Download any monthly yellow or green taxi `.parquet` file and point `--data` at it.

**Expected columns:**
- `PULocationID`, `DOLocationID`
- `tpep_pickup_datetime`, `tpep_dropoff_datetime` (yellow)
- `lpep_pickup_datetime`, `lpep_dropoff_datetime` (green)

---

## 🛠️ Dependencies

```
polars >= 1.0.0          # Fast DataFrame I/O with LazyFrames
pyarrow >= 15.0.0        # Parquet backend
pandas >= 2.2.0          # sklearn interface layer
numpy >= 2.0.0           # Vectorized haversine
scikit-learn >= 1.5.0    # Pipeline + model
joblib >= 1.4.0          # Model serialisation
matplotlib >= 3.9.0      # EDA plots
seaborn >= 0.13.0        # Styled EDA plots
folium >= 0.16.0         # Interactive maps
shapely >= 2.0.0         # GeoJSON centroid computation
streamlit >= 1.36.0      # Live demo app
pyshp >= 2.3.0           # Shapefile → GeoJSON conversion
```

---

## 🔮 Future Improvements

- [ ] Replace `map_elements` borough lookups with native Polars joins (faster on large datasets)
- [ ] Use `OrdinalEncoder` + native categoricals in `HistGBR` instead of one-hot (fewer features, faster training)
- [ ] Add `@st.cache_resource` to Streamlit for instant reloads
- [ ] Tighter duration clipping (`clip(60, 5400)`) for cleaner target distribution
- [ ] Add `haversine_km²` and cyclical hour encoding (`sin`/`cos`) as features
- [ ] Hyperparameter tuning with `Optuna`
- [ ] Deploy showcase page to GitHub Pages

---

## 📄 License

MIT — free to use, modify, and distribute.

---

<p align="center">Built with 🚕 · Polars · scikit-learn · Folium · Streamlit · Leaflet.js · Chart.js</p>
