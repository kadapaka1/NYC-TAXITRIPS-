from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


def _load_project_module() -> object:
    here = Path(__file__).resolve().parent
    path = here / "project-01.py"
    spec = importlib.util.spec_from_file_location("project_01", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to import: {path}")
    mod = importlib.util.module_from_spec(spec)
    # Ensure dataclasses (and similar) can resolve module globals during exec.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


project = _load_project_module()

@st.cache_resource
def load_model(path_str: str):
    return joblib.load(Path(path_str))


@st.cache_resource
def load_zone_centroids(path_str: str):
    return project.ZoneCentroids.from_geojson(Path(path_str))


@dataclass(frozen=True)
class Inputs:
    pu: int
    do: int
    pickup_dt: datetime


def _build_single_feature_row(inp: Inputs, zone_centroids) -> pd.DataFrame:
    # Recreate the exact feature schema used in training (subset is OK as long as columns match).
    pickup_hour = inp.pickup_dt.hour
    # Python: Monday=0..Sunday=6; script uses Polars weekday Mon=1..Sun=7
    pickup_weekday = inp.pickup_dt.weekday() + 1
    pickup_month = inp.pickup_dt.month
    pickup_yearday = int(inp.pickup_dt.strftime("%j"))

    is_rush_am = int((1 <= pickup_weekday <= 5) and (7 <= pickup_hour <= 10))
    is_rush_pm = int((1 <= pickup_weekday <= 5) and (16 <= pickup_hour <= 19))
    is_overnight = int(0 <= pickup_hour <= 5)
    is_weekend = int(pickup_weekday in (6, 7))

    pu_borough = ""
    do_borough = ""
    haversine_km = np.nan

    if zone_centroids is not None:
        pu_borough = zone_centroids.borough.get(int(inp.pu), "") or ""
        do_borough = zone_centroids.borough.get(int(inp.do), "") or ""

        la1 = zone_centroids.lat.get(int(inp.pu))
        lo1 = zone_centroids.lon.get(int(inp.pu))
        la2 = zone_centroids.lat.get(int(inp.do))
        lo2 = zone_centroids.lon.get(int(inp.do))
        if la1 is not None and lo1 is not None and la2 is not None and lo2 is not None:
            haversine_km = float(project.haversine_km(np.array([la1]), np.array([lo1]), np.array([la2]), np.array([lo2]))[0])

    hour_angle = 2.0 * np.pi * (pickup_hour / 24.0)
    pickup_hour_sin = float(np.sin(hour_angle))
    pickup_hour_cos = float(np.cos(hour_angle))
    haversine_km2 = float(haversine_km * haversine_km) if haversine_km == haversine_km else np.nan
    same_borough = int(pu_borough == do_borough) if (pu_borough or do_borough) else 0

    return pd.DataFrame(
        [
            {
                "PULocationID": int(inp.pu),
                "DOLocationID": int(inp.do),
                "PU_borough": pu_borough,
                "DO_borough": do_borough,
                "pickup_hour": int(pickup_hour),
                "pickup_hour_sin": pickup_hour_sin,
                "pickup_hour_cos": pickup_hour_cos,
                "pickup_weekday": int(pickup_weekday),
                "pickup_month": int(pickup_month),
                "pickup_yearday": int(pickup_yearday),
                "is_rush_am": int(is_rush_am),
                "is_rush_pm": int(is_rush_pm),
                "is_overnight": int(is_overnight),
                "is_weekend": int(is_weekend),
                "haversine_km": float(haversine_km) if haversine_km == haversine_km else np.nan,
                "haversine_km2": haversine_km2,
                "same_borough": int(same_borough),
            }
        ]
    )


st.set_page_config(page_title="NYC Taxi Trip Duration Demo", layout="centered")
st.title("NYC Taxi Trip Duration Estimator")

st.markdown(
    "Load a trained model (`model.joblib`) and optionally a taxi zones GeoJSON to enable borough + haversine features."
)

model_path = st.sidebar.text_input("Model path", value=str(Path("artifacts").resolve()))
zones_path = st.sidebar.text_input("Taxi zones GeoJSON path (optional)", value="")

col1, col2 = st.columns(2)
with col1:
    pu = st.number_input("Pickup LocationID (PULocationID)", min_value=1, max_value=999, value=161, step=1)
with col2:
    do = st.number_input("Dropoff LocationID (DOLocationID)", min_value=1, max_value=999, value=236, step=1)

pickup_dt = st.datetime_input("Pickup datetime", value=datetime.now())

if st.button("Estimate duration"):
    mp = Path(model_path)
    if mp.is_dir():
        # Pick the latest run directory containing model.joblib
        candidates = sorted(mp.glob("**/model.joblib"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            st.error(f"No model.joblib found under: {mp}")
            st.stop()
        mp = candidates[0]

    if not mp.exists():
        st.error(f"Model not found: {mp}")
        st.stop()

    pipe = load_model(str(mp))
    zone_centroids = None
    zp = zones_path.strip()
    if zp:
        zpp = Path(zp)
        if not zpp.exists():
            st.warning(f"Zones geojson not found; continuing without: {zpp}")
        else:
            zone_centroids = load_zone_centroids(str(zpp))

    X = _build_single_feature_row(Inputs(pu=int(pu), do=int(do), pickup_dt=pickup_dt), zone_centroids=zone_centroids)
    pred_s = float(pipe.predict(X)[0])

    st.subheader("Prediction")
    st.write(
        {
            "pred_duration_seconds": round(pred_s, 1),
            "pred_duration_minutes": round(pred_s / 60.0, 2),
            "model": str(mp),
        }
    )

    if zone_centroids is not None:
        st.subheader("Feature preview")
        st.dataframe(X)

