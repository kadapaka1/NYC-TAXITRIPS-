from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import polars as pl
from shapely.geometry import shape
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


ARTIFACTS_DIRNAME = "artifacts"
DEFAULT_ZONE_GEOJSON_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"


def _now_utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def haversine_km(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """
    Vectorized haversine distance in kilometers.
    Inputs in degrees.
    """
    r = 6371.0088  # mean Earth radius (km)
    lat1 = np.deg2rad(lat1)
    lon1 = np.deg2rad(lon1)
    lat2 = np.deg2rad(lat2)
    lon2 = np.deg2rad(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return r * c


@dataclass(frozen=True)
class ZoneCentroids:
    """
    Maps TLC LocationID -> (lat, lon, borough).
    """

    lat: dict[int, float]
    lon: dict[int, float]
    borough: dict[int, str]

    @staticmethod
    def from_geojson(path: Path) -> "ZoneCentroids":
        with path.open("r", encoding="utf-8") as f:
            gj = json.load(f)

        lat: dict[int, float] = {}
        lon: dict[int, float] = {}
        borough: dict[int, str] = {}

        features = gj.get("features", [])
        if not features:
            raise ValueError(f"GeoJSON has no features: {path}")

        for feat in features:
            props = feat.get("properties") or {}
            loc_raw = props.get("LocationID") or props.get("location_id") or props.get("OBJECTID")
            if loc_raw is None:
                continue
            try:
                loc_id = int(loc_raw)
            except Exception:
                continue

            geom = feat.get("geometry")
            if not geom:
                continue
            try:
                centroid = shape(geom).centroid
            except Exception:
                continue

            # Shapely returns x=lon, y=lat.
            lon[loc_id] = float(centroid.x)
            lat[loc_id] = float(centroid.y)

            b = props.get("borough") or props.get("Borough") or props.get("borough_name") or ""
            borough[loc_id] = str(b) if b is not None else ""

        if not lat or not lon:
            raise ValueError(f"Failed to parse centroids from geojson: {path}")

        return ZoneCentroids(lat=lat, lon=lon, borough=borough)


def download_file(url: str, out_path: Path) -> Path:
    _ensure_dir(out_path.parent)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    urllib.request.urlretrieve(url, tmp)  # noqa: S310 - user-driven URL, saved locally
    tmp.replace(out_path)
    return out_path


def taxi_zones_zip_to_geojson(zip_path: Path, out_geojson_path: Path) -> Path:
    """
    Convert official TLC taxi_zones.zip (shapefile) to GeoJSON.

    The resulting GeoJSON includes:
    - geometry: Polygon/MultiPolygon
    - properties: LocationID, borough
    """
    import shapefile  # pyshp

    _ensure_dir(out_geojson_path.parent)
    extract_dir = out_geojson_path.parent / "_taxi_zones_extract"
    if extract_dir.exists():
        # best-effort cleanup of old extracts
        for p in extract_dir.glob("*"):
            try:
                p.unlink()
            except Exception:
                pass
    _ensure_dir(extract_dir)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)

    shp_files = list(extract_dir.glob("*.shp"))
    if not shp_files:
        raise ValueError(f"No .shp found inside: {zip_path}")
    shp_path = shp_files[0]

    r = shapefile.Reader(str(shp_path))
    fields = [f[0] for f in r.fields[1:]]  # skip DeletionFlag

    # Common field names: LocationID, borough (or Borough)
    def _get(props: dict, *names: str) -> str:
        for n in names:
            if n in props:
                v = props.get(n)
                return "" if v is None else str(v)
        return ""

    feats = []
    for sr in r.iterShapeRecords():
        rec = dict(zip(fields, sr.record))
        loc_raw = _get(rec, "LocationID", "locationid", "OBJECTID")
        if not loc_raw:
            continue
        try:
            loc_id = int(float(loc_raw))
        except Exception:
            continue

        borough = _get(rec, "borough", "Borough")

        geom = sr.shape.__geo_interface__
        feats.append(
            {
                "type": "Feature",
                "properties": {"LocationID": loc_id, "borough": borough},
                "geometry": geom,
            }
        )

    gj = {"type": "FeatureCollection", "features": feats}
    out_geojson_path.write_text(json.dumps(gj), encoding="utf-8")
    return out_geojson_path


def _infer_tlc_kind(cols: set[str]) -> str:
    # Common variations
    if "tpep_pickup_datetime" in cols or "tpep_dropoff_datetime" in cols:
        return "yellow"
    if "lpep_pickup_datetime" in cols or "lpep_dropoff_datetime" in cols:
        return "green"
    if "request_datetime" in cols and "on_scene_datetime" in cols:
        return "fhv"
    return "unknown"


def _pickup_col(kind: str) -> str:
    return {"yellow": "tpep_pickup_datetime", "green": "lpep_pickup_datetime"}.get(kind, "tpep_pickup_datetime")


def _dropoff_col(kind: str) -> str:
    return {"yellow": "tpep_dropoff_datetime", "green": "lpep_dropoff_datetime"}.get(kind, "tpep_dropoff_datetime")


def _location_cols(cols: set[str]) -> tuple[str, str]:
    if "PULocationID" in cols and "DOLocationID" in cols:
        return "PULocationID", "DOLocationID"
    if "pulocationid" in cols and "dolocationid" in cols:
        return "pulocationid", "dolocationid"
    return "PULocationID", "DOLocationID"


def build_feature_frame(
    df: pl.DataFrame,
    zone_centroids: ZoneCentroids | None,
    kind: str,
) -> pl.DataFrame:
    cols = set(df.columns)
    pu_col, do_col = _location_cols(cols)
    pickup_col = _pickup_col(kind)
    dropoff_col = _dropoff_col(kind)

    # Ensure datetimes exist
    if pickup_col not in cols:
        raise ValueError(f"Missing pickup datetime column: {pickup_col}")
    if dropoff_col not in cols:
        raise ValueError(f"Missing dropoff datetime column: {dropoff_col}")
    if pu_col not in cols or do_col not in cols:
        raise ValueError(f"Missing location columns (need PULocationID/DOLocationID). Found: {sorted(cols)[:25]} ...")

    # Target: trip duration seconds (clipped)
    base = df.with_columns(
        pl.col(pickup_col).cast(pl.Datetime(time_unit="us")).alias("pickup_dt"),
        pl.col(dropoff_col).cast(pl.Datetime(time_unit="us")).alias("dropoff_dt"),
        pl.col(pu_col).cast(pl.Int32).alias("PULocationID"),
        pl.col(do_col).cast(pl.Int32).alias("DOLocationID"),
    ).with_columns(
        ((pl.col("dropoff_dt") - pl.col("pickup_dt")).dt.total_seconds()).alias("duration_s_raw")
    )

    # Clean + clip outliers (tighter than 4h; most legit NYC trips are < 90 min)
    base = base.with_columns(
        pl.col("duration_s_raw").clip(60, 90 * 60).alias("duration_s")
    ).filter(pl.col("duration_s").is_not_null())

    # Temporal features
    base = base.with_columns(
        pl.col("pickup_dt").dt.hour().alias("pickup_hour"),
        pl.col("pickup_dt").dt.weekday().alias("pickup_weekday"),  # Mon=1..Sun=7
        pl.col("pickup_dt").dt.month().alias("pickup_month"),
        pl.col("pickup_dt").dt.ordinal_day().alias("pickup_yearday"),
    )

    # Rush-hour flags (weekday commuting windows)
    base = base.with_columns(
        (
            (pl.col("pickup_weekday").is_between(1, 5))
            & (pl.col("pickup_hour").is_between(7, 10))
        ).cast(pl.Int8).alias("is_rush_am"),
        (
            (pl.col("pickup_weekday").is_between(1, 5))
            & (pl.col("pickup_hour").is_between(16, 19))
        ).cast(pl.Int8).alias("is_rush_pm"),
        (pl.col("pickup_hour").is_between(0, 5)).cast(pl.Int8).alias("is_overnight"),
        (pl.col("pickup_weekday").is_in([6, 7])).cast(pl.Int8).alias("is_weekend"),
    )

    # Cyclical hour encoding (helps with periodicity)
    base = base.with_columns(
        (pl.lit(2.0 * math.pi) * (pl.col("pickup_hour") / pl.lit(24.0))).alias("_hour_angle"),
    ).with_columns(
        pl.col("_hour_angle").sin().cast(pl.Float32).alias("pickup_hour_sin"),
        pl.col("_hour_angle").cos().cast(pl.Float32).alias("pickup_hour_cos"),
    ).drop("_hour_angle")

    # Borough + centroid join (native Polars hash join; no Python row loops)
    if zone_centroids is not None:
        loc_ids = sorted(set(zone_centroids.lat) | set(zone_centroids.lon) | set(zone_centroids.borough))
        lookup = pl.DataFrame(
            {
                "LocationID": loc_ids,
                "borough": [zone_centroids.borough.get(i, "") for i in loc_ids],
                "lat": [zone_centroids.lat.get(i, np.nan) for i in loc_ids],
                "lon": [zone_centroids.lon.get(i, np.nan) for i in loc_ids],
            }
        ).with_columns(pl.col("LocationID").cast(pl.Int32))

        pu_lu = lookup.rename({"LocationID": "PULocationID", "borough": "PU_borough", "lat": "PU_lat", "lon": "PU_lon"})
        do_lu = lookup.rename({"LocationID": "DOLocationID", "borough": "DO_borough", "lat": "DO_lat", "lon": "DO_lon"})
        base = (
            base.join(pu_lu, on="PULocationID", how="left")
            .join(do_lu, on="DOLocationID", how="left")
            .with_columns(
                pl.col("PU_borough").fill_null("").cast(pl.Utf8),
                pl.col("DO_borough").fill_null("").cast(pl.Utf8),
            )
        )
    else:
        base = base.with_columns(pl.lit("").alias("PU_borough"), pl.lit("").alias("DO_borough"))

    # Haversine distance between zone centroids (native Polars math when available)
    if zone_centroids is not None:
        r = pl.lit(6371.0088)  # km
        deg2rad = pl.lit(math.pi / 180.0)
        lat1 = (pl.col("PU_lat") * deg2rad).cast(pl.Float64)
        lon1 = (pl.col("PU_lon") * deg2rad).cast(pl.Float64)
        lat2 = (pl.col("DO_lat") * deg2rad).cast(pl.Float64)
        lon2 = (pl.col("DO_lon") * deg2rad).cast(pl.Float64)

        dlat = (lat2 - lat1)
        dlon = (lon2 - lon1)
        a = (dlat / 2).sin() ** 2 + (lat1.cos() * lat2.cos() * (dlon / 2).sin() ** 2)
        c = pl.lit(2.0) * a.sqrt().arcsin()
        base = base.with_columns((r * c).cast(pl.Float32).alias("haversine_km"))

        # Speed sanity filter (drop impossible trips; keep rows without haversine)
        base = base.with_columns(
            (pl.col("haversine_km") / (pl.col("duration_s") / pl.lit(3600.0))).alias("_speed_kmh")
        ).filter(
            pl.col("_speed_kmh").is_null() | pl.col("_speed_kmh").is_nan() | (pl.col("_speed_kmh") <= 120.0)
        ).drop("_speed_kmh")
    else:
        base = base.with_columns(pl.lit(np.nan).cast(pl.Float32).alias("haversine_km"))

    # Quick-win engineered features
    base = base.with_columns(
        (pl.col("PU_borough") == pl.col("DO_borough")).cast(pl.Int8).alias("same_borough"),
        (pl.col("haversine_km") * pl.col("haversine_km")).cast(pl.Float32).alias("haversine_km2"),
    )

    # Select final columns for modeling
    keep = [
        "duration_s",
        "PULocationID",
        "DOLocationID",
        "PU_borough",
        "DO_borough",
        "pickup_hour",
        "pickup_hour_sin",
        "pickup_hour_cos",
        "pickup_weekday",
        "pickup_month",
        "pickup_yearday",
        "is_rush_am",
        "is_rush_pm",
        "is_overnight",
        "is_weekend",
        "haversine_km",
        "haversine_km2",
        "same_borough",
    ]
    return base.select([c for c in keep if c in base.columns])


def scan_tlc(path: Path) -> pl.LazyFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    if path.is_dir():
        # Load all parquet/csv under a directory
        # NOTE: polars can scan parquet/glob natively; for mixed formats, user should preprocess.
        pq = list(path.glob("*.parquet"))
        if pq:
            return pl.scan_parquet(str(path / "*.parquet"))
        csvs = list(path.glob("*.csv"))
        if csvs:
            return pl.scan_csv(str(path / "*.csv"), infer_schema_length=10000, ignore_errors=True)
        raise ValueError(f"No .parquet or .csv in directory: {path}")

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pl.scan_parquet(str(path))
    if suffix == ".csv":
        return pl.scan_csv(str(path), infer_schema_length=10000, ignore_errors=True)
    raise ValueError(f"Unsupported file type: {path}")


def materialize_sample(lf: pl.LazyFrame, n: int, seed: int) -> pl.DataFrame:
    # Polars doesn't have true random sampling for LazyFrame across all scans.
    # We fetch a moderate prefix, then sample from it.
    df = lf.fetch(max(n * 3, n))
    if len(df) <= n:
        return df
    return df.sample(n=n, with_replacement=False, seed=seed)


def train_model(df_feat: pd.DataFrame, out_dir: Path) -> dict:
    y = df_feat["duration_s"].astype(float)
    X = df_feat.drop(columns=["duration_s"])

    categorical = [c for c in ["PULocationID", "DOLocationID", "PU_borough", "DO_borough"] if c in X.columns]
    numeric = [c for c in X.columns if c not in categorical]

    pre = ColumnTransformer(
        transformers=[
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                categorical,
            ),
            ("num", "passthrough", numeric),
        ],
        remainder="drop",
    )

    model = HistGradientBoostingRegressor(
        learning_rate=0.08,
        max_depth=8,
        max_iter=300,
        l2_regularization=0.0,
        categorical_features=list(range(len(categorical))),
        random_state=42,
    )

    pipe = Pipeline([("pre", pre), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    t0 = time.time()
    pipe.fit(X_train, y_train)
    fit_s = time.time() - t0
    pred = pipe.predict(X_test)

    rmse = float(math.sqrt(mean_squared_error(y_test, pred)))
    mae = float(mean_absolute_error(y_test, pred))
    r2 = float(r2_score(y_test, pred))

    _ensure_dir(out_dir)
    import joblib  # local import so base install stays minimal

    model_path = out_dir / "model.joblib"
    joblib.dump(pipe, model_path)

    metrics = {
        "rows": int(len(df_feat)),
        "fit_seconds": float(fit_s),
        "rmse_seconds": rmse,
        "mae_seconds": mae,
        "r2": r2,
        "model_path": str(model_path),
        "categorical_features": categorical,
        "numeric_features": numeric,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def make_geo_viz(
    df_raw: pl.DataFrame,
    zone_centroids: ZoneCentroids,
    kind: str,
    out_dir: Path,
    n_trips: int,
    seed: int,
) -> dict[str, str]:
    import folium
    from folium.plugins import HeatMap

    cols = set(df_raw.columns)
    pu_col, do_col = _location_cols(cols)
    pickup_col = _pickup_col(kind)

    df = df_raw.select(
        [
            pl.col(pu_col).cast(pl.Int32).alias("PULocationID"),
            pl.col(do_col).cast(pl.Int32).alias("DOLocationID"),
            pl.col(pickup_col).cast(pl.Datetime(time_unit="us")).alias("pickup_dt"),
        ]
    )

    if len(df) == 0:
        raise ValueError("No rows available for visualization.")

    # Sample trips for polyline visualization
    if len(df) > n_trips:
        df_lines = df.sample(n=n_trips, seed=seed)
    else:
        df_lines = df

    lat = zone_centroids.lat
    lon = zone_centroids.lon

    def _pt(loc_id: int) -> tuple[float, float] | None:
        la = lat.get(int(loc_id))
        lo = lon.get(int(loc_id))
        if la is None or lo is None:
            return None
        if math.isnan(la) or math.isnan(lo):
            return None
        return (la, lo)

    # Center map around Manhattan-ish
    center = (40.758, -73.9855)
    m_lines = folium.Map(location=center, zoom_start=11, tiles="CartoDB positron")

    rows = df_lines.to_dicts()
    random.Random(seed).shuffle(rows)
    drawn = 0
    for r in rows:
        a = _pt(r["PULocationID"])
        b = _pt(r["DOLocationID"])
        if a is None or b is None:
            continue
        folium.PolyLine([a, b], opacity=0.25, weight=2, color="#2b8cbe").add_to(m_lines)
        drawn += 1
        if drawn >= n_trips:
            break

    # Pickup heatmap
    m_heat = folium.Map(location=center, zoom_start=11, tiles="CartoDB positron")
    pts: list[list[float]] = []
    for r in df.to_dicts():
        p = _pt(r["PULocationID"])
        if p is not None:
            pts.append([p[0], p[1]])
    HeatMap(pts, radius=9, blur=12, min_opacity=0.2).add_to(m_heat)

    _ensure_dir(out_dir)
    out_lines = out_dir / "map_sample_trips.html"
    out_heat = out_dir / "map_pickup_heat.html"
    m_lines.save(str(out_lines))
    m_heat.save(str(out_heat))

    return {"sample_trips_map": str(out_lines), "pickup_heatmap": str(out_heat)}


def cmd_download_zones(args: argparse.Namespace) -> int:
    out = Path(args.out).resolve()
    url = args.url
    print(f"Downloading taxi zones geojson to: {out}")
    if url.lower().endswith(".zip"):
        zip_out = out.with_suffix(".zip")
        print(f"Downloading taxi zones zip to: {zip_out}")
        download_file(url, zip_out)
        taxi_zones_zip_to_geojson(zip_out, out_geojson_path=out)
        print("Converted zip -> geojson.")
    else:
        download_file(url, out)
        print("Downloaded geojson.")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    data_path = Path(args.data).resolve()
    out_dir = Path(args.out).resolve()
    zones_path = Path(args.zones).resolve() if args.zones else None

    lf = scan_tlc(data_path)
    # Determine dataset kind from schema
    schema_cols = set(lf.collect_schema().names())
    kind = _infer_tlc_kind(schema_cols)
    if kind == "unknown":
        print("Warning: couldn't infer TLC dataset kind; assuming yellow-style columns.", file=sys.stderr)
        kind = "yellow"

    zone_centroids = ZoneCentroids.from_geojson(zones_path) if zones_path else None
    raw = materialize_sample(lf, n=args.train_rows, seed=args.seed)
    feat = build_feature_frame(raw, zone_centroids=zone_centroids, kind=kind)

    df_feat = feat.to_pandas()
    df_feat = df_feat.replace([np.inf, -np.inf], np.nan).dropna()
    if len(df_feat) < 10_000:
        print(
            f"Warning: only {len(df_feat):,} usable rows after cleaning. "
            "Consider increasing --train-rows or provide zones geojson for haversine.",
            file=sys.stderr,
        )

    metrics = train_model(df_feat, out_dir=out_dir)
    print(json.dumps(metrics, indent=2))
    return 0


def cmd_viz(args: argparse.Namespace) -> int:
    data_path = Path(args.data).resolve()
    out_dir = Path(args.out).resolve()
    zones_path = Path(args.zones).resolve()

    lf = scan_tlc(data_path)
    kind = _infer_tlc_kind(set(lf.collect_schema().names()))
    if kind == "unknown":
        kind = "yellow"

    zone_centroids = ZoneCentroids.from_geojson(zones_path)
    raw = materialize_sample(lf, n=args.rows, seed=args.seed)
    out = make_geo_viz(
        raw,
        zone_centroids=zone_centroids,
        kind=kind,
        out_dir=out_dir,
        n_trips=args.trips,
        seed=args.seed,
    )
    print(json.dumps(out, indent=2))
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    import joblib

    model_path = Path(args.model).resolve()
    data_path = Path(args.data).resolve()
    out_path = Path(args.out).resolve()
    zones_path = Path(args.zones).resolve() if args.zones else None

    pipe = joblib.load(model_path)

    lf = scan_tlc(data_path)
    kind = _infer_tlc_kind(set(lf.collect_schema().names()))
    if kind == "unknown":
        kind = "yellow"

    zone_centroids = ZoneCentroids.from_geojson(zones_path) if zones_path else None
    raw = materialize_sample(lf, n=args.rows, seed=args.seed)
    feat = build_feature_frame(raw, zone_centroids=zone_centroids, kind=kind)
    df = feat.to_pandas().replace([np.inf, -np.inf], np.nan).dropna()
    X = df.drop(columns=["duration_s"])
    pred_s = pipe.predict(X)

    out = pd.DataFrame({"pred_duration_s": pred_s})
    _ensure_dir(out_path.parent)
    out.to_csv(out_path, index=False)
    print(f"Wrote predictions: {out_path} ({len(out):,} rows)")
    return 0


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="project-01.py",
        description="NYC TLC Trip Duration Regression (geospatial + temporal features, scalable IO).",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("download-zones", help="Download taxi zones GeoJSON for centroids/boroughs.")
    s.add_argument("--url", default=DEFAULT_ZONE_GEOJSON_URL)
    s.add_argument("--out", default=str(Path(ARTIFACTS_DIRNAME) / "taxi_zones.geojson"))
    s.set_defaults(func=cmd_download_zones)

    s = sub.add_parser("train", help="Train a regression model on a sample of rows.")
    s.add_argument("--data", required=True, help="Path to TLC .parquet/.csv or directory of parquet/csv.")
    s.add_argument("--zones", default=None, help="Path to taxi_zones.geojson (enables haversine + borough).")
    s.add_argument("--out", default=str(Path(ARTIFACTS_DIRNAME) / f"run_{_now_utc_compact()}"))
    s.add_argument("--train-rows", type=int, default=500_000, help="Rows to materialize for training (sampled).")
    s.add_argument("--seed", type=int, default=42)
    s.set_defaults(func=cmd_train)

    s = sub.add_parser("viz", help="Create geospatial HTML visualizations (folium).")
    s.add_argument("--data", required=True)
    s.add_argument("--zones", required=True)
    s.add_argument("--out", default=str(Path(ARTIFACTS_DIRNAME) / f"viz_{_now_utc_compact()}"))
    s.add_argument("--rows", type=int, default=200_000, help="Rows to use for the heatmap.")
    s.add_argument("--trips", type=int, default=2500, help="Trips to draw as lines.")
    s.add_argument("--seed", type=int, default=42)
    s.set_defaults(func=cmd_viz)

    s = sub.add_parser("predict", help="Run batch prediction on a sample.")
    s.add_argument("--model", required=True, help="Path to saved model.joblib")
    s.add_argument("--data", required=True)
    s.add_argument("--zones", default=None)
    s.add_argument("--rows", type=int, default=50_000)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--out", default=str(Path(ARTIFACTS_DIRNAME) / "predictions.csv"))
    s.set_defaults(func=cmd_predict)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

