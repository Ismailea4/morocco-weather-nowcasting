"""src.preprocess.wind_gridding

Grid EUMETSAT HRW wind observations (BUFR) onto the same ROI grid as satellite.

This script uses `pdbufr.read_bufr` to load BUFR into a Pandas DataFrame, then
bins observations onto a regular lat/lon grid (approx degree spacing computed
from `roi.resolution_m`).

Output format:
  <output-dir>/wind_<YYYYmmdd_HHMMSS>.npz containing:
    - data: (2, H, W) float32 where channel 0 is u, channel 1 is v
    - channels: ['u', 'v']
    - timestamp: ISO string

Notes:
  - HRW BUFR schemas vary. This script is best-effort and will print a clear
    error if expected columns can't be found.
"""

from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml


def _roi_from_config(roi_cfg_path: Path) -> Dict[str, float]:
    cfg = yaml.safe_load(roi_cfg_path.read_text(encoding="utf-8"))
    roi = cfg["roi"]
    return {
        "lat_min": float(roi["lat_min"]),
        "lat_max": float(roi["lat_max"]),
        "lon_min": float(roi["lon_min"]),
        "lon_max": float(roi["lon_max"]),
        "resolution_m": float(roi.get("resolution_m", 3000.0)),
    }


def _grid_params(roi: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
    lat_min, lat_max = roi["lat_min"], roi["lat_max"]
    lon_min, lon_max = roi["lon_min"], roi["lon_max"]
    res_m = roi["resolution_m"]
    mean_lat = 0.5 * (lat_min + lat_max)

    dlat = res_m / 111_000.0
    dlon = res_m / (111_000.0 * max(0.1, math.cos(math.radians(mean_lat))))

    lats = np.arange(lat_min, lat_max + 0.5 * dlat, dlat)
    lons = np.arange(lon_min, lon_max + 0.5 * dlon, dlon)
    return lats, lons


def _round_time(dt: datetime, bucket_minutes: int) -> datetime:
    if bucket_minutes <= 1:
        return dt.replace(second=0, microsecond=0)
    seconds = int(dt.timestamp())
    bucket = bucket_minutes * 60
    rounded = int(round(seconds / bucket) * bucket)
    return datetime.fromtimestamp(rounded)


def _try_build_datetime(df: pd.DataFrame) -> Optional[pd.Series]:
    # Prefer an existing datetime column
    for c in ("datetime", "time", "observation_time", "date_time"):
        if c in df.columns:
            try:
                return pd.to_datetime(df[c], errors="coerce", utc=False)
            except Exception:
                pass

    # Common split fields
    for y, m, d, hh, mm in (
        ("year", "month", "day", "hour", "minute"),
        ("typicalYear", "typicalMonth", "typicalDay", "typicalHour", "typicalMinute"),
    ):
        if all(col in df.columns for col in (y, m, d, hh, mm)):
            try:
                return pd.to_datetime(
                    {
                        "year": df[y],
                        "month": df[m],
                        "day": df[d],
                        "hour": df[hh],
                        "minute": df[mm],
                    },
                    errors="coerce",
                )
            except Exception:
                pass

    return None


def _pick_lat_lon(df: pd.DataFrame) -> Tuple[str, str]:
    lat_candidates = ["latitude", "lat", "Latitude"]
    lon_candidates = ["longitude", "lon", "Longitude"]
    lat = next((c for c in lat_candidates if c in df.columns), None)
    lon = next((c for c in lon_candidates if c in df.columns), None)
    if not lat or not lon:
        raise RuntimeError(
            "Could not find latitude/longitude columns in BUFR data. "
            f"Available columns include: {sorted(list(df.columns))[:50]}"
        )
    return lat, lon


def _extract_uv(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    # Prefer direct u/v
    u_candidates = ["u", "windU", "eastward_wind", "uWind"]
    v_candidates = ["v", "windV", "northward_wind", "vWind"]
    u = next((c for c in u_candidates if c in df.columns), None)
    v = next((c for c in v_candidates if c in df.columns), None)
    if u and v:
        return pd.to_numeric(df[u], errors="coerce"), pd.to_numeric(df[v], errors="coerce")

    # Else speed + direction
    spd_candidates = ["windSpeed", "speed", "wind_speed", "WindSpeed"]
    dir_candidates = ["windDirection", "direction", "wind_dir", "WindDirection"]
    spd = next((c for c in spd_candidates if c in df.columns), None)
    wdir = next((c for c in dir_candidates if c in df.columns), None)
    if not spd or not wdir:
        raise RuntimeError(
            "Could not find (u,v) or (windSpeed,windDirection) columns in BUFR data. "
            f"Available columns include: {sorted(list(df.columns))[:50]}"
        )

    speed = pd.to_numeric(df[spd], errors="coerce")
    direction_deg = pd.to_numeric(df[wdir], errors="coerce")

    # Meteorological convention: direction is FROM which wind blows.
    # u (east) = -speed * sin(dir), v (north) = -speed * cos(dir)
    rad = np.deg2rad(direction_deg.astype(float))
    u_arr = -speed.astype(float) * np.sin(rad)
    v_arr = -speed.astype(float) * np.cos(rad)
    return pd.Series(u_arr, index=df.index), pd.Series(v_arr, index=df.index)


def _grid_bucket(
    lat: np.ndarray,
    lon: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    roi: Dict[str, float],
    lats: np.ndarray,
    lons: np.ndarray,
) -> np.ndarray:
    lat_min, lat_max = roi["lat_min"], roi["lat_max"]
    lon_min, lon_max = roi["lon_min"], roi["lon_max"]

    # Compute indices
    dlat = float(lats[1] - lats[0])
    dlon = float(lons[1] - lons[0])
    i = np.floor((lat - lat_min) / dlat).astype(int)
    j = np.floor((lon - lon_min) / dlon).astype(int)

    H = len(lats)
    W = len(lons)

    valid = (
        (lat >= lat_min)
        & (lat <= lat_max)
        & (lon >= lon_min)
        & (lon <= lon_max)
        & (i >= 0)
        & (i < H)
        & (j >= 0)
        & (j < W)
        & np.isfinite(u)
        & np.isfinite(v)
    )

    i = i[valid]
    j = j[valid]
    u = u[valid]
    v = v[valid]

    sum_u = np.zeros((H, W), dtype=np.float64)
    sum_v = np.zeros((H, W), dtype=np.float64)
    cnt = np.zeros((H, W), dtype=np.float64)

    np.add.at(sum_u, (i, j), u)
    np.add.at(sum_v, (i, j), v)
    np.add.at(cnt, (i, j), 1.0)

    with np.errstate(invalid="ignore", divide="ignore"):
        out_u = sum_u / cnt
        out_v = sum_v / cnt

    out_u[cnt == 0] = np.nan
    out_v[cnt == 0] = np.nan

    return np.stack([out_u.astype(np.float32), out_v.astype(np.float32)], axis=0)


def _save_npz(out_path: Path, data: np.ndarray, timestamp: datetime) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        data=data.astype(np.float32, copy=False),
        channels=np.array(["u", "v"], dtype=object),
        timestamp=np.array(timestamp.isoformat(), dtype=object),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Grid HRW wind vectors to Morocco ROI")
    ap.add_argument("--input-dir", type=str, default="data/raw", help="Directory containing HRW BUFR downloads")
    ap.add_argument("--output-dir", type=str, default="data/processed/wind", help="Directory to write wind_*.npz")
    ap.add_argument("--roi-config", type=str, default="configs/project.yaml", help="ROI config YAML")
    ap.add_argument("--glob", type=str, default="**/*", help="Glob pattern under input-dir")
    ap.add_argument("--bucket-minutes", type=int, default=15, help="Time bucketing interval")
    ap.add_argument("--max-files", type=int, default=0, help="If set, process only first N files")
    ap.add_argument(
        "--reader",
        type=str,
        default="generic",
        help="pdbufr reader name (default: generic)",
    )
    ap.add_argument(
        "--fill-missing",
        action="store_true",
        help="Fill grid cells with 0 where no observations (default keeps NaNs)",
    )

    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    roi = _roi_from_config(Path(args.roi_config))
    lats, lons = _grid_params(roi)

    paths = sorted(in_dir.glob(args.glob))
    paths = [p for p in paths if p.is_file()]
    if not paths:
        raise RuntimeError(f"No files matched {args.glob!r} under {in_dir}")

    if args.max_files and int(args.max_files) > 0:
        paths = paths[: int(args.max_files)]

    out_dir.mkdir(parents=True, exist_ok=True)

    # Lazy import here so users can run other scripts without pandas/pdbufr issues.
    try:
        import pdbufr
    except Exception as e:
        raise RuntimeError("Missing dependency 'pdbufr'. Install with: pip install -r requirements.txt") from e

    all_buckets: Dict[datetime, List[np.ndarray]] = {}

    for p in paths:
        try:
            # Try reading all columns first (schema varies).
            df = pdbufr.read_bufr(str(p), columns=[], reader=str(args.reader))
            if df is None or len(df) == 0:
                continue

            dt_series = _try_build_datetime(df)
            if dt_series is None:
                raise RuntimeError("Could not infer datetime from BUFR columns")

            lat_col, lon_col = _pick_lat_lon(df)
            u_s, v_s = _extract_uv(df)

            lat = pd.to_numeric(df[lat_col], errors="coerce").to_numpy(dtype=float)
            lon = pd.to_numeric(df[lon_col], errors="coerce").to_numpy(dtype=float)
            u = pd.to_numeric(u_s, errors="coerce").to_numpy(dtype=float)
            v = pd.to_numeric(v_s, errors="coerce").to_numpy(dtype=float)

            for idx, dt in enumerate(pd.to_datetime(dt_series, errors="coerce")):
                if pd.isna(dt):
                    continue
                bucket = _round_time(dt.to_pydatetime(), int(args.bucket_minutes))
                all_buckets.setdefault(bucket, []).append(np.array([lat[idx], lon[idx], u[idx], v[idx]], dtype=float))

        except Exception as e:
            print(f"[wind_gridding] FAILED for {p}: {e}", file=sys.stderr)

    if not all_buckets:
        raise RuntimeError("No wind observations could be parsed/bucketed from inputs.")

    wrote = 0
    for bucket, rows in sorted(all_buckets.items()):
        arr = np.stack(rows, axis=0)
        lat = arr[:, 0]
        lon = arr[:, 1]
        u = arr[:, 2]
        v = arr[:, 3]

        grid = _grid_bucket(lat, lon, u, v, roi=roi, lats=lats, lons=lons)
        if args.fill_missing:
            grid = np.nan_to_num(grid, nan=0.0)

        out_path = out_dir / f"wind_{bucket.strftime('%Y%m%d_%H%M%S')}.npz"
        _save_npz(out_path, grid, timestamp=bucket)
        wrote += 1

    print(f"[wind_gridding] done. buckets_written={wrote}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise
