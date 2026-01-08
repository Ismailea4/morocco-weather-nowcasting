"""src.preprocess.create_combined

Create aligned satellite+wind combined samples used by training.

Inputs:
  - satellite_*.npz in --satellite-dir
  - wind_*.npz in --wind-dir

Outputs:
  - combined_<timestamp>.npz in --output-dir with key "data" shape (C, H, W)

By default, timestamps are matched exactly by filename timestamp. If wind is
missing, the sample is skipped unless --allow-missing-wind is set.
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


_TS_RE = re.compile(r"_(\d{8})_(\d{6})\.npz$")


def _parse_ts(path: Path) -> Optional[datetime]:
    m = _TS_RE.search(path.name)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    except Exception:
        return None


def _index_by_ts(paths: List[Path]) -> Dict[datetime, Path]:
    out: Dict[datetime, Path] = {}
    for p in paths:
        ts = _parse_ts(p)
        if ts is None:
            continue
        out[ts] = p
    return out


def _nearest(ts: datetime, candidates: List[datetime], tolerance: timedelta) -> Optional[datetime]:
    best = None
    best_dt = None
    for c in candidates:
        d = abs(c - ts)
        if d <= tolerance and (best_dt is None or d < best_dt):
            best = c
            best_dt = d
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description="Align and combine satellite + wind .npz into combined_*.npz")
    ap.add_argument("--satellite-dir", type=str, default="data/processed/satellite")
    ap.add_argument(
        "--satellite-glob",
        action="append",
        default=[],
        help="Glob(s) for satellite .npz inside --satellite-dir. Repeatable. Default: satellite_*.npz",
    )
    ap.add_argument("--wind-dir", type=str, default="data/processed/wind")
    ap.add_argument("--output-dir", type=str, default="data/combined")
    ap.add_argument("--tolerance-minutes", type=int, default=15)
    ap.add_argument(
        "--allow-missing-wind",
        action="store_true",
        help="If set, missing wind is filled with zeros (2 channels).",
    )
    ap.add_argument(
        "--allow-missing-satellite",
        action="store_true",
        help="If set, missing satellite sources are filled with zeros to maximize usable timesteps.",
    )
    ap.add_argument(
        "--nan-to-num",
        action="store_true",
        help="If set, replace NaNs with 0 in the final combined tensor.",
    )

    args = ap.parse_args()

    sat_dir = Path(args.satellite_dir)
    wind_dir = Path(args.wind_dir)
    out_dir = Path(args.output_dir)
    tol = timedelta(minutes=int(args.tolerance_minutes))

    sat_globs = args.satellite_glob or ["satellite_*.npz"]
    sat_sources = [sorted(sat_dir.glob(g)) for g in sat_globs]
    sat_sources = [lst for lst in sat_sources if lst]
    wind_files = sorted(wind_dir.glob("wind_*.npz"))

    include_wind = len(wind_files) > 0

    if not sat_sources:
        raise RuntimeError(f"No satellite inputs found under {sat_dir} for globs={sat_globs}")

    sat_by_ts_list = [_index_by_ts(lst) for lst in sat_sources]
    driver_ts = sorted(list(sat_by_ts_list[0].keys()))
    if not driver_ts:
        raise RuntimeError("No timestamped satellite files found")

    def _chan_count(path: Path) -> int:
        d = np.load(path, allow_pickle=True)
        return int(d["data"].shape[0])

    sat_chan_counts = [
        _chan_count(sat_by_ts_list[i][driver_ts[0]]) for i in range(len(sat_by_ts_list))
    ]

    wind_by_ts = _index_by_ts(wind_files) if include_wind else {}

    wind_ts = sorted(wind_by_ts.keys()) if include_wind else []

    out_dir.mkdir(parents=True, exist_ok=True)

    if not include_wind:
        print(
            f"[create_combined] no wind_*.npz found under {wind_dir}; producing satellite-only combined samples (no u/v placeholders)."
        )

    wrote = 0
    skipped = 0

    for ts in driver_ts:
        sat_datas: List[np.ndarray] = []
        sat_channels_all: List[str] = []

        for src_i, by_ts in enumerate(sat_by_ts_list):
            sp = by_ts.get(ts)
            if sp is None:
                if not args.allow_missing_satellite:
                    sat_datas = []
                    break
                ref_p = sat_by_ts_list[0].get(ts) or sat_by_ts_list[0][driver_ts[0]]
                ref = np.load(ref_p, allow_pickle=True)["data"]
                _, H, W = ref.shape
                sat_datas.append(np.zeros((sat_chan_counts[src_i], H, W), dtype=np.float32))
                sat_channels_all.extend([f"sat_missing_{src_i}_{k}" for k in range(sat_chan_counts[src_i])])
                continue

            sat = np.load(sp, allow_pickle=True)
            sat_data = sat["data"]
            if sat_data.ndim != 3:
                raise RuntimeError(f"Satellite data must be (C,H,W), got {sat_data.shape} in {sp}")
            sat_datas.append(sat_data.astype(np.float32))
            sat_channels = list(sat.get("channels", np.array([], dtype=object)))
            if sat_channels:
                sat_channels_all.extend([str(c) for c in sat_channels])
            else:
                sat_channels_all.extend([f"sat_{src_i}_{k}" for k in range(sat_data.shape[0])])

        if not sat_datas:
            skipped += 1
            continue

        sat_data = np.concatenate(sat_datas, axis=0)

        wind_data = None
        wind_channels: List[str] = []

        if include_wind:
            match_ts = _nearest(ts, wind_ts, tol) if wind_ts else None
            if match_ts is not None:
                w = np.load(wind_by_ts[match_ts], allow_pickle=True)
                wind_data = w["data"]
                if wind_data.ndim != 3:
                    raise RuntimeError(
                        f"Wind data must be (C,H,W), got {wind_data.shape} in {wind_by_ts[match_ts]}"
                    )
                wind_channels = list(w.get("channels", np.array(["u", "v"], dtype=object)))
            elif args.allow_missing_wind:
                # Fill zeros with 2 channels to keep channel count consistent when wind exists but a timestamp is missing.
                _, H, W = sat_data.shape
                wind_data = np.zeros((2, H, W), dtype=np.float32)
                wind_channels = ["u", "v"]
            else:
                skipped += 1
                continue

            if wind_data is None:
                skipped += 1
                continue

            if wind_data.shape[1:] != sat_data.shape[1:]:
                raise RuntimeError(
                    f"Shape mismatch: sat {sat_data.shape} vs wind {wind_data.shape} for timestamp {ts}. "
                    "Ensure both pipelines use the same ROI config/resolution."
                )

            combined = np.concatenate([sat_data.astype(np.float32), wind_data.astype(np.float32)], axis=0)
            channels = np.array(list(sat_channels_all) + list(wind_channels), dtype=object)
        else:
            combined = sat_data.astype(np.float32)
            channels = np.array(list(sat_channels_all), dtype=object)

        if args.nan_to_num:
            combined = np.nan_to_num(combined, nan=0.0)
        out_path = out_dir / f"combined_{ts.strftime('%Y%m%d_%H%M%S')}.npz"

        np.savez_compressed(
            out_path,
            data=combined,
            channels=channels,
            timestamp=np.array(ts.isoformat(), dtype=object),
        )
        wrote += 1

    print(f"[create_combined] done. wrote={wrote} skipped_no_wind={skipped}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise
