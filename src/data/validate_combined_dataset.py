"""src.data.validate_combined_dataset

Validate that `data/combined/combined_*.npz` is coherent for ViT training.

Checks:
- Timestamps are strictly increasing and have constant cadence (default 15 min)
- All samples share identical (C,H,W) shape
- All samples share identical channel ordering (if `channels` is present)
- No NaNs/Infs (optional)
- Config compatibility: (image_size divisible by patch_size), in/out channels match C

Usage:
  py -3.12 -m src.data.validate_combined_dataset --data-dir data/combined --expected-step-min 15
  py -3.12 -m src.data.validate_combined_dataset --config configs/vit_config_l2.yaml
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np


_TS_RE = re.compile(r"combined_(\d{8})_(\d{6})\.npz$")


def _parse_ts(path: str) -> datetime:
    m = _TS_RE.search(os.path.basename(path))
    if not m:
        # Fallback: try reading embedded timestamp from file.
        z = np.load(path, allow_pickle=True)
        ts = z.get("timestamp", None)
        if ts is None:
            raise ValueError(f"Could not parse timestamp from {path}")
        return datetime.fromisoformat(str(ts))
    return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")


@dataclass
class ValidationResult:
    ok: bool
    n_files: int
    time_start: datetime
    time_end: datetime
    unique_steps_min: List[float]
    step_min_min: float
    step_min_max: float
    n_bad_steps: int
    n_gaps: int
    longest_run: int
    shape: Tuple[int, int, int]
    channels: Optional[Tuple[str, ...]]
    nan_count: int
    inf_count: int


def validate(
    files: Sequence[str],
    expected_step_min: float = 15.0,
    step_tolerance_seconds: float = 90.0,
    check_finite: bool = True,
    max_finite_checks: int = 32,
) -> ValidationResult:
    if not files:
        raise RuntimeError("No combined_*.npz files found")

    times = [_parse_ts(f) for f in files]
    if any(times[i] >= times[i + 1] for i in range(len(times) - 1)):
        raise RuntimeError("Timestamps are not strictly increasing")

    expected_step_s = float(expected_step_min) * 60.0
    tol_s = float(step_tolerance_seconds)

    deltas_s = [(times[i + 1] - times[i]).total_seconds() for i in range(len(times) - 1)]
    deltas_m = [d / 60.0 for d in deltas_s]
    unique_steps = sorted(set(round(d, 6) for d in deltas_m))

    bad = [d for d in deltas_s if abs(d - expected_step_s) > tol_s]
    n_bad = len(bad)

    # A "gap" is a delta larger than expected + tol; those break contiguity.
    gap_idx = [i for i, d in enumerate(deltas_s) if d > expected_step_s + tol_s]
    n_gaps = len(gap_idx)

    # Longest contiguous run (in frames) given the tolerance
    longest = 1
    cur = 1
    for d in deltas_s:
        if abs(d - expected_step_s) <= tol_s:
            cur += 1
        else:
            longest = max(longest, cur)
            cur = 1
    longest = max(longest, cur)

    z0 = np.load(files[0], allow_pickle=True)
    data0 = z0["data"]
    if data0.ndim != 3:
        raise RuntimeError(f"Expected data (C,H,W); got {data0.shape} in {files[0]}")
    shape0 = tuple(int(x) for x in data0.shape)  # type: ignore[assignment]

    ch0 = z0.get("channels", None)
    channels0 = tuple(str(c) for c in list(ch0)) if ch0 is not None else None

    nan_count = 0
    inf_count = 0

    # validate all files: shape + channels ordering
    for f in files:
        z = np.load(f, allow_pickle=True)
        d = z["data"]
        if tuple(d.shape) != shape0:
            raise RuntimeError(f"Shape mismatch: expected {shape0} got {tuple(d.shape)} in {f}")

        ch = z.get("channels", None)
        if channels0 is not None:
            if ch is None:
                raise RuntimeError(f"Missing 'channels' in {f} (expected consistent channels)")
            ch_t = tuple(str(c) for c in list(ch))
            if ch_t != channels0:
                raise RuntimeError(f"Channels mismatch in {f}: expected {channels0} got {ch_t}")

    if check_finite:
        # sample a subset for finiteness checks (keeps it fast for large datasets)
        subset = list(files[: min(len(files), int(max_finite_checks))])
        for f in subset:
            d = np.load(f, allow_pickle=True)["data"].astype(np.float32, copy=False)
            nan_count += int(np.isnan(d).sum())
            inf_count += int(np.isinf(d).sum())
        if nan_count or inf_count:
            raise RuntimeError(f"Found NaNs/Infs in sampled files: nan={nan_count} inf={inf_count}")

    return ValidationResult(
        ok=True,
        n_files=len(files),
        time_start=times[0],
        time_end=times[-1],
        unique_steps_min=unique_steps,
        step_min_min=min(deltas_m) if deltas_m else float("nan"),
        step_min_max=max(deltas_m) if deltas_m else float("nan"),
        n_bad_steps=n_bad,
        n_gaps=n_gaps,
        longest_run=int(longest),
        shape=shape0,
        channels=channels0,
        nan_count=nan_count,
        inf_count=inf_count,
    )


def _load_config(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate combined_*.npz dataset for ViT")
    ap.add_argument("--data-dir", type=str, default="data/combined")
    ap.add_argument("--glob", type=str, default="combined_*.npz")
    ap.add_argument("--expected-step-min", type=float, default=15.0)
    ap.add_argument(
        "--step-tolerance-seconds",
        type=float,
        default=90.0,
        help="Allowed jitter around expected cadence before flagging as non-contiguous (default: 90s).",
    )
    ap.add_argument("--no-finite-check", action="store_true")
    ap.add_argument("--max-finite-checks", type=int, default=32)
    ap.add_argument(
        "--channel-stats",
        action="store_true",
        help="Print per-channel min/max/mean/std and fraction of zeros (streaming).",
    )
    ap.add_argument(
        "--max-stat-files",
        type=int,
        default=0,
        help="Limit number of files used for channel stats (0 = all).",
    )
    ap.add_argument("--config", type=str, default="", help="Optional training config to validate compatibility")

    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    files = sorted(glob.glob(str(data_dir / args.glob)))

    res = validate(
        files,
        expected_step_min=float(args.expected_step_min),
        step_tolerance_seconds=float(args.step_tolerance_seconds),
        check_finite=not bool(args.no_finite_check),
        max_finite_checks=int(args.max_finite_checks),
    )

    print("[validate_combined_dataset] OK")
    print(f"  files: {res.n_files}")
    print(
        f"  time:  {res.time_start} -> {res.time_end}"
        f" (expected_step_min={args.expected_step_min} tol_s={args.step_tolerance_seconds})"
    )
    if res.n_files > 1:
        print(
            f"  deltas_min: min={res.step_min_min:.6g} max={res.step_min_max:.6g}"
            f" unique_steps={len(res.unique_steps_min)} bad_steps={res.n_bad_steps} gaps={res.n_gaps} longest_run={res.longest_run}"
        )
    print(f"  shape: {res.shape} (C,H,W)")
    if res.channels is not None:
        print(f"  channels: {list(res.channels)}")

    if args.channel_stats:
        channels = res.channels if res.channels is not None else tuple(f"c{i}" for i in range(res.shape[0]))
        use_n = len(files) if int(args.max_stat_files) <= 0 else min(len(files), int(args.max_stat_files))
        stat_files = files[:use_n]

        c = res.shape[0]
        count = np.zeros((c,), dtype=np.int64)
        zeros = np.zeros((c,), dtype=np.int64)
        s1 = np.zeros((c,), dtype=np.float64)
        s2 = np.zeros((c,), dtype=np.float64)
        vmin = np.full((c,), np.inf, dtype=np.float64)
        vmax = np.full((c,), -np.inf, dtype=np.float64)

        for f in stat_files:
            d = np.load(f, allow_pickle=True)["data"].astype(np.float32, copy=False)
            flat = d.reshape(c, -1)
            count += flat.shape[1]
            zeros += (flat == 0).sum(axis=1, dtype=np.int64)
            s1 += flat.sum(axis=1, dtype=np.float64)
            s2 += (flat.astype(np.float64) ** 2).sum(axis=1)
            vmin = np.minimum(vmin, flat.min(axis=1))
            vmax = np.maximum(vmax, flat.max(axis=1))

        mean = s1 / np.maximum(count, 1)
        var = (s2 / np.maximum(count, 1)) - (mean**2)
        var = np.maximum(var, 0.0)
        std = np.sqrt(var)
        frac0 = zeros / np.maximum(count, 1)

        print(f"[validate_combined_dataset] Channel stats (files={use_n}/{len(files)}):")
        for i, name in enumerate(channels):
            warn = ""
            if std[i] < 1e-6:
                warn = "  (WARN: ~constant)"
            print(
                f"  {i:02d} {name:>18s}"
                f"  min={vmin[i]:.6g} max={vmax[i]:.6g} mean={mean[i]:.6g} std={std[i]:.6g} zero%={100.0*frac0[i]:.2f}%{warn}"
            )

    if args.config:
        cfg = _load_config(Path(args.config))
        data_cfg = cfg.get("data", {})
        model_cfg = cfg.get("model", {})

        image_size = int(data_cfg.get("image_size", model_cfg.get("image_size", res.shape[1])))
        patch_size = int(model_cfg.get("patch_size", 16))
        in_ch = int(data_cfg.get("in_channels", res.shape[0]))
        out_ch = int(data_cfg.get("out_channels", res.shape[0]))
        t_in = int(data_cfg.get("t_in", 4))
        t_out = int(data_cfg.get("t_out", 2))

        if image_size != res.shape[1] or image_size != res.shape[2]:
            print(
                "[validate_combined_dataset] WARNING: config image_size does not match dataset H,W; "
                "this is OK only if you resize/crop in the dataloader/training step."
            )
            print(f"  config image_size={image_size} dataset H,W={res.shape[1:]}")
        if image_size % patch_size != 0:
            raise RuntimeError(f"Config patch_size={patch_size} does not divide image_size={image_size}")
        if in_ch != res.shape[0] or out_ch != res.shape[0]:
            raise RuntimeError(f"Config in/out channels ({in_ch},{out_ch}) do not match dataset C={res.shape[0]}")
        if res.n_files < (t_in + t_out):
            raise RuntimeError(f"Not enough files for t_in+t_out={t_in+t_out} (have {res.n_files})")

        print("[validate_combined_dataset] Config compatibility: OK")
        print(f"  t_in={t_in} t_out={t_out} patch_size={patch_size}")


if __name__ == "__main__":
    main()
