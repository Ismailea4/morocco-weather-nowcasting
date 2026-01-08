"""src.eval.significance

Paired statistical significance tests for forecast artifacts.

This module operates purely on exported `forecasts.npz` artifacts (pred/target)
and does not require access to model code.

Currently implemented:
- Paired bootstrap on per-sample metric differences

Example:
  py -3.12 src/eval/significance.py \
    --baseline reports/evaluation/baseline/persistence_run_x/forecasts.npz \
    --vit reports/evaluation/vit/run_x/forecasts.npz \
    --out experiments/significance_run_x.md
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import os

import numpy as np

from src.eval.artifacts import load_forecast_npz


@dataclass
class BootstrapResult:
    metric: str
    n: int
    n_boot: int
    seed: int
    mean_diff: float
    ci_low: float
    ci_high: float
    p_one_sided: float


def _per_sample_mae(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    # pred/target: (B,T,C,H,W) (or any shape with batch as axis 0)
    err = np.abs(pred - target)
    return err.reshape(err.shape[0], -1).mean(axis=1)


def _per_sample_rmse(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    err2 = (pred - target) ** 2
    return np.sqrt(err2.reshape(err2.shape[0], -1).mean(axis=1))


def _metric_fn(name: str):
    name = name.lower().strip()
    if name == "mae":
        return _per_sample_mae
    if name == "rmse":
        return _per_sample_rmse
    raise ValueError(f"Unknown metric: {name}. Use mae|rmse")


def paired_bootstrap(
    vit_pred: np.ndarray,
    baseline_pred: np.ndarray,
    target: np.ndarray,
    *,
    metric: str,
    n_boot: int,
    seed: int,
) -> BootstrapResult:
    fn = _metric_fn(metric)
    vit = fn(vit_pred, target)
    base = fn(baseline_pred, target)

    if vit.shape != base.shape:
        raise ValueError(f"Per-sample arrays must match. Got vit={vit.shape} base={base.shape}")

    # Define improvement as base - vit (positive => ViT better/lower error).
    diff = base - vit
    n = int(diff.shape[0])

    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, n, size=(int(n_boot), n), endpoint=False)
    boot_means = diff[idx].mean(axis=1)

    mean_diff = float(diff.mean())
    ci_low = float(np.percentile(boot_means, 2.5))
    ci_high = float(np.percentile(boot_means, 97.5))

    # One-sided p-value for H0: mean_diff <= 0 (ViT not better).
    p_one_sided = float((boot_means <= 0.0).mean())

    return BootstrapResult(
        metric=str(metric),
        n=n,
        n_boot=int(n_boot),
        seed=int(seed),
        mean_diff=mean_diff,
        ci_low=ci_low,
        ci_high=ci_high,
        p_one_sided=p_one_sided,
    )


def _format_md(res: BootstrapResult, *, vit_path: str, base_path: str) -> str:
    lines = []
    lines.append("# Statistical Significance (Paired Bootstrap)\n\n")
    lines.append("This report uses paired bootstrap resampling over per-sample errors.\n\n")
    lines.append(f"- ViT artifact: `{vit_path.replace('\\\\','/')}`\n")
    lines.append(f"- Baseline artifact: `{base_path.replace('\\\\','/')}`\n\n")

    lines.append("## Result\n\n")
    lines.append(
        "Improvement is defined as `baseline_error - vit_error` (positive means ViT is better).\n\n"
    )
    lines.append("| Metric | N samples | N bootstrap | Mean improvement | 95% CI | p (one-sided) |\n")
    lines.append("|---|---:|---:|---:|---:|---:|\n")
    lines.append(
        f"| {res.metric.upper()} | {res.n} | {res.n_boot} | {res.mean_diff:.6f} | [{res.ci_low:.6f}, {res.ci_high:.6f}] | {res.p_one_sided:.6f} |\n"
    )
    return "".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", type=str, required=True)
    ap.add_argument("--vit", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--metric", type=str, default="mae", help="mae|rmse")
    ap.add_argument("--n-bootstrap", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    base = load_forecast_npz(args.baseline)
    vit = load_forecast_npz(args.vit)

    if base.target.shape != vit.target.shape:
        raise ValueError(
            f"Targets differ between artifacts: baseline={base.target.shape} vit={vit.target.shape}. "
            "Paired significance requires identical targets."
        )

    res = paired_bootstrap(
        vit_pred=vit.pred,
        baseline_pred=base.pred,
        target=vit.target,
        metric=args.metric,
        n_boot=args.n_bootstrap,
        seed=args.seed,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_format_md(res, vit_path=args.vit, base_path=args.baseline), encoding="utf-8")
    try:
        print(f"Wrote {os.path.relpath(out_path)}")
    except Exception:
        print("Wrote significance report")


if __name__ == "__main__":
    main()
