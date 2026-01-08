"""src.eval.efficiency

Computational efficiency analysis for the ViT nowcasting checkpoint.

- Parameter count
- Checkpoint size
- Simple inference latency benchmark (synthetic input)

Example:
  py -3.12 src/eval/efficiency.py \
    --checkpoint models/vit/run_x/best.pt \
    --out experiments/efficiency_run_x.md
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from src.models.vit_nowcasting import ViTNowcaster, ViTNowcasterConfig


def _param_count(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def _checkpoint_size_bytes(path: Path) -> int:
    try:
        return int(path.stat().st_size)
    except Exception:
        return 0


def _bench_inference(
    model: torch.nn.Module,
    *,
    device: torch.device,
    t_in: int,
    in_channels: int,
    image_size: int,
    batch_size: int,
    n_warmup: int,
    n_iters: int,
) -> dict:
    model.eval()

    x = torch.randn(batch_size, t_in, in_channels, image_size, image_size, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(int(n_warmup)):
            _y, _extras = model(x, return_attn=False)

    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(int(n_iters)):
            t0 = time.perf_counter()
            _y, _extras = model(x, return_attn=False)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

    t = np.array(times, dtype=np.float64)
    return {
        "batch_size": int(batch_size),
        "n_warmup": int(n_warmup),
        "n_iters": int(n_iters),
        "ms_per_batch_mean": float(t.mean() * 1000.0),
        "ms_per_batch_p50": float(np.percentile(t, 50) * 1000.0),
        "ms_per_batch_p90": float(np.percentile(t, 90) * 1000.0),
        "ms_per_batch_p99": float(np.percentile(t, 99) * 1000.0),
        "ms_per_sample_mean": float(t.mean() * 1000.0 / max(1, int(batch_size))),
    }


def _format_md(
    *,
    checkpoint_path: Path,
    vit_cfg: ViTNowcasterConfig,
    params: int,
    ckpt_bytes: int,
    bench: dict,
    device: str,
) -> str:
    lines = []
    lines.append("# Efficiency Report\n\n")
    lines.append(f"- Checkpoint: `{checkpoint_path.as_posix()}`\n")
    lines.append(f"- Device: `{device}`\n\n")

    lines.append("## Model Size\n\n")
    lines.append("| Item | Value |\n")
    lines.append("|---|---:|\n")
    lines.append(f"| Parameters | {params} |\n")
    lines.append(f"| Checkpoint size (MB) | {ckpt_bytes / (1024.0 * 1024.0):.3f} |\n\n")

    lines.append("## Inference Latency (Synthetic Input)\n\n")
    lines.append("Benchmark uses random normal input with the configured tensor shape.\n\n")
    lines.append("| Item | Value |\n")
    lines.append("|---|---:|\n")
    for k in [
        "batch_size",
        "n_warmup",
        "n_iters",
        "ms_per_batch_mean",
        "ms_per_batch_p50",
        "ms_per_batch_p90",
        "ms_per_batch_p99",
        "ms_per_sample_mean",
    ]:
        v = bench.get(k)
        if isinstance(v, float):
            lines.append(f"| {k} | {v:.4f} |\n")
        else:
            lines.append(f"| {k} | {v} |\n")

    lines.append("\n## Config Snapshot\n\n")
    lines.append("```json\n")
    lines.append(f"{json.dumps(asdict(vit_cfg), indent=2)}\n")
    lines.append("```\n")
    return "".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda")
    ap.add_argument("--t-in", type=int, default=4, help="Number of input frames (T_in) for the benchmark")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--n-warmup", type=int, default=10)
    ap.add_argument("--n-iters", type=int, default=50)
    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(str(ckpt_path))

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(ckpt_path, map_location=device)
    vit_cfg = ViTNowcasterConfig(**ckpt["vit_cfg"])

    model = ViTNowcaster(vit_cfg).to(device)
    model.load_state_dict(ckpt["model"], strict=True)

    params = _param_count(model)
    ckpt_bytes = _checkpoint_size_bytes(ckpt_path)

    bench = _bench_inference(
        model,
        device=device,
        t_in=int(args.t_in),
        in_channels=int(vit_cfg.in_channels),
        image_size=int(vit_cfg.image_size),
        batch_size=int(args.batch_size),
        n_warmup=int(args.n_warmup),
        n_iters=int(args.n_iters),
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        _format_md(
            checkpoint_path=ckpt_path,
            vit_cfg=vit_cfg,
            params=params,
            ckpt_bytes=ckpt_bytes,
            bench=bench,
            device=str(device),
        ),
        encoding="utf-8",
    )

    try:
        print(f"Wrote {os.path.relpath(out_path)}")
    except Exception:
        print("Wrote efficiency report")


if __name__ == "__main__":
    main()
