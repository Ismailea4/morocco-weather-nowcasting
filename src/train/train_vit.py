"""src.train.train_vit

Config-driven training script for the ViT nowcasting model.

This script is intentionally self-contained and avoids any baseline model code.
Baseline comparison should be done by loading baseline evaluation outputs.

Usage (example):
    py -3.12 src/train/train_vit.py --config configs/vit_config_l2.yaml

Expected data format (current repo stub):
  data/combined/combined_*.npz with key "data".
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from src.datasets.morocco_dataset import MoroccoWeatherDataset
from src.eval.artifacts import save_forecast_npz
from src.eval import metrics as M
from src.models.vit_nowcasting import ViTNowcaster, ViTNowcasterConfig


def _now_run_id() -> str:
    return datetime.now().strftime("run_%Y%m%d_%H%M%S")


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _repo_root() -> Path:
    # train_vit.py -> src/train/train_vit.py, so repo root is 2 parents up.
    return Path(__file__).resolve().parents[2]


def _rel_to_repo(p: Path, repo_root: Path) -> str:
    try:
        return p.resolve().relative_to(repo_root.resolve()).as_posix()
    except Exception:
        # Fall back to a normalized string without forcing absolute paths.
        return p.as_posix()


def _try_git_hash(repo_root: Path) -> str:
    try:
        p = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            text=True,
            capture_output=True,
            check=False,
        )
        if p.returncode == 0:
            h = (p.stdout or "").strip()
            return h if h else "unknown"
    except Exception:
        pass
    return "unknown"


def _dataset_fingerprint(paths) -> str:
    """Stable-ish dataset identifier from file list + basic file metadata."""

    h = hashlib.sha256()
    for p in paths:
        try:
            st = Path(p).stat()
            h.update(str(Path(p).name).encode("utf-8"))
            h.update(b"\0")
            h.update(str(int(st.st_size)).encode("utf-8"))
            h.update(b"\0")
            h.update(str(int(st.st_mtime)).encode("utf-8"))
            h.update(b"\n")
        except Exception:
            # Fall back to name only.
            h.update(str(Path(p).name).encode("utf-8"))
            h.update(b"\n")
    return h.hexdigest()


def _grad_norm_l2(model: torch.nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is None:
            continue
        v = float(p.grad.detach().data.norm(2).cpu().item())
        total += v * v
    return float(total ** 0.5)


def _as_torch_batch(x_np: np.ndarray, y_np: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert dataset arrays into torch tensors.

        Expected:
            x_np: (T_in, C, H, W)
            y_np: (T_out, C, H, W)
        """

        if x_np.ndim != 4 or y_np.ndim != 4:
                raise ValueError(f"Expected x,y 4D (T,C,H,W). Got x={x_np.shape} y={y_np.shape}")
        return torch.from_numpy(x_np).float(), torch.from_numpy(y_np).float()


def _time_split_file_indices(n_files: int, val_frac: float = 0.2, gap: int = 0) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Return (train_slice, val_slice) in file-index space.

    We split by time (ordered files), optionally leaving a gap of `gap` files
    between train and val to avoid any overlap/leakage.

    Each slice is (start_idx, end_idx) with end exclusive.
    """

    if n_files <= 0:
        raise ValueError("n_files must be > 0")

    n_val_files = max(1, int(round(n_files * float(val_frac))))
    n_train_files = n_files - n_val_files

    # Leave a gap so that windows (t_in+t_out) don't share frames across splits.
    train_end = max(0, n_train_files - int(gap))
    val_start = min(n_files, n_train_files + int(gap))
    if train_end <= 0 or (n_files - val_start) <= 0:
        # Fallback: no gap
        train_end = n_train_files
        val_start = n_train_files
    return (0, train_end), (val_start, n_files)


def _largest_contiguous_segment_indices(
    times, *, expected_step_min: float, step_tolerance_seconds: float
) -> Tuple[int, int]:
    """Return (start,end) file indices for the longest contiguous time run."""

    expected_s = float(expected_step_min) * 60.0
    tol_s = float(step_tolerance_seconds)
    n = len(times)
    if n <= 1:
        return (0, n)

    seg_start = 0
    best_start = 0
    best_end = 1
    for i in range(n - 1):
        d = (times[i + 1] - times[i]).total_seconds()
        if abs(d - expected_s) > tol_s:
            # segment ended at i+1
            seg_end = i + 1
            if (seg_end - seg_start) > (best_end - best_start):
                best_start, best_end = seg_start, seg_end
            seg_start = i + 1

    # last segment
    if (n - seg_start) > (best_end - best_start):
        best_start, best_end = seg_start, n

    return best_start, best_end


def _time_split_within_segment(
    seg: Tuple[int, int], *, val_frac: float, gap: int, seq_len: int
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Time split within a contiguous segment, leaving a gap to avoid leakage."""

    seg_start, seg_end = seg
    n = seg_end - seg_start
    if n <= 0:
        return (0, 0), (0, 0)

    # Ensure both slices have at least seq_len files.
    n_val = max(seq_len, int(round(n * float(val_frac))))
    n_val = min(n - seq_len, n_val) if n > seq_len else seq_len
    val_start = max(seg_start + seq_len, seg_end - n_val)
    train_end = max(seg_start + seq_len, val_start - int(gap))
    if (train_end - seg_start) < seq_len:
        # Fallback: drop gap if it's too restrictive.
        train_end = val_start
    return (seg_start, train_end), (val_start, seg_end)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/vit_config_l2.yaml")
    ap.add_argument("--run-id", type=str, default="")
    args = ap.parse_args()

    repo_root = _repo_root()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = (repo_root / cfg_path).resolve()
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    seed = int(cfg.get("seed", 42))
    _set_seed(seed)

    run_id = args.run_id or _now_run_id()
    log_cfg = cfg["logging"]

    def _resolve_out(p: str) -> Path:
        pp = Path(p)
        return pp if pp.is_absolute() else (repo_root / pp)

    exp_root = _resolve_out(log_cfg["experiment_root"]) / run_id
    ckpt_dir = _resolve_out(log_cfg["checkpoints_dir"]) / run_id
    report_dir = _resolve_out(log_cfg["report_dir"]) / run_id
    exp_root.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    # Save config + environment metadata
    (exp_root / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    meta = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(_device()),
        "seed": seed,
        "platform": platform.platform(),
        "git_hash": _try_git_hash(repo_root),
        "cpu_count": int(os.cpu_count() or 0),
        "torch_threads": int(torch.get_num_threads()),
    }
    (exp_root / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    data_cfg = cfg["data"]

    # Build a full list of files once, then split by time so train/val don't overlap.
    data_dir = str(data_cfg["data_dir"])
    data_dir_path = Path(data_dir)
    if not data_dir_path.is_absolute():
        data_dir_path = (repo_root / data_dir_path)
    data_dir = str(data_dir_path)
    all_ds = MoroccoWeatherDataset(
        data_dir=data_dir,
        t_in=int(data_cfg["t_in"]),
        t_out=int(data_cfg["t_out"]),
        stride=1,
    )

    n_files = len(all_ds.files)
    seq_len = int(data_cfg["t_in"]) + int(data_cfg["t_out"])

    # Record dataset fingerprint + planned split indices for reproducibility.
    meta.update(
        {
            "data_dir": _rel_to_repo(data_dir_path, repo_root),
            "data_files_total": int(n_files),
            "dataset_fingerprint": _dataset_fingerprint(all_ds.files),
        }
    )

    # Prefer splitting within the longest contiguous time segment to avoid crossing gaps.
    seg = _largest_contiguous_segment_indices(
        all_ds.times,
        expected_step_min=all_ds.expected_step_min,
        step_tolerance_seconds=all_ds.step_tolerance_seconds,
    )
    train_slice, val_slice = _time_split_within_segment(seg, val_frac=0.2, gap=seq_len, seq_len=seq_len)

    # If something went wrong (e.g., segment too short), fall back to whole-series split.
    if (train_slice[1] - train_slice[0]) < seq_len or (val_slice[1] - val_slice[0]) < seq_len:
        train_slice, val_slice = _time_split_file_indices(n_files, val_frac=0.2, gap=seq_len)

    meta.update({"longest_contiguous_segment": list(seg), "train_file_slice": list(train_slice), "val_file_slice": list(val_slice)})
    (exp_root / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    train_ds = MoroccoWeatherDataset(
        data_dir=data_dir,
        t_in=int(data_cfg["t_in"]),
        t_out=int(data_cfg["t_out"]),
        stride=1,
        start_file_index=train_slice[0],
        end_file_index=train_slice[1],
    )
    val_ds = MoroccoWeatherDataset(
        data_dir=data_dir,
        t_in=int(data_cfg["t_in"]),
        t_out=int(data_cfg["t_out"]),
        stride=1,
        start_file_index=val_slice[0],
        end_file_index=val_slice[1],
    )

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError(
            f"Empty split: train={len(train_ds)} val={len(val_ds)}. "
            "Try reducing t_in/t_out or val_frac, or remove the gap."
        )

    # Simple iterable loaders (dataset is numpy-based, so we keep it simple)
    target_image_size = int(data_cfg["image_size"])

    def _resize_if_needed(x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,C,H,W)
        if x.ndim != 5:
            return x
        b, t, c, h, w = x.shape
        if h == target_image_size and w == target_image_size:
            return x
        x2 = x.reshape(b * t, c, h, w)
        x2 = F.interpolate(x2, size=(target_image_size, target_image_size), mode="bilinear", align_corners=False)
        return x2.reshape(b, t, c, target_image_size, target_image_size)

    def batch_iter(ds: MoroccoWeatherDataset, batch_size: int):
        bs = int(batch_size)
        for s in range(0, len(ds), bs):
            Xs, Ys = [], []
            for i in range(s, min(len(ds), s + bs)):
                x_np, y_np = ds[int(i)]
                x_t, y_t = _as_torch_batch(x_np, y_np)
                Xs.append(x_t)
                Ys.append(y_t)
            x = torch.stack(Xs, dim=0)
            y = torch.stack(Ys, dim=0)
            x = _resize_if_needed(x)
            y = _resize_if_needed(y)
            yield x, y

    model_cfg = cfg["model"]
    vit_cfg = ViTNowcasterConfig(
        in_channels=int(data_cfg["in_channels"]),
        out_channels=int(data_cfg["out_channels"]),
        image_size=int(data_cfg["image_size"]),
        patch_size=int(model_cfg["patch_size"]),
        embed_dim=int(model_cfg["embed_dim"]),
        depth=int(model_cfg["depth"]),
        num_heads=int(model_cfg["num_heads"]),
        mlp_ratio=float(model_cfg["mlp_ratio"]),
        dropout=float(model_cfg["dropout"]),
        attn_dropout=float(model_cfg.get("attn_dropout", 0.0)),
        temporal_fusion=str(model_cfg["temporal_fusion"]),
        temporal_heads=int(model_cfg.get("temporal_heads", 4)),
        t_out=int(model_cfg["t_out"]),
    )

    device = _device()
    model = ViTNowcaster(vit_cfg).to(device)

    train_cfg = cfg["train"]
    opt_name = str(train_cfg.get("optimizer", "adamw")).lower()
    lr = float(train_cfg["lr"])
    weight_decay = float(train_cfg.get("weight_decay", 0.0))

    if opt_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == "lion":
        try:
            from lion_pytorch import Lion
            optimizer = Lion(model.parameters(), lr=lr, weight_decay=weight_decay)
        except ImportError:
            print("Warning: Lion optimizer not installed, falling back to AdamW.")
            print("Install with: pip install lion-pytorch")
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == "sgd":
        momentum = float(train_cfg.get("momentum", 0.9))
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")

    warmup_steps = int(train_cfg.get("warmup_steps", 0))
    total_steps = int(train_cfg.get("total_steps", 1000))
    min_lr = float(train_cfg.get("min_lr", 0.0))

    def lr_at(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return lr * float(step + 1) / float(warmup_steps)
        # cosine decay
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        t = min(max(t, 0.0), 1.0)
        return min_lr + 0.5 * (lr - min_lr) * (1.0 + np.cos(np.pi * t))

    use_amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    # Use modern torch.amp APIs to avoid FutureWarnings (which print temp absolute paths in notebooks).
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if hasattr(torch, "amp") else torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val = float("inf")
    global_step = 0
    log_every = int(cfg["logging"].get("log_every_steps", 50))
    export_forecasts = bool(cfg["logging"].get("export_forecasts", True))
    export_batches = int(cfg["logging"].get("export_batches", 8))
    export_persistence_baseline = bool(cfg["logging"].get("export_persistence_baseline", True))
    event_thr = float(cfg.get("metrics", {}).get("event_threshold", 0.5))

    metrics_path = exp_root / "metrics.jsonl"

    epochs = int(train_cfg["epochs"])
    batch_size = int(train_cfg["batch_size"])
    grad_clip = float(train_cfg.get("grad_clip", 0.0))

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        train_losses = []

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        for x, y in batch_iter(train_ds, batch_size=batch_size):
            x = x.to(device)
            y = y.to(device)

            # set LR
            lr_now = lr_at(global_step)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_now

            optimizer.zero_grad(set_to_none=True)

            if hasattr(torch, "amp"):
                autocast_ctx = torch.amp.autocast(device_type="cuda", enabled=use_amp)
            else:
                autocast_ctx = torch.cuda.amp.autocast(enabled=use_amp)

            with autocast_ctx:
                yhat, _extras = model(x, return_attn=False)
                loss = F.mse_loss(yhat, y)

            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip).detach().cpu().item())
            else:
                # If we don't clip, still log an L2 grad norm.
                if use_amp:
                    scaler.unscale_(optimizer)
                gnorm = _grad_norm_l2(model)
            scaler.step(optimizer)
            scaler.update()

            train_losses.append(float(loss.detach().cpu().item()))

            if global_step % log_every == 0:
                row = {
                    "step": global_step,
                    "epoch": epoch,
                    "phase": "train",
                    "loss": float(np.mean(train_losses[-min(len(train_losses), 20):])),
                    "lr": lr_now,
                    "grad_norm": float(gnorm),
                }
                metrics_path.write_text("", encoding="utf-8") if not metrics_path.exists() else None
                with metrics_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")

            global_step += 1
            if global_step >= total_steps:
                break

        # Validation
        model.eval()
        val_losses = []
        # compute summary metrics on a few batches for speed
        with torch.no_grad():
            for bi, (x, y) in enumerate(batch_iter(val_ds, batch_size=batch_size)):
                x = x.to(device)
                y = y.to(device)
                yhat, _extras = model(x, return_attn=False)
                vloss = F.mse_loss(yhat, y)
                val_losses.append(float(vloss.detach().cpu().item()))
                if bi >= 4:
                    break

        val_loss = float(np.mean(val_losses)) if val_losses else float("inf")

        # Save best checkpoint
        ckpt = {
            "model": model.state_dict(),
            "vit_cfg": asdict(vit_cfg),
            "epoch": epoch,
            "step": global_step,
            "val_loss": val_loss,
        }
        torch.save(ckpt, ckpt_dir / "last.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, ckpt_dir / "best.pt")

        # Quick metric summary (numpy)
        # Note: event metrics threshold is generic; adjust per-channel precip later if needed.
        try:
            s = M.summary_table(yhat, y, threshold=event_thr)
        except Exception:
            s = {"mae": float(M.mae(yhat, y)), "rmse": float(M.rmse(yhat, y))}

        epoch_row = {
            "epoch": epoch,
            "phase": "val",
            "val_loss": val_loss,
            "train_loss": float(np.mean(train_losses)) if train_losses else float("nan"),
            "best_val": best_val,
            "sec_epoch": float(time.time() - t0),
            "max_cuda_mem_mb": float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)) if device.type == "cuda" else None,
            **s,
        }
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(epoch_row) + "\n")

        print(f"Epoch {epoch}/{epochs} | train={epoch_row['train_loss']:.4f} val={val_loss:.4f} best={best_val:.4f}")

        if global_step >= total_steps:
            break

    best_ckpt_path = ckpt_dir / "best.pt"
    last_ckpt_path = ckpt_dir / "last.pt"

    if export_forecasts:
        # Export a small validation artifact for later comparison (baseline vs ViT).
        # We run inference from the best checkpoint for consistency.
        if best_ckpt_path.exists():
            best_ckpt = torch.load(best_ckpt_path, map_location=device)
            model.load_state_dict(best_ckpt["model"], strict=True)

        model.eval()
        preds = []
        base_preds = []
        targets = []
        with torch.no_grad():
            for bi, (x, y) in enumerate(batch_iter(val_ds, batch_size=batch_size)):
                x = x.to(device)
                y = y.to(device)
                yhat, _extras = model(x, return_attn=False)
                preds.append(yhat.detach().cpu())
                targets.append(y.detach().cpu())

                # Infrastructure-only baseline proxy: persistence (repeat last input frame).
                if export_persistence_baseline:
                    t_out = int(y.shape[1])
                    base = x[:, -1:, ...].repeat(1, t_out, 1, 1, 1)
                    base_preds.append(base.detach().cpu())
                if bi + 1 >= max(1, export_batches):
                    break

        if preds and targets:
            pred_t = torch.cat(preds, dim=0)
            target_t = torch.cat(targets, dim=0)
            save_forecast_npz(
                report_dir / "forecasts.npz",
                pred=pred_t,
                target=target_t,
                meta={
                    **meta,
                    "checkpoint": _rel_to_repo(best_ckpt_path if best_ckpt_path.exists() else last_ckpt_path, repo_root),
                    "export_batches": int(export_batches),
                    "export_num_samples": int(pred_t.shape[0]),
                    "event_threshold": float(event_thr),
                    "vit_cfg": asdict(vit_cfg),
                },
            )

            if export_persistence_baseline and base_preds:
                base_t = torch.cat(base_preds, dim=0)
                base_report_dir = (repo_root / "reports" / "evaluation" / "baseline" / f"persistence_{run_id}").resolve()
                base_report_dir.mkdir(parents=True, exist_ok=True)
                save_forecast_npz(
                    base_report_dir / "forecasts.npz",
                    pred=base_t,
                    target=target_t,
                    meta={
                        **meta,
                        "baseline_type": "persistence",
                        "paired_vit_run_id": run_id,
                        "export_batches": int(export_batches),
                        "export_num_samples": int(base_t.shape[0]),
                        "event_threshold": float(event_thr),
                        "note": "Baseline proxy only; no baseline model training code included.",
                    },
                )

    # Final artifact pointers
    (exp_root / "artifacts.json").write_text(
        json.dumps(
            {
                "best_checkpoint": _rel_to_repo(best_ckpt_path, repo_root),
                "last_checkpoint": _rel_to_repo(last_ckpt_path, repo_root),
                "forecasts_npz": _rel_to_repo((report_dir / 'forecasts.npz'), repo_root),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
