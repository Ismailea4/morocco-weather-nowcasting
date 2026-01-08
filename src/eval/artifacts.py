"""src.eval.artifacts

File formats for storing model forecasts + ground truth for later comparison.

Goal: allow ViT engineer to wire comparison infrastructure without needing
baseline training code.

Recommended artifact: NPZ with the following keys:

- `pred`: float32 array shaped (B, T_out, C, H, W)
- `target`: float32 array shaped (B, T_out, C, H, W)
- `meta_json`: UTF-8 JSON string with metadata (optional)

This is intentionally simple and model-agnostic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach") and hasattr(x, "cpu") and hasattr(x, "numpy"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


@dataclass
class ForecastArtifact:
    pred: np.ndarray
    target: np.ndarray
    meta: Dict[str, Any]


def save_forecast_npz(
    out_path: str | Path,
    pred,
    target,
    meta: Optional[Dict[str, Any]] = None,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    p = _to_numpy(pred).astype(np.float32)
    t = _to_numpy(target).astype(np.float32)
    if p.shape != t.shape:
        raise ValueError(f"pred/target shape mismatch: {p.shape} vs {t.shape}")

    meta_json = json.dumps(meta or {}, indent=None)
    np.savez_compressed(out_path, pred=p, target=t, meta_json=np.array(meta_json))
    return out_path


def load_forecast_npz(path: str | Path) -> ForecastArtifact:
    path = Path(path)
    data = np.load(path, allow_pickle=True)
    pred = data["pred"].astype(np.float32)
    target = data["target"].astype(np.float32)
    meta_json = data.get("meta_json", "{}").item() if hasattr(data.get("meta_json", "{}"), "item") else data.get("meta_json", "{}")
    try:
        meta = json.loads(meta_json) if isinstance(meta_json, str) else {}
    except Exception:
        meta = {}
    return ForecastArtifact(pred=pred, target=target, meta=meta)
