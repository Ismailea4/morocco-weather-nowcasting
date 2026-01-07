"""
Satpy preprocessing pipeline (stub).
- Unzip MSG zips, load .nat via `seviri_l1b_native`.
- Crop/resample to Morocco ROI.
- Save per-timestamp npz + optional PNG.
"""
from pathlib import Path
import numpy as np

from src.utils.paths import SAT_PROCESSED_DIR, ensure_dirs

def process_zip(zip_path: Path) -> Path:
    """Process a single zip to ROI arrays. Returns output npz path."""
    ensure_dirs()
    # TODO: implement with Satpy/Pyresample as in notebook.
    raise NotImplementedError("Implement Satpy-based processing here")
