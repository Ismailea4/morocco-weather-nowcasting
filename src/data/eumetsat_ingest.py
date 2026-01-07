"""
EUMETSAT ingestion script (stub).
Responsibilities:
- List and download MSG SEVIRI and HRW BUFR products via `eumdac`.
- Save zips/BUFR to data/raw and append to manifest.csv.
- Filter by time window and approximate ROI (post-crop happens in preprocessing).
"""
from pathlib import Path
from datetime import datetime
import csv

from src.utils.paths import RAW_DIR, ensure_dirs

MANIFEST = RAW_DIR / "manifest.csv"

def append_manifest(file_path: Path, product: str, timestamp: datetime, checksum: str = ""):
    ensure_dirs()
    header = ["file", "product", "timestamp", "checksum"]
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    write_header = not MANIFEST.exists()
    with MANIFEST.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow([str(file_path), product, timestamp.isoformat(), checksum])

# TODO: Implement list/download using eumdac Python SDK or CLI wrappers.
