from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "project.yaml"

with open(CONFIG, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

paths = cfg["paths"]

RAW_DIR = ROOT / paths["raw"]
SAT_PROCESSED_DIR = ROOT / paths["sat_processed"]
WIND_PROCESSED_DIR = ROOT / paths["wind_processed"]
COMBINED_DIR = ROOT / paths["combined"]

def ensure_dirs():
    for p in [RAW_DIR, SAT_PROCESSED_DIR, WIND_PROCESSED_DIR, COMBINED_DIR]:
        Path(p).mkdir(parents=True, exist_ok=True)
