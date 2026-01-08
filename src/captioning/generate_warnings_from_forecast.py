from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

from .warnings import generate_warning, render_warning_message, optional_llm_caption


def load_features(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Generate textual weather warnings from forecast features.")
    parser.add_argument("--features", required=True, help="Path to JSON file containing forecast_map_features.")
    parser.add_argument("--outdir", default=str(Path(__file__).resolve().parents[2] / "warnings"), help="Output directory for warning files.")
    parser.add_argument("--llm", action="store_true", help="Enable optional LLM captioning (placeholder).")
    args = parser.parse_args()

    features = load_features(args.features)
    alerts = generate_warning(features)
    message = render_warning_message(alerts, features)

    # Optional LLM caption (currently template-based placeholder)
    llm_text = optional_llm_caption(features, enabled=args.llm)
    if llm_text:
        message = message + "\n\n" + llm_text

    ts = features.get("timestamp", "unknown_time")
    outdir = Path(args.outdir)
    ensure_dir(outdir)
    outfile = outdir / f"{ts}_warning.txt"

    with open(outfile, "w", encoding="utf-8") as f:
        f.write(message + "\n")

    print(str(outfile))


if __name__ == "__main__":
    main()
