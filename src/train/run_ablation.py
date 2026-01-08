"""src.train.run_ablation

Run ablation studies across multiple ViT configurations.

Usage:
    py -3.12 -m src.train.run_ablation --configs configs/ablation/*.yaml
    py -3.12 -m src.train.run_ablation --configs configs/ablation/vit_patch16_base.yaml configs/ablation/vit_deep.yaml
"""

from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import yaml


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _run_single_config(config_path: Path, run_id: str) -> Dict:
    """Run training for a single config and return results."""
    
    repo_root = _repo_root()
    cmd = [
        sys.executable,
        "-m", "src.train.train_vit",
        "--config", str(config_path),
        "--run-id", run_id,
    ]
    
    print(f"\n{'='*60}")
    print(f"Running: {config_path.name} as {run_id}")
    print(f"{'='*60}")
    
    t0 = time.time()
    result = subprocess.run(
        cmd,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - t0
    
    return {
        "config": str(config_path.relative_to(repo_root)),
        "run_id": run_id,
        "returncode": result.returncode,
        "elapsed_seconds": elapsed,
        "stdout": result.stdout[-2000:] if result.stdout else "",  # Last 2000 chars
        "stderr": result.stderr[-2000:] if result.stderr else "",
    }


def _load_metrics(run_id: str) -> Optional[Dict]:
    """Load final metrics from an experiment run."""
    
    repo_root = _repo_root()
    metrics_path = repo_root / "experiments" / "vit_nowcasting" / run_id / "metrics.jsonl"
    
    if not metrics_path.exists():
        return None
    
    # Get the last validation row
    val_rows = []
    with open(metrics_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line.strip())
            if row.get("phase") == "val":
                val_rows.append(row)
    
    return val_rows[-1] if val_rows else None


def _load_config_summary(config_path: Path) -> Dict:
    """Extract key config values for the summary table."""
    
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model = cfg.get("model", {})
    train = cfg.get("train", {})
    
    return {
        "patch_size": model.get("patch_size", "?"),
        "embed_dim": model.get("embed_dim", "?"),
        "depth": model.get("depth", "?"),
        "num_heads": model.get("num_heads", "?"),
        "temporal_fusion": model.get("temporal_fusion", "?"),
        "lr": train.get("lr", "?"),
        "batch_size": train.get("batch_size", "?"),
    }


def _generate_summary_table(results: List[Dict]) -> str:
    """Generate markdown summary table of ablation results."""
    
    lines = [
        "# Ablation Study Results",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Configuration Summary",
        "",
        "| Config | Patch | Dim | Depth | Heads | Fusion | Val Loss | MAE | RMSE | Time (s) |",
        "|--------|-------|-----|-------|-------|--------|----------|-----|------|----------|",
    ]
    
    for r in results:
        cfg = r.get("config_summary", {})
        metrics = r.get("metrics", {})
        
        val_loss = metrics.get("val_loss", float("nan"))
        mae = metrics.get("mae", float("nan"))
        rmse = metrics.get("rmse", float("nan"))
        
        row = (
            f"| {r.get('config_name', '?')} "
            f"| {cfg.get('patch_size', '?')} "
            f"| {cfg.get('embed_dim', '?')} "
            f"| {cfg.get('depth', '?')} "
            f"| {cfg.get('num_heads', '?')} "
            f"| {cfg.get('temporal_fusion', '?')[:4]} "
            f"| {val_loss:.4f} "
            f"| {mae:.4f} "
            f"| {rmse:.4f} "
            f"| {r.get('elapsed_seconds', 0):.1f} |"
        )
        lines.append(row)
    
    # Add analysis section
    lines.extend([
        "",
        "## Analysis",
        "",
    ])
    
    # Find best config
    valid_results = [r for r in results if r.get("metrics", {}).get("val_loss") is not None]
    if valid_results:
        best = min(valid_results, key=lambda x: x["metrics"].get("val_loss", float("inf")))
        lines.append(f"**Best configuration**: {best.get('config_name', '?')} (val_loss={best['metrics']['val_loss']:.4f})")
    
    # Observations
    lines.extend([
        "",
        "### Key Observations",
        "",
        "1. **Patch Size Effect**: Smaller patches capture finer details but increase computation.",
        "2. **Model Depth**: Deeper models may overfit on small datasets.",
        "3. **Temporal Fusion**: Attention-based fusion provides interpretable weights.",
        "",
        "## Run Details",
        "",
    ])
    
    for r in results:
        status = "✅ Success" if r.get("returncode") == 0 else "❌ Failed"
        lines.append(f"- **{r.get('config_name', '?')}**: {status} ({r.get('run_id', '?')})")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Run ViT ablation studies")
    parser.add_argument(
        "--configs",
        nargs="+",
        required=True,
        help="Config files to run (supports glob patterns)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/ablation_summary.md",
        help="Output summary file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configs that would be run without executing",
    )
    args = parser.parse_args()
    
    repo_root = _repo_root()
    
    # Expand glob patterns
    config_paths = []
    for pattern in args.configs:
        if "*" in pattern:
            matches = glob.glob(str(repo_root / pattern))
            config_paths.extend(Path(m) for m in matches)
        else:
            p = Path(pattern)
            if not p.is_absolute():
                p = repo_root / p
            if p.exists():
                config_paths.append(p)
    
    config_paths = sorted(set(config_paths))
    
    if not config_paths:
        print("No config files found!")
        return
    
    print(f"Found {len(config_paths)} configurations:")
    for p in config_paths:
        print(f"  - {p.relative_to(repo_root)}")
    
    if args.dry_run:
        print("\nDry run - exiting without training.")
        return
    
    # Run each config
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = []
    
    for i, config_path in enumerate(config_paths, 1):
        config_name = config_path.stem
        run_id = f"ablation_{config_name}_{timestamp}"
        
        print(f"\n[{i}/{len(config_paths)}] Processing {config_name}...")
        
        run_result = _run_single_config(config_path, run_id)
        run_result["config_name"] = config_name
        run_result["config_summary"] = _load_config_summary(config_path)
        
        # Load metrics if training succeeded
        if run_result["returncode"] == 0:
            metrics = _load_metrics(run_id)
            run_result["metrics"] = metrics or {}
        else:
            run_result["metrics"] = {}
            print(f"  ERROR: {run_result['stderr'][:500]}")
        
        results.append(run_result)
    
    # Generate summary
    summary = _generate_summary_table(results)
    output_path = repo_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(summary, encoding="utf-8")
    
    print(f"\n{'='*60}")
    print(f"Ablation study complete! Summary: {output_path}")
    print(f"{'='*60}")
    
    # Also save raw results as JSON
    json_path = output_path.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        # Don't include stdout/stderr in JSON (too verbose)
        clean_results = [
            {k: v for k, v in r.items() if k not in ("stdout", "stderr")}
            for r in results
        ]
        json.dump(clean_results, f, indent=2)
    print(f"Raw results: {json_path}")


if __name__ == "__main__":
    main()
