# AI Meteorologist (Weather Nowcasting) — Project Plan

Goal: Predict immediate future weather (next 0–4 hours) using satellite imagery (Meteosat SEVIRI) and time-series weather products (EUMETSAT HRW winds), and generate both maps and concise textual warnings.

## Overview
- Primary source (numbers): ClimateBench for benchmark climate variables (temp/precip) — optional auxiliary/regression targets.
- Visual data (core): Meteosat Second Generation (MSG/SEVIRI) geostationary satellites from EUMETSAT (15-minute cadence) for cloud movement and water vapor.
- Wind mapping: EUMETSAT HRW (Atmospheric Motion Vectors) in BUFR format for U/V/speed derived from tracked cloud features.
- Region of Interest (ROI): Northern Morocco. Suggested bbox for nowcasting: lat 28°–37°, lon −13° to −1° (adjustable in configs).
- Deliverables: Spatial forecasts (maps) + textual warnings; baseline ConvLSTM; upgraded ViT; optional LLM captions.

## Dataset Form & Constraints
- Satellite files: Zipped SEVIRI L1b native (.nat inside .zip) or HRIT/NetCDF variants. Channels of interest: IR_108, IR_120, VIS006, VIS008, Water Vapor.
- Wind files: Zipped BUFR with HRW vectors; parse to latitude, longitude, windSpeed, windDirection, pressure, datetime.
- Temporal cadence: ~15 min (satellite) with rapid scan availability over Europe; HRW near-synchronous windows.
- Storage: Expect 0.5–2 GB/day cropped ROI; combined/processed datasets 2–3× raw size; plan for 500 GB for pilot.

## Team Roles (3 members)
1. Data Engineer
   - Own ingestion, catalog, preprocessing, wind gridding, and combined dataset creation.
2. ML Engineer
   - Own dataset loaders, baseline ConvLSTM training, ViT upgrade, and optional LLM captioning.
3. Integration & Evaluator
   - Own metrics, visuals/animations, experiment tracking, and repo hygiene.

## Work Breakdown (Tasks, Deliverables, Acceptance)

### A. Data Engineering
1. Ingestion (EUMETSAT via eumdac CLI/SDK)
   - Inputs: Credentials; product codes `EO:EUM:DAT:MSG:HRSEVIRI` (SEVIRI), HRW BUFR collection.
   - Outputs: Raw zips under `data/raw/` + `manifest.csv` with file, product, timestamp, checksum.
   - Acceptance: Automated for any date range; retries; logs.

2. Raw Catalog & Validation
   - Generate `data/raw/manifest.csv`; checksum verification; incomplete/corrupt zip detection.
   - Acceptance: Manifest reflects 100% of files; bad files quarantined or re-downloaded.

3. Satpy Preprocessing (SEVIRI)
   - Unzip → read `.nat` via `seviri_l1b_native`; load channels; resample/crop to ROI using Pyresample.
   - Save `.npz` per timestamp to `data/processed/satellite/` with channel arrays; save PNG preview.
   - Acceptance: ≥90% files processed; shapes/NaN rates logged.

4. Wind BUFR Gridding (HRW)
   - Parse BUFR → filter ROI; convert speed/direction to U/V; bin to regular grid.
   - Save `.npz` to `data/processed/wind/` with grid arrays and bin metadata.
   - Acceptance: Non-empty grids; stats report (counts per cell, mean speed).

5. Temporal Alignment & Combination
   - Match satellite/wind within ±15 min; resize wind grids to satellite resolution; stack `[satellite, wind_u, wind_v, wind_speed]`.
   - Save combined `.npz` to `data/combined/` + `matched_pairs.csv`.
   - Acceptance: Matched pairs ≥70% of available timestamps; integrity checks (shapes/time deltas).

### B. ML Engineering
6. Dataset Loader & Splits
   - Implement `MoroccoWeatherDataset` with sequence sampling (e.g., 6 past → 1 next), normalizations, train/val/test split.
   - Acceptance: Can iterate batches on combined data without errors; summary stats printed.

7. Baseline Model: ConvLSTM
   - Simple encoder–decoder ConvLSTM; train to predict next-step satellite + wind maps.
   - Acceptance: Trains on sample set; logs loss/metrics; checkpoint saved to `models/baseline/`.

8. Vision Transformer (ViT) Upgrade
   - ViT-based spatial encoder + temporal fusion (e.g., temporal Conv/VL); compare against baseline.
   - Acceptance: Reproducible run (config/seed); metrics table; visual examples.

9. LLM Captioning (Optional/MVP)
   - Use frozen lightweight LLM (local or API-free mock) to generate warnings from high-level features (e.g., low pressure proxy, strong wind shear, convective signatures).
   - Acceptance: Captions per timestamp saved; qualitative examples.

### C. Integration & Evaluation
10. Metrics & Visuals
   - Implement RMSE for continuous maps; event-based metrics (CSI, POD, FAR) with thresholds.
   - Produce plots and animations over time.
   - Acceptance: Figures stored in `reports/evaluation/`; metric summary per run.

11. Experiment Tracking & Reproducibility
   - Config-driven runs (`configs/`), run metadata to `experiments/` (config, seed, dataset version, git commit, metrics).
   - Acceptance: Any run can be reproduced exactly; comparison table across runs.

12. Repo Hygiene & Contracts
   - `requirements.txt`, `.gitignore`, folder contracts, README sections; optional pre-commit hooks.
   - Acceptance: Clean structure; documented paths and conventions.

## Milestones (Indicative 4-week cadence)
- Week 1: Ingestion, catalog, satpy preprocessing, wind gridding (Tasks 1–4).
- Week 2: Alignment/combination, dataset loader, baseline ConvLSTM (Tasks 5–7).
- Week 3: ViT upgrade, metrics foundations, first visuals (Tasks 8, 10).
- Week 4: LLM captions, full evaluation, experiment tracking, repo hygiene (Tasks 9–12).

## Folder Structure & Contracts
- Raw: `data/raw/` — zipped satellite/BUFR files; `manifest.csv` required.
- Processed: `data/processed/satellite/` `.npz` + PNG; `data/processed/wind/` `.npz`.
- Combined: `data/combined/` `combined_YYYYmmdd_HHMMSS.npz` stacks; `matched_pairs.csv`.
- Code: `src/data/`, `src/preprocess/`, `src/datasets/`, `src/models/`, `src/train/`, `src/eval/`, `src/utils/`.
- Configs: `configs/project.yaml` (ROI, paths, cadence) + run configs.
- Reports: `reports/evaluation/` figures/animations.
- Experiments: `experiments/` run metadata and summaries.

## Return Artifacts After Each Task
- Data Engineer: `data/raw/manifest.csv`, processed satellite/wind `.npz`, combined stacks, matching CSV.
- ML Engineer: dataset loader, training logs/checkpoints, metrics tables.
- Integration & Evaluator: metrics scripts, figures/animations, experiment records.

## Risks & Mitigations
- Data gaps/missing timestamps → implement tolerance windows, interpolation, and robust matching.
- Large storage/compute → crop ROI early; Dask for lazy processing; sample-based training.
- Format quirks (.nat/HRIT/BUFR) → Satpy/Pyresample/PDBUFR with fallbacks; enhanced diagnostics.

## Tools & Dependencies
- Data: EUMETSAT Data Store (`eumdac`), Satpy, Pyresample, PDBUFR, Xarray, Dask.
- ML: PyTorch or TensorFlow; scikit-learn; optional small LLM.
- Visualization: Matplotlib, Cartopy.
- Config: YAML; experiment logs in JSON/CSV.

## Acceptance Summary
- End-to-end pipeline: Raw → Processed → Combined → Sequences.
- Baseline ConvLSTM and ViT trained; metrics reported; visuals produced.
- Optional captions generated; all runs tracked and reproducible.

---

References to current repo scaffolding:
- Utils and paths: `src/utils/paths.py`
- Ingestion: `src/data/eumetsat_ingest.py`
- Preprocessing: `src/preprocess/satpy_pipeline.py`
- Dataset loader: `src/datasets/morocco_dataset.py`
- Models: `src/models/conv_lstm_baseline.py`, `src/models/vit_nowcasting.py`
- Training stub: `src/train/train_baseline.py`
- Metrics stub: `src/eval/metrics.py`
- Config: `configs/project.yaml`
