# Data Engineer Deliverables

## Overview
Responsible for the complete data pipeline: ingestion, preprocessing, validation, and preparation of combined datasets for model training.

## Development Workflow

⚠️ **Important**: Start with **notebook prototyping** before implementing production code in `src/`.

### Phase 1: Notebook Prototyping & Exploration
- **Notebook**: `notebooks/01_data_pipeline_prototype.ipynb`
- Prototype and validate the entire data pipeline interactively
- Include comprehensive visualizations:
  - Sample satellite images (raw and processed)
  - Channel-by-channel comparisons
  - Wind vector overlay maps
  - Data quality statistics (histograms, spatial coverage)
  - Temporal alignment diagnostics
- Document data quirks, edge cases, and processing decisions
- Validate data formats and contracts
- Calculate dataset statistics and coverage metrics
- **Acceptance**: Fully functional pipeline in notebook with clear visualizations demonstrating data quality

### Phase 2: Production Code Implementation
- Refactor validated notebook code into modular scripts in `src/`
- Add error handling, logging, and robustness
- Maintain same logic validated in notebooks

## Core Deliverables

### 1. Data Ingestion & Catalog
- **Raw data directory**: `data/raw/` populated with Meteosat SEVIRI and HRW wind files
- **Manifest file**: `data/raw/manifest.csv` containing:
  - Timestamp
  - Product type (SEVIRI/HRW)
  - File path
  - Checksum for integrity verification
- **Ingestion script**: `src/data/eumetsat_ingest.py`
  - Automated download from EUMETSAT Data Store
  - Date range handling
  - Retry logic for failed downloads
  - Logging and error handling

### 2. Satellite Data Preprocessing
- **Processed satellite data**: `data/processed/satellite/*.npz`
  - Multi-channel arrays (IR_108, IR_120, VIS006, VIS008, Water Vapor)
  - Cropped to Morocco ROI (lat 21-36°N, lon -17 to -1°E)
  - Resampled to consistent resolution
- **Preview images**: PNG files for quick visual inspection
- **Processing script**: `src/preprocess/satpy_pipeline.py`
  - Satpy-based SEVIRI native format reader
  - Pyresample integration for georeferencing
  - Quality control and NaN handling

### 3. Wind Data Processing
- **Processed wind data**: `data/processed/wind/*.npz`
  - U/V wind components
  - Wind speed grids
  - Pressure levels
  - Gridded to match satellite resolution
- **BUFR parsing script**: Handles HRW Atmospheric Motion Vectors
  - Convert wind speed/direction to U/V components
  - Spatial binning to regular grid
  - Temporal alignment with satellite data

### 4. Combined Dataset Creation
- **Combined arrays**: `data/combined/combined_YYYYmmdd_HHMMSS.npz`
  - Stacked channels: [satellite_channels, wind_u, wind_v, wind_speed]
  - Consistent spatial dimensions
  - Normalized and quality-checked
- **Matching metadata**: `data/combined/matched_pairs.csv`
  - Timestamp pairs (satellite-wind)
  - Time delta statistics
  - Data quality flags
- **Alignment logic**: Match satellite and wind data within ±15 minute tolerance

## Quality Assurance
- Coverage reports: percentage of successfully processed files
- Data integrity checks: shape validation, NaN statistics, value ranges
- Processing logs with timestamps and error tracking
- Documentation of any data gaps or anomalies

## Scripts & Tools
- `src/data/eumetsat_ingest.py` - Data ingestion from EUMETSAT
- `src/preprocess/satpy_pipeline.py` - Satellite preprocessing
- `src/preprocess/wind_gridding.py` - Wind data gridding (if applicable)
- `src/utils/paths.py` - Path management utilities

## Success Metrics
- ≥90% of raw files successfully processed
- ≥70% temporal matching success rate between satellite and wind data
- All data contracts (shapes, types, ranges) validated
- Complete documentation and reproducible pipeline
