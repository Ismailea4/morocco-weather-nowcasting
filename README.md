# Morocco Weather Nowcasting

A deep learning project for short-term weather forecasting (0-4 hours) over Northern Morocco using satellite imagery from Meteosat SEVIRI and wind data from EUMETSAT HRW.

## 🌤️ Project Overview

This project implements state-of-the-art deep learning models for **weather nowcasting** - predicting immediate future weather conditions from current observations. We combine:

- **Satellite imagery**: Meteosat Second Generation (MSG/SEVIRI) multi-channel data (IR, VIS, Water Vapor)
- **Wind field data**: EUMETSAT High Resolution Winds (HRW) atmospheric motion vectors
- **Region of Interest**: Northern Morocco (lat 21-36°N, lon -17 to -1°E)
- **Temporal resolution**: 15-minute cadence predictions

### Models
1. **Baseline**: ConvLSTM encoder-decoder for spatiotemporal forecasting
2. **Advanced**: Vision Transformer (ViT) with temporal fusion for enhanced spatial pattern recognition

### Key Features
- End-to-end pipeline from raw satellite data to predictions
- Multi-channel weather data fusion (satellite + wind)
- Comprehensive evaluation metrics (RMSE, MAE, SSIM, CSI, POD, FAR)
- Visualization of predictions and attention maps
- Reproducible experiments with configuration management

## 🎯 Suggested GitHub Repository Name

**`morocco-weather-nowcasting`**

Alternative names:
- `seviri-weather-forecasting`
- `satellite-nowcast-morocco`
- `ai-meteorologist-morocco`

## 👥 Team Structure

This project is organized into three specialized roles:

### 1. Data Engineer
**Responsibilities**: Data acquisition, preprocessing, and pipeline management
- EUMETSAT data ingestion (SEVIRI & HRW)
- Satellite image preprocessing with Satpy
- Wind field gridding from BUFR format
- Temporal alignment and dataset creation

📋 [See detailed deliverables](team/data-engineer/DELIVERABLES.md)

### 2. ConvLSTM Baseline Engineer
**Responsibilities**: Baseline model development and evaluation
- Dataset loader implementation
- ConvLSTM architecture design and training
- Comprehensive testing and metrics
- Visualization and experiment tracking

📋 [See detailed deliverables](team/convlstm-engineer/DELIVERABLES.md)

### 3. ViT Advanced Model Engineer
**Responsibilities**: Advanced model development and comparative analysis
- Vision Transformer architecture for weather prediction
- Advanced training techniques and optimization
- Attention visualization and interpretability
- Performance comparison with baseline

📋 [See detailed deliverables](team/vit-engineer/DELIVERABLES.md)

## 📁 Project Structure

```
morocco-weather-nowcasting/
├── configs/                      # Configuration files
│   └── project.yaml             # Project-wide settings (ROI, paths)
├── data/                        # Data directory (not in git)
│   ├── raw/                     # Raw satellite and wind files
│   ├── processed/               # Preprocessed data
│   │   ├── satellite/          # Processed SEVIRI .npz + previews
│   │   └── wind/               # Gridded wind data .npz
│   └── combined/               # Aligned satellite+wind datasets
├── experiments/                 # Experiment logs and results
│   ├── baseline_convlstm/
│   └── vit_nowcasting/
├── models/                      # Saved model checkpoints
│   ├── baseline/
│   └── vit/
├── reports/                     # Analysis and visualizations
│   └── evaluation/
│       ├── baseline/
│       └── vit/
├── src/                        # Source code
│   ├── data/                   # Data ingestion scripts
│   ├── preprocess/             # Preprocessing pipelines
│   ├── datasets/               # PyTorch Dataset classes
│   ├── models/                 # Model architectures
│   ├── train/                  # Training scripts
│   ├── eval/                   # Evaluation metrics
│   └── utils/                  # Utilities and helpers
├── team/                       # Team-specific deliverables
│   ├── data-engineer/
│   ├── convlstm-engineer/
│   └── vit-engineer/
├── .gitignore
├── requirements.txt
├── PROJECT_PLAN.md             # Detailed project plan
└── README.md
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- EUMETSAT Data Store credentials (for data ingestion)
- ~500 GB storage for pilot dataset
- GPU recommended for model training (CUDA-compatible)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/morocco-weather-nowcasting.git
   cd morocco-weather-nowcasting
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure EUMETSAT credentials**
   ```bash
   # Set up your EUMETSAT API credentials
   export EUMETSAT_KEY="your_key_here"
   export EUMETSAT_SECRET="your_secret_here"
   ```

### Data Pipeline Quick Start

#### 1. Ingest Raw Data
```bash
# Download satellite and wind data for a date range
python src/data/eumetsat_ingest.py \
    --start-date 2020-03-21 \
    --end-date 2020-03-21 \
    --products SEVIRI HRW \
    --output-dir data/raw
```

#### 2. Preprocess Satellite Data
```bash
# Process SEVIRI images to cropped, calibrated .npz arrays
python src/preprocess/satpy_pipeline.py \
    --input-dir data/raw \
    --output-dir data/processed/satellite \
    --roi-config configs/project.yaml
```

#### 3. Process Wind Data
```bash
# Grid HRW wind vectors to match satellite resolution
python src/preprocess/wind_gridding.py \
    --input-dir data/raw \
    --output-dir data/processed/wind \
    --roi-config configs/project.yaml
```

#### 4. Create Combined Dataset
```bash
# Align and combine satellite + wind data
python src/preprocess/create_combined.py \
    --satellite-dir data/processed/satellite \
    --wind-dir data/processed/wind \
    --output-dir data/combined \
    --tolerance-minutes 15
```

### Model Training Quick Start

#### Train Baseline ConvLSTM
```bash
# Train the ConvLSTM baseline model
python src/train/train_baseline.py \
    --config configs/convlstm_config.yaml \
    --data-dir data/combined \
    --output-dir models/baseline \
    --experiment-name baseline_run_001
```

Expected output:
- Model checkpoints in `models/baseline/`
- Training logs in `experiments/baseline_convlstm/baseline_run_001/`
- Metrics and visualizations in `reports/evaluation/baseline/`

#### Train ViT Advanced Model
```bash
# Train the Vision Transformer model
python src/train/train_vit.py \
    --config configs/vit_config.yaml \
    --data-dir data/combined \
    --output-dir models/vit \
    --experiment-name vit_run_001
```

Expected output:
- Model checkpoints in `models/vit/`
- Training logs in `experiments/vit_nowcasting/vit_run_001/`
- Metrics and visualizations in `reports/evaluation/vit/`

#### Evaluate Models
```bash
# Evaluate baseline model on test set
python src/eval/evaluate.py \
    --model-path models/baseline/best_model.pth \
    --model-type convlstm \
    --data-dir data/combined \
    --output-dir reports/evaluation/baseline

# Evaluate ViT model on test set
python src/eval/evaluate.py \
    --model-path models/vit/best_model.pth \
    --model-type vit \
    --data-dir data/combined \
    --output-dir reports/evaluation/vit
```

#### Compare Models
```bash
# Generate comparative analysis
python src/eval/compare_models.py \
    --baseline-results reports/evaluation/baseline \
    --vit-results reports/evaluation/vit \
    --output-dir experiments/comparison_summary
```

## 📊 Dataset

### Sources
- **Satellite**: EUMETSAT Meteosat Second Generation (MSG) SEVIRI Level 1B
  - Product: `EO:EUM:DAT:MSG:HRSEVIRI`
  - Channels: IR_108, IR_120, VIS006, VIS008, WV_062, WV_073
  - Cadence: 15 minutes
  - Format: Native (.nat) or HRIT

- **Wind**: EUMETSAT High Resolution Winds (HRW)
  - Atmospheric Motion Vectors from tracked cloud features
  - Format: BUFR
  - Variables: U/V components, wind speed, direction, pressure level

### Processed Dataset Structure
Each combined sample (`combined_YYYYmmdd_HHMMSS.npz`) contains:
```python
{
    'satellite': (H, W, C_sat),  # Multi-channel satellite imagery
    'wind_u': (H, W),             # U-component wind field
    'wind_v': (H, W),             # V-component wind field
    'wind_speed': (H, W),         # Wind speed magnitude
    'timestamp': datetime,        # Observation timestamp
    'metadata': {...}             # ROI bounds, resolution, etc.
}
```

## 🔬 Evaluation Metrics

### Continuous Metrics
- **RMSE** (Root Mean Squared Error): Per-channel prediction accuracy
- **MAE** (Mean Absolute Error): Average prediction deviation
- **SSIM** (Structural Similarity Index): Spatial pattern preservation

### Event-Based Metrics
- **CSI** (Critical Success Index): Overall detection skill
- **POD** (Probability of Detection): Hit rate for events
- **FAR** (False Alarm Rate): False positive rate

## 📖 Documentation

- [**PROJECT_PLAN.md**](PROJECT_PLAN.md) - Comprehensive project plan with tasks, milestones, and technical details
- [**Team Deliverables**](team/) - Role-specific responsibilities and acceptance criteria
- [**Config Guide**](configs/project.yaml) - Configuration options and ROI settings

## 🛠️ Technologies

- **Data Processing**: Satpy, Pyresample, Xarray, Dask, PDBUFR
- **Deep Learning**: PyTorch (or TensorFlow)
- **Geospatial**: Cartopy, NetCDF4
- **Visualization**: Matplotlib, Pillow
- **Experiment Tracking**: Config-based with YAML, manual logging

## 📝 Development Workflow

1. **Branch naming**: `feature/task-description`, `bugfix/issue-description`
2. **Commits**: Clear, descriptive messages
3. **Testing**: Validate data integrity and model outputs before committing
4. **Experiments**: Always use versioned configs and document in `experiments/`

## 🔄 Reproducibility

All experiments are designed to be fully reproducible:
- Configuration files specify all hyperparameters
- Random seeds are fixed and logged
- Dataset versions tracked via timestamps
- Git commit hashes recorded in experiment metadata

## 🤝 Contributing

Each team member should:
1. Create feature branches for new work
2. Document code with docstrings and comments
3. Update relevant `DELIVERABLES.md` upon completion
4. Coordinate data formats and interfaces between components

## 📜 License

[Specify your license here - e.g., MIT, Apache 2.0]

## 🙏 Acknowledgments

- EUMETSAT for providing open access to MSG/SEVIRI and HRW data
- Satpy and Pyresample communities for excellent geospatial tools
- [Add your institution/university if applicable]

## 📧 Contact

[Add team contact information or links to individual profiles]

---

**Last Updated**: January 7, 2026
