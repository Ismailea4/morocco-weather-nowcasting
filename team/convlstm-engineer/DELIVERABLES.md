# ConvLSTM Baseline Model Engineer Deliverables

## Overview
Responsible for implementing, training, and evaluating the baseline ConvLSTM model for weather nowcasting, including comprehensive testing, experimentation, and result visualization.

## Development Workflow

⚠️ **Important**: Start with **notebook prototyping** before implementing production code in `src/`.

### Phase 1: Notebook Prototyping & Exploration
- **Notebook**: `notebooks/02_convlstm_baseline_prototype.ipynb`
- Prototype the complete baseline pipeline interactively:
  - Data loading and exploration
  - Dataset class with sequence sampling
  - ConvLSTM architecture design and testing
  - Training loop with loss tracking
  - Evaluation metrics calculation
- Include rich visualizations:
  - Sample input sequences and ground truth
  - Model predictions vs actual (side-by-side)
  - Loss curves (train/validation)
  - Prediction error maps and residuals
  - Per-channel performance plots
  - Animated prediction sequences (GIFs)
  - Attention/feature map visualizations (if applicable)
- Experiment with hyperparameters and architecture variants
- Document model behavior and failure cases
- **Acceptance**: End-to-end training and evaluation in notebook with comprehensive visualizations

### Phase 2: Production Code Implementation
- Refactor validated notebook code into modular scripts in `src/`
- Separate concerns: datasets, models, training, evaluation
- Add configuration management and experiment tracking
- Maintain same architecture and logic validated in notebooks

## Core Deliverables

### 1. Dataset Loader Implementation
- **Dataset class**: `src/datasets/morocco_dataset.py`
  - PyTorch Dataset implementation for combined weather data
  - Sequence sampling: 6 past timesteps → 1 future prediction
  - Train/validation/test splits (70%/15%/15%)
  - Data normalization and augmentation
  - Efficient loading from `.npz` files
- **Data statistics**: Summary of dataset characteristics
  - Temporal coverage
  - Spatial dimensions
  - Channel statistics (mean, std, min, max)

### 2. ConvLSTM Baseline Model
- **Model architecture**: `src/models/conv_lstm_baseline.py`
  - Encoder-decoder ConvLSTM structure
  - Convolutional layers for spatial feature extraction
  - LSTM cells for temporal dynamics
  - Skip connections (optional)
  - Multi-channel output for satellite and wind predictions
- **Model documentation**: Architecture diagram and parameter counts

### 3. Training Pipeline
- **Training script**: `src/train/train_baseline.py`
  - Config-driven training (hyperparameters from YAML)
  - Learning rate scheduling
  - Early stopping with validation monitoring
  - Gradient clipping for stability
  - Mixed precision training (optional)
- **Training logs**: Detailed logs per epoch
  - Loss curves (train/val)
  - Learning rate progression
  - Training time metrics
- **Model checkpoints**: `models/baseline/`
  - Best model (lowest validation loss)
  - Final model
  - Checkpoint with optimizer state for resuming

### 4. Evaluation & Testing
- **Metrics implementation**: `src/eval/metrics.py`
  - RMSE for continuous predictions (per channel)
  - MAE (Mean Absolute Error)
  - Structural Similarity Index (SSIM) for spatial patterns
  - Event-based metrics with thresholds:
    - Critical Success Index (CSI)
    - Probability of Detection (POD)
    - False Alarm Rate (FAR)
- **Test results**: `experiments/baseline_convlstm/`
  - Quantitative metrics on test set
  - Per-channel performance breakdown
  - Temporal forecasting horizon analysis (0-4 hours)

### 5. Visualization & Analysis
- **Prediction visualizations**: `reports/evaluation/baseline/`
  - Side-by-side comparisons: ground truth vs predictions
  - Difference maps (error visualization)
  - Multi-timestep predictions
  - Channel-specific visualizations (IR, VIS, wind)
- **Animations**: Time-lapse videos of predictions
  - `.gif` or `.mp4` format
  - Overlay of wind vectors on satellite imagery
- **Performance plots**:
  - Loss curves over training
  - Metric trends across forecast horizons
  - Spatial error distributions

### 6. Experiment Tracking
- **Experiment metadata**: `experiments/baseline_convlstm/run_YYYYMMDD_HHMMSS/`
  - Configuration file (hyperparameters, data splits, random seeds)
  - Git commit hash for reproducibility
  - Dataset version/timestamp
  - Hardware specifications
  - Training duration
- **Results summary**: Consolidated metrics table comparing different runs
- **Model card**: Documentation of model capabilities and limitations

## Scripts & Tools
- `src/datasets/morocco_dataset.py` - Data loading
- `src/models/conv_lstm_baseline.py` - Model architecture
- `src/train/train_baseline.py` - Training loop
- `src/eval/metrics.py` - Evaluation metrics
- `src/utils/visualization.py` - Plotting utilities

## Success Metrics
- Model successfully trains without divergence
- Validation loss decreases consistently
- Test RMSE better than persistence baseline
- Visualizations clearly show spatial patterns
- All experiments fully documented and reproducible
- Code passes unit tests (if implemented)
