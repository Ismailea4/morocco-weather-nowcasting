# Vision Transformer (ViT) Advanced Model Engineer Deliverables

## Overview
Responsible for implementing, training, and evaluating an advanced Vision Transformer-based model for weather nowcasting, with comprehensive testing, experimentation, and comparative analysis against the baseline.

## Development Workflow

⚠️ **Important**: Start with **notebook prototyping** before implementing production code in `src/`.

### Phase 1: Notebook Prototyping & Exploration
- **Notebook**: `notebooks/03_vit_advanced_prototype.ipynb`
- Prototype the complete ViT pipeline interactively:
  - Data loading and preprocessing for ViT (patch extraction)
  - Vision Transformer architecture design
  - Temporal fusion mechanism exploration
  - Training loop with learning rate scheduling
  - Evaluation against baseline ConvLSTM
- Include rich visualizations:
  - Patch embedding visualization
  - Attention map heatmaps (spatial and temporal)
  - Model predictions vs baseline vs ground truth (3-way comparison)
  - Loss curves comparison (ViT vs ConvLSTM)
  - Performance metrics comparison tables
  - Prediction error analysis (where ViT excels/fails)
  - Animated sequences showing ViT predictions
  - Feature importance and attention flow diagrams
- Experiment with architecture variants:
  - Different patch sizes
  - Number of attention heads
  - Temporal fusion strategies
- Document interpretability insights from attention patterns
- **Acceptance**: End-to-end ViT training and comparative evaluation in notebook with comprehensive visualizations

### Phase 2: Production Code Implementation
- Refactor validated notebook code into modular scripts in `src/`
- Add configuration management for architecture variants
- Implement efficient attention computation for production
- Maintain same architecture validated in notebooks

## Core Deliverables

### 1. ViT Model Architecture
- **Model implementation**: `src/models/vit_nowcasting.py`
  - Vision Transformer encoder for spatial features
    - Patch embedding layer
    - Multi-head self-attention blocks
    - Positional encoding for spatial context
  - Temporal fusion module
    - Temporal attention or temporal convolutions
    - Integration of sequential timesteps
  - Decoder for spatial reconstruction
    - Upsampling/unpatchifying
    - Multi-channel output heads
- **Architecture variants**: Explore different configurations
  - Patch sizes (8x8, 16x16, 32x32)
  - Number of attention heads
  - Model depth (number of transformer blocks)
- **Model documentation**: Detailed architecture diagram and design choices

### 2. Training Pipeline
- **Training script**: `src/train/train_vit.py`
  - Config-driven training with hyperparameter management
  - Optimizer selection (Adam, AdamW, Lion)
  - Learning rate scheduling (warmup + cosine decay)
  - Regularization techniques:
    - Dropout in attention layers
    - Weight decay
    - Gradient clipping
  - Mixed precision training for efficiency
- **Transfer learning** (optional):
  - Pre-trained ViT weights from computer vision
  - Fine-tuning strategies
- **Training logs**: Comprehensive tracking
  - Loss curves (train/val) per epoch
  - Learning rate and gradient norms
  - Training time and memory usage
- **Model checkpoints**: `models/vit/`
  - Best model (validation performance)
  - Intermediate checkpoints for analysis
  - Final trained model

### 3. Evaluation & Testing
- **Metrics implementation**: Using shared `src/eval/metrics.py`
  - Same metrics as baseline for fair comparison:
    - RMSE, MAE per channel
    - SSIM for spatial quality
    - Event-based: CSI, POD, FAR
  - Additional attention analysis:
    - Attention map visualizations
    - Feature importance across patches
- **Test results**: `experiments/vit_nowcasting/`
  - Quantitative comparison with ConvLSTM baseline
  - Statistical significance tests
  - Computational efficiency analysis (inference time, memory)
  - Ablation studies on architecture choices

### 4. Visualization & Analysis
- **Prediction visualizations**: `reports/evaluation/vit/`
  - Ground truth vs ViT predictions
  - Side-by-side comparison with baseline ConvLSTM
  - Error/residual maps
  - Multi-timestep forecasts
- **Attention visualizations**:
  - Attention heatmaps showing model focus
  - Temporal attention weights across timesteps
  - Patch-level importance scores
- **Animations**: Time-lapse predictions
  - `.gif` or `.mp4` format
  - Synchronized baseline vs ViT comparison
- **Performance analysis plots**:
  - Comparative loss curves (ViT vs ConvLSTM)
  - Metric improvements across forecast horizons
  - Computational cost vs accuracy trade-offs

### 5. Comparative Experiments
- **Baseline comparison table**: `experiments/comparison_summary.md`
  - Metrics: RMSE, MAE, SSIM, CSI, POD, FAR
  - Training time and convergence speed
  - Inference latency
  - Model size (parameters, disk space)
  - Memory footprint
- **Ablation studies**: Impact of key design choices
  - Patch size effects
  - Number of attention heads
  - Temporal fusion strategies
  - Pre-training benefits
- **Error analysis**: Where does ViT excel or fail compared to baseline?
  - Performance by weather patterns
  - Spatial error distributions
  - Temporal stability

### 6. Experiment Tracking & Reproducibility
- **Experiment metadata**: `experiments/vit_nowcasting/run_YYYYMMDD_HHMMSS/`
  - Full configuration (architecture, hyperparameters)
  - Random seeds for reproducibility
  - Dataset version and splits
  - Git commit hash
  - Hardware specifications
  - Training duration
- **Model card**: Comprehensive documentation
  - Model capabilities and limitations
  - Recommended use cases
  - Known failure modes
  - Computational requirements

## Scripts & Tools
- `src/models/vit_nowcasting.py` - ViT architecture
- `src/train/train_vit.py` - Training pipeline
- `src/eval/metrics.py` - Shared evaluation metrics
- `src/utils/visualization.py` - Plotting utilities
- `src/utils/attention_viz.py` - Attention visualization (if needed)

## Success Metrics
- ViT model trains successfully without instability
- Performance meets or exceeds ConvLSTM baseline on key metrics
- Attention visualizations provide interpretable insights
- All experiments documented with reproducible configs
- Comprehensive comparison report delivered
- Code quality maintained (comments, docstrings, type hints)

## Optional Extensions
- Multi-scale ViT with different patch resolutions
- Ensemble of ViT models for uncertainty quantification
- Integration with LLM for textual weather warnings
- Real-time inference optimization
