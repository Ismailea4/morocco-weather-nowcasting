
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
- **🤖 AI Weather Agent**: Interactive LLM-powered assistant using ReAct pattern for real-time weather queries

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
│   ├── project.yaml             # Project-wide settings (ROI, paths)
│   └── agent_config.yaml        # AI agent configuration
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
├── docs/                        # Documentation
│   ├── VIT_ARCHITECTURE.md
│   └── AGENT_EXAMPLES.md        # AI agent example queries
├── src/                        # Source code
│   ├── agent/                  # AI Weather Agent (ReAct)
│   │   ├── react_agent.py      # Core ReAct implementation
│   │   ├── weather_tools.py    # Weather prediction tools
│   │   ├── chat_interface.py   # Interactive chat interface
│   │   └── run_agent.py        # CLI runner
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
- ~50 GB storage for pilot dataset
- GPU recommended for model training (CUDA-compatible)
- **LLM API key** (for AI Weather Agent): OpenAI, Groq (free tier), or Anthropic

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
4. **Configure EUMETSAT credentials** (for data pipeline)

   ```bash
   # Set up your EUMETSAT API credentials
   export EUMETSAT_KEY="your_key_here"
   export EUMETSAT_SECRET="your_secret_here"
   ```

5. **Configure LLM API** (for AI Weather Agent)

   ```bash
   # Option 1: OpenAI (free trial credits available)
   export OPENAI_API_KEY="your_openai_key"
   
   # Option 2: Groq (fast inference, free tier - RECOMMENDED)
   export GROQ_API_KEY="your_groq_key"
   
   # Option 3: Anthropic Claude
   export ANTHROPIC_API_KEY="your_anthropic_key"
   ```

   Get free API keys:
   - **Groq** (Recommended): [https://console.groq.com](https://console.groq.com) - Fast inference with generous free tier
   - **OpenAI**: [https://platform.openai.com](https://platform.openai.com) - Free trial credits
   - **Anthropic**: [https://console.anthropic.com](https://console.anthropic.com)

## 🤖 AI Weather Agent (Interactive Demo)

The Morocco Weather Agent is an AI-powered assistant that uses the ReAct (Reason + Act) pattern to answer weather queries intelligently. It decides which weather tools to call based on your questions and provides comprehensive, safety-focused responses.

### 🌐 Web Interface (Recommended)

**Beautiful Streamlit interface with chat, visual safety indicators, and easy configuration!**

```bash
# Install Streamlit
pip install streamlit

# Launch web interface
python launch_streamlit.py
```

The app opens at `http://localhost:8501` with:
- 💬 Interactive chat interface
- 🎨 Modern, responsive design
- 🚦 Visual safety badges (GO/CAUTION/NO-GO)
- 📝 Click-to-use example questions
- ⚙️ Easy configuration in sidebar
- 📊 Optional agent reasoning display

**See [docs/STREAMLIT_GUIDE.md](docs/STREAMLIT_GUIDE.md) for complete guide**

### 💻 Command-Line Interface

```bash
# Interactive chat mode
python -m src.agent.run_agent

# Single query mode
python -m src.agent.run_agent -q "Is it safe for a helicopter flight to Ceuta?"

# Use Groq for faster, free inference
python -m src.agent.run_agent --provider groq

# Use specific model
python -m src.agent.run_agent --provider openai --model gpt-4
```

### Example Conversations

**Flight Safety Assessment:**
```
You: Is it safe for a helicopter flight to Ceuta right now?

Agent Thought: I need to check comprehensive flight safety for Ceuta.
Agent Action: get_flight_safety_assessment(location="Ceuta", aircraft_type="helicopter")
Agent Observation: Overall assessment: NO-GO. Wind speed 45 knots exceeds limits.

Agent: No, it is not safe for helicopter operations to Ceuta right now. The current 
wind speed is 23.5 m/s (45.7 knots), which exceeds the safe limit of 20 m/s for 
helicopters. Wind gusts are reaching 28 m/s (54.4 knots). I strongly recommend 
waiting for winds to decrease before attempting the flight.
```

**Multi-Factor Weather Query:**
```
You: What are the weather conditions in Tangier?

Agent: Let me check multiple weather factors for Tangier.
[Calls: get_wind_conditions, get_visibility_conditions, get_temperature_conditions]

Agent: Current conditions in Tangier:

🌬️ Wind: 12.3 m/s (23.9 knots) from 285°, moderate wind. Gusts up to 15.8 m/s.
👁️ Visibility: Excellent at 15 km, clear conditions suitable for flight.
🌡️ Temperature: 18.4°C (65.1°F), comfortable conditions.
⚠️ Assessment: Moderate wind conditions. Safe for commercial aviation but 
   approaching caution limits for light aircraft and helicopters.
```

**Severe Weather Detection:**
```
You: Are there any weather hazards in Casablanca?

Agent: [Calls detect_severe_weather]

Agent: Currently no severe weather detected in Casablanca. Conditions are favorable 
with light winds (8.2 m/s), good visibility (12 km), and no precipitation. All 
parameters are within safe limits for flight operations.
```

### Agent Capabilities

The agent has access to 5 specialized weather tools:

1. **`get_wind_conditions`** - Surface and upper-level winds, wind shear
2. **`get_visibility_conditions`** - Visibility, cloud cover, fog risk
3. **`get_temperature_conditions`** - Temperature, humidity, icing risk
4. **`detect_severe_weather`** - Thunderstorms, hazards, risk assessment
5. **`get_flight_safety_assessment`** - Comprehensive GO/NO-GO decision

The agent autonomously decides which tools to use based on your question!

### Supported Locations

**Moroccan Cities**: Ceuta, Tangier, Casablanca, Rabat, Marrakech, Fes, Agadir, Tetouan, Nador, Oujda, Meknes, Kenitra, Safi

**Coordinates**: You can also use `latitude,longitude` format (e.g., "35.8,-5.3")

### Interactive Commands

- `/help` - Show help and available commands
- `/reset` - Reset conversation history
- `/history` - View conversation history
- `/quit` - Exit the agent

### Agent Architecture (ReAct Pattern)

```
User Question
     ↓
[LLM Reasoning] ← "What information do I need?"
     ↓
[Tool Selection] ← "Which weather tool should I call?"
     ↓
[Tool Execution] ← Real-time weather data fetch
     ↓
[Observation] ← Process tool results
     ↓
[Final Answer] ← Natural language response with specific data
```

### Configuration

Edit [configs/agent_config.yaml](configs/agent_config.yaml) to customize:

- LLM provider and model
- Weather thresholds for different aircraft types
- API timeout and retry settings
- Verbosity and output options

### Example Use Cases

- ✈️ **Aviation Safety**: Pre-flight weather briefings for pilots
- 🚁 **Helicopter Operations**: Real-time go/no-go decisions
- 🌊 **Maritime**: Coastal wind and visibility assessment
- 📊 **Weather Analysis**: Quick insights into current conditions
- 🎓 **Education**: Interactive weather learning tool

For more examples, see [docs/AGENT_EXAMPLES.md](docs/AGENT_EXAMPLES.md).

## 🧪 Testing the Agent

### Quick Demo (No API Key Required)

Test the weather tools directly without needing an LLM API key:

```bash
# Test all weather tools for Ceuta
python demo_weather_tools.py --location Ceuta --all-tools

# Test specific tool
python demo_weather_tools.py --location Tangier --tool wind

# Compare flight safety across multiple cities
python demo_weather_tools.py --compare

# List available locations
python demo_weather_tools.py --list-locations
```

### Full Agent Test Suite

Run automated tests with your LLM provider:

```bash
# Run all tests with Groq
python test_agent.py --provider groq

# Run specific test
python test_agent.py --provider groq --test 1

# Run with verbose output (see agent reasoning)
python test_agent.py --provider groq --verbose
```

### Manual Testing

1. Start the interactive agent
2. Try each test query from [docs/AGENT_EXAMPLES.md](docs/AGENT_EXAMPLES.md)
3. Verify the agent uses appropriate tools
4. Check response quality and safety recommendations

## 📚 Additional Documentation

- **[Quick Start Guide](docs/AGENT_QUICKSTART.md)** - Get started in 5 minutes
- **[Agent Architecture](docs/AGENT_ARCHITECTURE.md)** - Technical deep-dive
- **[Example Queries](docs/AGENT_EXAMPLES.md)** - Sample questions and expected behavior
- **[ViT Architecture](docs/VIT_ARCHITECTURE.md)** - Vision Transformer model details

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
- **AI Agent**: LangChain-compatible ReAct pattern, OpenAI/Groq/Anthropic APIs
- **Weather Data**: Open-Meteo API (real-time weather data)

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
- ENSIAS

## 📧 Contact

* Ismail ELADRAOUI
* Saad QACIF
* Youssra TAFIH

---

**Last Updated**: January 7, 2026
