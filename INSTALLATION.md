# Installation Guide for Morocco Weather Agent

This guide covers complete installation and setup for the AI Weather Agent.

## System Requirements

- **Python**: 3.8 or higher
- **OS**: Windows, Linux, or macOS
- **Memory**: 500 MB RAM minimum
- **Storage**: 100 MB for dependencies
- **Network**: Internet connection required

## Installation Steps

### Step 1: Install Core Dependencies

These are required for the weather tools (no API key needed for testing):

```bash
pip install requests pyyaml numpy
```

### Step 2: Install LLM Provider Client

Choose **ONE** provider and install its client library:

#### Option A: Groq (Recommended - Free & Fast) ⭐

```bash
pip install groq
```

Get your free API key: [https://console.groq.com](https://console.groq.com)

#### Option B: OpenAI

```bash
pip install openai
```

Get API key: [https://platform.openai.com](https://platform.openai.com)

#### Option C: Anthropic

```bash
pip install anthropic
```

Get API key: [https://console.anthropic.com](https://console.anthropic.com)

### Step 3: Set API Key

Set the environment variable for your chosen provider:

#### Linux / macOS

```bash
# Add to ~/.bashrc or ~/.zshrc for persistence
export GROQ_API_KEY="your_api_key_here"

# Or for current session only
export GROQ_API_KEY="your_api_key_here"
```

#### Windows PowerShell

```powershell
# Current session
$env:GROQ_API_KEY="your_api_key_here"

# Permanent (user-level)
[System.Environment]::SetEnvironmentVariable('GROQ_API_KEY', 'your_api_key_here', 'User')
```

#### Windows Command Prompt

```cmd
# Current session
set GROQ_API_KEY=your_api_key_here

# Permanent
setx GROQ_API_KEY "your_api_key_here"
```

### Step 4: Verify Installation

Test the weather tools without API key:

```bash
python demo_weather_tools.py --location Ceuta --all-tools
```

You should see weather data printed for Ceuta.

### Step 5: Test the Agent

Run a single query to verify the agent works:

```bash
python -m src.agent.run_agent --provider groq -q "What are the wind conditions in Tangier?"
```

### Step 6: Start Interactive Chat

```bash
python -m src.agent.run_agent --provider groq
```

## Complete Installation (All Providers)

If you want to support all providers:

```bash
pip install openai groq anthropic requests pyyaml numpy
```

## Optional: Install Full Project Dependencies

For the complete weather nowcasting project (ViT model, data pipeline, etc.):

```bash
pip install -r requirements.txt
```

This includes PyTorch, Satpy, and other heavy dependencies (~2GB download).

## Installation for Different Use Cases

### Use Case 1: Just the Agent (Minimal)

```bash
pip install groq requests pyyaml
export GROQ_API_KEY="your_key"
python -m src.agent.run_agent --provider groq
```

**Size**: ~50 MB

### Use Case 2: Agent + Testing

```bash
pip install groq openai anthropic requests pyyaml
# Set API keys for providers you want to test
python test_agent.py --provider groq
```

**Size**: ~150 MB

### Use Case 3: Full Project

```bash
pip install -r requirements.txt
# Includes deep learning libraries and geospatial tools
```

**Size**: ~3 GB (includes PyTorch, CUDA if available)

## Verifying Your Installation

### Check Python Version

```bash
python --version
# Should be 3.8 or higher
```

### Check Installed Packages

```bash
pip list | grep groq
pip list | grep openai
pip list | grep anthropic
```

### Check API Key

```bash
# Linux/macOS
echo $GROQ_API_KEY

# Windows PowerShell
$env:GROQ_API_KEY

# Windows CMD
echo %GROQ_API_KEY%
```

### Run Diagnostic

```bash
# Test weather tools (no API key required)
python demo_weather_tools.py --list-locations

# Test agent with API key
python -m src.agent.run_agent --provider groq -q "Test query"
```

## Troubleshooting

### Issue: "No module named 'groq'"

**Solution**: Install the package
```bash
pip install groq
```

### Issue: "API key not found"

**Solution**: Set the environment variable
```bash
export GROQ_API_KEY="your_key_here"
```

### Issue: "ModuleNotFoundError: No module named 'src'"

**Solution**: Run from project root directory
```bash
cd morocco-weather-nowcasting
python -m src.agent.run_agent
```

### Issue: "Connection timeout" or "Network error"

**Solution**: 
1. Check internet connection
2. Try again (API might be temporarily unavailable)
3. Increase timeout in config:
   ```yaml
   weather:
     api:
       timeout: 30  # Increase from 10 to 30 seconds
   ```

### Issue: "Rate limit exceeded"

**Solution**: 
1. Wait a few minutes
2. Use a different provider
3. Check your API plan limits

### Issue: Import errors in IDE (VS Code, PyCharm)

**Solution**: These are expected if packages aren't installed. They'll resolve once you install the required packages. The code will work fine at runtime.

## Virtual Environment (Recommended)

It's best practice to use a virtual environment:

### Linux / macOS

```bash
cd morocco-weather-nowcasting
python -m venv venv
source venv/bin/activate
pip install groq requests pyyaml
```

### Windows

```powershell
cd morocco-weather-nowcasting
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install groq requests pyyaml
```

## Docker Installation (Advanced)

Create a `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir groq requests pyyaml

COPY . .

ENV GROQ_API_KEY=""

CMD ["python", "-m", "src.agent.run_agent", "--provider", "groq"]
```

Build and run:

```bash
docker build -t morocco-weather-agent .
docker run -it -e GROQ_API_KEY="your_key" morocco-weather-agent
```

## Updating

To update the agent code:

```bash
git pull origin main
```

To update dependencies:

```bash
pip install --upgrade groq openai anthropic
```

## Uninstallation

To remove the agent dependencies:

```bash
pip uninstall groq openai anthropic requests pyyaml
```

To remove environment variables:

```bash
# Linux/macOS: Remove from ~/.bashrc or ~/.zshrc
unset GROQ_API_KEY

# Windows PowerShell
Remove-Item Env:\GROQ_API_KEY

# Windows (permanent)
[System.Environment]::SetEnvironmentVariable('GROQ_API_KEY', $null, 'User')
```

## Getting Help

If you encounter issues:

1. Check this installation guide
2. Review [AGENT_QUICKSTART.md](AGENT_QUICKSTART.md)
3. Try the demo: `python demo_weather_tools.py`
4. Check Python version: `python --version`
5. Verify API key is set: `echo $GROQ_API_KEY`

## Success Checklist

✅ Python 3.8+ installed
✅ LLM provider client installed (groq/openai/anthropic)
✅ API key obtained from provider
✅ Environment variable set
✅ Demo script works: `python demo_weather_tools.py`
✅ Agent starts: `python -m src.agent.run_agent --provider groq`

## Next Steps

Once installed:

1. Read [AGENT_QUICKSTART.md](docs/AGENT_QUICKSTART.md)
2. Try example queries from [AGENT_EXAMPLES.md](docs/AGENT_EXAMPLES.md)
3. Run tests: `python test_agent.py --provider groq`
4. Explore the architecture: [AGENT_ARCHITECTURE.md](docs/AGENT_ARCHITECTURE.md)

---

**Ready to start?**

```bash
python -m src.agent.run_agent --provider groq
```

Ask: "Is it safe for a helicopter flight to Ceuta right now?"
