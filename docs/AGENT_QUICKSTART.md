# Morocco Weather Agent - Quick Start Guide

This guide will help you get the AI Weather Agent up and running in under 5 minutes.

## Prerequisites

- Python 3.8 or higher
- An API key from one of these providers:
  - **Groq** (Recommended - Free tier, fast): [https://console.groq.com](https://console.groq.com)
  - **OpenAI** (Free trial credits): [https://platform.openai.com](https://platform.openai.com)
  - **Anthropic** (Claude): [https://console.anthropic.com](https://console.anthropic.com)

## Step 1: Install Dependencies

```bash
# Install required packages for the agent
pip install openai groq anthropic requests pyyaml
```

**Note**: You only need to install the client library for your chosen provider. For example, if using Groq:
```bash
pip install groq requests pyyaml
```

## Step 2: Set Your API Key

Choose one provider and set the corresponding environment variable:

### Option A: Groq (Recommended for Free Usage)

```bash
# Linux/Mac
export GROQ_API_KEY="your_groq_api_key_here"

# Windows PowerShell
$env:GROQ_API_KEY="your_groq_api_key_here"

# Windows CMD
set GROQ_API_KEY=your_groq_api_key_here
```

### Option B: OpenAI

```bash
# Linux/Mac
export OPENAI_API_KEY="your_openai_api_key_here"

# Windows PowerShell
$env:OPENAI_API_KEY="your_openai_api_key_here"

# Windows CMD
set OPENAI_API_KEY=your_openai_api_key_here
```

### Option C: Anthropic

```bash
# Linux/Mac
export ANTHROPIC_API_KEY="your_anthropic_api_key_here"

# Windows PowerShell
$env:ANTHROPIC_API_KEY="your_anthropic_api_key_here"

# Windows CMD
set ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

## Step 3: Run the Agent

### Interactive Chat Mode (Recommended)

```bash
# Using Groq (recommended - fast and free)
python -m src.agent.run_agent --provider groq

# Using OpenAI
python -m src.agent.run_agent --provider openai

# Using Anthropic
python -m src.agent.run_agent --provider anthropic
```

You'll see a welcome screen and can start asking questions:

```
🌤️  Morocco Weather Assistant (ReAct Agent)
====================================================================

Ask me about weather conditions in Morocco!

Example questions:
  • Is it safe for a helicopter flight to Ceuta right now?
  • What are the wind conditions in Tangier?
  • Can I fly a small plane to Casablanca today?
  
You: 
```

### Single Query Mode

For quick questions without interactive chat:

```bash
python -m src.agent.run_agent \
  --provider groq \
  -q "Is it safe for a helicopter flight to Ceuta?"
```

## Step 4: Try Example Queries

Here are some questions you can ask:

### Flight Safety
- "Is it safe for a helicopter flight to Ceuta right now?"
- "Can I fly a small plane to Casablanca today?"
- "What are the flight conditions for Tangier?"

### Weather Conditions
- "What are the wind conditions in Marrakech?"
- "What's the visibility like in Agadir?"
- "What's the temperature in Fes?"

### Hazard Detection
- "Are there any severe weather risks in Rabat?"
- "Is there a thunderstorm expected in Tangier?"
- "What weather hazards should I be aware of near Ceuta?"

### Multi-Turn Conversations
```
You: What are the wind conditions in Ceuta?
Agent: [Provides wind details]

You: Is that safe for helicopters?
Agent: [Analyzes safety based on context]

You: What about Tangier?
Agent: [Checks Tangier conditions]
```

## Step 5: Interactive Commands

While in chat mode, you can use these commands:

- `/help` - Show help message and available commands
- `/reset` - Clear conversation history and start fresh
- `/history` - Display your conversation history
- `/quit` - Exit the agent (or press Ctrl+C)

## Testing the Agent

Run the test suite to verify everything is working:

```bash
# Run all tests
python test_agent.py --provider groq

# Run a specific test
python test_agent.py --provider groq --test 1

# Run with verbose output (see agent reasoning)
python test_agent.py --provider groq --verbose
```

## Troubleshooting

### Error: API Key Not Found

```
❌ Error: GROQ_API_KEY not found in environment variables.
```

**Solution**: Make sure you've set the environment variable for your chosen provider (see Step 2).

### Error: Module Not Found

```
ImportError: No module named 'groq'
```

**Solution**: Install the required package:
```bash
pip install groq
```

### Error: Connection Timeout

```
Error: Could not fetch weather data
```

**Solution**: Check your internet connection. The agent needs to fetch real-time weather data from Open-Meteo API.

### Slow Response Times

**Solution**: 
- Use Groq provider for faster inference: `--provider groq`
- Use a smaller model if available
- Disable verbose mode: `--no-verbose`

## Configuration

You can customize the agent by editing `configs/agent_config.yaml`:

```yaml
llm:
  provider: groq
  model: llama-3.3-70b-versatile
  max_iterations: 5
  verbose: true

weather:
  api:
    timeout: 10
```

## Next Steps

- Read [docs/AGENT_EXAMPLES.md](AGENT_EXAMPLES.md) for more example queries
- Explore the code in `src/agent/` to understand how it works
- Customize weather thresholds in `configs/agent_config.yaml`
- Add new weather tools in `src/agent/weather_tools.py`

## Getting Help

If you encounter issues:

1. Check that your API key is correctly set
2. Verify internet connectivity (for weather data API)
3. Run the test suite: `python test_agent.py --provider groq`
4. Check the [docs/AGENT_EXAMPLES.md](AGENT_EXAMPLES.md) for usage patterns

## API Provider Comparison

| Provider | Free Tier | Speed | Best For |
|----------|-----------|-------|----------|
| **Groq** | ✅ Generous | ⚡ Very Fast | Quick demos, testing |
| **OpenAI** | ⚠️ Trial credits | 🚀 Fast | Production use |
| **Anthropic** | ⚠️ Limited | 🚀 Fast | Complex reasoning |

**Recommendation**: Start with Groq for free, fast testing. Switch to OpenAI or Anthropic for production use.

---

**Ready to fly?** 🚁 Start the agent and ask about weather conditions in Morocco!

```bash
python -m src.agent.run_agent --provider groq
```
