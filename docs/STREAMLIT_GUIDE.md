# Morocco Weather Agent - Streamlit Web Interface Guide

## Quick Start

### Option 1: Simple Launch

```bash
# Install Streamlit
pip install streamlit

# Launch the app
python launch_streamlit.py
```

The app will automatically open in your browser at `http://localhost:8501`

### Option 2: Direct Streamlit Command

```bash
streamlit run src/agent/streamlit_app.py
```

## First Time Setup

1. **Install Dependencies**
   ```bash
   pip install streamlit groq requests pyyaml
   ```

2. **Get API Key**
   - Visit [console.groq.com](https://console.groq.com) (recommended - free)
   - Or use OpenAI/Anthropic

3. **Launch App**
   ```bash
   python launch_streamlit.py
   ```

4. **Configure in Sidebar**
   - Select provider (Groq recommended)
   - Enter API key
   - Click "Initialize Agent"

5. **Start Chatting!**
   - Type questions in the chat box
   - Or click example questions

## Features

### 🎨 Beautiful Interface
- Modern, responsive design
- Chat-style conversation
- Color-coded safety assessments
- Example questions for quick start

### 💬 Interactive Chat
- Natural conversation flow
- Conversation history preserved
- Clear user/assistant distinction
- Safety badges for flight assessments

### ⚙️ Flexible Configuration
- Multiple LLM providers (Groq, OpenAI, Anthropic)
- Model selection
- Verbose mode to see agent reasoning
- Easy API key management

### 🚁 Aviation Safety Focus
- Visual GO/CAUTION/NO-GO indicators
- Real-time weather data
- Comprehensive assessments
- Clear safety recommendations

### 📍 Location Support
13 Moroccan cities: Ceuta, Tangier, Casablanca, Rabat, Marrakech, Fes, Agadir, Tetouan, Nador, Oujda, Meknes, Kenitra, Safi

## Interface Overview

### Sidebar
- **Configuration**: Provider, model selection
- **API Key**: Secure input (not visible)
- **Show Reasoning**: Toggle verbose mode
- **Controls**: Initialize, clear chat
- **Resources**: Quick links to get API keys
- **Locations**: Supported cities

### Main Area
- **Chat Window**: Conversation history with styled messages
- **Safety Badges**: Visual indicators for flight safety
- **Example Questions**: Click to try pre-made queries
- **Chat Input**: Type your questions

## Example Questions

Try clicking these in the interface:

1. "Is it safe for a helicopter flight to Ceuta right now?"
2. "What are the wind conditions in Tangier?"
3. "Can I fly a small plane to Casablanca today?"
4. "What's the visibility like in Agair?"
5. "Are there any weather hazards in Fes?"
6. "What's the temperature in Marrakech?"

## Safety Assessment Badges

The interface automatically detects safety assessments and displays:

- **🟢 SAFE FOR FLIGHT** (Green) - Conditions are favorable
- **🟡 PROCEED WITH CAUTION** (Orange) - Marginal conditions
- **🔴 NOT SAFE - DO NOT FLY** (Red) - Dangerous conditions

## Verbose Mode

Enable "Show Reasoning Steps" to see:
- Agent's thought process
- Tool calls being made
- Observations from tools
- Decision-making process

This is educational and helps understand how the agent works!

## Configuration Examples

### Using Groq (Free & Fast)
```
Provider: groq
Model: llama-3.3-70b-versatile (default)
API Key: gsk_your_key_here
```

### Using OpenAI
```
Provider: openai
Model: gpt-3.5-turbo (default)
API Key: sk-your_key_here
```

### Using Anthropic
```
Provider: anthropic
Model: claude-3-haiku-20240307 (default)
API Key: sk-ant-your_key_here
```

## Environment Variables

You can set API keys as environment variables (they'll be auto-detected):

```bash
# Linux/Mac
export GROQ_API_KEY="your_key_here"

# Windows PowerShell
$env:GROQ_API_KEY="your_key_here"

# Windows CMD
set GROQ_API_KEY=your_key_here
```

Then just launch the app - no need to enter key in UI!

## Keyboard Shortcuts

- **Enter** - Send message
- **Ctrl+C** (in terminal) - Stop the server
- **F5** - Refresh page

## Troubleshooting

### Issue: "Please initialize the agent first"
**Solution**: Enter your API key in the sidebar and click "Initialize Agent"

### Issue: "Error initializing agent"
**Solution**: 
1. Check API key is correct
2. Verify internet connection
3. Try a different provider

### Issue: "Module not found: streamlit"
**Solution**: `pip install streamlit`

### Issue: Page won't load
**Solution**: 
1. Check terminal for errors
2. Try a different port: `streamlit run src/agent/streamlit_app.py --server.port 8502`
3. Clear browser cache

### Issue: Slow responses
**Solution**: 
1. Use Groq provider (fastest)
2. Try lighter model (llama-3.1-8b-instant)
3. Check internet connection

## Advanced Usage

### Running on Custom Port

```bash
streamlit run src/agent/streamlit_app.py --server.port 8080
```

### Running in Headless Mode (Server)

```bash
streamlit run src/agent/streamlit_app.py --server.headless true
```

### Sharing with Others (Temporary)

```bash
streamlit run src/agent/streamlit_app.py --server.enableCORS false
```

Then share the URL shown in terminal.

## Deployment

### Deploy to Streamlit Cloud (Free)

1. Push code to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect repository
4. Add API key in Secrets management
5. Deploy!

### Deploy to Heroku

Create `Procfile`:
```
web: streamlit run src/agent/streamlit_app.py --server.port $PORT
```

### Deploy with Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .
RUN pip install streamlit groq requests pyyaml

EXPOSE 8501

CMD ["streamlit", "run", "src/agent/streamlit_app.py"]
```

## Screenshots Reference

### Main Interface
- Clean, modern design
- Chat window on left (2/3 width)
- Examples and controls on right (1/3 width)
- Sidebar for configuration

### Safety Badges
- Green badge: Safe conditions
- Orange badge: Caution required
- Red badge: Unsafe conditions

### Verbose Mode
- Expandable section showing agent reasoning
- Tool calls and results
- Helpful for learning

## Tips for Best Experience

1. **Start with examples** - Click example questions to learn
2. **Enable verbose mode** - See how the agent thinks
3. **Use Groq** - Fastest responses, generous free tier
4. **Keep chat history** - Build context for follow-up questions
5. **Clear when switching topics** - Reset for unrelated queries

## Getting Help

- 📖 Read [AGENT_QUICKSTART.md](AGENT_QUICKSTART.md)
- 🏗️ See [AGENT_ARCHITECTURE.md](AGENT_ARCHITECTURE.md)
- 💡 Try [AGENT_EXAMPLES.md](AGENT_EXAMPLES.md)
- 🐛 Check terminal for error messages

## Comparison: CLI vs Streamlit

| Feature | CLI | Streamlit |
|---------|-----|-----------|
| Interface | Terminal | Web Browser |
| Ease of Use | Moderate | Easy |
| Visual Appeal | Basic | Beautiful |
| Configuration | Command-line args | Interactive UI |
| Chat History | Text-based | Formatted messages |
| Safety Indicators | Text only | Color badges |
| Examples | Manual typing | Click buttons |
| Sharing | Screenshot only | Share URL |
| Best For | Quick queries | Extended use |

## Next Steps

1. **Launch the app**: `python launch_streamlit.py`
2. **Try example questions**
3. **Experiment with different locations**
4. **Enable verbose mode** to learn
5. **Share with your team**!

---

**Ready to fly?** 🚁

```bash
python launch_streamlit.py
```

The app will open automatically in your browser!
