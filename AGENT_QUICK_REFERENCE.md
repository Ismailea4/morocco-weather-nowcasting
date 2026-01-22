# 🚀 Morocco Weather Agent - Quick Reference Card

## 30-Second Setup

```bash
# 1. Install dependencies
pip install groq requests pyyaml

# 2. Get free API key from https://console.groq.com

# 3. Set environment variable
export GROQ_API_KEY="your_key_here"

# 4. Run the agent
python -m src.agent.run_agent --provider groq
```

## Quick Commands

| Command | What It Does |
|---------|--------------|
| `python -m src.agent.run_agent --provider groq` | Start interactive chat |
| `python -m src.agent.run_agent -q "Question?"` | Single query mode |
| `python demo_weather_tools.py --location Ceuta` | Test tools (no API key) |
| `python test_agent.py --provider groq` | Run automated tests |

## Example Questions

### Flight Safety
```
Is it safe for a helicopter flight to Ceuta right now?
Can I fly a small plane to Casablanca today?
```

### Weather Conditions
```
What are the wind conditions in Tangier?
What's the visibility like in Agadir?
What's the temperature in Marrakech?
```

### Hazards
```
Are there any severe weather risks in Fes?
Is there a thunderstorm expected in Rabat?
```

## Available Locations

🗺️ **13 Moroccan Cities**:
Ceuta, Tangier, Casablanca, Rabat, Marrakech, Fes, Agadir, Tetouan, Nador, Oujda, Meknes, Kenitra, Safi

📍 **Or use coordinates**: `35.8,-5.3`

## Interactive Commands

While chatting with the agent:

- `/help` - Show available commands
- `/reset` - Clear conversation history
- `/history` - View your chat history
- `/quit` - Exit the agent

## Supported LLM Providers

| Provider | Free Tier | Speed | API Key Setup |
|----------|-----------|-------|---------------|
| **Groq** ⭐ | ✅ Generous | ⚡ Very Fast | [console.groq.com](https://console.groq.com) |
| **OpenAI** | Trial credits | Fast | [platform.openai.com](https://platform.openai.com) |
| **Anthropic** | Limited | Fast | [console.anthropic.com](https://console.anthropic.com) |

## File Structure

```
morocco-weather-nowcasting/
├── src/agent/              # Agent source code
│   ├── weather_tools.py    # 5 weather tools
│   ├── react_agent.py      # ReAct implementation
│   ├── chat_interface.py   # Interactive chat
│   └── run_agent.py        # CLI runner
├── docs/
│   ├── AGENT_QUICKSTART.md      # 5-min setup guide
│   ├── AGENT_EXAMPLES.md        # Example queries
│   ├── AGENT_ARCHITECTURE.md    # Technical details
│   └── AGENT_IMPLEMENTATION_SUMMARY.md
├── configs/
│   └── agent_config.yaml   # Configuration
├── test_agent.py           # Automated tests
└── demo_weather_tools.py   # Demo without API key
```

## 5 Weather Tools

1. **🌬️ Wind Conditions** - Surface/upper winds, shear
2. **👁️ Visibility** - Distance, clouds, fog risk
3. **🌡️ Temperature** - Temp, humidity, icing risk
4. **⚠️ Severe Weather** - Storms, hazards
5. **✈️ Flight Safety** - Comprehensive GO/NO-GO

## Troubleshooting

**Problem**: `API key not found`
**Solution**: Set environment variable for your provider

**Problem**: `Module not found`
**Solution**: `pip install openai groq anthropic`

**Problem**: `Connection timeout`
**Solution**: Check internet connection

## Documentation

📖 Read the guides in order:
1. [AGENT_QUICKSTART.md](AGENT_QUICKSTART.md) - Setup
2. [AGENT_EXAMPLES.md](AGENT_EXAMPLES.md) - Usage
3. [AGENT_ARCHITECTURE.md](AGENT_ARCHITECTURE.md) - Deep dive

## Sample Conversation

```
You: Is it safe for a helicopter flight to Ceuta right now?

Agent: Let me check the flight conditions for Ceuta...
[Calls get_flight_safety_assessment(location="Ceuta", aircraft_type="helicopter")]

Agent: No, it is not safe for helicopter operations to Ceuta right now. 
The current wind speed is 23.5 m/s (45.7 knots), which exceeds the safe 
limit of 20 m/s for helicopters. Wind gusts are reaching 28 m/s (54.4 knots).
I strongly recommend waiting for winds to decrease before attempting the flight.

You: What about Tangier?

Agent: Let me check Tangier...
[Calls get_flight_safety_assessment(location="Tangier", aircraft_type="helicopter")]

Agent: Conditions in Tangier are MARGINAL. Wind speed is 12.3 m/s (23.9 knots),
approaching the caution limit. Visibility is good at 15 km. Proceed with caution
and monitor conditions closely.
```

## Key Features

✅ Real-time weather data (Open-Meteo API)
✅ Aviation safety focus with specific thresholds
✅ Intelligent tool selection (agent decides)
✅ Conversation history and context
✅ Multiple LLM providers supported
✅ Free options available (Groq)
✅ No model training required
✅ Easy to extend with new tools

## Quick Test (No API Key)

Want to test the weather tools without setting up an API key?

```bash
python demo_weather_tools.py --location Ceuta --all-tools
```

This shows all weather data for Ceuta without requiring LLM access!

## Getting Help

🐛 **Issues?** Check:
- API key is set correctly
- Internet connection is working
- Required packages are installed

📚 **Learn More**:
- Full documentation in `docs/` folder
- Example queries in `AGENT_EXAMPLES.md`
- Architecture in `AGENT_ARCHITECTURE.md`

## Production Tips

- Use environment variables for API keys (never hardcode)
- Groq for fast demos, OpenAI/Anthropic for production
- Set `verbose=False` to hide reasoning steps
- Cache weather data for repeated queries (future enhancement)
- Monitor API rate limits

---

**Ready to fly?** 🚁

```bash
python -m src.agent.run_agent --provider groq
```

Then ask: **"Is it safe for a helicopter flight to Ceuta right now?"**
