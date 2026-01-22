# Morocco Weather Agent Architecture

## Overview

The Morocco Weather Agent is an AI-powered assistant that uses the **ReAct (Reason + Act)** pattern to intelligently answer weather-related queries. It combines Large Language Model (LLM) reasoning with real-time weather data tools to provide accurate, contextual responses.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          USER INTERFACE                          │
│  (Interactive Chat / Single Query / API)                        │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                      REACT AGENT CORE                            │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  LLM (GPT-3.5 / Mixtral / Claude)                         │ │
│  │  - Reasoning Engine                                        │ │
│  │  - Tool Selection                                          │ │
│  │  - Response Generation                                     │ │
│  └──────────────────┬─────────────────────────────────────────┘ │
│                     │                                            │
│                     ▼                                            │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │           TOOL EXECUTION LAYER                             │ │
│  └──────────────────┬─────────────────────────────────────────┘ │
└─────────────────────┼────────────────────────────────────────────┘
                      │
        ┌─────────────┴─────────────┐
        │                           │
        ▼                           ▼
┌──────────────────┐        ┌──────────────────┐
│  WEATHER TOOLS   │        │  DATA SOURCES    │
│                  │        │                  │
│ • Wind Cond.     │◄───────┤ • Open-Meteo API │
│ • Visibility     │        │ • Simulated HRW  │
│ • Temperature    │        │ • Location DB    │
│ • Severe Weather │        │                  │
│ • Flight Safety  │        │                  │
└──────────────────┘        └──────────────────┘
```

## ReAct Loop Flow

The agent follows a continuous reasoning loop until it can answer the user's question:

```
User Query: "Is it safe for a helicopter flight to Ceuta?"
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ Iteration 1                                             │
│                                                         │
│ [THOUGHT] LLM Reasoning                                 │
│ "I need to check comprehensive flight safety            │
│  for Ceuta with helicopter-specific thresholds."        │
│    │                                                    │
│    ▼                                                    │
│ [ACTION] Tool Selection & Execution                     │
│ Call: get_flight_safety_assessment(                     │
│         location="Ceuta",                               │
│         aircraft_type="helicopter"                      │
│       )                                                 │
│    │                                                    │
│    ▼                                                    │
│ [OBSERVATION] Tool Results                              │
│ {                                                       │
│   "overall_assessment": "NO-GO",                        │
│   "limiting_factors": [                                 │
│     {                                                   │
│       "factor": "wind_speed",                           │
│       "value": "23.5 m/s (45.7 knots)",                 │
│       "status": "NO-GO"                                 │
│     }                                                   │
│   ],                                                    │
│   "recommendation": "Flight NOT RECOMMENDED..."         │
│ }                                                       │
│    │                                                    │
│    ▼                                                    │
│ [DECISION] Have enough information?                     │
│ Yes ✓ → Generate final answer                          │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ [FINAL ANSWER] Natural Language Response                │
│                                                         │
│ "No, it is not safe for helicopter operations to       │
│  Ceuta right now. The current wind speed is 23.5 m/s   │
│  (45.7 knots), which exceeds the safe limit of 20 m/s  │
│  for helicopters. I strongly recommend waiting for     │
│  winds to decrease before attempting the flight."      │
└─────────────────────────────────────────────────────────┘
```

## Component Details

### 1. ReAct Agent (`react_agent.py`)

**Responsibilities**:
- Maintains conversation context
- Calls LLM with system prompt and conversation history
- Parses LLM responses to identify tool calls
- Executes tools and feeds results back to LLM
- Manages the reasoning loop (max 5 iterations)

**Key Features**:
- Multi-provider support (OpenAI, Groq, Anthropic)
- Conversation history management
- Automatic tool result formatting
- Error handling and retries

### 2. Weather Tools (`weather_tools.py`)

Five specialized tools providing granular weather information:

#### Tool 1: `get_wind_conditions`
```python
Input:  location (city name or coordinates)
Output: {
  "surface_wind": {...},
  "upper_level_wind": {...},
  "wind_shear": {...}
}
Use Case: Wind-specific queries, turbulence assessment
```

#### Tool 2: `get_visibility_conditions`
```python
Input:  location
Output: {
  "visibility": {...},
  "cloud_cover": {...},
  "precipitation": {...},
  "fog_risk": {...}
}
Use Case: Visual flight rules (VFR) assessment
```

#### Tool 3: `get_temperature_conditions`
```python
Input:  location
Output: {
  "temperature": {...},
  "humidity": {...},
  "dew_point": {...},
  "freezing_level": {...}
}
Use Case: Icing risk, thermal conditions
```

#### Tool 4: `detect_severe_weather`
```python
Input:  location
Output: {
  "overall_severity": "none|moderate|high",
  "active_risks": [...],
  "summary": "..."
}
Use Case: Hazard detection, thunderstorm warning
```

#### Tool 5: `get_flight_safety_assessment`
```python
Input:  location, aircraft_type
Output: {
  "overall_assessment": "GO|CAUTION|NO-GO",
  "limiting_factors": [...],
  "recommendation": "..."
}
Use Case: Comprehensive go/no-go decision
```

### 3. Chat Interface (`chat_interface.py`)

**Features**:
- Interactive REPL with command support
- Conversation history display
- Clean formatting and error messages
- Single-query mode for scripting

**Commands**:
- `/help` - Display help
- `/reset` - Clear history
- `/history` - Show conversation
- `/quit` - Exit

### 4. Weather Data Fetcher

**Data Sources**:

1. **Open-Meteo API** (Primary)
   - Free, no API key required
   - Real-time weather data
   - Global coverage
   - Parameters: temp, wind, humidity, precipitation, cloud cover

2. **Simulated High-Altitude Winds**
   - Based on surface winds with altitude scaling
   - Provides wind profile (surface → 5500m)
   - Calculates wind shear

**Location Database**:
- 13 pre-configured Moroccan cities with coordinates
- Support for arbitrary lat/lon coordinates
- Name normalization (case-insensitive)

## Tool Selection Strategy

The LLM agent intelligently selects tools based on question semantics:

| User Question Type | Likely Tool(s) Used |
|-------------------|---------------------|
| "Is it safe to fly?" | `get_flight_safety_assessment` |
| "What are the winds?" | `get_wind_conditions` |
| "Can I see clearly?" | `get_visibility_conditions` |
| "How cold is it?" | `get_temperature_conditions` |
| "Any storms?" | `detect_severe_weather` |
| "Weather conditions?" | Multiple tools (wind + visibility + temp) |

## Example Multi-Tool Query

**User**: "Give me a complete weather briefing for Tangier"

```
Iteration 1:
  Thought: "Need comprehensive weather data"
  Action: get_wind_conditions("Tangier")
  Observation: [Wind data received]

Iteration 2:
  Thought: "Need visibility info too"
  Action: get_visibility_conditions("Tangier")
  Observation: [Visibility data received]

Iteration 3:
  Thought: "Should include temperature"
  Action: get_temperature_conditions("Tangier")
  Observation: [Temperature data received]

Iteration 4:
  Thought: "Have enough data now"
  Final Answer: [Comprehensive briefing with all data]
```

## Safety-First Design

The agent is designed with aviation safety as the primary concern:

1. **Conservative Thresholds**: Uses strict safety margins for aircraft types
2. **Explicit Warnings**: Always states when conditions are unsafe
3. **Specific Numbers**: Cites actual measurements (not vague descriptions)
4. **Multi-Factor Assessment**: Considers wind, visibility, precipitation together
5. **Aircraft-Specific**: Tailors recommendations to helicopter/small plane/commercial

## Extensibility

### Adding New Tools

1. Define tool function in `weather_tools.py`:
```python
def get_new_weather_parameter(location: str) -> str:
    # Fetch data
    # Process
    return json.dumps(result)
```

2. Add tool definition to `WEATHER_TOOLS` list:
```python
{
    "type": "function",
    "function": {
        "name": "get_new_weather_parameter",
        "description": "What this tool does...",
        "parameters": {...}
    }
}
```

3. Register in `TOOL_FUNCTIONS` dict:
```python
TOOL_FUNCTIONS = {
    "get_new_weather_parameter": get_new_weather_parameter,
    ...
}
```

### Adding New Data Sources

1. Extend `WeatherDataFetcher` class
2. Add API endpoint configuration
3. Implement data transformation
4. Update tool functions to use new data

### Supporting New LLM Providers

1. Add provider in `ReActAgent.__init__()`
2. Implement `_call_llm()` for new provider
3. Handle provider-specific response format
4. Update documentation

## Performance Characteristics

| Metric | Value | Notes |
|--------|-------|-------|
| Average Response Time | 2-5 seconds | With Groq (fast inference) |
| API Calls per Query | 1-4 | Depends on query complexity |
| Tool Execution Time | 0.5-1 second | Open-Meteo API latency |
| Max Iterations | 5 | Prevents infinite loops |
| Supported Locations | 13 cities + arbitrary coords | Easily extensible |

## System Requirements

- **Runtime**: Python 3.8+
- **Memory**: ~500 MB (including LLM client libraries)
- **Network**: Internet connection required for:
  - LLM API calls
  - Weather data API calls
- **API Keys**: One of OpenAI/Groq/Anthropic

## Security Considerations

1. **API Keys**: Never hardcode, use environment variables
2. **Input Validation**: Location names sanitized
3. **Rate Limiting**: Respect LLM provider rate limits
4. **Error Handling**: Graceful degradation on API failures
5. **No Sensitive Data**: Weather data is public information

## Future Enhancements

Potential improvements:

1. **ViT Model Integration**: Use trained nowcasting model for predictions
2. **Caching**: Cache weather data for repeated queries
3. **Batch Processing**: Process multiple locations in parallel
4. **Alert System**: Proactive notifications for severe weather
5. **Historical Analysis**: "Was it safe to fly yesterday?"
6. **Route Planning**: "Best route from A to B considering weather"
7. **Voice Interface**: Speech-to-text for queries
8. **Web Dashboard**: Visual interface with maps

---

**Architecture Benefits**:

✅ **Modular**: Each component is independent and testable
✅ **Extensible**: Easy to add new tools and data sources
✅ **Provider-Agnostic**: Works with multiple LLM providers
✅ **Safety-Focused**: Aviation safety is the primary design goal
✅ **Intelligent**: LLM decides which tools to use based on context
✅ **Conversational**: Maintains context across multiple queries
