# Example Queries for Morocco Weather ReAct Agent

This document contains example queries to test the weather agent's capabilities.

## Basic Weather Queries

### Wind Conditions
```
What are the wind conditions in Tangier?
```

```
How strong are the winds in Ceuta right now?
```

### Visibility
```
What's the visibility like in Casablanca?
```

```
Is there fog in Rabat?
```

### Temperature
```
What's the temperature in Marrakech?
```

```
Is there icing risk at Fes?
```

## Aviation Safety Queries

### Helicopter Safety
```
Is it safe for a helicopter flight to Ceuta right now?
```

```
Can I fly a helicopter from Tangier to Tetouan?
```

### General Aviation
```
Can I fly a small plane to Agadir today?
```

```
What are the flight conditions for Nador?
```

### Specific Concerns
```
Are there strong winds for helicopter operations in Ceuta?
```

```
Is visibility good enough for VFR flight to Oujda?
```

## Severe Weather Detection

```
Are there any severe weather risks in Fes?
```

```
Is a thunderstorm expected in Casablanca?
```

```
What weather hazards should I be aware of in the Tangier area?
```

## Complex Multi-Factor Queries

```
I'm planning a helicopter flight from Rabat to Casablanca in 2 hours. What should I know about the weather?
```

```
What are the main weather concerns for a small plane flight to Marrakech?
```

```
Give me a complete weather briefing for Ceuta including wind, visibility, and any hazards.
```

## Comparison Queries

```
Which location has better flight conditions: Tangier or Ceuta?
```

```
Is it safer to fly to Agadir or Essaouira right now?
```

## Location Formats

The agent supports multiple location formats:

```
# City name (case-insensitive)
What are conditions in ceuta?
What are conditions in TANGIER?
```

```
# Coordinates (lat,lon)
What's the weather at 35.8,-5.3?
```

## Expected Agent Behavior

For each query, the agent should:

1. **Thought**: Reason about what information is needed
2. **Action**: Call appropriate weather tool(s)
3. **Observation**: Process the tool results
4. **Final Answer**: Provide clear, actionable response with specific data

### Example Flow

**User**: "Is it safe for a helicopter flight to Ceuta right now?"

**Agent Thought**: "I need to check comprehensive flight safety for Ceuta with helicopter-specific thresholds."

**Agent Action**: `get_flight_safety_assessment(location="Ceuta", aircraft_type="helicopter")`

**Agent Observation**: 
```json
{
  "overall_assessment": "NO-GO",
  "limiting_factors": [
    {
      "factor": "wind_speed",
      "status": "NO-GO",
      "value": "23.5 m/s (45.7 knots)"
    }
  ]
}
```

**Agent Final Answer**: "No, it is not safe for helicopter operations to Ceuta right now. The current wind speed is 23.5 m/s (45.7 knots), which exceeds the safe limit of 20 m/s for helicopters. I recommend waiting for winds to decrease before attempting the flight."

## Testing Recommendations

1. **Start with simple queries** to verify basic tool execution
2. **Test each tool** individually before complex queries
3. **Try edge cases**: unknown locations, extreme weather
4. **Verify reasoning**: Check that agent uses appropriate tools
5. **Test conversation context**: Ask follow-up questions

## Sample Test Session

```
You: What are the wind conditions in Ceuta?
Agent: [Calls get_wind_conditions, provides detailed wind report]

You: Is that safe for helicopters?
Agent: [Uses context from previous query, calls get_flight_safety_assessment]

You: What about Tangier?
Agent: [Understands implicit question, checks Tangier conditions]
```
