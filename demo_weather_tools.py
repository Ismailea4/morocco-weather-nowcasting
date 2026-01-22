"""
Demo script to test weather tools without requiring LLM API keys.

This script demonstrates the weather tools in action by calling them directly,
bypassing the LLM agent. Useful for:
- Testing weather API connectivity
- Validating tool outputs
- Quick weather checks without LLM overhead

Usage:
    python demo_weather_tools.py
    python demo_weather_tools.py --location Tangier
    python demo_weather_tools.py --all-tools
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from src.agent.weather_tools import (
    get_wind_conditions,
    get_visibility_conditions,
    get_temperature_conditions,
    detect_severe_weather,
    get_flight_safety_assessment,
    MOROCCAN_LOCATIONS
)


def print_section(title: str):
    """Print a section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def print_tool_result(tool_name: str, result: str):
    """Pretty print a tool result."""
    print(f"🔧 Tool: {tool_name}")
    print(f"{'─'*70}")
    
    try:
        data = json.loads(result)
        print(json.dumps(data, indent=2))
    except:
        print(result)
    
    print()


def demo_single_tool(tool_name: str, location: str):
    """Demo a single weather tool."""
    print_section(f"Testing: {tool_name}")
    print(f"Location: {location}\n")
    
    tools = {
        "wind": get_wind_conditions,
        "visibility": get_visibility_conditions,
        "temperature": get_temperature_conditions,
        "severe": detect_severe_weather,
        "flight": lambda loc: get_flight_safety_assessment(loc, "helicopter")
    }
    
    if tool_name not in tools:
        print(f"❌ Unknown tool: {tool_name}")
        print(f"Available tools: {', '.join(tools.keys())}")
        return
    
    func = tools[tool_name]
    result = func(location)
    print_tool_result(tool_name, result)


def demo_all_tools(location: str):
    """Demo all weather tools for a location."""
    print_section(f"Complete Weather Report for {location.title()}")
    
    tools = [
        ("Wind Conditions", get_wind_conditions),
        ("Visibility Conditions", get_visibility_conditions),
        ("Temperature Conditions", get_temperature_conditions),
        ("Severe Weather Detection", detect_severe_weather),
        ("Flight Safety Assessment", lambda loc: get_flight_safety_assessment(loc, "helicopter"))
    ]
    
    for tool_name, func in tools:
        print(f"\n📊 {tool_name}")
        print(f"{'─'*70}")
        result = func(location)
        
        try:
            data = json.loads(result)
            
            # Pretty print key information
            if "error" in data:
                print(f"❌ Error: {data['error']}")
            else:
                # Print summary based on tool type
                if "wind" in tool_name.lower():
                    surface = data.get("surface_wind", {})
                    print(f"  Surface Wind: {surface.get('speed_knots', 'N/A')} knots")
                    print(f"  Direction: {surface.get('direction_degrees', 'N/A')}°")
                    print(f"  Description: {surface.get('description', 'N/A')}")
                
                elif "visibility" in tool_name.lower():
                    vis = data.get("visibility", {})
                    print(f"  Visibility: {vis.get('distance_km', 'N/A')} km")
                    print(f"  Cloud Cover: {data.get('cloud_cover', {}).get('percentage', 'N/A')}%")
                    print(f"  Flight Suitability: {vis.get('flight_suitability', 'N/A')}")
                
                elif "temperature" in tool_name.lower():
                    temp = data.get("temperature", {})
                    print(f"  Temperature: {temp.get('celsius', 'N/A')}°C ({temp.get('fahrenheit', 'N/A')}°F)")
                    print(f"  Humidity: {data.get('humidity', {}).get('percentage', 'N/A')}%")
                    print(f"  Freezing Level: {data.get('freezing_level', {}).get('altitude_m', 'N/A')} m")
                
                elif "severe" in tool_name.lower():
                    print(f"  Overall Severity: {data.get('overall_severity', 'N/A').upper()}")
                    print(f"  Active Risks: {data.get('risk_count', 0)}")
                    print(f"  Recommendation: {data.get('flight_recommendation', 'N/A')}")
                
                elif "flight" in tool_name.lower():
                    print(f"  Overall Assessment: {data.get('overall_assessment', 'N/A')}")
                    print(f"  Aircraft Type: {data.get('aircraft_type', 'N/A')}")
                    limiting = data.get('limiting_factors', [])
                    if limiting:
                        print(f"  Limiting Factors:")
                        for factor in limiting:
                            print(f"    • {factor.get('factor', 'N/A')}: {factor.get('status', 'N/A')}")
                    else:
                        print(f"  ✅ All parameters within safe limits")
        
        except Exception as e:
            print(f"❌ Error parsing result: {e}")
        
        print()


def demo_multiple_locations():
    """Demo flight safety for multiple locations."""
    print_section("Flight Safety Comparison - Multiple Locations")
    
    locations = ["Ceuta", "Tangier", "Casablanca", "Rabat"]
    
    results = []
    for location in locations:
        result = get_flight_safety_assessment(location, "helicopter")
        try:
            data = json.loads(result)
            assessment = data.get("overall_assessment", "UNKNOWN")
            results.append((location, assessment, data))
        except:
            results.append((location, "ERROR", {}))
    
    # Print comparison table
    print("Location        | Assessment | Key Factors")
    print("-" * 70)
    
    for location, assessment, data in results:
        # Determine emoji based on assessment
        if assessment == "GO":
            emoji = "✅"
        elif assessment == "CAUTION":
            emoji = "⚠️"
        elif assessment == "NO-GO":
            emoji = "❌"
        else:
            emoji = "❓"
        
        # Get key limiting factor if any
        limiting = data.get("limiting_factors", [])
        factor = limiting[0]["factor"] if limiting else "None"
        
        print(f"{emoji} {location:13} | {assessment:10} | {factor}")
    
    print()


def main():
    """Main demo runner."""
    parser = argparse.ArgumentParser(
        description="Demo weather tools without LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "-l", "--location",
        type=str,
        default="Ceuta",
        help="Location to check (default: Ceuta)"
    )
    
    parser.add_argument(
        "-t", "--tool",
        type=str,
        choices=["wind", "visibility", "temperature", "severe", "flight"],
        help="Test specific tool only"
    )
    
    parser.add_argument(
        "-a", "--all-tools",
        action="store_true",
        help="Test all tools for the location"
    )
    
    parser.add_argument(
        "-c", "--compare",
        action="store_true",
        help="Compare flight safety across multiple locations"
    )
    
    parser.add_argument(
        "--list-locations",
        action="store_true",
        help="List available locations"
    )
    
    args = parser.parse_args()
    
    # List locations
    if args.list_locations:
        print_section("Available Locations")
        for i, (name, (lat, lon)) in enumerate(sorted(MOROCCAN_LOCATIONS.items()), 1):
            print(f"{i:2}. {name.title():15} ({lat:.2f}, {lon:.2f})")
        print()
        return
    
    # Compare mode
    if args.compare:
        demo_multiple_locations()
        return
    
    # Single tool mode
    if args.tool:
        demo_single_tool(args.tool, args.location)
        return
    
    # All tools mode (default)
    if args.all_tools or not args.tool:
        demo_all_tools(args.location)
        return


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user.\n")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
