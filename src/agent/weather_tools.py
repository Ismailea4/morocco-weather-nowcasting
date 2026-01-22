"""
Weather prediction tools for the ReAct agent.

This module provides separate tools for different weather aspects to enable
fine-grained reasoning by the LLM agent. Tools fetch real-time weather data
from Meteosat/EUMETSAT APIs and process it into actionable information.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests


# Moroccan cities coordinates (lat, lon)
MOROCCAN_LOCATIONS = {
    "ceuta": (35.8894, -5.3213),
    "tangier": (35.7595, -5.8339),
    "casablanca": (33.5731, -7.5898),
    "rabat": (34.0209, -6.8416),
    "marrakech": (31.6295, -7.9811),
    "fes": (34.0181, -5.0078),
    "agadir": (30.4278, -9.5981),
    "tetouan": (35.5889, -5.3626),
    "nador": (35.1681, -2.9332),
    "oujda": (34.6814, -1.9086),
    "meknes": (33.8935, -5.5473),
    "kenitra": (34.2610, -6.5802),
    "safi": (32.2994, -9.2372),
}


class WeatherDataFetcher:
    """
    Fetches weather data from various sources.
    
    For demo purposes, this uses:
    1. Open-Meteo API (free, no key required) for general weather
    2. Simulated Meteosat satellite-derived wind data (placeholder for EUMETSAT API)
    """
    
    def __init__(self):
        self.open_meteo_base = "https://api.open-meteo.com/v1/forecast"
        
    def get_coordinates(self, location: str) -> Optional[Tuple[float, float]]:
        """Get coordinates for a location name."""
        location_lower = location.lower().strip()
        if location_lower in MOROCCAN_LOCATIONS:
            return MOROCCAN_LOCATIONS[location_lower]
        return None
    
    def fetch_current_weather(self, lat: float, lon: float) -> Dict[str, Any]:
        """
        Fetch current weather conditions from Open-Meteo API.
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            Dictionary with weather data
        """
        try:
            params = {
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,precipitation,weather_code,cloud_cover,wind_speed_10m,wind_direction_10m,wind_gusts_10m",
                "timezone": "auto"
            }
            response = requests.get(self.open_meteo_base, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def fetch_wind_data_high_altitude(self, lat: float, lon: float) -> Dict[str, Any]:
        """
        Fetch high-altitude wind data (simulated for demo).
        
        In production, this would call EUMETSAT HRW API or similar.
        For now, we simulate based on surface winds with typical altitude corrections.
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            Dictionary with wind data at various altitudes
        """
        try:
            # Get surface wind first
            weather = self.fetch_current_weather(lat, lon)
            if "error" in weather or "current" not in weather:
                return {"error": "Could not fetch base weather data"}
            
            surface_wind = weather["current"].get("wind_speed_10m", 0)
            wind_dir = weather["current"].get("wind_direction_10m", 0)
            
            # Simulate wind at different altitudes (winds typically increase with altitude)
            # This is a simplified model for demo purposes
            wind_profile = {
                "surface": {
                    "speed_ms": surface_wind,
                    "speed_knots": surface_wind * 1.944,
                    "direction": wind_dir,
                    "altitude_m": 10
                },
                "850mb": {
                    "speed_ms": surface_wind * 1.5,
                    "speed_knots": surface_wind * 1.5 * 1.944,
                    "direction": (wind_dir + 15) % 360,
                    "altitude_m": 1500
                },
                "700mb": {
                    "speed_ms": surface_wind * 2.0,
                    "speed_knots": surface_wind * 2.0 * 1.944,
                    "direction": (wind_dir + 25) % 360,
                    "altitude_m": 3000
                },
                "500mb": {
                    "speed_ms": surface_wind * 2.5,
                    "speed_knots": surface_wind * 2.5 * 1.944,
                    "direction": (wind_dir + 35) % 360,
                    "altitude_m": 5500
                }
            }
            
            return {
                "location": {"lat": lat, "lon": lon},
                "timestamp": datetime.utcnow().isoformat(),
                "wind_profile": wind_profile
            }
        except Exception as e:
            return {"error": str(e)}


# Initialize global fetcher
_fetcher = WeatherDataFetcher()


def get_wind_conditions(location: str) -> str:
    """
    Get current wind conditions for a location.
    
    Args:
        location: City name or "lat,lon" coordinates
        
    Returns:
        JSON string with wind information
    """
    # Parse location
    if "," in location:
        try:
            lat, lon = map(float, location.split(","))
        except:
            return json.dumps({"error": "Invalid coordinates format. Use 'lat,lon' or city name."})
    else:
        coords = _fetcher.get_coordinates(location)
        if not coords:
            return json.dumps({"error": f"Unknown location: {location}. Try: {', '.join(list(MOROCCAN_LOCATIONS.keys())[:5])}"})
        lat, lon = coords
    
    # Fetch wind data
    wind_data = _fetcher.fetch_wind_data_high_altitude(lat, lon)
    
    if "error" in wind_data:
        return json.dumps(wind_data)
    
    # Format response
    surface = wind_data["wind_profile"]["surface"]
    upper = wind_data["wind_profile"]["700mb"]
    
    result = {
        "location": location,
        "coordinates": {"lat": lat, "lon": lon},
        "timestamp": wind_data["timestamp"],
        "surface_wind": {
            "speed_ms": round(surface["speed_ms"], 1),
            "speed_knots": round(surface["speed_knots"], 1),
            "direction_degrees": surface["direction"],
            "description": _wind_description(surface["speed_ms"])
        },
        "upper_level_wind": {
            "altitude_m": upper["altitude_m"],
            "speed_ms": round(upper["speed_ms"], 1),
            "speed_knots": round(upper["speed_knots"], 1),
            "direction_degrees": upper["direction"]
        },
        "wind_shear": {
            "magnitude_ms": round(upper["speed_ms"] - surface["speed_ms"], 1),
            "assessment": "moderate" if (upper["speed_ms"] - surface["speed_ms"]) > 10 else "low"
        }
    }
    
    return json.dumps(result, indent=2)


def get_visibility_conditions(location: str) -> str:
    """
    Get visibility and cloud cover conditions for a location.
    
    Args:
        location: City name or "lat,lon" coordinates
        
    Returns:
        JSON string with visibility information
    """
    # Parse location
    if "," in location:
        try:
            lat, lon = map(float, location.split(","))
        except:
            return json.dumps({"error": "Invalid coordinates format. Use 'lat,lon' or city name."})
    else:
        coords = _fetcher.get_coordinates(location)
        if not coords:
            return json.dumps({"error": f"Unknown location: {location}"})
        lat, lon = coords
    
    # Fetch weather data
    weather = _fetcher.fetch_current_weather(lat, lon)
    
    if "error" in weather:
        return json.dumps(weather)
    
    current = weather["current"]
    cloud_cover = current.get("cloud_cover", 0)
    precip = current.get("precipitation", 0)
    humidity = current.get("relative_humidity_2m", 0)
    
    # Estimate visibility based on conditions (simplified model)
    if precip > 5:
        visibility_km = 2
        visibility_desc = "Poor (heavy precipitation)"
    elif precip > 1:
        visibility_km = 5
        visibility_desc = "Moderate (light precipitation)"
    elif cloud_cover > 80:
        visibility_km = 8
        visibility_desc = "Good but overcast"
    elif humidity > 90:
        visibility_km = 6
        visibility_desc = "Moderate (high humidity, possible fog)"
    else:
        visibility_km = 15
        visibility_desc = "Excellent (clear conditions)"
    
    result = {
        "location": location,
        "coordinates": {"lat": lat, "lon": lon},
        "timestamp": weather["current"]["time"],
        "visibility": {
            "distance_km": visibility_km,
            "description": visibility_desc,
            "flight_suitability": "unsafe" if visibility_km < 5 else ("marginal" if visibility_km < 10 else "good")
        },
        "cloud_cover": {
            "percentage": cloud_cover,
            "description": _cloud_description(cloud_cover)
        },
        "precipitation": {
            "current_mm": precip,
            "active": precip > 0.1
        },
        "humidity": {
            "percentage": humidity,
            "fog_risk": "high" if humidity > 95 else ("moderate" if humidity > 85 else "low")
        }
    }
    
    return json.dumps(result, indent=2)


def get_temperature_conditions(location: str) -> str:
    """
    Get temperature and related thermal conditions for a location.
    
    Args:
        location: City name or "lat,lon" coordinates
        
    Returns:
        JSON string with temperature information
    """
    # Parse location
    if "," in location:
        try:
            lat, lon = map(float, location.split(","))
        except:
            return json.dumps({"error": "Invalid coordinates format. Use 'lat,lon' or city name."})
    else:
        coords = _fetcher.get_coordinates(location)
        if not coords:
            return json.dumps({"error": f"Unknown location: {location}"})
        lat, lon = coords
    
    # Fetch weather data
    weather = _fetcher.fetch_current_weather(lat, lon)
    
    if "error" in weather:
        return json.dumps(weather)
    
    current = weather["current"]
    temp_c = current.get("temperature_2m", 0)
    humidity = current.get("relative_humidity_2m", 0)
    
    # Calculate derived values
    # Simplified dew point calculation
    dew_point_c = temp_c - ((100 - humidity) / 5)
    
    # Freezing level estimate (rough approximation: -6.5°C per 1000m)
    if temp_c > 0:
        freezing_level_m = (temp_c / 6.5) * 1000
    else:
        freezing_level_m = 0
    
    result = {
        "location": location,
        "coordinates": {"lat": lat, "lon": lon},
        "timestamp": weather["current"]["time"],
        "temperature": {
            "celsius": round(temp_c, 1),
            "fahrenheit": round(temp_c * 9/5 + 32, 1),
            "description": _temp_description(temp_c)
        },
        "humidity": {
            "percentage": humidity,
            "comfort": "humid" if humidity > 70 else ("comfortable" if humidity > 30 else "dry")
        },
        "dew_point": {
            "celsius": round(dew_point_c, 1),
            "spread": round(temp_c - dew_point_c, 1),
            "fog_risk": "high" if (temp_c - dew_point_c) < 3 else "low"
        },
        "freezing_level": {
            "altitude_m": int(freezing_level_m),
            "altitude_ft": int(freezing_level_m * 3.281),
            "icing_risk": "high" if freezing_level_m < 3000 else ("moderate" if freezing_level_m < 5000 else "low")
        }
    }
    
    return json.dumps(result, indent=2)


def detect_severe_weather(location: str) -> str:
    """
    Detect and assess severe weather risks for a location.
    
    Args:
        location: City name or "lat,lon" coordinates
        
    Returns:
        JSON string with severe weather assessment
    """
    # Parse location
    if "," in location:
        try:
            lat, lon = map(float, location.split(","))
        except:
            return json.dumps({"error": "Invalid coordinates format. Use 'lat,lon' or city name."})
    else:
        coords = _fetcher.get_coordinates(location)
        if not coords:
            return json.dumps({"error": f"Unknown location: {location}"})
        lat, lon = coords
    
    # Fetch weather data
    weather = _fetcher.fetch_current_weather(lat, lon)
    wind_data = _fetcher.fetch_wind_data_high_altitude(lat, lon)
    
    if "error" in weather or "error" in wind_data:
        return json.dumps({"error": "Could not fetch weather data"})
    
    current = weather["current"]
    
    # Extract relevant parameters
    wind_speed = current.get("wind_speed_10m", 0)
    wind_gusts = current.get("wind_gusts_10m", 0)
    precip = current.get("precipitation", 0)
    temp = current.get("temperature_2m", 0)
    
    # Wind shear from profile
    surface_wind = wind_data["wind_profile"]["surface"]["speed_ms"]
    upper_wind = wind_data["wind_profile"]["700mb"]["speed_ms"]
    wind_shear = upper_wind - surface_wind
    
    # Assess risks
    risks = []
    severity = "none"
    
    # Strong wind check
    if wind_speed > 15:
        risks.append({
            "type": "strong_winds",
            "severity": "high" if wind_speed > 25 else "moderate",
            "description": f"Strong winds at {wind_speed:.1f} m/s ({wind_speed*1.944:.1f} knots)",
            "aviation_impact": "Significant crosswind and turbulence risk"
        })
        severity = "high" if wind_speed > 25 else "moderate"
    
    # Gust check
    if wind_gusts > 20:
        risks.append({
            "type": "wind_gusts",
            "severity": "high" if wind_gusts > 30 else "moderate",
            "description": f"Strong gusts up to {wind_gusts:.1f} m/s ({wind_gusts*1.944:.1f} knots)",
            "aviation_impact": "Dangerous for light aircraft and helicopters"
        })
        severity = "high"
    
    # Wind shear check
    if wind_shear > 15:
        risks.append({
            "type": "wind_shear",
            "severity": "moderate",
            "description": f"Significant wind shear: {wind_shear:.1f} m/s between surface and 3km",
            "aviation_impact": "Turbulence expected, especially during approach/departure"
        })
        if severity == "none":
            severity = "moderate"
    
    # Heavy precipitation
    if precip > 5:
        risks.append({
            "type": "heavy_precipitation",
            "severity": "moderate",
            "description": f"Heavy precipitation: {precip:.1f} mm/h",
            "aviation_impact": "Reduced visibility, possible icing"
        })
        if severity == "none":
            severity = "moderate"
    
    # Thunderstorm potential (simplified heuristic)
    if temp > 15 and precip > 2 and wind_shear > 10:
        risks.append({
            "type": "thunderstorm_potential",
            "severity": "high",
            "description": "Conditions favorable for thunderstorm development",
            "aviation_impact": "Lightning, severe turbulence, hail risk"
        })
        severity = "high"
    
    result = {
        "location": location,
        "coordinates": {"lat": lat, "lon": lon},
        "timestamp": weather["current"]["time"],
        "overall_severity": severity,
        "active_risks": risks,
        "risk_count": len(risks),
        "flight_recommendation": _flight_recommendation(severity, wind_speed, wind_gusts),
        "summary": _risk_summary(risks) if risks else "No severe weather detected. Conditions are favorable."
    }
    
    return json.dumps(result, indent=2)


def get_flight_safety_assessment(location: str, aircraft_type: str = "helicopter") -> str:
    """
    Comprehensive flight safety assessment combining all weather factors.
    
    Args:
        location: City name or "lat,lon" coordinates
        aircraft_type: Type of aircraft (helicopter, small_plane, commercial)
        
    Returns:
        JSON string with comprehensive flight safety assessment
    """
    # Parse location
    if "," in location:
        try:
            lat, lon = map(float, location.split(","))
        except:
            return json.dumps({"error": "Invalid coordinates format. Use 'lat,lon' or city name."})
    else:
        coords = _fetcher.get_coordinates(location)
        if not coords:
            return json.dumps({"error": f"Unknown location: {location}"})
        lat, lon = coords
    
    # Gather all weather data
    weather = _fetcher.fetch_current_weather(lat, lon)
    wind_data = _fetcher.fetch_wind_data_high_altitude(lat, lon)
    
    if "error" in weather or "error" in wind_data:
        return json.dumps({"error": "Could not fetch weather data"})
    
    current = weather["current"]
    
    # Extract parameters
    wind_speed = current.get("wind_speed_10m", 0)
    wind_gusts = current.get("wind_gusts_10m", 0)
    precip = current.get("precipitation", 0)
    cloud_cover = current.get("cloud_cover", 0)
    temp = current.get("temperature_2m", 0)
    
    # Determine thresholds based on aircraft type
    if aircraft_type.lower() == "helicopter":
        max_wind = 20  # m/s (~40 knots)
        max_gusts = 25  # m/s (~50 knots)
        min_visibility = 5  # km
    elif aircraft_type.lower() == "small_plane":
        max_wind = 15
        max_gusts = 20
        min_visibility = 8
    else:  # commercial
        max_wind = 25
        max_gusts = 30
        min_visibility = 3
    
    # Assess each factor
    factors = []
    go_no_go = "GO"
    
    # Wind check
    if wind_speed > max_wind * 0.7:
        status = "CAUTION" if wind_speed < max_wind else "NO-GO"
        factors.append({
            "factor": "wind_speed",
            "status": status,
            "value": f"{wind_speed:.1f} m/s ({wind_speed*1.944:.1f} knots)",
            "threshold": f"{max_wind} m/s",
            "message": f"Wind speed is {'approaching' if status == 'CAUTION' else 'exceeding'} limits for {aircraft_type}"
        })
        if status == "NO-GO":
            go_no_go = "NO-GO"
    
    # Gust check
    if wind_gusts > max_gusts * 0.7:
        status = "CAUTION" if wind_gusts < max_gusts else "NO-GO"
        factors.append({
            "factor": "wind_gusts",
            "status": status,
            "value": f"{wind_gusts:.1f} m/s ({wind_gusts*1.944:.1f} knots)",
            "threshold": f"{max_gusts} m/s",
            "message": f"Wind gusts are {'approaching' if status == 'CAUTION' else 'exceeding'} limits"
        })
        if status == "NO-GO":
            go_no_go = "NO-GO"
    
    # Visibility estimate (same logic as get_visibility_conditions)
    if precip > 5:
        visibility_km = 2
    elif precip > 1:
        visibility_km = 5
    elif cloud_cover > 80:
        visibility_km = 8
    else:
        visibility_km = 15
    
    if visibility_km < min_visibility * 1.5:
        status = "CAUTION" if visibility_km >= min_visibility else "NO-GO"
        factors.append({
            "factor": "visibility",
            "status": status,
            "value": f"{visibility_km} km",
            "threshold": f"{min_visibility} km",
            "message": f"Visibility is {'marginal' if status == 'CAUTION' else 'below minimum'} for {aircraft_type}"
        })
        if status == "NO-GO":
            go_no_go = "NO-GO"
    
    # Precipitation
    if precip > 1:
        status = "CAUTION" if precip < 5 else "NO-GO"
        factors.append({
            "factor": "precipitation",
            "status": status,
            "value": f"{precip:.1f} mm/h",
            "message": f"{'Moderate' if status == 'CAUTION' else 'Heavy'} precipitation present"
        })
        if status == "NO-GO" and go_no_go == "GO":
            go_no_go = "CAUTION"
    
    # Set overall status
    if any(f["status"] == "NO-GO" for f in factors):
        go_no_go = "NO-GO"
    elif any(f["status"] == "CAUTION" for f in factors):
        go_no_go = "CAUTION"
    
    result = {
        "location": location,
        "coordinates": {"lat": lat, "lon": lon},
        "aircraft_type": aircraft_type,
        "timestamp": weather["current"]["time"],
        "overall_assessment": go_no_go,
        "recommendation": _flight_recommendation_detailed(go_no_go, factors),
        "limiting_factors": [f for f in factors if f["status"] in ["CAUTION", "NO-GO"]],
        "all_factors_checked": factors,
        "summary": _create_flight_summary(go_no_go, factors, aircraft_type)
    }
    
    return json.dumps(result, indent=2)


# Helper functions

def _wind_description(speed_ms: float) -> str:
    """Convert wind speed to descriptive text."""
    if speed_ms < 2:
        return "Calm"
    elif speed_ms < 6:
        return "Light breeze"
    elif speed_ms < 12:
        return "Moderate wind"
    elif speed_ms < 20:
        return "Strong wind"
    else:
        return "Very strong wind"


def _cloud_description(cover_pct: int) -> str:
    """Convert cloud cover percentage to descriptive text."""
    if cover_pct < 10:
        return "Clear sky"
    elif cover_pct < 30:
        return "Few clouds"
    elif cover_pct < 60:
        return "Partly cloudy"
    elif cover_pct < 90:
        return "Mostly cloudy"
    else:
        return "Overcast"


def _temp_description(temp_c: float) -> str:
    """Convert temperature to descriptive text."""
    if temp_c < 0:
        return "Freezing"
    elif temp_c < 10:
        return "Cold"
    elif temp_c < 20:
        return "Cool"
    elif temp_c < 30:
        return "Warm"
    else:
        return "Hot"


def _flight_recommendation(severity: str, wind_speed: float, wind_gusts: float) -> str:
    """Generate flight recommendation based on severity."""
    if severity == "high" or wind_speed > 25 or wind_gusts > 30:
        return "NOT RECOMMENDED - Dangerous conditions for flight operations"
    elif severity == "moderate" or wind_speed > 15 or wind_gusts > 20:
        return "CAUTION - Marginal conditions, experienced pilots only"
    else:
        return "SAFE - Conditions suitable for flight operations"


def _flight_recommendation_detailed(status: str, factors: List[Dict]) -> str:
    """Generate detailed flight recommendation."""
    if status == "NO-GO":
        issues = [f["factor"] for f in factors if f["status"] == "NO-GO"]
        return f"Flight NOT RECOMMENDED due to: {', '.join(issues)}. Wait for conditions to improve."
    elif status == "CAUTION":
        concerns = [f["factor"] for f in factors if f["status"] == "CAUTION"]
        return f"Flight MARGINAL - proceed with caution. Monitor: {', '.join(concerns)}. Experienced pilots only."
    else:
        return "Flight conditions are FAVORABLE. Standard precautions apply."


def _risk_summary(risks: List[Dict]) -> str:
    """Create summary of active risks."""
    if not risks:
        return "No significant risks detected."
    
    high_risks = [r for r in risks if r["severity"] == "high"]
    if high_risks:
        risk_types = [r["type"].replace("_", " ").title() for r in high_risks]
        return f"HIGH RISK: {', '.join(risk_types)}. Flight operations not recommended."
    else:
        risk_types = [r["type"].replace("_", " ").title() for r in risks]
        return f"MODERATE RISK: {', '.join(risk_types)}. Proceed with caution."


def _create_flight_summary(status: str, factors: List[Dict], aircraft_type: str) -> str:
    """Create human-readable flight summary."""
    if status == "GO":
        return f"Weather conditions are suitable for {aircraft_type} operations. All parameters within safe limits."
    elif status == "CAUTION":
        limiting = [f["factor"].replace("_", " ") for f in factors if f["status"] == "CAUTION"]
        return f"Marginal conditions for {aircraft_type}. Concerns: {', '.join(limiting)}. Exercise increased caution."
    else:
        limiting = [f["factor"].replace("_", " ") for f in factors if f["status"] == "NO-GO"]
        return f"Unsafe conditions for {aircraft_type}. Flight not recommended due to: {', '.join(limiting)}."


# Tool definitions for agent
WEATHER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_wind_conditions",
            "description": "Get current wind conditions including surface and upper-level winds, wind shear assessment. Useful for assessing wind hazards for aviation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name (e.g., 'Ceuta', 'Tangier', 'Casablanca') or coordinates as 'lat,lon'"
                    }
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_visibility_conditions",
            "description": "Get visibility, cloud cover, precipitation, and fog risk information. Essential for flight planning and safety assessment.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name (e.g., 'Ceuta', 'Tangier') or coordinates as 'lat,lon'"
                    }
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_temperature_conditions",
            "description": "Get temperature, humidity, dew point, and freezing level information. Important for icing risk assessment.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name (e.g., 'Ceuta', 'Tangier') or coordinates as 'lat,lon'"
                    }
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "detect_severe_weather",
            "description": "Detect and assess severe weather risks including thunderstorms, strong winds, and hazardous conditions. Use this for comprehensive hazard assessment.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name (e.g., 'Ceuta', 'Tangier') or coordinates as 'lat,lon'"
                    }
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_flight_safety_assessment",
            "description": "Comprehensive flight safety assessment combining wind, visibility, precipitation, and other factors. Provides GO/NO-GO/CAUTION recommendation for specific aircraft types.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name (e.g., 'Ceuta', 'Tangier') or coordinates as 'lat,lon'"
                    },
                    "aircraft_type": {
                        "type": "string",
                        "enum": ["helicopter", "small_plane", "commercial"],
                        "description": "Type of aircraft for tailored safety thresholds",
                        "default": "helicopter"
                    }
                },
                "required": ["location"]
            }
        }
    }
]


# Map function names to actual functions
TOOL_FUNCTIONS = {
    "get_wind_conditions": get_wind_conditions,
    "get_visibility_conditions": get_visibility_conditions,
    "get_temperature_conditions": get_temperature_conditions,
    "detect_severe_weather": detect_severe_weather,
    "get_flight_safety_assessment": get_flight_safety_assessment,
}
