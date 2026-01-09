"""
Phase 5: LLM Captioning (Optional) - MVP

This module provides a minimal rule-based warning generator and an optional
LLM hook placeholder. It is designed to be easily integrated post-inference
with your existing pipeline.

Usage (programmatic):
    from src.captioning.warnings import generate_warning, render_warning_message

    features = {
        "max_wind_speed": 12.3,  # m/s
        "temp_moyenne": 271.5,  # Kelvin
        "shear": 12.0,          # m/s (0-6 km layer proxy)
        "timestamp": "20200321_120000",
        "region": "Morocco"
    }
    alerts = generate_warning(features)
    message = render_warning_message(alerts, features)

Design notes:
- Deterministic rules ensure reproducibility and zero external dependencies.
- The LLM hook is disabled by default and can be wired to a small local model
  later if desired.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


# Default thresholds (tune as needed)
WIND_STRONG_MS = 8.0         # m/s
TEMP_HAIL_K = 273.0          # K (0C)
SHEAR_CONVECTIVE_MS = 10.0   # m/s (proxy for deep-layer shear)


@dataclass(frozen=True)
class Alert:
    code: str
    message: str
    severity: str = "info"  # e.g., info | watch | warning


def _coerce_float(value: Optional[float], fallback: float = 0.0) -> float:
    try:
        return float(value) if value is not None else fallback
    except Exception:
        return fallback


def generate_warning(forecast_map_features: Dict) -> List[Alert]:
    """
    MVP rule-based generator.

    Input dict is expected to include at least:
      - max_wind_speed (float, m/s)
      - temp_moyenne (float, K)
      - shear (float, m/s)
    Optional:
      - timestamp (str), region (str)

    Returns a list of Alert objects.
    """
    alerts: List[Alert] = []

    max_wind = _coerce_float(forecast_map_features.get("max_wind_speed"))
    temp_k = _coerce_float(forecast_map_features.get("temp_moyenne"), fallback=300.0)
    shear = _coerce_float(forecast_map_features.get("shear"))

    # Wind rule
    if max_wind > WIND_STRONG_MS:
        alerts.append(Alert(code="WIND_STRONG", message="Alerte vent fort", severity="warning"))

    # Temperature/hail proxy
    if temp_k < TEMP_HAIL_K:
        alerts.append(Alert(code="HAIL_RISK", message="Risque grêle", severity="watch"))

    # Shear/convective instability proxy
    if shear > SHEAR_CONVECTIVE_MS:
        alerts.append(Alert(code="CONVECTIVE_INSTABILITY", message="Instabilité convective", severity="info"))

    # If no specific alerts, return a benign status
    if not alerts:
        alerts.append(Alert(code="NO_ALERT", message="Aucune alerte spécifique", severity="info"))

    return alerts


def render_warning_message(alerts: List[Alert], meta: Optional[Dict] = None) -> str:
    """Compose a human-readable multi-line message from alerts and metadata."""
    meta = meta or {}
    timestamp = meta.get("timestamp", "unknown_time")
    region = meta.get("region", "unknown_region")

    header = f"Alerte(s) météo — {region} — {timestamp}"

    lines = [header, "-" * len(header)]
    for a in alerts:
        lines.append(f"- [{a.severity.upper()}] {a.message} ({a.code})")

    # Optionally include key metrics for traceability
    if any(k in meta for k in ("max_wind_speed", "temp_moyenne", "shear")):
        lines.append("")
        lines.append("Indicateurs:")
        if "max_wind_speed" in meta:
            lines.append(f"  • Vent max: {meta['max_wind_speed']} m/s")
        if "temp_moyenne" in meta:
            lines.append(f"  • Temp moyenne: {meta['temp_moyenne']} K")
        if "shear" in meta:
            lines.append(f"  • Shear: {meta['shear']} m/s")

    return "\n".join(lines)


def optional_llm_caption(context: Dict, enabled: bool = False) -> Optional[str]:
    """
    Placeholder for a future small-LLM captioning step. Returns None when disabled.

    To enable later, set enabled=True and implement a call to your preferred
    lightweight model (e.g., quantized 7B) or use templating enhanced by
    additional context.
    """
    if not enabled:
        return None
    # Example scaffold (to be replaced with actual LLM integration):
    features = {
        "wind": context.get("max_wind_speed"),
        "temp_k": context.get("temp_moyenne"),
        "shear": context.get("shear"),
        "region": context.get("region", "unknown_region"),
        "time": context.get("timestamp", "unknown_time"),
    }
    template = (
        "Synthèse automatique: Vent max {wind} m/s, Temp {temp_k} K, Shear {shear} m/s "
        "sur {region} à {time}."
    )
    return template.format(**features)
