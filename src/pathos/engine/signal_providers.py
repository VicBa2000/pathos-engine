"""Real-world signal providers — Convert actual sensor data into emotional signals.

Each provider takes raw data from the real world and maps it to
valence/arousal/dominance hints with appropriate confidence levels.

Providers:
  - time_of_day: Circadian rhythm effects on mood (Thayer 1989)
  - weather: Weather-mood link (Schwarz 1983, Denissen 2008)
  - keyboard_dynamics: Typing patterns as arousal/stress proxy
  - facial_au: Facial expression to emotion mapping (Ekman FACS)
"""

from __future__ import annotations

import math
from datetime import datetime, timezone, timedelta


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


# ── Time of Day ──

def compute_time_of_day_signal(
    hour: float,
    minute: float = 0,
) -> dict[str, float]:
    """Map time of day to emotional modulation using circadian rhythm model.

    Based on Thayer (1989) — energy and tension follow predictable daily cycles.
    Hour should be in user's local time (0-23).

    Returns dict with valence_hint, arousal_hint, confidence.
    """
    t = hour + minute / 60.0

    # Arousal follows a rough sinusoidal: peaks ~10am and ~4pm, low at ~3am and ~2pm
    # Simplified dual-peak model
    arousal_raw = (
        0.3 * math.sin(2 * math.pi * (t - 6) / 24)   # Main wake cycle
        + 0.15 * math.sin(2 * math.pi * (t - 10) / 12)  # Afternoon dip
    )
    arousal = _clamp(0.5 + arousal_raw, 0.0, 1.0)

    # Valence: slightly positive in morning (fresh start), neutral midday,
    # slightly negative late night (fatigue)
    valence_raw = 0.15 * math.sin(2 * math.pi * (t - 8) / 24)
    valence = _clamp(valence_raw, -1.0, 1.0)

    # Confidence is moderate — circadian effects are real but subtle
    confidence = 0.4

    return {
        "valence_hint": round(valence, 4),
        "arousal_hint": round(arousal, 4),
        "confidence": confidence,
        "detail": {
            "hour": round(t, 2),
            "period": _get_period(t),
        },
    }


def _get_period(hour: float) -> str:
    if hour < 6:
        return "night"
    if hour < 12:
        return "morning"
    if hour < 14:
        return "midday"
    if hour < 18:
        return "afternoon"
    if hour < 22:
        return "evening"
    return "night"


# ── Weather ──

def compute_weather_signal(
    temp_celsius: float,
    humidity: float,
    cloud_cover: float,
    wind_speed_kmh: float,
    condition: str = "",
    is_raining: bool = False,
    is_snowing: bool = False,
) -> dict[str, float]:
    """Map weather conditions to emotional modulation.

    Based on Schwarz (1983) — sunshine increases positive affect,
    Denissen (2008) — wind and rain decrease mood, temperature has
    an inverted-U effect (comfortable range = positive).

    Args:
        temp_celsius: Temperature in Celsius.
        humidity: 0-100 percent.
        cloud_cover: 0-100 percent.
        wind_speed_kmh: Wind speed in km/h.
        condition: Weather condition string (e.g., "clear", "rain").
        is_raining: Whether it's currently raining.
        is_snowing: Whether it's currently snowing.

    Returns dict with valence_hint, arousal_hint, confidence.
    """
    # Valence: sunshine = positive, clouds/rain = negative
    sun_factor = 1.0 - (cloud_cover / 100.0)
    valence = sun_factor * 0.3  # Max +0.3 for full sun

    # Rain/snow penalty
    if is_raining:
        valence -= 0.25
    if is_snowing:
        valence -= 0.1  # Snow is less negative (can be cozy)

    # Temperature comfort: inverted U, peak around 22°C
    temp_comfort = 1.0 - min(abs(temp_celsius - 22) / 20, 1.0)
    valence += temp_comfort * 0.15

    # Wind: high wind = negative affect (Denissen)
    wind_penalty = min(wind_speed_kmh / 50, 1.0) * 0.15
    valence -= wind_penalty

    # Arousal: storms = high arousal, calm = low
    arousal = 0.3  # Base
    if is_raining:
        arousal += 0.1
    if wind_speed_kmh > 30:
        arousal += 0.15
    if "storm" in condition.lower() or "thunder" in condition.lower():
        arousal += 0.25

    # Humidity: high humidity = discomfort = slightly negative
    if humidity > 70:
        valence -= (humidity - 70) / 100 * 0.1

    valence = _clamp(valence, -1.0, 1.0)
    arousal = _clamp(arousal, 0.0, 1.0)

    return {
        "valence_hint": round(valence, 4),
        "arousal_hint": round(arousal, 4),
        "confidence": 0.45,
        "detail": {
            "temp_celsius": round(temp_celsius, 1),
            "cloud_cover": round(cloud_cover),
            "is_raining": is_raining,
            "condition": condition,
        },
    }


# ── Keyboard Dynamics ──

def compute_keyboard_signal(
    chars_per_second: float,
    avg_pause_ms: float,
    delete_ratio: float,
    total_chars: int,
) -> dict[str, float]:
    """Map typing dynamics to emotional signal.

    Fast typing with few pauses → high arousal, engaged.
    Slow typing with many pauses → low arousal, uncertain.
    Many deletes → frustration/anxiety proxy.

    Args:
        chars_per_second: Average typing speed.
        avg_pause_ms: Average pause between keystrokes.
        delete_ratio: Ratio of delete/backspace to total keystrokes (0-1).
        total_chars: Total characters typed.

    Returns dict with valence_hint, arousal_hint, dominance_hint, confidence.
    """
    if total_chars < 5:
        return {
            "valence_hint": 0.0,
            "arousal_hint": 0.5,
            "dominance_hint": None,
            "confidence": 0.1,
        }

    # Arousal from typing speed (normal ~5 cps, fast ~8+, slow ~2)
    speed_norm = _clamp(chars_per_second / 8.0, 0.0, 1.0)
    arousal = 0.3 + speed_norm * 0.4  # 0.3-0.7

    # Valence: many deletes = frustration (negative)
    valence = 0.0
    if delete_ratio > 0.15:
        valence = -(delete_ratio - 0.15) * 2.0  # Gets negative with many deletes
    valence = _clamp(valence, -0.5, 0.2)

    # Dominance: long pauses suggest hesitation = lower dominance
    if avg_pause_ms > 500:
        dominance = _clamp(0.5 - (avg_pause_ms - 500) / 2000, 0.1, 0.5)
    else:
        dominance = 0.5 + (500 - avg_pause_ms) / 2000 * 0.3
    dominance = _clamp(dominance, 0.0, 1.0)

    # Confidence scales with total chars (more data = more confident)
    confidence = _clamp(total_chars / 50, 0.2, 0.6)

    return {
        "valence_hint": round(valence, 4),
        "arousal_hint": round(arousal, 4),
        "dominance_hint": round(dominance, 4),
        "confidence": round(confidence, 4),
        "detail": {
            "chars_per_second": round(chars_per_second, 2),
            "avg_pause_ms": round(avg_pause_ms),
            "delete_ratio": round(delete_ratio, 3),
            "total_chars": total_chars,
        },
    }


# ── Facial AU ──

def compute_facial_signal(
    expressions: dict[str, float],
) -> dict[str, float]:
    """Map facial expression detection results to emotional signal.

    Expects a dict of expression probabilities from face-api.js or similar:
    {
        "neutral": 0.7, "happy": 0.1, "sad": 0.05, "angry": 0.02,
        "fearful": 0.01, "disgusted": 0.01, "surprised": 0.11
    }

    Maps to valence/arousal using Ekman's basic emotion framework.
    """
    if not expressions:
        return {
            "valence_hint": 0.0,
            "arousal_hint": 0.5,
            "confidence": 0.1,
        }

    # Valence mapping
    positive = expressions.get("happy", 0) + expressions.get("surprised", 0) * 0.3
    negative = (
        expressions.get("sad", 0) * 0.8
        + expressions.get("angry", 0)
        + expressions.get("fearful", 0) * 0.7
        + expressions.get("disgusted", 0) * 0.6
    )
    valence = _clamp(positive - negative, -1.0, 1.0)

    # Arousal mapping
    high_arousal = (
        expressions.get("angry", 0)
        + expressions.get("fearful", 0)
        + expressions.get("surprised", 0)
        + expressions.get("happy", 0) * 0.5
    )
    low_arousal = expressions.get("sad", 0) * 0.5 + expressions.get("neutral", 0) * 0.3
    arousal = _clamp(0.3 + high_arousal * 0.5 - low_arousal * 0.3, 0.0, 1.0)

    # Confidence: higher when dominant expression is strong, lower when mostly neutral
    dominant_value = max(expressions.values()) if expressions else 0
    neutral_value = expressions.get("neutral", 0)
    confidence = _clamp(dominant_value * 0.8 - neutral_value * 0.3, 0.15, 0.7)

    return {
        "valence_hint": round(valence, 4),
        "arousal_hint": round(arousal, 4),
        "confidence": round(confidence, 4),
        "detail": {
            "dominant": max(expressions, key=expressions.get) if expressions else "neutral",
            "dominant_score": round(dominant_value, 3),
        },
    }
