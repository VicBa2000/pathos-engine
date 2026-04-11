"""Tests for real-world signal providers — time, weather, keyboard, facial."""

import pytest

from pathos.engine.signal_providers import (
    compute_facial_signal,
    compute_keyboard_signal,
    compute_time_of_day_signal,
    compute_weather_signal,
)


class TestTimeOfDay:

    def test_morning_positive_valence(self) -> None:
        r = compute_time_of_day_signal(hour=9, minute=0)
        assert r["valence_hint"] > 0  # Mid-morning = peak positive

    def test_late_night_lower_arousal(self) -> None:
        r_night = compute_time_of_day_signal(hour=3)
        r_morning = compute_time_of_day_signal(hour=10)
        assert r_night["arousal_hint"] < r_morning["arousal_hint"]

    def test_midday_period(self) -> None:
        r = compute_time_of_day_signal(hour=13)
        assert r["detail"]["period"] == "midday"

    def test_evening_period(self) -> None:
        r = compute_time_of_day_signal(hour=20)
        assert r["detail"]["period"] == "evening"

    def test_confidence_moderate(self) -> None:
        r = compute_time_of_day_signal(hour=12)
        assert 0.3 <= r["confidence"] <= 0.6

    def test_values_in_range(self) -> None:
        for h in range(24):
            r = compute_time_of_day_signal(hour=h)
            assert -1 <= r["valence_hint"] <= 1
            assert 0 <= r["arousal_hint"] <= 1


class TestWeather:

    def test_sunny_positive(self) -> None:
        r = compute_weather_signal(temp_celsius=22, humidity=40, cloud_cover=10,
                                    wind_speed_kmh=5)
        assert r["valence_hint"] > 0

    def test_rainy_negative(self) -> None:
        r = compute_weather_signal(temp_celsius=15, humidity=85, cloud_cover=95,
                                    wind_speed_kmh=20, is_raining=True)
        assert r["valence_hint"] < 0

    def test_storm_high_arousal(self) -> None:
        r = compute_weather_signal(temp_celsius=18, humidity=90, cloud_cover=100,
                                    wind_speed_kmh=60, condition="thunderstorm",
                                    is_raining=True)
        assert r["arousal_hint"] > 0.5

    def test_comfortable_temp_positive(self) -> None:
        r_comfy = compute_weather_signal(temp_celsius=22, humidity=50, cloud_cover=50,
                                          wind_speed_kmh=10)
        r_cold = compute_weather_signal(temp_celsius=-5, humidity=50, cloud_cover=50,
                                         wind_speed_kmh=10)
        assert r_comfy["valence_hint"] > r_cold["valence_hint"]

    def test_high_wind_negative(self) -> None:
        r_calm = compute_weather_signal(temp_celsius=20, humidity=50, cloud_cover=50,
                                         wind_speed_kmh=5)
        r_windy = compute_weather_signal(temp_celsius=20, humidity=50, cloud_cover=50,
                                          wind_speed_kmh=45)
        assert r_calm["valence_hint"] > r_windy["valence_hint"]

    def test_values_clamped(self) -> None:
        r = compute_weather_signal(temp_celsius=-30, humidity=100, cloud_cover=100,
                                    wind_speed_kmh=100, is_raining=True)
        assert -1 <= r["valence_hint"] <= 1
        assert 0 <= r["arousal_hint"] <= 1


class TestKeyboard:

    def test_fast_typing_high_arousal(self) -> None:
        r = compute_keyboard_signal(chars_per_second=8, avg_pause_ms=100,
                                     delete_ratio=0.02, total_chars=50)
        assert r["arousal_hint"] > 0.5

    def test_slow_typing_low_arousal(self) -> None:
        r = compute_keyboard_signal(chars_per_second=2, avg_pause_ms=600,
                                     delete_ratio=0.05, total_chars=30)
        assert r["arousal_hint"] < 0.5

    def test_many_deletes_negative_valence(self) -> None:
        r = compute_keyboard_signal(chars_per_second=4, avg_pause_ms=200,
                                     delete_ratio=0.35, total_chars=40)
        assert r["valence_hint"] < 0  # Frustration proxy

    def test_few_chars_low_confidence(self) -> None:
        r = compute_keyboard_signal(chars_per_second=3, avg_pause_ms=200,
                                     delete_ratio=0.1, total_chars=3)
        assert r["confidence"] < 0.3

    def test_long_pauses_low_dominance(self) -> None:
        r = compute_keyboard_signal(chars_per_second=2, avg_pause_ms=1500,
                                     delete_ratio=0.05, total_chars=30)
        assert r["dominance_hint"] is not None
        assert r["dominance_hint"] < 0.5

    def test_values_clamped(self) -> None:
        r = compute_keyboard_signal(chars_per_second=20, avg_pause_ms=10,
                                     delete_ratio=0.9, total_chars=200)
        assert -1 <= r["valence_hint"] <= 1
        assert 0 <= r["arousal_hint"] <= 1


class TestFacial:

    def test_happy_positive_valence(self) -> None:
        r = compute_facial_signal({"happy": 0.8, "neutral": 0.15, "sad": 0.05})
        assert r["valence_hint"] > 0

    def test_sad_negative_valence(self) -> None:
        r = compute_facial_signal({"sad": 0.7, "neutral": 0.2, "happy": 0.1})
        assert r["valence_hint"] < 0

    def test_angry_high_arousal(self) -> None:
        r = compute_facial_signal({"angry": 0.8, "neutral": 0.1, "happy": 0.1})
        assert r["arousal_hint"] > 0.5

    def test_neutral_low_confidence(self) -> None:
        r = compute_facial_signal({"neutral": 0.95, "happy": 0.05})
        assert r["confidence"] < 0.5  # Mostly neutral = lower confidence

    def test_strong_expression_higher_confidence(self) -> None:
        r_strong = compute_facial_signal({"happy": 0.9, "neutral": 0.1})
        r_weak = compute_facial_signal({"happy": 0.3, "neutral": 0.5, "sad": 0.2})
        assert r_strong["confidence"] > r_weak["confidence"]

    def test_empty_expressions(self) -> None:
        r = compute_facial_signal({})
        assert r["confidence"] <= 0.15

    def test_surprised_positive_arousal(self) -> None:
        r = compute_facial_signal({"surprised": 0.8, "neutral": 0.2})
        assert r["arousal_hint"] > 0.4
