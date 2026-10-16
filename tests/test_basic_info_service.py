"""Unit tests for services.basic_info_service."""

import datetime
from unittest.mock import MagicMock, patch
import pytest

from services.basic_info_service import BasicInfoService


def test_time_response_daytime():
    """Verify time response during normal daytime hours."""
    service = BasicInfoService()
    # 2:30 PM
    dt = datetime.datetime(2026, 10, 11, 14, 30, 0)
    resp = service.get_time_response(now=dt)

    assert "2:30 PM" in resp
    assert "past your bedtime" not in resp.lower()
    assert "watch your fingers" in resp.lower()


def test_time_response_past_bedtime_night():
    """Verify time response at 10:00 PM and later triggers past bedtime warning."""
    service = BasicInfoService()
    # 10:15 PM (22:15)
    dt_10pm = datetime.datetime(2026, 10, 11, 22, 15, 0)
    resp = service.get_time_response(now=dt_10pm)

    assert "10:15 PM" in resp
    assert "past your bedtime" in resp.lower()
    assert "midnight ghouls" in resp.lower()


def test_time_response_past_bedtime_midnight():
    """Verify time response after midnight triggers past bedtime warning."""
    service = BasicInfoService()
    # 1:45 AM (01:45)
    dt_midnight = datetime.datetime(2026, 10, 12, 1, 45, 0)
    resp = service.get_time_response(now=dt_midnight)

    assert "1:45 AM" in resp
    assert "past your bedtime" in resp.lower()


def test_time_response_morning_after_5am():
    """Verify time response after 5:00 AM does not trigger past bedtime."""
    service = BasicInfoService()
    # 8:05 AM (08:05)
    dt_morning = datetime.datetime(2026, 10, 12, 8, 5, 0)
    resp = service.get_time_response(now=dt_morning)

    assert "8:05 AM" in resp
    assert "past your bedtime" not in resp.lower()


def test_weather_response_success():
    """Verify weather response format when weather data is retrieved."""
    service = BasicInfoService()
    fake_data = {"temp_f": "55", "temp_c": "13", "desc": "Light rain"}
    with patch.object(service, "get_weather_data", return_value=fake_data):
        resp = service.get_weather_response()
        assert "55 degrees Fahrenheit" in resp
        assert "light rain" in resp.lower()
        assert "crisp and chilly" in resp.lower()


def test_weather_response_freezing():
    """Verify weather response cold commentary."""
    service = BasicInfoService()
    fake_data = {"temp_f": "32", "temp_c": "0", "desc": "Clear"}
    with patch.object(service, "get_weather_data", return_value=fake_data):
        resp = service.get_weather_response()
        assert "32 degrees Fahrenheit" in resp
        assert "freezing cold" in resp.lower()


def test_weather_response_offline_fallback():
    """Verify weather response falls back gracefully when weather data is None."""
    service = BasicInfoService()
    with patch.object(service, "get_weather_data", return_value=None):
        resp = service.get_weather_response()
        assert "internal barometer" in resp.lower()
        assert "ribs rattle" in resp.lower()


def test_custom_profile_time_and_weather_responses():
    """Verify custom profile templates are used when provided."""
    from core.intent_builder import PersonalityProfile

    profile = PersonalityProfile(
        personality="custom_bot",
        display_name="Custom Bot",
        description="A helpful assistant bot.",
        voice="models/piper/custom.onnx",
        time_responses={
            "bedtime": "Custom bedtime: {time}",
            "standard": "Custom standard: {time}",
        },
        weather_responses={
            "freezing": "Custom freezing!",
            "cold": "Custom cold!",
            "mild": "Custom mild!",
            "hot": "Custom hot!",
            "fallback": "Custom fallback!",
        },
    )
    service = BasicInfoService(profile=profile)
    # Daytime
    dt = datetime.datetime(2026, 10, 11, 14, 30, 0)
    assert service.get_time_response(now=dt) == "Custom standard: 2:30 PM"

    # Bedtime
    dt_bed = datetime.datetime(2026, 10, 11, 23, 0, 0)
    assert service.get_time_response(now=dt_bed) == "Custom bedtime: 11:00 PM"

    # Weather freezing
    fake_freezing = {"temp_f": "20", "temp_c": "-6", "desc": "Clear"}
    with patch.object(service, "get_weather_data", return_value=fake_freezing):
        assert "Custom freezing!" in service.get_weather_response()

    # Weather fallback
    with patch.object(service, "get_weather_data", return_value=None):
        assert service.get_weather_response() == "Custom fallback!"
