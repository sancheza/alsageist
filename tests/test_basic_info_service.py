"""Unit tests for services.basic_info_service."""

import datetime
import time
from unittest.mock import MagicMock, patch
import pytest

from services.basic_info_service import BasicInfoService


def test_time_response_daytime():
    """Verify time response during normal daytime hours."""
    service = BasicInfoService()
    # 2:30 PM
    dt = datetime.datetime(2026, 10, 11, 14, 30, 0)
    resp = service.get_time_response(now=dt)

    assert resp == "It is 2:30 PM."


def test_time_response_past_bedtime_night():
    """Verify time response at 10:00 PM and later triggers past bedtime warning."""
    service = BasicInfoService()
    # 10:15 PM (22:15)
    dt_10pm = datetime.datetime(2026, 10, 11, 22, 15, 0)
    resp = service.get_time_response(now=dt_10pm)

    assert resp == "It is currently 10:15 PM. It is late."


def test_time_response_past_bedtime_midnight():
    """Verify time response after midnight triggers past bedtime warning."""
    service = BasicInfoService()
    # 1:45 AM (01:45)
    dt_midnight = datetime.datetime(2026, 10, 12, 1, 45, 0)
    resp = service.get_time_response(now=dt_midnight)

    assert resp == "It is currently 1:45 AM. It is late."


def test_time_response_morning_after_5am():
    """Verify time response after 5:00 AM does not trigger past bedtime."""
    service = BasicInfoService()
    # 8:05 AM (08:05)
    dt_morning = datetime.datetime(2026, 10, 12, 8, 5, 0)
    resp = service.get_time_response(now=dt_morning)

    assert resp == "It is 8:05 AM."


def test_weather_response_success():
    """Verify weather response format when weather data is retrieved."""
    service = BasicInfoService()
    fake_data = {"temp_f": "55", "temp_c": "13", "desc": "Light rain"}
    with patch.object(service, "get_weather_data", return_value=fake_data):
        resp = service.get_weather_response()
        assert "55 degrees Fahrenheit" in resp
        assert "light rain" in resp.lower()
        assert resp.endswith("It is chilly.")


def test_weather_response_freezing():
    """Verify weather response cold commentary."""
    service = BasicInfoService()
    fake_data = {"temp_f": "32", "temp_c": "0", "desc": "Clear"}
    with patch.object(service, "get_weather_data", return_value=fake_data):
        resp = service.get_weather_response()
        assert "32 degrees Fahrenheit" in resp
        assert resp.endswith("It is freezing.")


def test_weather_response_offline_fallback():
    """Verify weather response falls back gracefully when weather data is None."""
    service = BasicInfoService()
    with patch.object(service, "get_weather_data", return_value=None):
        assert service.get_weather_response() == "I cannot check the weather right now."


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


def test_weather_fresh_cache_skips_network():
    """A fresh cache entry is returned without any request."""
    service = BasicInfoService()
    service._weather_cache = {"temp_f": "60", "temp_c": "16", "desc": "Clear"}
    service._weather_cache_time = time.time()
    with patch("services.basic_info_service.http_client.get_json") as get_json:
        assert service.get_weather_data()["temp_f"] == "60"
    get_json.assert_not_called()


def test_weather_stale_cache_served_while_refreshing():
    """A stale entry is returned immediately and refreshed on a background thread."""
    service = BasicInfoService(cache_ttl_sec=1.0)
    service._weather_cache = {"temp_f": "60", "temp_c": "16", "desc": "Clear"}
    service._weather_cache_time = time.time() - 100
    payload = {"current_condition": [{"temp_F": "70", "temp_C": "21", "weatherDesc": [{"value": "Sunny"}]}]}
    with patch("services.basic_info_service.http_client.get_json", return_value=payload):
        assert service.get_weather_data()["temp_f"] == "60"
        service._fetch_thread.join(timeout=2)
    assert service._weather_cache["temp_f"] == "70"


def test_weather_offline_never_touches_network():
    """allow_network=False answers from cache or falls back, without a request."""
    service = BasicInfoService()
    with patch("services.basic_info_service.http_client.get_json") as get_json:
        assert service.get_weather_data(allow_network=False) is None
        assert service.get_weather_response(allow_network=False) == "I cannot check the weather right now."
    get_json.assert_not_called()


def test_weather_fetch_failure_returns_none():
    """A network error yields None instead of raising."""
    service = BasicInfoService()
    with patch("services.basic_info_service.http_client.get_json", side_effect=OSError("down")):
        assert service.get_weather_data() is None


def test_prefetch_weather_runs_one_fetch_at_a_time():
    """A second prefetch while one is running does not start another thread."""
    import threading
    service = BasicInfoService()
    release = threading.Event()

    def slow_fetch(*args, **kwargs):
        release.wait(timeout=2)
        raise OSError("down")

    with patch("services.basic_info_service.http_client.get_json", side_effect=slow_fetch) as get_json:
        service.prefetch_weather()
        first = service._fetch_thread
        service.prefetch_weather()
        assert service._fetch_thread is first
        release.set()
        first.join(timeout=2)
    assert get_json.call_count == 1


def test_defaults_carry_no_persona_wording():
    """Without profile text the answers are neutral; no character voice leaks in."""
    service = BasicInfoService()
    texts = [
        service.get_time_response(now=datetime.datetime(2026, 10, 11, 14, 30)),
        service.get_time_response(now=datetime.datetime(2026, 10, 11, 23, 30)),
    ]
    for temp in ("20", "50", "70", "90"):
        with patch.object(service, "get_weather_data", return_value={"temp_f": temp, "temp_c": "0", "desc": "Clear"}):
            texts.append(service.get_weather_response())
    with patch.object(service, "get_weather_data", return_value=None):
        texts.append(service.get_weather_response())

    for text in texts:
        for word in ("candy", "bones", "ribs", "crypt", "skeleton", "ghoul", "mortal"):
            assert word not in text.lower(), (word, text)


def test_grimjaw_profile_still_supplies_its_own_voice():
    """The shipped persona keeps its wording because its profile defines every template."""
    from core.intent_builder import load_personality

    service = BasicInfoService(profile=load_personality("grimjaw"))
    assert "candy bowl" in service.get_time_response(now=datetime.datetime(2026, 10, 11, 14, 30))
    assert "goosebumps" not in service.get_time_response(now=datetime.datetime(2026, 10, 11, 14, 30))
    with patch.object(service, "get_weather_data", return_value=None):
        assert "barometer" in service.get_weather_response().lower()


def test_weather_negative_caching_avoids_repeated_blocking():
    """A failed weather fetch is negatively cached to avoid repeated synchronous network attempts."""
    service = BasicInfoService()
    with patch("services.basic_info_service.http_client.get_json", side_effect=OSError("network down")) as mock_get:
        res1 = service.get_weather_data()
        assert res1 is None
        assert mock_get.call_count == 1

        # Second immediate call within negative TTL must return None without calling http_client
        res2 = service.get_weather_data()
        assert res2 is None
        assert mock_get.call_count == 1


def test_weather_location_env_var(monkeypatch):
    """WEATHER_LOCATION environment variable configures the wttr.in target location."""
    monkeypatch.setenv("WEATHER_LOCATION", "Austin,Texas")
    service = BasicInfoService()
    with patch(
        "services.basic_info_service.http_client.get_json",
        return_value={
            "current_condition": [
                {"temp_F": "75", "temp_C": "24", "weatherDesc": [{"value": "Sunny"}]}
            ]
        },
    ) as mock_get:
        service.get_weather_data()
        assert mock_get.call_count == 1
        url = mock_get.call_args[0][0]
        assert "Austin%2CTexas" in url

