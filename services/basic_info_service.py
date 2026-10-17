"""Local time and live weather information service for voice assist personalities.

Fetches accurate local time and ambient weather/temperature for real-time
spoken responses with caching and fallback.
"""

import datetime
import os
import re
import threading
import time
from typing import Dict, Optional
import urllib.parse

from core import http_client
from core.env_utils import env_bool
from core.intent_builder import PersonalityProfile

INTENT_TIME = "time_and_clock"
INTENT_WEATHER = "weather_and_temperature"

# Phrases that ask for the time or the weather, for dialogue engines (the LLM) that have no
# intent rules of their own.
_TIME_QUESTION = re.compile(r"\bwhat('?s| is)? the time\b|\bwhat time\b|\bcurrent time\b|\btell me the time\b", re.IGNORECASE)
_WEATHER_QUESTION = re.compile(r"\bweather\b|\btemperature\b|\bhow (cold|hot|warm)\b|\bis it (cold|hot|warm|raining|snowing)\b", re.IGNORECASE)


def safe_format(template: str, fallback: str, **values: str) -> str:
    """Fill a profile template, returning fallback when the template has unknown placeholders.

    Args:
        template: Text from a personality profile, with {name} placeholders.
        fallback: Text returned when formatting fails.
        **values: Placeholder values.

    Returns:
        The formatted template, or fallback.
    """
    try:
        return template.format(**values)
    except (KeyError, IndexError, ValueError):
        return fallback


class BasicInfoService:
    """Provides local time and weather responses for voice assist personalities."""

    def __init__(
        self,
        cache_ttl_sec: float = 600.0,
        profile: Optional[PersonalityProfile] = None,
        cached_weather: Optional[Dict[str, str]] = None,
        cached_weather_time: float = 0.0,
    ) -> None:
        """Initialize the basic info service.

        Args:
            cache_ttl_sec: Weather cache time-to-live in seconds (default: 10 minutes).
            profile: Optional personality profile with custom time and weather response templates.
            cached_weather: Optional pre-existing weather cache dictionary.
            cached_weather_time: Optional timestamp when pre-existing cache was recorded.
        """
        self._cache_ttl_sec = cache_ttl_sec
        self._profile = profile
        self._weather_cache: Optional[Dict[str, str]] = dict(cached_weather) if cached_weather else None
        self._weather_cache_time: float = cached_weather_time
        self._last_fetch_attempt_time: float = 0.0
        self._negative_cache_ttl_sec: float = 60.0
        # WEATHER_ENABLED=false stops all requests to wttr.in (which locates the device by IP address
        # when WEATHER_LOCATION is unset)
        self._enabled = env_bool("WEATHER_ENABLED", True)
        self._lock = threading.Lock()
        self._fetch_thread: Optional[threading.Thread] = None

    @property
    def cached_weather(self) -> Optional[Dict[str, str]]:
        """Return the raw cached weather data."""
        with self._lock:
            return dict(self._weather_cache) if self._weather_cache else None

    @property
    def cached_weather_time(self) -> float:
        """Return the timestamp when the weather cache was last updated."""
        with self._lock:
            return self._weather_cache_time

    @staticmethod
    def match_intent(user_text: str) -> Optional[str]:
        """Return INTENT_TIME or INTENT_WEATHER when the text asks for that, else None."""
        if _TIME_QUESTION.search(user_text):
            return INTENT_TIME
        if _WEATHER_QUESTION.search(user_text):
            return INTENT_WEATHER
        return None

    def respond(self, intent: str, allow_network: bool = True) -> Optional[str]:
        """Return the live answer for a time or weather intent, or None for any other intent.

        Args:
            intent: Intent name.
            allow_network: If False, weather is answered from cache only.

        Returns:
            Spoken answer, or None when this service does not handle the intent.
        """
        if intent == INTENT_TIME:
            return self.get_time_response()
        if intent == INTENT_WEATHER:
            return self.get_weather_response(allow_network=allow_network)
        return None

    def get_time_response(self, now: Optional[datetime.datetime] = None) -> str:
        """Generate a spoken response indicating the current local time.

        If the current time is at or past 10:00 PM (or before 5:00 AM),
        includes a reminder that it is past bedtime.

        Args:
            now: Optional datetime override for testing. Defaults to local now.

        Returns:
            A spoken response with the current time.
        """
        if now is None:
            now = datetime.datetime.now()

        # Format 12-hour time (e.g. 8:15 PM)
        # %I has leading zero on some platforms, strip it
        hour_12 = now.strftime("%I").lstrip("0")
        if not hour_12:
            hour_12 = "12"
        minute = now.strftime("%M")
        am_pm = now.strftime("%p")
        time_str = f"{hour_12}:{minute} {am_pm}"

        # Check bedtime condition: >= 10 PM (22:00) or early morning before 5 AM (05:00)
        is_past_bedtime = now.hour >= 22 or now.hour < 5

        time_responses = self._profile.time_responses if self._profile else {}
        if is_past_bedtime:
            template = time_responses.get("bedtime")
            default = f"It is currently {time_str}. It is late."
            if template:
                return safe_format(template, default, time=time_str)
            return default

        template = time_responses.get("standard")
        default = f"It is {time_str}."
        if template:
            return safe_format(template, default, time=time_str)
        return default

    def get_weather_data(self, allow_network: bool = True) -> Optional[Dict[str, str]]:
        """Return current weather from the cache, fetching from wttr.in when needed.

        A fresh cache entry is returned directly. A stale entry is returned at once
        while a background refresh runs. Only when nothing is cached does the call
        wait for the network.

        Args:
            allow_network: If False, never contact the network; return cached data or None.

        Returns:
            Dictionary with 'temp_f', 'temp_c', and 'desc' keys, or None if unavailable.
        """
        with self._lock:
            cached = self._weather_cache
            age = time.time() - self._weather_cache_time
            last_attempt_age = time.time() - self._last_fetch_attempt_time
        if cached is not None and age < self._cache_ttl_sec:
            return cached
        if not allow_network or not self._enabled:
            return cached
        if cached is not None:
            self.prefetch_weather()
            return cached
        if last_attempt_age < self._negative_cache_ttl_sec:
            return None
        return self._fetch_weather()

    def prefetch_weather(self) -> None:
        """Refresh the weather cache on a background thread, if no refresh is already running."""
        if not self._enabled:
            return
        with self._lock:
            if self._fetch_thread is not None and self._fetch_thread.is_alive():
                return
            self._fetch_thread = threading.Thread(
                target=self._fetch_weather, daemon=True, name="AlsaGeist-WeatherFetch"
            )
            self._fetch_thread.start()

    def _fetch_weather(self) -> Optional[Dict[str, str]]:
        """Fetch weather from wttr.in (short timeout) and update the cache."""
        with self._lock:
            self._last_fetch_attempt_time = time.time()

        location = os.environ.get("WEATHER_LOCATION", "").strip()
        loc_path = f"{urllib.parse.quote(location)}" if location else ""
        url = f"https://wttr.in/{loc_path}?format=j1" if loc_path else "https://wttr.in/?format=j1"

        try:
            data = http_client.get_json(
                url,
                headers={"User-Agent": "curl/7.88.1"},
                timeout=3.5,
            )
            current = data.get("current_condition", [{}])[0]
            temp_f = str(current.get("temp_F", "")).strip()
            temp_c = str(current.get("temp_C", "")).strip()
            weather_descs = current.get("weatherDesc", [{}])
            desc = str(weather_descs[0].get("value", "")).strip() if weather_descs else ""

            if temp_f:
                result = {
                    "temp_f": temp_f,
                    "temp_c": temp_c,
                    "desc": desc,
                }
                with self._lock:
                    self._weather_cache = result
                    self._weather_cache_time = time.time()
                return result
        except Exception:
            pass

        return None

    def get_weather_response(self, allow_network: bool = True) -> str:
        """Generate a spoken response describing the current temperature and weather.

        Args:
            allow_network: If False, answer from cached data only (for offline mode).

        Returns:
            A spoken response with current weather or offline fallback.
        """
        weather = self.get_weather_data(allow_network=allow_network)
        weather_resp = self._profile.weather_responses if self._profile else {}
        if weather and weather.get("temp_f"):
            temp_f = weather["temp_f"]
            desc = weather.get("desc", "").lower()

            try:
                temp_val = int(temp_f)
            except ValueError:
                temp_val = 50

            comment = ""
            if temp_val <= 40:
                comment = weather_resp.get("freezing", "It is freezing.")
            elif temp_val <= 60:
                comment = weather_resp.get("cold", "It is chilly.")
            elif temp_val <= 75:
                comment = weather_resp.get("mild", "It is comfortable.")
            else:
                comment = weather_resp.get("hot", "It is hot.")

            if desc and desc not in ("clear", "sunny"):
                return (
                    f"It is currently {temp_f} degrees Fahrenheit with {desc}. {comment}"
                )
            return f"It is currently {temp_f} degrees Fahrenheit outside. {comment}"

        return weather_resp.get("fallback", "I cannot check the weather right now.")
