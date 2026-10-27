"""Unit tests for core/wifi_manager.py."""

import subprocess
from unittest.mock import MagicMock, call, patch
import pytest

from core import wifi_manager
from core.intent_builder import DEFAULT_ANNOUNCEMENTS


class TestWifiManagerParsing:
    """Tests for parse_fallback_networks."""

    def test_parse_empty_and_none(self):
        """Verify empty or None inputs return an empty list."""
        assert wifi_manager.parse_fallback_networks(None) == []
        assert wifi_manager.parse_fallback_networks("") == []
        assert wifi_manager.parse_fallback_networks("   ") == []

    def test_parse_standard_pairs(self):
        """Verify standard SSID:PASSWORD pairs with whitespace."""
        raw = "HomeNet:SecretPass; CoffeeShop:Latte123"
        result = wifi_manager.parse_fallback_networks(raw)
        assert result == [("HomeNet", "SecretPass"), ("CoffeeShop", "Latte123")]

    def test_parse_trailing_semicolons_and_extra_spaces(self):
        """Verify trailing semicolons and irregular spacing are handled cleanly."""
        raw = "  Net1:Pass1 ;   Net2:Pass2;   ; "
        result = wifi_manager.parse_fallback_networks(raw)
        assert result == [("Net1", "Pass1"), ("Net2", "Pass2")]

    def test_parse_quoted_special_characters(self):
        """Verify passwords and SSIDs with colons and semicolons inside quotes are preserved."""
        raw = "'My SSID;Office':'pass:word;123'; \"SimpleNet\":pwd"
        result = wifi_manager.parse_fallback_networks(raw)
        assert result == [("My SSID;Office", "pass:word;123"), ("SimpleNet", "pwd")]

    def test_parse_open_networks_no_password(self):
        """Verify open networks without passwords are supported."""
        raw = "FreeWiFi:; PublicPark"
        result = wifi_manager.parse_fallback_networks(raw)
        assert result == [("FreeWiFi", ""), ("PublicPark", "")]


class TestWifiManagerNmcliOperations:
    """Tests for nmcli querying, scanning, and connecting."""

    @patch("shutil.which", return_value="/usr/bin/nmcli")
    @patch("subprocess.run")
    def test_get_current_wifi_ssid_connected(self, mock_run, mock_which):
        """Verify get_current_wifi_ssid extracts the active connection name."""
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = "wlan0:wifi:connected:MyHomeWiFi\neth0:ethernet:disconnected:--\n"
        mock_run.return_value = mock_proc

        assert wifi_manager.get_current_wifi_ssid() == "MyHomeWiFi"

    @patch("shutil.which", return_value="/usr/bin/nmcli")
    @patch("subprocess.run")
    def test_get_current_wifi_ssid_disconnected(self, mock_run, mock_which):
        """Verify get_current_wifi_ssid returns None when disconnected."""
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = "wlan0:wifi:disconnected:--\n"
        mock_run.return_value = mock_proc

        assert wifi_manager.get_current_wifi_ssid() is None

    @patch("shutil.which", return_value=None)
    def test_get_current_wifi_ssid_no_nmcli(self, mock_which):
        """Verify get_current_wifi_ssid returns None if nmcli is not installed."""
        assert wifi_manager.get_current_wifi_ssid() is None

    @patch("shutil.which", return_value="/usr/bin/nmcli")
    @patch("subprocess.run")
    def test_get_saved_wifi_connections(self, mock_run, mock_which):
        """Verify get_saved_wifi_connections filters 802-11-wireless profiles."""
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = "HomeWiFi:802-11-wireless\nOfficeEth:802-3-ethernet\nGuestHotspot:wifi\n"
        mock_run.return_value = mock_proc

        saved = wifi_manager.get_saved_wifi_connections()
        assert saved == ["HomeWiFi", "GuestHotspot"]

    @patch("shutil.which", return_value="/usr/bin/nmcli")
    @patch("subprocess.run")
    def test_scan_visible_ssids(self, mock_run, mock_which):
        """Verify scan_visible_ssids strips duplicates and empty entries."""
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = "SSID1\nSSID2\nSSID1\n--\n\nSSID3\n"
        mock_run.return_value = mock_proc

        visible = wifi_manager.scan_visible_ssids()
        assert visible == ["SSID1", "SSID2", "SSID3"]

    @patch("core.wifi_manager.get_saved_wifi_connections", return_value=["SavedNet"])
    @patch("shutil.which", return_value="/usr/bin/nmcli")
    @patch("subprocess.run")
    def test_connect_to_wifi_saved_success(self, mock_run, mock_which, mock_saved):
        """Verify saved connections trigger 'con up id'."""
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = "Connection activated\n"
        mock_run.return_value = mock_proc

        assert wifi_manager.connect_to_wifi("SavedNet") is True
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert args == ["/usr/bin/nmcli", "con", "up", "id", "SavedNet"]

    @patch("core.wifi_manager.get_saved_wifi_connections", return_value=[])
    @patch("shutil.which", return_value="/usr/bin/nmcli")
    @patch("subprocess.run")
    def test_connect_to_wifi_new_with_password(self, mock_run, mock_which, mock_saved):
        """Verify new connection triggers 'dev wifi connect' with password."""
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = "Device successfully connected\n"
        mock_run.return_value = mock_proc

        assert wifi_manager.connect_to_wifi("NewNet", password="SecretPass") is True
        args = mock_run.call_args[0][0]
        assert args == ["/usr/bin/nmcli", "dev", "wifi", "connect", "NewNet", "password", "SecretPass"]


class TestWifiStartupFallbackOrchestration:
    """Tests for manage_wifi_startup_fallback flow and announcements."""

    def test_fallback_skipped_if_already_online(self):
        """Verify nothing is announced or attempted if already online."""
        speak_mock = MagicMock()
        announcement_mock = MagicMock()
        is_online_mock = MagicMock(return_value=True)

        result = wifi_manager.manage_wifi_startup_fallback(
            speak_fn=speak_mock,
            announcement_fn=announcement_mock,
            fallback_env_str="Net1:pwd1",
            is_online_fn=is_online_mock,
        )
        assert result is True
        speak_mock.assert_not_called()

    @patch("core.wifi_manager.scan_visible_ssids", return_value=[])
    def test_fallback_no_visible_networks(self, mock_scan):
        """Verify announces wifi_not_found and exits when no networks are visible."""
        spoken = []
        result = wifi_manager.manage_wifi_startup_fallback(
            speak_fn=spoken.append,
            announcement_fn=lambda k: DEFAULT_ANNOUNCEMENTS.get(k, k),
            fallback_env_str="Net1:pwd1",
            is_online_fn=lambda: False,
        )
        assert result is False
        assert spoken == ["WiFi network not found."]

    @patch("core.wifi_manager.scan_visible_ssids", return_value=["UnrelatedNet"])
    @patch("core.wifi_manager.get_saved_wifi_connections", return_value=["SavedNet"])
    def test_fallback_candidate_not_in_scan(self, mock_saved, mock_scan):
        """Verify candidate networks not visible in scan are ignored."""
        spoken = []
        result = wifi_manager.manage_wifi_startup_fallback(
            speak_fn=spoken.append,
            announcement_fn=lambda k: DEFAULT_ANNOUNCEMENTS.get(k, k),
            fallback_env_str="EnvNet:pwd1",
            is_online_fn=lambda: False,
        )
        assert result is False
        # Only initial wifi_not_found was spoken; no connection attempts made
        assert spoken == ["WiFi network not found."]

    @patch("core.wifi_manager.connect_to_wifi")
    @patch("core.wifi_manager.scan_visible_ssids", return_value=["Net1", "Net2"])
    @patch("core.wifi_manager.get_saved_wifi_connections", return_value=["Net1"])
    def test_fallback_first_fails_second_succeeds(self, mock_saved, mock_scan, mock_connect):
        """Verify failover cycles through visible candidates until one succeeds."""
        # Net1 fails, Net2 succeeds
        mock_connect.side_effect = [False, True]
        spoken = []
        # Initial check is False; verification after Net2 succeeds is True
        online_state = [False, True]

        def check_online():
            return online_state.pop(0)

        result = wifi_manager.manage_wifi_startup_fallback(
            speak_fn=spoken.append,
            announcement_fn=lambda k: DEFAULT_ANNOUNCEMENTS.get(k, k),
            fallback_env_str="Net2:SecretPass",
            is_online_fn=check_online,
        )
        assert result is True
        assert spoken == [
            "WiFi network not found.",
            "Attempting to connect to WiFi network Net1.",
            "Attempting to connect to WiFi network Net1 failed.",
            "Attempting to connect to WiFi network Net2.",
            "Attempting to connect to WiFi network Net2 succeeded.",
        ]

    @patch("core.wifi_manager.connect_to_wifi", return_value=False)
    @patch("core.wifi_manager.scan_visible_ssids", return_value=["Net1"])
    @patch("core.wifi_manager.get_saved_wifi_connections", return_value=[])
    def test_fallback_all_candidates_fail(self, mock_saved, mock_scan, mock_connect):
        """Verify handles all visible candidates failing."""
        spoken = []
        result = wifi_manager.manage_wifi_startup_fallback(
            speak_fn=spoken.append,
            announcement_fn=lambda k: DEFAULT_ANNOUNCEMENTS.get(k, k),
            fallback_env_str="Net1:WrongPass",
            is_online_fn=lambda: False,
        )
        assert result is False
        assert spoken == [
            "WiFi network not found.",
            "Attempting to connect to WiFi network Net1.",
            "Attempting to connect to WiFi network Net1 failed.",
        ]


class TestAgentWifiIntegration:
    """Tests for AlsaGeistAgent Wi-Fi fallback startup integration."""

    @patch("core.wifi_manager.manage_wifi_startup_fallback", return_value=True)
    def test_agent_handle_wifi_startup_fallback_success(self, mock_manage):
        """Verify _handle_wifi_startup_fallback sets is_online to True on success."""
        from services.agent import AlsaGeistAgent
        agent = MagicMock(spec=AlsaGeistAgent)
        agent.is_online = False
        agent.speak = MagicMock()
        agent._announcement = MagicMock(side_effect=lambda k: DEFAULT_ANNOUNCEMENTS.get(k, k))
        agent._check_internet = MagicMock(return_value=True)

        # Call real unbound method
        success = AlsaGeistAgent._handle_wifi_startup_fallback(agent)
        assert success is True
        assert agent.is_online is True
        mock_manage.assert_called_once()

    @patch("core.wifi_manager.manage_wifi_startup_fallback", return_value=False)
    def test_agent_handle_wifi_startup_fallback_failure(self, mock_manage):
        """Verify _handle_wifi_startup_fallback retains is_online as False on failure."""
        from services.agent import AlsaGeistAgent
        agent = MagicMock(spec=AlsaGeistAgent)
        agent.is_online = False
        agent.speak = MagicMock()
        agent._announcement = MagicMock(side_effect=lambda k: DEFAULT_ANNOUNCEMENTS.get(k, k))
        agent._check_internet = MagicMock(return_value=False)

        success = AlsaGeistAgent._handle_wifi_startup_fallback(agent)
        assert success is False
        assert agent.is_online is False

