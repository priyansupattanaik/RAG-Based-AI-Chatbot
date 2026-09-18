"""
Unit tests for run_api port detection, PID identification, and auto-fallback logic.
"""

import socket
import sys
from unittest.mock import patch
import pytest

from run_api import (
    is_port_available,
    find_available_port,
    get_process_using_port,
    print_port_conflict_help,
    start_server,
    main,
    DEFAULT_FALLBACK_PORTS,
)


def test_is_port_available_on_free_port():
    """Verify is_port_available returns True for an open port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        free_port = s.getsockname()[1]
    # Socket is closed, so free_port should now be free
    assert is_port_available(free_port, host="127.0.0.1") is True


def test_is_port_available_on_occupied_port():
    """Verify is_port_available returns False when a port is actively bound."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        occupied_port = s.getsockname()[1]
        # While socket is open and listening, is_port_available must return False
        assert is_port_available(occupied_port, host="127.0.0.1") is False


def test_is_port_available_on_wildcard_listener():
    """
    Verify that a process listening on 0.0.0.0 is properly detected as occupied
    even when checking 127.0.0.1.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("0.0.0.0", 0))
        s.listen(1)
        wildcard_port = s.getsockname()[1]
        assert is_port_available(wildcard_port, host="127.0.0.1") is False
        assert is_port_available(wildcard_port, host="0.0.0.0") is False


def test_find_available_port_prefers_open_port():
    """If the preferred port is open, find_available_port returns it directly."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        free_port = s.getsockname()[1]

    chosen = find_available_port(preferred_port=free_port, host="127.0.0.1")
    assert chosen == free_port


def test_find_available_port_fallback_when_occupied():
    """When preferred port is occupied, find_available_port selects a fallback port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        occupied_port = s.getsockname()[1]

        chosen = find_available_port(
            preferred_port=occupied_port,
            fallback_ports=[occupied_port, 9911, 9912],
            host="127.0.0.1",
        )
        assert chosen != occupied_port
        assert is_port_available(chosen, host="127.0.0.1") is True


def test_occupied_port_8000_detection():
    """
    If port 8000 is occupied in this environment, verify fallback chooses an alternative.
    """
    if not is_port_available(8000, host="127.0.0.1"):
        chosen = find_available_port(preferred_port=8000, host="127.0.0.1")
        assert chosen != 8000
        assert is_port_available(chosen, host="127.0.0.1") is True
        assert chosen in DEFAULT_FALLBACK_PORTS or chosen > 8000


def test_get_process_using_port_on_occupied_8000():
    """Verify that PID detection finds the process currently bound to port 8000."""
    if not is_port_available(8000, host="127.0.0.1"):
        proc_info = get_process_using_port(8000)
        assert proc_info is not None
        assert "pid" in proc_info
        assert isinstance(proc_info["pid"], int)
        assert proc_info["pid"] > 0


def test_print_port_conflict_help_executes(capsys):
    """Verify print_port_conflict_help outputs diagnostic banner without errors."""
    print_port_conflict_help(blocked_port=8000, chosen_port=8001, host="127.0.0.1")
    captured = capsys.readouterr()
    assert "[PORT CONFLICT] 127.0.0.1:8000 is in use or blocked." in captured.out
    assert "--> Automatically falling back to available port: 8001" in captured.out
    assert "http://127.0.0.1:8001/docs" in captured.out


@patch("uvicorn.run")
def test_start_server_with_auto_fallback(mock_uvicorn_run):
    """Verify start_server falls back when preferred port is occupied."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        busy_port = s.getsockname()[1]

        final_port = start_server(
            host="127.0.0.1",
            port=busy_port,
            reload=False,
            auto_fallback=True,
        )

        assert final_port != busy_port
        mock_uvicorn_run.assert_called_once()
        _, kwargs = mock_uvicorn_run.call_args
        assert kwargs["port"] == final_port
        assert kwargs["reload"] is False


def test_start_server_no_fallback_exits_on_occupied():
    """Verify start_server exits with code 1 if port occupied and auto_fallback=False."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        busy_port = s.getsockname()[1]

        with pytest.raises(SystemExit) as exc_info:
            start_server(
                host="127.0.0.1",
                port=busy_port,
                reload=False,
                auto_fallback=False,
            )
        assert exc_info.value.code == 1


@patch("run_api.start_server")
def test_main_cli_argument_parsing(mock_start_server):
    """Verify main() properly forwards parsed arguments to start_server."""
    mock_start_server.return_value = 8080
    res = main(["--port", "8080", "--host", "0.0.0.0", "--no-reload", "--no-fallback"])
    assert res == 8080
    mock_start_server.assert_called_once_with(
        host="0.0.0.0",
        port=8080,
        reload=False,
        auto_fallback=False,
    )
