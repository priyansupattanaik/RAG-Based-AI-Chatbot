"""
Server runner script for Agentic AI FastAPI application with port conflict handling.

Resolves Windows [WinError 10013] and port collision errors by:
1. Detecting if the desired port is blocked or already bound by another process.
2. Automatically falling back to an open port (e.g. 8001, 8080, 8502, or next free port).
3. Displaying diagnostic guidance with conflicting PID and commands to kill it if desired.
"""

import argparse
from pathlib import Path
import socket
import subprocess
import sys
from typing import Any, Dict, List, Optional
import uvicorn

# Ensure safe console output encoding on Windows
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_FALLBACK_PORTS = [8001, 8080, 8502, 8002, 8003, 8004, 8005, 8081]


def is_port_available(port: int, host: str = "127.0.0.1") -> bool:
    """
    Check if a given port can be bound on the specified host.
    Handles:
    1. Active connection check (catches processes listening on 0.0.0.0 or 127.0.0.1).
    2. Windows SO_EXCLUSIVEADDRUSE to accurately catch WinError 10013 / 10048.
    """
    # 1. Connection check: if something is actively accepting connections, the port is occupied
    connect_hosts = ["127.0.0.1"]
    if host not in ("127.0.0.1", "0.0.0.0", "", "localhost"):
        connect_hosts.append(host)

    for ch in connect_hosts:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as conn_s:
                conn_s.settimeout(0.2)
                if conn_s.connect_ex((ch, port)) == 0:
                    return False
        except OSError:
            pass

    # 2. Bind check: attempt exclusive bind to detect reserved ranges or existing bindings
    hosts_to_bind = [host]
    if host in ("0.0.0.0", ""):
        hosts_to_bind.append("127.0.0.1")

    for bh in hosts_to_bind:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                    try:
                        s.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
                    except OSError:
                        pass
                s.bind((bh, port))
        except OSError:
            return False

    return True


def get_process_using_port(port: int) -> Optional[Dict[str, Any]]:
    """
    Identifies the PID and process name listening on a specified port (Windows-focused).
    Returns a dict with 'pid' and 'name' if found, else None.
    """
    try:
        out = subprocess.check_output(
            ["netstat", "-ano", "-p", "tcp"],
            text=True,
            stderr=subprocess.DEVNULL
        )
        for line in out.splitlines():
            line = line.strip()
            if "LISTENING" in line and f":{port} " in line:
                parts = line.split()
                pid = int(parts[-1])
                name = None
                try:
                    pname_out = subprocess.check_output(
                        ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                        text=True,
                        stderr=subprocess.DEVNULL
                    )
                    if pname_out.strip() and not pname_out.startswith("INFO:"):
                        name = pname_out.split(",")[0].strip('"')
                except Exception:
                    pass
                return {"pid": pid, "name": name}
    except Exception:
        pass
    return None


def find_available_port(
    preferred_port: int = 8000,
    fallback_ports: Optional[List[int]] = None,
    host: str = "127.0.0.1",
    max_scan: int = 50,
) -> int:
    """
    Finds the preferred port if open, or scans fallback ports / sequential ports
    until an open, bindable port is found.
    """
    if is_port_available(preferred_port, host=host):
        return preferred_port

    if fallback_ports is None:
        fallback_ports = DEFAULT_FALLBACK_PORTS

    # Check designated fallback candidates first
    for port in fallback_ports:
        if port != preferred_port and is_port_available(port, host=host):
            return port

    # Scan sequentially starting from preferred_port + 1
    for port in range(preferred_port + 1, preferred_port + max_scan + 1):
        if is_port_available(port, host=host):
            return port

    # Final fallback: let the OS kernel assign an ephemeral port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def print_port_conflict_help(blocked_port: int, chosen_port: int, host: str) -> None:
    """Print informative diagnostics for the user regarding port collision on Windows."""
    proc_info = get_process_using_port(blocked_port)
    if proc_info and proc_info.get("pid"):
        pid = proc_info["pid"]
        pname = f" ({proc_info['name']})" if proc_info.get("name") else ""
        proc_line = f"   Conflicting process: PID {pid}{pname}\n"
        kill_cmd = f"taskkill /PID {pid} /F"
        kill_ps = f"Stop-Process -Id {pid} -Force"
    else:
        proc_line = ""
        kill_cmd = f"taskkill /PID <PID> /F  (find PID with: netstat -ano | findstr :{blocked_port})"
        kill_ps = f"Stop-Process -Id <PID> -Force"

    msg = (
        "=" * 70 + "\n"
        f"[PORT CONFLICT] {host}:{blocked_port} is in use or blocked.\n"
        f"   Windows error: [WinError 10013 / 10048] - Access forbidden / port occupied.\n"
        + proc_line +
        f"--> Automatically falling back to available port: {chosen_port}\n"
        f"--> FastAPI starting on: http://{host}:{chosen_port}\n"
        f"--> Swagger Docs at:   http://{host}:{chosen_port}/docs\n"
        + "-" * 70 + "\n"
        f"To free port {blocked_port} on Windows:\n"
        f"   Command Prompt: {kill_cmd}\n"
        f"   PowerShell:     {kill_ps}\n"
        + "=" * 70 + "\n"
    )
    print(msg, flush=True)


def start_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    reload: bool = True,
    auto_fallback: bool = True,
) -> int:
    """
    Starts the FastAPI Uvicorn server with automatic port fallback if port is occupied.
    Returns the port the server was started on.
    """
    final_port = port
    if not is_port_available(port, host=host):
        if auto_fallback:
            final_port = find_available_port(preferred_port=port, host=host)
            print_port_conflict_help(blocked_port=port, chosen_port=final_port, host=host)
        else:
            print(
                f"ERROR: Port {port} on host {host} is unavailable and auto_fallback is disabled.",
                file=sys.stderr,
                flush=True,
            )
            sys.exit(1)
    else:
        print(f"[INFO] Starting FastAPI server on http://{host}:{final_port}", flush=True)
        print(f"[INFO] Swagger Docs at: http://{host}:{final_port}/docs", flush=True)

    uvicorn.run(
        "api:app",
        host=host,
        port=final_port,
        reload=reload,
        app_dir=str(PROJECT_ROOT),
    )
    return final_port


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run Agentic AI FastAPI Server with auto-fallback port management"
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Preferred port (default: 8000)")
    parser.add_argument("--reload", action="store_true", default=True, help="Enable auto-reload (default: True)")
    parser.add_argument("--no-reload", dest="reload", action="store_false", help="Disable auto-reload")
    parser.add_argument("--fallback", dest="auto_fallback", action="store_true", default=True, help="Enable automatic port fallback (default: True)")
    parser.add_argument("--no-fallback", dest="auto_fallback", action="store_false", help="Disable port fallback")

    args = parser.parse_args(argv)
    return start_server(
        host=args.host,
        port=args.port,
        reload=args.reload,
        auto_fallback=args.auto_fallback,
    )


if __name__ == "__main__":
    main()
