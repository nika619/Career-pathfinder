"""
Career PathFinder — Launch Script (Enterprise Auto-Recovery & Instant Ctrl+C Shutdown)
======================================================================================
- Automatically frees stale zombie processes holding target port (e.g. 8000).
- Binds 127.0.0.1 locally so terminal hyperlinks open cleanly in browsers.
- Traps Ctrl+C / SIGINT / SIGTERM for instantaneous, clean shutdown with zero orphan sockets.
- Dynamically switches to 0.0.0.0 when deployed on cloud container environments via $PORT.
"""

import os
import signal
import socket
import sys
import subprocess
import uvicorn

# Configure UTF-8 encoding on Windows if supported
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def kill_stale_port_holder(port: int) -> None:
    """Terminate any stale process holding the target port before launch."""
    if sys.platform == "win32":
        try:
            out = subprocess.check_output(f"netstat -ano | findstr :{port}", shell=True).decode(errors="ignore")
            pids = set()
            for line in out.strip().splitlines():
                parts = line.split()
                if len(parts) >= 5 and "LISTENING" in parts:
                    pids.add(parts[-1])
            for pid in pids:
                if pid and pid != "0" and int(pid) != os.getpid():
                    subprocess.call(f"taskkill /F /PID {pid}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass


def is_port_available(port: int, host: str = "127.0.0.1") -> bool:
    """Check if a port is free to bind."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
            return True
        except OSError:
            return False


def find_available_port(start_port: int, max_attempts: int = 20, host: str = "127.0.0.1") -> int:
    """Find start_port or the next available port after clearing stale holders."""
    kill_stale_port_holder(start_port)
    for p in range(start_port, start_port + max_attempts):
        if is_port_available(p, host=host):
            return p
    return start_port


def setup_clean_shutdown(port: int):
    """Ensure Ctrl+C or kill signals terminate Python and free the socket immediately."""
    def _shutdown_handler(sig, frame):
        print("\n\n[+] Shutting down Career PathFinder Server cleanly...")
        kill_stale_port_holder(port)
        os._exit(0)

    signal.signal(signal.SIGINT, _shutdown_handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _shutdown_handler)


if __name__ == "__main__":
    is_cloud_container = "PORT" in os.environ
    bind_host = "0.0.0.0" if is_cloud_container else "127.0.0.1"
    
    preferred_port = int(os.environ.get("PORT", 8000))
    selected_port = find_available_port(preferred_port, host=bind_host)
    
    setup_clean_shutdown(selected_port)

    if selected_port != preferred_port:
        print(f"[NOTE] Port {preferred_port} was occupied. Auto-shifted to port {selected_port}.")

    print("=" * 68)
    print("  [+] Career PathFinder -- AI Learning Path Recommender Engine")
    print(f"  [+] Open in browser:  http://127.0.0.1:{selected_port}")
    print("  [+] Press CTRL+C at any time to stop the server cleanly.")
    print("=" * 68)

    try:
        uvicorn.run(
            "backend.app.main:app",
            host=bind_host,
            port=selected_port,
            reload=False,
            log_level="info"
        )
    except (KeyboardInterrupt, SystemExit):
        print("\n[+] Server stopped cleanly.")
        kill_stale_port_holder(selected_port)
        os._exit(0)
