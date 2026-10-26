"""
Application Launcher for Gemini Conversational Chatbot.
Usage:
    python ui.py
"""

import os
import sys
import time
import socket
import tempfile
import subprocess
import webbrowser
import urllib.request

# Ensure Windows console supports UTF-8 characters without charmap errors
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# 1. Environment Forwarding: If not running within the local .venv, re-execute using .venv
VENV_PYTHON = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "Scripts", "python.exe")
if os.path.exists(VENV_PYTHON):
    current_python = os.path.abspath(sys.executable).lower()
    target_python = os.path.abspath(VENV_PYTHON).lower()
    if current_python != target_python:
        # Re-run under .venv Python to ensure all dependencies are available
        proc = subprocess.run([VENV_PYTHON, os.path.abspath(__file__)] + sys.argv[1:])
        sys.exit(proc.returncode)


def find_free_port(default_port: int = 8501) -> int:
    """Check if the default port is available; find a free port if occupied."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", default_port))
            return default_port
        except OSError:
            pass

    # Find any free port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for_server(url: str, server_proc: subprocess.Popen, timeout: float = 25.0) -> bool:
    """Wait until the Streamlit server responds with HTTP 200, verifying it hasn't crashed."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        # Check if server process terminated unexpectedly
        if server_proc.poll() is not None:
            return False

        try:
            req = urllib.request.urlopen(url, timeout=1.5)
            if req.status == 200:
                return True
        except Exception:
            time.sleep(0.4)
    return False


def get_app_browser() -> str | None:
    """Find Microsoft Edge or Google Chrome to open as a dedicated application window."""
    candidate_paths = [
        # Microsoft Edge
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        # Google Chrome
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        # Local AppData paths
        os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\Edge\Application\msedge.exe"),
        os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
    ]
    for path in candidate_paths:
        if os.path.isfile(path):
            return path
    return None


def open_as_application(url: str) -> subprocess.Popen | None:
    """
    Launch the URL in a dedicated, borderless application window (App Mode),
    or fall back to the default web browser.
    """
    browser_exe = get_app_browser()
    if browser_exe:
        print(f"[*] Opening application window via: {os.path.basename(browser_exe)}")
        # Use an isolated profile in TEMP so Edge/Chrome runs as a standalone window
        # without delegating and immediately exiting.
        profile_dir = os.path.join(tempfile.gettempdir(), "gemini_chat_app_profile")
        cmd = [
            browser_exe,
            f"--app={url}",
            f"--user-data-dir={profile_dir}",
            "--window-size=1240,860",
            "--no-first-run",
            "--no-default-browser-check",
        ]
        try:
            return subprocess.Popen(cmd)
        except Exception as e:
            print(f"[!] Could not launch app mode: {e}. Opening default browser...")
            webbrowser.open(url)
            return None
    else:
        print("[*] Opening application in your default web browser...")
        webbrowser.open(url)
        return None


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    chatbot_file = os.path.join(script_dir, "chatbot.py")

    if not os.path.isfile(chatbot_file):
        print(f"Error: Could not locate {chatbot_file}", file=sys.stderr)
        sys.exit(1)

    port = find_free_port(8501)
    app_url = f"http://127.0.0.1:{port}"

    print("=" * 60)
    print(" [*] Launching SanviBot Application")
    print(f" URL: {app_url}")
    print("=" * 60)

    # Launch Streamlit server as a subprocess bound specifically to 127.0.0.1
    server_cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        chatbot_file,
        f"--server.port={port}",
        "--server.address=127.0.0.1",
        "--server.headless=true",
        "--browser.gatherUsageStats=false",
        "--server.enableCORS=false",
        "--server.enableXsrfProtection=false",
    ]

    print("[*] Starting backend server...")
    server_proc = subprocess.Popen(server_cmd)

    try:
        # Wait for the web server to be ready and responding
        print("[*] Waiting for server to initialize...")
        if wait_for_server(app_url, server_proc=server_proc, timeout=25.0):
            print("[*] Server is ready! Launching application window...")
            app_proc = open_as_application(app_url)
        else:
            if server_proc.poll() is not None:
                print(f"[!] Server terminated early with exit code {server_proc.returncode}.", file=sys.stderr)
                return
            print("[!] Server startup took longer than expected. Opening browser...")
            webbrowser.open(app_url)
            app_proc = None

        print("\n[+] Application is running successfully!")
        print(f"[+] Access URL: {app_url}")
        print("[+] Press Ctrl+C in this terminal to shut down the application.\n")

        # Monitor lifecycle
        start_wait = time.time()
        if app_proc:
            # Wait for app window process
            app_proc.wait()
            elapsed = time.time() - start_wait
            # If the process exited in under 2 seconds, it attached to an existing browser instance.
            # In that case, DO NOT terminate the server, keep it alive for the user!
            if elapsed < 2.5:
                print("[*] Application opened in browser session. Server active.")
                server_proc.wait()
            else:
                print("[*] Application window closed.")
        else:
            server_proc.wait()

    except KeyboardInterrupt:
        print("\n[*] Stopping application...")
    finally:
        # Cleanly terminate the Streamlit server
        if server_proc.poll() is None:
            print("[*] Shutting down backend server...")
            server_proc.terminate()
            try:
                server_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                server_proc.kill()
        print("[*] Application closed.")


def _is_running_inside_streamlit() -> bool:
    """Detect if ui.py is being executed by `streamlit run ui.py` (e.g. on Streamlit Community Cloud)."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


if __name__ == "__main__":
    if _is_running_inside_streamlit():
        from chatbot import run_streamlit_app
        run_streamlit_app()
    else:
        main()
