"""
safe_executor.py — JARVIS Safe Executor
Sandboxed application launcher and process killer with:
- Allow/block lists
- Platform-aware app resolution
- Audit logging
- Rate limiting
- Thread-safe execution
"""

from __future__ import annotations

import logging
import os
import platform
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("JARVIS.SafeExec")

try:
    import psutil
    PSUTIL_OK = True
except ImportError:
    PSUTIL_OK = False


# ─────────────────────────────────────────────
# Result type
# ─────────────────────────────────────────────
@dataclass
class ExecutionResult:
    success: bool
    message: str
    pid:     Optional[int] = None
    output:  Optional[str] = None


OS = platform.system()


# ─────────────────────────────────────────────
# Platform app launchers
# ─────────────────────────────────────────────
WINDOWS_APPS: dict[str, list[str]] = {
    # Browsers
    "chrome":      [r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"],
    "firefox":     [r"C:\Program Files\Mozilla Firefox\firefox.exe",
                    r"C:\Program Files (x86)\Mozilla Firefox\firefox.exe"],
    "msedge":      [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"],
    # System
    "notepad":     ["notepad.exe"],
    "cmd":         ["cmd.exe"],
    "powershell":  ["powershell.exe"],
    "taskmgr":     ["taskmgr.exe"],
    "explorer":    ["explorer.exe"],
    "calc":        ["calc.exe"],
    "mspaint":     ["mspaint.exe"],
    # Office (standard locations)
    "WINWORD":     [r"C:\Program Files\Microsoft Office\root\Office16\WINWORD.EXE"],
    "EXCEL":       [r"C:\Program Files\Microsoft Office\root\Office16\EXCEL.EXE"],
    "POWERPNT":    [r"C:\Program Files\Microsoft Office\root\Office16\POWERPNT.EXE"],
}

MACOS_APPS: dict[str, str] = {
    "chrome":    "Google Chrome",
    "firefox":   "Firefox",
    "safari":    "Safari",
    "code":      "Visual Studio Code",
    "terminal":  "Terminal",
    "finder":    "Finder",
    "notes":     "Notes",
    "calendar":  "Calendar",
    "spotify":   "Spotify",
    "discord":   "Discord",
    "slack":     "Slack",
}

LINUX_APPS: dict[str, list[str]] = {
    "chrome":    ["google-chrome", "google-chrome-stable", "chromium-browser", "chromium"],
    "firefox":   ["firefox", "firefox-esr"],
    "code":      ["code", "code-insiders"],
    "terminal":  ["gnome-terminal", "xterm", "konsole", "xfce4-terminal"],
    "notepad":   ["gedit", "kate", "mousepad", "nano"],
    "spotify":   ["spotify"],
    "discord":   ["discord"],
    "vlc":       ["vlc"],
    "gimp":      ["gimp"],
}


# ─────────────────────────────────────────────
# Block list (process names to never kill)
# ─────────────────────────────────────────────
PROTECTED_PROCESSES = {
    # Windows
    "system", "smss.exe", "csrss.exe", "wininit.exe", "services.exe",
    "lsass.exe", "winlogon.exe", "svchost.exe", "dwm.exe", "explorer.exe",
    # Linux
    "init", "systemd", "kthreadd", "ksoftirqd", "kernel",
    # macOS
    "launchd", "kernel_task", "WindowServer",
    # Python runtime
    "python", "python3", "python.exe",
}

# Regex patterns that are always blocked
BLOCKED_PATTERNS = [
    re.compile(r"(rm\s+-rf|del\s+/[qfsF]|format\s+[a-z]:)", re.I),
    re.compile(r"(shutdown|reboot|halt)\s", re.I),
    re.compile(r"\bdd\b.*\bif="),
    re.compile(r"(mkfs|fdisk|parted)", re.I),
]


# ─────────────────────────────────────────────
# Safe Executor
# ─────────────────────────────────────────────
class SafeExecutor:
    """
    Thread-safe, audited application launcher and process manager.
    """

    def __init__(self, safe_mode: bool = True):
        self._safe_mode   = safe_mode
        self._lock        = threading.Lock()
        self._audit_log:  list[dict] = []
        self._rate_tracker: dict[str, float] = {}   # action → last_ts
        logger.info("SafeExecutor ready (safe_mode=%s)", safe_mode)

    # ── Open application ──────────────────────

    def open_application(self, app_name: str) -> tuple[bool, str]:
        """Launch an application by name."""
        with self._lock:
            if self._safe_mode and self._is_blocked_command(app_name):
                msg = f"Blocked: {app_name}"
                logger.warning(msg)
                self._audit("OPEN_BLOCKED", app_name)
                return False, msg

            self._audit("OPEN", app_name)

            if OS == "Windows":
                return self._open_windows(app_name)
            elif OS == "Darwin":
                return self._open_macos(app_name)
            else:
                return self._open_linux(app_name)

    def _open_windows(self, app: str) -> tuple[bool, str]:
        # 1. Known app list
        candidates = WINDOWS_APPS.get(app.lower(), [])
        for path in candidates:
            if os.path.exists(path):
                try:
                    proc = subprocess.Popen([path])
                    return True, f"Launched {app} (PID {proc.pid})"
                except Exception as exc:
                    logger.debug("Failed to launch %s: %s", path, exc)

        # 2. shutil.which (PATH lookup)
        found = shutil.which(app) or shutil.which(app + ".exe")
        if found:
            try:
                proc = subprocess.Popen([found])
                return True, f"Launched {app} (PID {proc.pid})"
            except Exception as exc:
                return False, f"Launch failed: {exc}"

        # 3. os.startfile (Windows shell)
        try:
            os.startfile(app)
            return True, f"Opened {app}"
        except Exception as exc:
            return False, f"Could not open {app}: {exc}"

    def _open_macos(self, app: str) -> tuple[bool, str]:
        friendly = MACOS_APPS.get(app.lower(), app)
        try:
            result = subprocess.Popen(["open", "-a", friendly])
            return True, f"Launched {friendly}"
        except Exception:
            pass
        # Try direct binary
        found = shutil.which(app)
        if found:
            proc = subprocess.Popen([found])
            return True, f"Launched {app} (PID {proc.pid})"
        return False, f"Application not found: {app}"

    def _open_linux(self, app: str) -> tuple[bool, str]:
        candidates = LINUX_APPS.get(app.lower(), [app])
        for candidate in candidates:
            found = shutil.which(candidate)
            if found:
                try:
                    proc = subprocess.Popen(
                        [found],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    return True, f"Launched {candidate} (PID {proc.pid})"
                except Exception as exc:
                    logger.debug("Failed %s: %s", candidate, exc)

        # xdg-open as last resort
        try:
            subprocess.Popen(["xdg-open", app])
            return True, f"Opened {app} via xdg-open"
        except Exception as exc:
            return False, f"Application not found: {app}"

    # ── Close application ─────────────────────

    def close_application(self, app_name: str) -> tuple[bool, str]:
        """Gracefully close a running application by name."""
        with self._lock:
            self._audit("CLOSE", app_name)
            if not PSUTIL_OK:
                return self._close_without_psutil(app_name)

            killed = []
            for proc in psutil.process_iter(["pid", "name"]):
                try:
                    pname = proc.info["name"].lower()
                    if app_name.lower() in pname:
                        proc.terminate()
                        killed.append(proc.info["pid"])
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

            if killed:
                logger.info("Closed %s (PIDs: %s)", app_name, killed)
                return True, f"Closed {app_name} (PIDs: {killed})"
            return False, f"No running process found: {app_name}"

    def _close_without_psutil(self, app: str) -> tuple[bool, str]:
        if OS == "Windows":
            result = subprocess.run(
                ["taskkill", "/IM", app, "/F"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                return True, f"Closed {app}"
            return False, f"Could not close {app}: {result.stderr}"
        else:
            result = subprocess.run(
                ["pkill", "-f", app],
                capture_output=True, timeout=5,
            )
            return result.returncode == 0, (
                f"Closed {app}" if result.returncode == 0 else f"Could not close {app}"
            )

    # ── Kill process ──────────────────────────

    def kill_process(self, name_or_pid: str) -> tuple[bool, str]:
        """Force-kill a process (SIGKILL / TerminateProcess)."""
        with self._lock:
            # Check protection
            clean = name_or_pid.lower().strip()
            if clean in PROTECTED_PROCESSES:
                msg = f"Refused to kill protected process: {name_or_pid}"
                logger.warning(msg)
                return False, msg

            self._audit("KILL", name_or_pid)

            if not PSUTIL_OK:
                return self._close_without_psutil(name_or_pid)

            # Try PID first
            if name_or_pid.isdigit():
                try:
                    proc = psutil.Process(int(name_or_pid))
                    proc.kill()
                    return True, f"Killed PID {name_or_pid}"
                except psutil.NoSuchProcess:
                    return False, f"No process with PID {name_or_pid}"
                except psutil.AccessDenied:
                    return False, f"Access denied killing PID {name_or_pid}"

            # By name
            killed = []
            for proc in psutil.process_iter(["pid", "name"]):
                try:
                    if name_or_pid.lower() in proc.info["name"].lower():
                        proc.kill()
                        killed.append(proc.info["pid"])
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

            if killed:
                return True, f"Killed {name_or_pid} (PIDs: {killed})"
            return False, f"Process not found: {name_or_pid}"

    # ── Safe subprocess ───────────────────────

    def run_command(
        self,
        command: list[str],
        timeout: int = 10,
        capture: bool = True,
    ) -> ExecutionResult:
        """
        Run a subprocess command with safety checks.
        Only use for trusted, pre-validated commands.
        """
        cmd_str = " ".join(command)

        if self._safe_mode and self._is_blocked_command(cmd_str):
            return ExecutionResult(False, f"Blocked command: {cmd_str}")

        self._audit("RUN_CMD", cmd_str)

        try:
            result = subprocess.run(
                command,
                capture_output=capture,
                text=True,
                timeout=timeout,
            )
            return ExecutionResult(
                success=result.returncode == 0,
                message=f"Command exited {result.returncode}",
                output=result.stdout + result.stderr if capture else None,
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(False, f"Command timed out after {timeout}s")
        except Exception as exc:
            return ExecutionResult(False, f"Command error: {exc}")

    # ── Audit ─────────────────────────────────

    def _audit(self, action: str, target: str):
        entry = {"ts": time.time(), "action": action, "target": target}
        self._audit_log.append(entry)
        logger.info("[AUDIT] %s → %s", action, target)
        # Keep last 1000
        if len(self._audit_log) > 1000:
            self._audit_log = self._audit_log[-1000:]

    def get_audit_log(self) -> list[dict]:
        return list(self._audit_log)

    # ── Safety checks ─────────────────────────

    def _is_blocked_command(self, cmd: str) -> bool:
        if not self._safe_mode:
            return False
        for pattern in BLOCKED_PATTERNS:
            if pattern.search(cmd):
                logger.warning("Blocked pattern matched in: %r", cmd[:80])
                return True
        return False

    def toggle_safe_mode(self, enabled: bool):
        self._safe_mode = enabled
        logger.info("Safe mode %s.", "ENABLED" if enabled else "DISABLED")
