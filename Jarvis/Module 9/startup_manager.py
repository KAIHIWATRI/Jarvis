"""
startup_manager.py — JARVIS Windows Startup Registration
Registers/removes JARVIS from Windows auto-start using two methods:
  1. HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run  (registry — preferred)
  2. Task Scheduler (schtasks.exe — fallback / elevated option)

Both methods are tried gracefully; errors are logged, not raised.
"""

from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger("JARVIS.Startup")

IS_WINDOWS = platform.system() == "Windows"

if IS_WINDOWS:
    try:
        import winreg
        WINREG_OK = True
    except ImportError:
        WINREG_OK = False
else:
    WINREG_OK = False


# ─────────────────────────────────────────────
# Startup Manager
# ─────────────────────────────────────────────
class StartupManager:
    """
    Cross-method Windows startup registration with fallback chain:
    Registry  →  Task Scheduler  →  Startup folder shortcut

    Non-Windows platforms log a warning and no-op gracefully.

    Usage
    -----
    sm = StartupManager("JARVIS", sys.executable, __file__)
    sm.enable()
    sm.is_enabled()   # True
    sm.disable()
    """

    # Registry path for current user autorun
    REG_PATH  = r"Software\Microsoft\Windows\CurrentVersion\Run"
    # Startup folder (per-user)
    STARTUP_FOLDER = Path(os.environ.get("APPDATA", "")) / \
                     "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"

    def __init__(
        self,
        app_name:   str,
        exe_path:   str,
        script_path: Optional[str] = None,
        extra_args:  str = "--silent",
    ):
        self._name        = app_name
        self._exe         = exe_path
        self._script      = script_path
        self._extra_args  = extra_args

        # Build the full launch command
        if script_path:
            self._cmd = f'"{exe_path}" "{script_path}" {extra_args}'.strip()
        else:
            self._cmd = f'"{exe_path}" {extra_args}'.strip()

        logger.debug("StartupManager: cmd=%r", self._cmd)

    # ── Public API ────────────────────────────

    def enable(self) -> bool:
        """Register JARVIS to start with Windows. Returns True on success."""
        if not IS_WINDOWS:
            logger.warning("Startup registration is Windows-only.")
            return False

        # Try registry first (most reliable, no elevation needed)
        if self._registry_write(self._cmd):
            logger.info("Startup enabled via registry.")
            return True

        # Fallback: Task Scheduler (survives user-profile moves)
        if self._task_scheduler_create():
            logger.info("Startup enabled via Task Scheduler.")
            return True

        # Last resort: shortcut in Startup folder
        if self._startup_folder_shortcut():
            logger.info("Startup enabled via Startup folder shortcut.")
            return True

        logger.error("All startup registration methods failed.")
        return False

    def disable(self) -> bool:
        """Remove all JARVIS startup registrations."""
        if not IS_WINDOWS:
            return False

        removed = False
        removed |= self._registry_delete()
        removed |= self._task_scheduler_delete()
        removed |= self._startup_folder_remove()

        if removed:
            logger.info("Startup registration removed.")
        else:
            logger.warning("No startup registration found to remove.")
        return removed

    def is_enabled(self) -> bool:
        """Check if any startup method is currently active."""
        if not IS_WINDOWS:
            return False
        return (
            self._registry_exists()
            or self._task_scheduler_exists()
            or self._startup_folder_exists()
        )

    def get_method(self) -> Optional[str]:
        """Return which startup method is currently active."""
        if not IS_WINDOWS:
            return None
        if self._registry_exists():
            return "registry"
        if self._task_scheduler_exists():
            return "task_scheduler"
        if self._startup_folder_exists():
            return "startup_folder"
        return None

    # ── Registry ──────────────────────────────

    def _registry_write(self, cmd: str) -> bool:
        if not WINREG_OK:
            return False
        try:
            key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                self.REG_PATH,
                0,
                winreg.KEY_SET_VALUE,
            )
            winreg.SetValueEx(key, self._name, 0, winreg.REG_SZ, cmd)
            winreg.CloseKey(key)
            return True
        except Exception as exc:
            logger.debug("Registry write failed: %s", exc)
            return False

    def _registry_delete(self) -> bool:
        if not WINREG_OK:
            return False
        try:
            key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                self.REG_PATH,
                0,
                winreg.KEY_SET_VALUE,
            )
            winreg.DeleteValue(key, self._name)
            winreg.CloseKey(key)
            return True
        except FileNotFoundError:
            return False
        except Exception as exc:
            logger.debug("Registry delete failed: %s", exc)
            return False

    def _registry_exists(self) -> bool:
        if not WINREG_OK:
            return False
        try:
            key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                self.REG_PATH,
                0,
                winreg.KEY_READ,
            )
            winreg.QueryValueEx(key, self._name)
            winreg.CloseKey(key)
            return True
        except Exception:
            return False

    # ── Task Scheduler ────────────────────────

    def _task_scheduler_create(self) -> bool:
        """Create a Task Scheduler entry that runs at logon, no elevation."""
        if not shutil.which("schtasks"):
            return False
        try:
            result = subprocess.run(
                [
                    "schtasks", "/create",
                    "/tn",  f"JARVIS\\{self._name}",
                    "/tr",  self._cmd,
                    "/sc",  "ONLOGON",
                    "/rl",  "LIMITED",     # No elevation
                    "/f",                  # Force overwrite
                ],
                capture_output=True,
                text=True,
                timeout=15,
            )
            return result.returncode == 0
        except Exception as exc:
            logger.debug("schtasks create failed: %s", exc)
            return False

    def _task_scheduler_delete(self) -> bool:
        if not shutil.which("schtasks"):
            return False
        try:
            result = subprocess.run(
                ["schtasks", "/delete", "/tn", f"JARVIS\\{self._name}", "/f"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.returncode == 0
        except Exception:
            return False

    def _task_scheduler_exists(self) -> bool:
        if not shutil.which("schtasks"):
            return False
        try:
            result = subprocess.run(
                ["schtasks", "/query", "/tn", f"JARVIS\\{self._name}"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.returncode == 0
        except Exception:
            return False

    # ── Startup folder shortcut ───────────────

    def _startup_folder_shortcut(self) -> bool:
        """Drop a .bat launcher in the Windows Startup folder."""
        try:
            self.STARTUP_FOLDER.mkdir(parents=True, exist_ok=True)
            bat_path = self.STARTUP_FOLDER / f"{self._name}.bat"
            bat_content = (
                f"@echo off\n"
                f"start \"\" {self._cmd}\n"
            )
            bat_path.write_text(bat_content, encoding="utf-8")
            return True
        except Exception as exc:
            logger.debug("Startup folder shortcut failed: %s", exc)
            return False

    def _startup_folder_remove(self) -> bool:
        try:
            bat_path = self.STARTUP_FOLDER / f"{self._name}.bat"
            if bat_path.exists():
                bat_path.unlink()
                return True
        except Exception:
            pass
        return False

    def _startup_folder_exists(self) -> bool:
        return (self.STARTUP_FOLDER / f"{self._name}.bat").exists()
