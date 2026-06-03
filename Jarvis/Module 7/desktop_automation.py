"""
desktop_automation.py — JARVIS Desktop Automation
pyautogui + keyboard wrappers for mouse, keyboard, volume, and screenshots.
All actions include safety guards and human-like timing.
"""

from __future__ import annotations

import logging
import os
import platform
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger("JARVIS.Desktop")

# ─────────────────────────────────────────────
# Optional imports
# ─────────────────────────────────────────────
try:
    import pyautogui
    pyautogui.FAILSAFE     = True    # Move mouse to top-left to abort
    pyautogui.PAUSE        = 0.05    # Small pause between actions
    PYAUTOGUI_OK = True
except ImportError:
    PYAUTOGUI_OK = False
    logger.warning("pyautogui not installed — mouse/keyboard automation unavailable.")

try:
    import keyboard
    KEYBOARD_OK = True
except ImportError:
    KEYBOARD_OK = False
    logger.warning("keyboard not installed — hotkey support limited.")


OS = platform.system()   # "Windows" | "Darwin" | "Linux"


# ─────────────────────────────────────────────
# Screenshot directory
# ─────────────────────────────────────────────
SCREENSHOT_DIR = Path.home() / "JARVIS_Screenshots"
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# Desktop Automation
# ─────────────────────────────────────────────
class DesktopAutomation:
    """
    High-level desktop automation actions.
    Each method returns (success: bool, message: str).
    """

    # ── Typing ────────────────────────────────

    def type_text(self, text: str, interval: float = 0.03) -> tuple[bool, str]:
        """Type text at the current cursor position."""
        if not PYAUTOGUI_OK:
            return False, "pyautogui not available."
        try:
            pyautogui.write(text, interval=interval)
            logger.info("Typed text: %.40s…", text)
            return True, f"Typed: {text[:50]}"
        except Exception as exc:
            logger.error("type_text error: %s", exc)
            return False, f"Type failed: {exc}"

    def type_text_clipboard(self, text: str) -> tuple[bool, str]:
        """
        Type text via clipboard paste (faster for long strings, Unicode-safe).
        """
        try:
            import pyperclip
            pyperclip.copy(text)
            time.sleep(0.1)
            self.hotkey("ctrl", "v")
            return True, f"Pasted: {text[:50]}"
        except ImportError:
            return self.type_text(text)
        except Exception as exc:
            return False, f"Clipboard type failed: {exc}"

    # ── Keys ──────────────────────────────────

    def press_key(self, key: str) -> tuple[bool, str]:
        """Press a single key."""
        if not PYAUTOGUI_OK:
            return False, "pyautogui not available."
        try:
            pyautogui.press(key)
            logger.debug("Pressed key: %s", key)
            return True, f"Pressed: {key}"
        except Exception as exc:
            logger.error("press_key error: %s", exc)
            return False, f"Key press failed: {exc}"

    def hotkey(self, *keys: str) -> tuple[bool, str]:
        """
        Press a key combination (e.g. 'ctrl', 'c').
        Tries pyautogui first, falls back to keyboard module.
        """
        combo = "+".join(keys)
        if PYAUTOGUI_OK:
            try:
                pyautogui.hotkey(*keys)
                logger.debug("Hotkey: %s", combo)
                return True, f"Hotkey: {combo}"
            except Exception as exc:
                logger.debug("pyautogui hotkey failed, trying keyboard: %s", exc)

        if KEYBOARD_OK:
            try:
                keyboard.send(combo)
                return True, f"Hotkey: {combo}"
            except Exception as exc:
                return False, f"Hotkey failed: {exc}"

        return False, "No keyboard library available."

    def key_down(self, key: str) -> tuple[bool, str]:
        if PYAUTOGUI_OK:
            pyautogui.keyDown(key)
            return True, f"Key down: {key}"
        return False, "pyautogui not available."

    def key_up(self, key: str) -> tuple[bool, str]:
        if PYAUTOGUI_OK:
            pyautogui.keyUp(key)
            return True, f"Key up: {key}"
        return False, "pyautogui not available."

    # ── Mouse ─────────────────────────────────

    def click(
        self,
        x: Optional[int] = None,
        y: Optional[int] = None,
        button: str = "left",
        double: bool = False,
    ) -> tuple[bool, str]:
        """Click at (x, y) or current cursor position."""
        if not PYAUTOGUI_OK:
            return False, "pyautogui not available."
        try:
            clicks = 2 if double else 1
            if x is not None and y is not None:
                pyautogui.click(x, y, button=button, clicks=clicks)
                loc = f"({x}, {y})"
            else:
                pyautogui.click(button=button, clicks=clicks)
                loc = "current position"
            kind = "Double-clicked" if double else "Clicked"
            logger.debug("%s %s at %s", kind, button, loc)
            return True, f"{kind} {button} at {loc}"
        except Exception as exc:
            logger.error("click error: %s", exc)
            return False, f"Click failed: {exc}"

    def move_mouse(self, x: int, y: int, duration: float = 0.3) -> tuple[bool, str]:
        """Move mouse to absolute coordinates with smooth animation."""
        if not PYAUTOGUI_OK:
            return False, "pyautogui not available."
        try:
            pyautogui.moveTo(x, y, duration=duration, tween=pyautogui.easeInOutQuad)
            return True, f"Mouse moved to ({x}, {y})"
        except Exception as exc:
            return False, f"Mouse move failed: {exc}"

    def scroll(self, direction: str = "down", amount: int = 3) -> tuple[bool, str]:
        """Scroll the mouse wheel."""
        if not PYAUTOGUI_OK:
            return False, "pyautogui not available."
        try:
            clicks = -amount if direction == "down" else amount
            pyautogui.scroll(clicks)
            return True, f"Scrolled {direction} {amount} units"
        except Exception as exc:
            return False, f"Scroll failed: {exc}"

    def drag(self, x1: int, y1: int, x2: int, y2: int, duration: float = 0.5):
        if not PYAUTOGUI_OK:
            return False, "pyautogui not available."
        try:
            pyautogui.drag(x1, y1, x2 - x1, y2 - y1, duration=duration, button="left")
            return True, f"Dragged from ({x1},{y1}) to ({x2},{y2})"
        except Exception as exc:
            return False, f"Drag failed: {exc}"

    def get_mouse_position(self) -> tuple[int, int]:
        if PYAUTOGUI_OK:
            return pyautogui.position()
        return (0, 0)

    def get_screen_size(self) -> tuple[int, int]:
        if PYAUTOGUI_OK:
            return pyautogui.size()
        return (1920, 1080)

    # ── Screenshot ────────────────────────────

    def take_screenshot(
        self, filename: Optional[str] = None
    ) -> tuple[bool, str, Optional[str]]:
        """
        Capture the full desktop.
        Returns (success, message, filepath).
        """
        if not PYAUTOGUI_OK:
            return False, "pyautogui not available.", None
        try:
            if filename is None:
                ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"jarvis_screenshot_{ts}.png"
            if not filename.endswith(".png"):
                filename += ".png"

            path = SCREENSHOT_DIR / filename
            img  = pyautogui.screenshot()
            img.save(str(path))
            logger.info("Screenshot saved: %s", path)
            return True, f"Screenshot saved: {path}", str(path)
        except Exception as exc:
            logger.error("screenshot error: %s", exc)
            return False, f"Screenshot failed: {exc}", None

    def take_region_screenshot(
        self, x: int, y: int, width: int, height: int,
        filename: Optional[str] = None,
    ) -> tuple[bool, str, Optional[str]]:
        """Capture a region of the screen."""
        try:
            if not filename:
                ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"jarvis_region_{ts}.png"
            path = SCREENSHOT_DIR / filename
            img  = pyautogui.screenshot(region=(x, y, width, height))
            img.save(str(path))
            return True, f"Region screenshot saved: {path}", str(path)
        except Exception as exc:
            return False, f"Region screenshot failed: {exc}", None

    # ── Volume ────────────────────────────────

    def set_volume(self, level: int) -> tuple[bool, str]:
        """Set system volume (0-100)."""
        level = max(0, min(100, level))
        try:
            if OS == "Windows":
                return self._set_volume_windows(level)
            elif OS == "Darwin":
                return self._set_volume_macos(level)
            else:
                return self._set_volume_linux(level)
        except Exception as exc:
            logger.error("set_volume error: %s", exc)
            return False, f"Volume change failed: {exc}"

    def get_volume(self) -> tuple[Optional[int], str]:
        """Get current system volume (0-100)."""
        try:
            if OS == "Windows":
                return self._get_volume_windows()
            elif OS == "Darwin":
                return self._get_volume_macos()
            else:
                return self._get_volume_linux()
        except Exception as exc:
            return None, f"Get volume failed: {exc}"

    def mute(self) -> tuple[bool, str]:
        if OS == "Windows":
            try:
                from ctypes import cast, POINTER
                from comtypes import CLSCTX_ALL
                from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
                devices = AudioUtilities.GetSpeakers()
                interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                volume = cast(interface, POINTER(IAudioEndpointVolume))
                volume.SetMute(1, None)
                return True, "System muted."
            except Exception:
                pass
        # Fallback: press mute key
        return self.press_key("volumemute")

    def unmute(self) -> tuple[bool, str]:
        if OS == "Windows":
            try:
                from ctypes import cast, POINTER
                from comtypes import CLSCTX_ALL
                from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
                devices = AudioUtilities.GetSpeakers()
                interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                volume = cast(interface, POINTER(IAudioEndpointVolume))
                volume.SetMute(0, None)
                return True, "System unmuted."
            except Exception:
                pass
        return self.press_key("volumemute")

    # ── Volume OS implementations ─────────────

    def _set_volume_windows(self, level: int) -> tuple[bool, str]:
        # Method 1: pycaw (precise)
        try:
            from ctypes import cast, POINTER
            from comtypes import CLSCTX_ALL
            from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
            import math
            devices   = AudioUtilities.GetSpeakers()
            interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            vol       = cast(interface, POINTER(IAudioEndpointVolume))
            # Convert 0-100 to dB scalar
            scalar = level / 100.0
            vol.SetMasterVolumeLevelScalar(scalar, None)
            logger.info("Volume set to %d%% (pycaw)", level)
            return True, f"Volume set to {level}%"
        except ImportError:
            pass

        # Method 2: PowerShell
        try:
            ps = (
                f"$obj = New-Object -ComObject WScript.Shell; "
                f"$vol = [int]([math]::Round({level} / 2)); "
                f"1..$vol | %{{ $obj.SendKeys([char]175) }}"
            )
            subprocess.run(["powershell", "-Command", ps],
                           capture_output=True, timeout=5)
            return True, f"Volume set to ~{level}%"
        except Exception as exc:
            return False, f"Volume set failed: {exc}"

    def _get_volume_windows(self) -> tuple[Optional[int], str]:
        try:
            from ctypes import cast, POINTER
            from comtypes import CLSCTX_ALL
            from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
            devices   = AudioUtilities.GetSpeakers()
            interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            vol       = cast(interface, POINTER(IAudioEndpointVolume))
            level     = int(vol.GetMasterVolumeLevelScalar() * 100)
            return level, f"Volume is {level}%"
        except Exception:
            return None, "Could not read volume."

    def _set_volume_macos(self, level: int) -> tuple[bool, str]:
        try:
            subprocess.run(["osascript", "-e", f"set volume output volume {level}"],
                           check=True, timeout=5)
            return True, f"Volume set to {level}%"
        except Exception as exc:
            return False, f"macOS volume failed: {exc}"

    def _get_volume_macos(self) -> tuple[Optional[int], str]:
        try:
            result = subprocess.run(
                ["osascript", "-e", "output volume of (get volume settings)"],
                capture_output=True, text=True, timeout=5,
            )
            level = int(result.stdout.strip())
            return level, f"Volume is {level}%"
        except Exception:
            return None, "Could not read volume."

    def _set_volume_linux(self, level: int) -> tuple[bool, str]:
        # Try amixer (ALSA)
        try:
            subprocess.run(
                ["amixer", "-D", "pulse", "sset", "Master", f"{level}%"],
                check=True, capture_output=True, timeout=5,
            )
            return True, f"Volume set to {level}%"
        except FileNotFoundError:
            pass
        # Try pactl (PulseAudio)
        try:
            subprocess.run(
                ["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{level}%"],
                check=True, timeout=5,
            )
            return True, f"Volume set to {level}%"
        except Exception as exc:
            return False, f"Linux volume failed: {exc}"

    def _get_volume_linux(self) -> tuple[Optional[int], str]:
        try:
            result = subprocess.run(
                ["amixer", "-D", "pulse", "sget", "Master"],
                capture_output=True, text=True, timeout=5,
            )
            import re
            match = re.search(r"\[(\d+)%\]", result.stdout)
            if match:
                level = int(match.group(1))
                return level, f"Volume is {level}%"
        except Exception:
            pass
        return None, "Could not read volume."

    # ── Window management ─────────────────────

    def minimize_window(self):
        if OS == "Windows":
            return self.hotkey("winleft", "down")
        elif OS == "Darwin":
            return self.hotkey("command", "m")
        return self.hotkey("super", "h")

    def maximize_window(self):
        if OS == "Windows":
            return self.hotkey("winleft", "up")
        elif OS == "Darwin":
            return self.hotkey("ctrl", "command", "f")
        return self.hotkey("super", "up")

    def close_window(self):
        if OS == "Darwin":
            return self.hotkey("command", "w")
        return self.hotkey("alt", "f4")
