"""
tray_icon.py — JARVIS System Tray Icon Manager
Manages the pystray icon, dynamic context menu, balloon notifications,
and animated icon states (idle / listening / thinking / speaking / error).
"""

from __future__ import annotations

import io
import logging
import math
import threading
import time
from typing import Callable, Optional

logger = logging.getLogger("JARVIS.TrayIcon")

# ─────────────────────────────────────────────
# Pillow + pystray imports
# ─────────────────────────────────────────────
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_OK = True
except ImportError:
    PIL_OK = False
    logger.error("Pillow not installed — tray icon will be blank.")

try:
    import pystray
    from pystray import MenuItem as Item, Menu
    PYSTRAY_OK = True
except ImportError:
    PYSTRAY_OK = False
    logger.error("pystray not installed — tray integration unavailable.")


# ─────────────────────────────────────────────
# Icon renderer
# ─────────────────────────────────────────────
class IconRenderer:
    """
    Generates 64×64 JARVIS tray icon frames using Pillow.
    States: idle | listening | thinking | speaking | error | muted
    """

    SIZE = 64

    PALETTE = {
        "idle":      {"ring": (0, 180, 180),   "core": (0, 212, 212),  "bg": (5, 8, 16)},
        "listening": {"ring": (0, 220, 100),   "core": (0, 255, 136),  "bg": (5, 8, 16)},
        "thinking":  {"ring": (160, 80, 240),  "core": (168, 85, 247), "bg": (5, 8, 16)},
        "speaking":  {"ring": (0, 220, 255),   "core": (0, 255, 255),  "bg": (5, 8, 16)},
        "error":     {"ring": (200, 50, 50),   "core": (255, 61, 61),  "bg": (5, 8, 16)},
        "muted":     {"ring": (80, 80, 100),   "core": (120, 120, 140),"bg": (5, 8, 16)},
    }

    def __init__(self):
        self._cache: dict[str, list[Image.Image]] = {}

    def get_frame(self, state: str, frame_idx: int = 0) -> "Image.Image":
        if not PIL_OK:
            return self._blank()
        if state not in self._cache:
            self._cache[state] = self._render_all_frames(state)
        frames = self._cache[state]
        return frames[frame_idx % len(frames)]

    def _render_all_frames(self, state: str, n_frames: int = 12) -> list["Image.Image"]:
        return [self._render(state, i, n_frames) for i in range(n_frames)]

    def _render(self, state: str, frame: int, total: int) -> "Image.Image":
        s    = self.SIZE
        pal  = self.PALETTE.get(state, self.PALETTE["idle"])
        img  = Image.new("RGBA", (s, s), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        cx = cy = s // 2
        phase = (frame / total) * 2 * math.pi

        # Background circle
        draw.ellipse([2, 2, s-2, s-2], fill=pal["bg"] + (255,))

        # Pulsing outer ring
        pulse = 0.85 + 0.12 * math.sin(phase)
        r_outer = int(cx * 0.88 * pulse)
        for offset, alpha in [(0, 80), (1, 140), (2, 200)]:
            r = r_outer - offset
            if r > 0:
                ring_col = pal["ring"] + (alpha,)
                draw.ellipse(
                    [cx-r, cy-r, cx+r, cy+r],
                    outline=ring_col, width=1,
                )

        # Rotating arc (state-dependent speed)
        speeds = {"idle": 0.3, "listening": 1.2, "thinking": 2.0, "speaking": 0.8, "error": 0.1, "muted": 0}
        arc_rot = phase * speeds.get(state, 0.3) * (180 / math.pi)
        r_arc = int(cx * 0.72)
        if r_arc > 0:
            draw.arc(
                [cx-r_arc, cy-r_arc, cx+r_arc, cy+r_arc],
                start=arc_rot % 360,
                end=(arc_rot + 110) % 360,
                fill=pal["ring"] + (180,),
                width=2,
            )

        # Core dot — pulses with state
        core_r = int(cx * 0.32 * (0.9 + 0.15 * math.sin(phase * 1.5)))
        if core_r > 0:
            draw.ellipse(
                [cx-core_r, cy-core_r, cx+core_r, cy+core_r],
                fill=pal["core"] + (230,),
            )

        # State indicator dot (top-right corner)
        indicator = {
            "idle":      (80, 80, 100),
            "listening": (0, 255, 136),
            "thinking":  (168, 85, 247),
            "speaking":  (0, 255, 255),
            "error":     (255, 61, 61),
            "muted":     (200, 80, 80),
        }.get(state, (80, 80, 100))

        dot_r = 5
        draw.ellipse(
            [s-dot_r*2-2, 2, s-2, dot_r*2+2],
            fill=indicator + (220,),
        )

        return img

    def _blank(self) -> "Image.Image":
        if PIL_OK:
            img = Image.new("RGBA", (self.SIZE, self.SIZE), (30, 30, 30, 255))
            return img
        return None


# ─────────────────────────────────────────────
# Tray Icon Manager
# ─────────────────────────────────────────────
class TrayIconManager:
    """
    Manages the Windows system tray icon with:
    - Animated icon cycling through state frames
    - Dynamic context menu with live state
    - Balloon/toast notifications
    - Thread-safe updates
    """

    ANIMATION_FPS = 8    # low enough to be cheap

    def __init__(
        self,
        app_name:          str  = "JARVIS",
        version:           str  = "2.0",
        on_show:           Optional[Callable] = None,
        on_hide:           Optional[Callable] = None,
        on_quit:           Optional[Callable] = None,
        on_restart:        Optional[Callable] = None,
        on_toggle_startup: Optional[Callable] = None,
        on_mute_toggle:    Optional[Callable] = None,
        on_open_logs:      Optional[Callable] = None,
        startup_enabled_fn:Optional[Callable[[], bool]] = None,
        state_fn:          Optional[Callable] = None,
        show_notifications:bool = True,
    ):
        self._app_name          = app_name
        self._version           = version
        self._on_show           = on_show or (lambda: None)
        self._on_hide           = on_hide or (lambda: None)
        self._on_quit           = on_quit or (lambda: None)
        self._on_restart        = on_restart or (lambda: None)
        self._on_toggle_startup = on_toggle_startup or (lambda: None)
        self._on_mute_toggle    = on_mute_toggle or (lambda: None)
        self._on_open_logs      = on_open_logs or (lambda: None)
        self._startup_enabled   = startup_enabled_fn or (lambda: False)
        self._state_fn          = state_fn or (lambda: None)
        self._show_notifications= show_notifications

        self._renderer    = IconRenderer()
        self._icon: Optional[pystray.Icon] = None
        self._ui_window   = None

        # Animation state
        self._anim_state  = "idle"
        self._frame_idx   = 0
        self._anim_thread: Optional[threading.Thread] = None
        self._anim_stop   = threading.Event()
        self._lock        = threading.Lock()
        self._muted       = False

        logger.debug("TrayIconManager created.")

    def set_ui_window(self, window):
        self._ui_window = window

    # ── Lifecycle ─────────────────────────────

    def run(self):
        """Start the pystray event loop (blocks — must run on main thread)."""
        if not PYSTRAY_OK:
            logger.error("pystray unavailable — running without tray.")
            # Fallback: just block forever
            while True:
                time.sleep(1)

        icon_img = self._renderer.get_frame("idle", 0)
        self._icon = pystray.Icon(
            name=self._app_name,
            icon=icon_img,
            title=f"{self._app_name} Assistant",
            menu=self._build_menu(),
        )

        self._start_animation()
        logger.info("Tray icon running.")
        self._icon.run()

    def stop(self):
        """Stop the tray icon and animation."""
        self._anim_stop.set()
        if self._icon:
            try:
                self._icon.stop()
            except Exception as exc:
                logger.debug("Icon stop error: %s", exc)

    def update_tooltip(self, text: str):
        if self._icon:
            try:
                self._icon.title = text[:63]   # Windows tooltip limit
            except Exception:
                pass

    def set_state(self, state: str):
        """Update icon animation state: idle/listening/thinking/speaking/error/muted."""
        with self._lock:
            self._anim_state = state
        self._rebuild_menu()

    def notify(self, title: str, message: str, timeout: int = 3):
        """Show a Windows balloon notification from the tray."""
        if not self._show_notifications or not self._icon:
            return
        try:
            self._icon.notify(message, title)
        except Exception as exc:
            logger.debug("Notification error: %s", exc)

    # ── Menu ──────────────────────────────────

    def _build_menu(self) -> "Menu":
        from service_state import State

        current = self._state_fn()
        is_running = True   # always True if tray is alive
        startup_on = self._startup_enabled()
        muted      = self._muted

        menu = Menu(
            # ── Header (non-interactive) ──────
            Item(
                f"JARVIS v{self._version}",
                action=None,
                enabled=False,
            ),
            Item(
                f"Status: {self._anim_state.upper()}",
                action=None,
                enabled=False,
            ),
            Menu.SEPARATOR,

            # ── Window controls ───────────────
            Item("Show Dashboard",   self._cb_show,  default=True),
            Item("Hide to Tray",     self._cb_hide),
            Menu.SEPARATOR,

            # ── Assistant controls ────────────
            Item(
                "🔇 Unmute" if muted else "🔇 Mute",
                self._cb_mute_toggle,
            ),
            Item(
                "▶  Start Listening",
                self._cb_start_listening,
                enabled=not muted,
            ),
            Item(
                "⬛  Stop / Interrupt",
                self._cb_stop,
            ),
            Menu.SEPARATOR,

            # ── System ────────────────────────
            Item(
                "✓ Start with Windows" if startup_on else "  Start with Windows",
                self._cb_toggle_startup,
            ),
            Item("📂 Open Logs",     self._cb_open_logs),
            Item("🔁 Restart",       self._cb_restart),
            Menu.SEPARATOR,

            # ── Quit ──────────────────────────
            Item("✕  Quit JARVIS",   self._cb_quit),
        )
        return menu

    def _rebuild_menu(self):
        """Refresh menu (reflects live state changes)."""
        if self._icon:
            try:
                self._icon.menu = self._build_menu()
            except Exception:
                pass

    # ── Menu callbacks ────────────────────────

    def _cb_show(self, icon, item):
        self._on_show()

    def _cb_hide(self, icon, item):
        self._on_hide()

    def _cb_quit(self, icon, item):
        self._on_quit()

    def _cb_restart(self, icon, item):
        self._on_restart()

    def _cb_toggle_startup(self, icon, item):
        self._on_toggle_startup()
        self._rebuild_menu()

    def _cb_mute_toggle(self, icon, item):
        self._muted = not self._muted
        self._anim_state = "muted" if self._muted else "idle"
        self._on_mute_toggle()
        self._rebuild_menu()

    def _cb_open_logs(self, icon, item):
        self._on_open_logs()

    def _cb_start_listening(self, icon, item):
        """Trigger listening mode — route to BackgroundServiceManager via callback."""
        self.set_state("listening")
        self.notify("JARVIS", "Listening…")

    def _cb_stop(self, icon, item):
        self.set_state("idle")
        self.notify("JARVIS", "Stopped.")

    # ── Animation ─────────────────────────────

    def _start_animation(self):
        self._anim_stop.clear()
        self._anim_thread = threading.Thread(
            target=self._animate_loop,
            name="JARVIS-TrayAnim",
            daemon=True,
        )
        self._anim_thread.start()

    def _animate_loop(self):
        interval = 1.0 / self.ANIMATION_FPS
        while not self._anim_stop.is_set():
            with self._lock:
                state = self._anim_state
                self._frame_idx = (self._frame_idx + 1) % 12

            if self._icon:
                try:
                    frame = self._renderer.get_frame(state, self._frame_idx)
                    self._icon.icon = frame
                except Exception as exc:
                    logger.debug("Animation frame error: %s", exc)

            time.sleep(interval)
