"""
tray_service.py — JARVIS Windows Background Service
Central orchestrator for system tray, background tasks, startup registration,
and graceful lifecycle management.

Architecture
────────────
JARVISTrayService
├── TrayIconManager       — pystray icon + menu
├── BackgroundServiceManager — async task runner
├── StartupManager        — Windows registry / Task Scheduler
├── ServiceState          — thread-safe state machine
└── ShutdownCoordinator   — safe multi-component teardown
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Optional

# Internal modules
from tray_icon import TrayIconManager
from background_service import BackgroundServiceManager
from startup_manager import StartupManager
from service_state import ServiceState, State
from shutdown_coordinator import ShutdownCoordinator

# ─────────────────────────────────────────────
# Logging — dual output: console + rotating file
# ─────────────────────────────────────────────
LOG_DIR = Path.home() / "JARVIS" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def _build_logger() -> logging.Logger:
    from logging.handlers import RotatingFileHandler

    logger = logging.getLogger("JARVIS.Tray")
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)-8s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    fh = RotatingFileHandler(
        LOG_DIR / "jarvis_tray.log",
        maxBytes=5 * 1024 * 1024,   # 5 MB
        backupCount=3,
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


logger = _build_logger()


# ─────────────────────────────────────────────
# JARVIS Tray Service
# ─────────────────────────────────────────────
class JARVISTrayService:
    """
    Top-level coordinator for the JARVIS assistant running as a
    persistent Windows background service with system-tray presence.

    Lifecycle
    ---------
    start()  → initialise all sub-systems, enter tray loop
    stop()   → graceful teardown (from tray menu, signal, or API)

    Thread model
    ------------
    Main thread   : pystray icon loop (required by pystray)
    BG thread     : asyncio event loop for background tasks
    Worker threads: spawned per-task by BackgroundServiceManager
    """

    VERSION = "2.0.0"
    APP_NAME = "JARVIS"

    def __init__(
        self,
        autostart:        bool = False,
        show_notifications: bool = True,
        hotkey:           str  = "ctrl+alt+j",
    ):
        self._autostart          = autostart
        self._show_notifications = show_notifications
        self._hotkey             = hotkey

        # Sub-systems (created in start())
        self._state:    Optional[ServiceState]          = None
        self._bg:       Optional[BackgroundServiceManager] = None
        self._tray:     Optional[TrayIconManager]       = None
        self._startup:  Optional[StartupManager]        = None
        self._shutdown: Optional[ShutdownCoordinator]   = None

        # Async loop lives on bg thread
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._bg_thread: Optional[threading.Thread]     = None

        # UI window reference (set by connect_ui)
        self._ui_window = None

        logger.info("JARVISTrayService v%s created.", self.VERSION)

    # ── Lifecycle ─────────────────────────────

    def start(self):
        """
        Initialise all sub-systems and start the tray icon loop.
        Blocks until the service is stopped (pystray runs on main thread).
        """
        logger.info("Starting JARVIS tray service…")

        try:
            self._init_state()
            self._init_background_loop()
            self._init_startup_manager()
            self._init_tray()
            self._register_signals()

            if self._autostart:
                self._startup.enable()

            self._state.transition(State.RUNNING)
            logger.info("JARVIS tray service running.")

            # Blocks here — pystray owns the main thread
            self._tray.run()

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received.")
        except Exception as exc:
            logger.exception("Fatal error in tray service: %s", exc)
        finally:
            self._do_shutdown()

    def stop(self, reason: str = "user request"):
        """Request a graceful shutdown from any thread."""
        logger.info("Stop requested: %s", reason)
        self._state.transition(State.STOPPING)
        if self._tray:
            self._tray.stop()

    def connect_ui(self, window):
        """
        Attach a CustomTkinter window.
        The tray service will show/hide it via the tray menu.
        """
        self._ui_window = window
        if self._tray:
            self._tray.set_ui_window(window)
        logger.debug("UI window connected.")

    # ── Sub-system initialisers ───────────────

    def _init_state(self):
        self._state = ServiceState()
        self._state.transition(State.STARTING)

    def _init_background_loop(self):
        """Start asyncio event loop on a dedicated daemon thread."""
        self._loop = asyncio.new_event_loop()
        self._bg   = BackgroundServiceManager(
            loop=self._loop,
            on_status_change=self._on_bg_status,
        )

        self._bg_thread = threading.Thread(
            target=self._run_bg_loop,
            name="JARVIS-BG-Loop",
            daemon=True,
        )
        self._bg_thread.start()
        logger.debug("Background asyncio loop started.")

    def _run_bg_loop(self):
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._bg.run())
        except Exception as exc:
            logger.error("Background loop error: %s", exc)
        finally:
            self._loop.close()
            logger.debug("Background asyncio loop closed.")

    def _init_startup_manager(self):
        self._startup = StartupManager(
            app_name=self.APP_NAME,
            exe_path=sys.executable,
            script_path=str(Path(__file__).resolve()),
        )

    def _init_tray(self):
        self._shutdown = ShutdownCoordinator(service=self)
        self._tray = TrayIconManager(
            app_name=self.APP_NAME,
            version=self.VERSION,
            on_show=self._on_tray_show,
            on_hide=self._on_tray_hide,
            on_quit=self._on_tray_quit,
            on_restart=self._on_tray_restart,
            on_toggle_startup=self._on_toggle_startup,
            on_mute_toggle=self._on_mute_toggle,
            on_open_logs=self._on_open_logs,
            startup_enabled_fn=lambda: self._startup.is_enabled() if self._startup else False,
            state_fn=lambda: self._state.current if self._state else State.IDLE,
            show_notifications=self._show_notifications,
        )

    def _register_signals(self):
        """Register OS-level signal handlers for clean shutdown."""
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, self._signal_handler)
            except (OSError, ValueError):
                pass   # Not all signals work on all platforms/threads

    def _signal_handler(self, signum, frame):
        logger.info("Signal %d received — shutting down.", signum)
        self.stop(reason=f"signal {signum}")

    # ── Shutdown ──────────────────────────────

    def _do_shutdown(self):
        """Perform the full shutdown sequence."""
        if self._state and self._state.current == State.STOPPED:
            return
        logger.info("Performing shutdown sequence…")

        if self._state:
            self._state.transition(State.STOPPING)

        # Stop background tasks
        if self._bg and self._loop and not self._loop.is_closed():
            future = asyncio.run_coroutine_threadsafe(
                self._bg.stop(), self._loop
            )
            try:
                future.result(timeout=5)
            except Exception as exc:
                logger.warning("BG stop error: %s", exc)

        # Hide/destroy UI
        if self._ui_window:
            try:
                self._ui_window.after(0, self._ui_window.destroy)
            except Exception:
                pass

        if self._state:
            self._state.transition(State.STOPPED)
        logger.info("JARVIS tray service stopped.")

    # ── Tray callbacks ────────────────────────

    def _on_tray_show(self):
        """Show the main JARVIS window."""
        if self._ui_window:
            try:
                self._ui_window.after(0, self._show_window)
            except Exception as exc:
                logger.debug("Show window error: %s", exc)

    def _show_window(self):
        if self._ui_window:
            self._ui_window.deiconify()
            self._ui_window.lift()
            self._ui_window.focus_force()

    def _on_tray_hide(self):
        """Minimise the main window to tray."""
        if self._ui_window:
            try:
                self._ui_window.after(0, self._ui_window.withdraw)
            except Exception:
                pass

    def _on_tray_quit(self):
        self.stop(reason="tray menu quit")

    def _on_tray_restart(self):
        logger.info("Restart requested from tray.")
        self.stop(reason="restart")
        # Re-launch in new process
        import subprocess
        subprocess.Popen([sys.executable, str(Path(__file__).resolve())])

    def _on_toggle_startup(self):
        if self._startup:
            if self._startup.is_enabled():
                self._startup.disable()
                logger.info("Startup disabled.")
            else:
                self._startup.enable()
                logger.info("Startup enabled.")

    def _on_mute_toggle(self):
        if self._bg:
            self._bg.toggle_mute()

    def _on_open_logs(self):
        try:
            os.startfile(str(LOG_DIR))
        except Exception:
            pass

    def _on_bg_status(self, status: str):
        """Called by BackgroundServiceManager when its status changes."""
        logger.debug("BG status: %s", status)
        if self._tray:
            self._tray.update_tooltip(f"JARVIS — {status}")

    # ── Public API ────────────────────────────

    def schedule_task(self, coro):
        """Schedule an async coroutine on the background loop from any thread."""
        if self._loop and not self._loop.is_closed():
            asyncio.run_coroutine_threadsafe(coro, self._loop)

    def get_status(self) -> dict:
        return {
            "state":           self._state.current.name if self._state else "UNKNOWN",
            "bg_tasks":        self._bg.active_task_count if self._bg else 0,
            "startup_enabled": self._startup.is_enabled() if self._startup else False,
            "version":         self.VERSION,
        }


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser(description="JARVIS Background Service")
    parser.add_argument("--autostart",    action="store_true", help="Register startup on launch")
    parser.add_argument("--silent",       action="store_true", help="Suppress notifications")
    parser.add_argument("--hotkey",       default="ctrl+alt+j", help="Global show/hide hotkey")
    parser.add_argument("--with-ui",      action="store_true", help="Launch full JARVIS UI")
    args = parser.parse_args()

    service = JARVISTrayService(
        autostart=args.autostart,
        show_notifications=not args.silent,
        hotkey=args.hotkey,
    )

    if args.with_ui:
        # Launch UI on a separate thread so pystray keeps the main thread
        def _launch_ui():
            time.sleep(1.5)   # Let tray settle first
            try:
                import customtkinter as ctk
                # Import from the JARVIS UI module (built in previous step)
                from jarvis_ui import JARVISApp
                app = JARVISApp()
                app.protocol("WM_DELETE_WINDOW", lambda: (app.withdraw(), None))
                service.connect_ui(app)
                app.mainloop()
            except ImportError:
                logger.warning("jarvis_ui not found — running tray-only mode.")
        threading.Thread(target=_launch_ui, name="JARVIS-UI", daemon=True).start()

    service.start()


if __name__ == "__main__":
    main()
