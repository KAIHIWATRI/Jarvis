"""
background_service.py — JARVIS Background Service Manager
Async task runner that keeps JARVIS alive in the background:
- Heartbeat / watchdog
- Hotkey listener
- Periodic system checks
- Task registry with named handles
- Mute/unmute state
"""

from __future__ import annotations

import asyncio
import logging
import time
import threading
from typing import Callable, Coroutine, Dict, Optional

logger = logging.getLogger("JARVIS.BGService")

try:
    import keyboard
    KEYBOARD_OK = True
except ImportError:
    KEYBOARD_OK = False
    logger.warning("keyboard not installed — global hotkeys unavailable.")

try:
    import psutil
    PSUTIL_OK = True
except ImportError:
    PSUTIL_OK = False


# ─────────────────────────────────────────────
# Background Service Manager
# ─────────────────────────────────────────────
class BackgroundServiceManager:
    """
    Runs a set of named async background tasks on a shared event loop.
    Tasks can be added, cancelled, and restarted dynamically.

    Built-in tasks
    --------------
    - heartbeat   : logs every 60 s, updates uptime
    - hotkey      : global keyboard shortcut (show/hide)
    - sys_watch   : periodic CPU/RAM alert checks
    """

    HEARTBEAT_INTERVAL = 60     # seconds
    SYS_WATCH_INTERVAL = 30     # seconds

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        on_status_change: Optional[Callable[[str], None]] = None,
        hotkey: str = "ctrl+alt+j",
        on_hotkey: Optional[Callable] = None,
    ):
        self._loop             = loop
        self._on_status_change = on_status_change or (lambda s: None)
        self._hotkey           = hotkey
        self._on_hotkey        = on_hotkey or (lambda: None)

        self._tasks: Dict[str, asyncio.Task] = {}
        self._stop_event = asyncio.Event()
        self._muted      = False
        self._start_ts   = time.time()

        # Callbacks registered by other modules
        self._on_listening: Optional[Callable] = None
        self._on_interrupt: Optional[Callable] = None

    # ── Lifecycle ─────────────────────────────

    async def run(self):
        """Start all background tasks and wait for stop signal."""
        logger.info("BackgroundServiceManager starting…")
        self._stop_event.clear()

        await self._spawn("heartbeat",  self._task_heartbeat())
        await self._spawn("sys_watch",  self._task_sys_watch())

        if KEYBOARD_OK:
            await self._spawn("hotkey", self._task_hotkey_listener())

        self._on_status_change("Running")
        logger.info("All background tasks running.")

        # Block until stop() is called
        await self._stop_event.wait()
        logger.info("Stop event received — cancelling tasks…")
        await self._cancel_all()

    async def stop(self):
        """Signal the run() loop to exit."""
        self._stop_event.set()

    # ── Task management ───────────────────────

    async def _spawn(self, name: str, coro: Coroutine):
        """Create and register a named task."""
        if name in self._tasks and not self._tasks[name].done():
            self._tasks[name].cancel()
        task = asyncio.ensure_future(coro, loop=self._loop)
        task.add_done_callback(lambda t: self._on_task_done(name, t))
        self._tasks[name] = task
        logger.debug("Spawned task: %s", name)

    def _on_task_done(self, name: str, task: asyncio.Task):
        if task.cancelled():
            logger.debug("Task cancelled: %s", name)
        elif task.exception():
            logger.error("Task %s raised: %s", name, task.exception())
        else:
            logger.debug("Task completed: %s", name)

    async def _cancel_all(self):
        for name, task in list(self._tasks.items()):
            if not task.done():
                task.cancel()
                try:
                    await asyncio.wait_for(asyncio.shield(task), timeout=2)
                except Exception:
                    pass
        self._tasks.clear()

    # ── Built-in tasks ────────────────────────

    async def _task_heartbeat(self):
        """Periodic heartbeat — logs uptime, updates tray tooltip."""
        while not self._stop_event.is_set():
            uptime_s  = int(time.time() - self._start_ts)
            h, rem    = divmod(uptime_s, 3600)
            m, s      = divmod(rem, 60)
            uptime_str= f"{h:02d}:{m:02d}:{s:02d}"
            logger.debug("Heartbeat | uptime=%s | tasks=%d", uptime_str, len(self._tasks))
            self._on_status_change(f"Running — uptime {uptime_str}")
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self.HEARTBEAT_INTERVAL,
                )
                break
            except asyncio.TimeoutError:
                pass

    async def _task_sys_watch(self):
        """
        Lightweight system resource watcher.
        Logs warnings if CPU or RAM hit thresholds.
        """
        if not PSUTIL_OK:
            logger.debug("psutil unavailable — sys_watch disabled.")
            return

        while not self._stop_event.is_set():
            try:
                cpu = psutil.cpu_percent(interval=None)
                ram = psutil.virtual_memory().percent
                if cpu > 90:
                    logger.warning("HIGH CPU: %.0f%%", cpu)
                if ram > 90:
                    logger.warning("HIGH RAM: %.0f%%", ram)
                logger.debug("SysWatch: CPU=%.0f%% RAM=%.0f%%", cpu, ram)
            except Exception as exc:
                logger.debug("SysWatch error: %s", exc)

            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self.SYS_WATCH_INTERVAL,
                )
                break
            except asyncio.TimeoutError:
                pass

    async def _task_hotkey_listener(self):
        """
        Register global hotkey in a thread-executor (keyboard library is blocking).
        """
        loop = asyncio.get_event_loop()

        def _register():
            try:
                keyboard.add_hotkey(self._hotkey, self._on_hotkey)
                logger.info("Global hotkey registered: %s", self._hotkey)
                # Block thread until stop event fires
                while not self._stop_event.is_set():
                    time.sleep(0.5)
                keyboard.remove_hotkey(self._hotkey)
            except Exception as exc:
                logger.error("Hotkey registration error: %s", exc)

        await loop.run_in_executor(None, _register)

    # ── Dynamic task API ──────────────────────

    def add_task(self, name: str, coro: Coroutine):
        """Schedule a custom coroutine from any thread."""
        asyncio.run_coroutine_threadsafe(
            self._spawn(name, coro), self._loop
        )

    def cancel_task(self, name: str):
        """Cancel a named task from any thread."""
        task = self._tasks.get(name)
        if task and not task.done():
            self._loop.call_soon_threadsafe(task.cancel)

    # ── Mute ──────────────────────────────────

    def toggle_mute(self):
        self._muted = not self._muted
        status = "Muted" if self._muted else "Running"
        self._on_status_change(status)
        logger.info("JARVIS %s.", status)

    @property
    def is_muted(self) -> bool:
        return self._muted

    @property
    def active_task_count(self) -> int:
        return sum(1 for t in self._tasks.values() if not t.done())

    def register_callbacks(
        self,
        on_listening: Optional[Callable] = None,
        on_interrupt: Optional[Callable] = None,
    ):
        self._on_listening = on_listening
        self._on_interrupt = on_interrupt
