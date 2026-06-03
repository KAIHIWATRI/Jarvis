"""
automation_engine.py — JARVIS Automation Engine
Central orchestrator: parses commands, dispatches to subsystems,
enforces safety, handles errors, and manages thread-safe execution.

Architecture
────────────
AutomationEngine
├── CommandParser          — NLP-style intent + slot extraction
├── BrowserManager         — Selenium Chrome/Firefox wrapper
├── DesktopAutomation      — pyautogui + keyboard + subprocess
├── SystemMonitor          — psutil resource monitor
├── VolumeController       — platform-aware volume control
├── ScreenshotManager      — capture, annotate, save
└── SafeExecutor           — sandboxed, rate-limited, logged dispatcher
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional

# Sub-module imports (same package)
from command_parser   import CommandParser, ParsedCommand, Intent
from browser_manager  import BrowserManager
from desktop_automation import DesktopAutomation
from system_monitor   import SystemMonitor
from safe_executor    import SafeExecutor, ExecutionResult


# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
def _setup_logger() -> logging.Logger:
    logger = logging.getLogger("JARVIS.Automation")
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        fmt = logging.Formatter(
            "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
            datefmt="%H:%M:%S",
        )
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        ch.setLevel(logging.DEBUG)

        fh = logging.FileHandler("jarvis_automation.log", encoding="utf-8")
        fh.setFormatter(fmt)
        fh.setLevel(logging.INFO)

        logger.addHandler(ch)
        logger.addHandler(fh)
    return logger


logger = _setup_logger()


# ─────────────────────────────────────────────
# Engine result
# ─────────────────────────────────────────────
@dataclass
class AutomationResult:
    success:   bool
    message:   str
    data:      Any           = None
    intent:    Optional[str] = None
    duration:  float         = 0.0
    error:     Optional[str] = None

    def __str__(self) -> str:
        status = "✓" if self.success else "✗"
        return f"[{status}] {self.message}" + (f" | {self.error}" if self.error else "")


# ─────────────────────────────────────────────
# Automation Engine
# ─────────────────────────────────────────────
class AutomationEngine:
    """
    Single entry-point for all JARVIS automation tasks.

    Usage
    -----
    engine = AutomationEngine()
    result = engine.execute("open notepad")
    result = engine.execute("search google for Python tutorials")
    result = engine.execute("take a screenshot")
    result = engine.execute("set volume to 60")

    Thread-safe: multiple threads can call execute() concurrently.
    """

    def __init__(
        self,
        browser:        str  = "chrome",
        headless:       bool = False,
        safe_mode:      bool = True,
        rate_limit_sec: float = 0.5,
        on_event:       Optional[Callable[[str, dict], None]] = None,
    ):
        """
        Parameters
        ----------
        browser        : 'chrome' or 'firefox'
        headless       : run browser without GUI
        safe_mode      : block dangerous commands
        rate_limit_sec : min seconds between consecutive commands
        on_event       : optional callback(event_name, data) for UI hooks
        """
        logger.info("Initialising AutomationEngine…")

        self._on_event      = on_event
        self._lock          = threading.Lock()
        self._last_exec_ts  = 0.0
        self._rate_limit    = rate_limit_sec
        self._exec_count    = 0

        # Sub-systems (lazy-init for browser)
        self.parser   = CommandParser()
        self.desktop  = DesktopAutomation()
        self.monitor  = SystemMonitor()
        self.executor = SafeExecutor(safe_mode=safe_mode)

        self._browser_cfg  = {"browser": browser, "headless": headless}
        self._browser: Optional[BrowserManager] = None   # lazy

        # Intent → handler dispatch table
        self._handlers: dict[Intent, Callable] = {
            Intent.OPEN_APP:        self._handle_open_app,
            Intent.CLOSE_APP:       self._handle_close_app,
            Intent.GOOGLE_SEARCH:   self._handle_google_search,
            Intent.YOUTUBE_SEARCH:  self._handle_youtube_search,
            Intent.OPEN_URL:        self._handle_open_url,
            Intent.TYPE_TEXT:       self._handle_type_text,
            Intent.PRESS_KEY:       self._handle_press_key,
            Intent.SCREENSHOT:      self._handle_screenshot,
            Intent.SET_VOLUME:      self._handle_set_volume,
            Intent.GET_VOLUME:      self._handle_get_volume,
            Intent.SYSTEM_STATS:    self._handle_system_stats,
            Intent.KILL_PROCESS:    self._handle_kill_process,
            Intent.SCROLL:          self._handle_scroll,
            Intent.CLICK:           self._handle_click,
            Intent.HOTKEY:          self._handle_hotkey,
            Intent.MOVE_MOUSE:      self._handle_move_mouse,
            Intent.CLOSE_BROWSER:   self._handle_close_browser,
            Intent.UNKNOWN:         self._handle_unknown,
        }

        logger.info("AutomationEngine ready.")

    # ── Public API ────────────────────────────────────────────────────────────

    def execute(self, command: str) -> AutomationResult:
        """
        Parse and execute a natural-language automation command.
        Thread-safe; rate-limited.
        """
        t0 = time.perf_counter()

        with self._lock:
            # Rate limiting
            elapsed = time.time() - self._last_exec_ts
            if elapsed < self._rate_limit:
                time.sleep(self._rate_limit - elapsed)
            self._last_exec_ts = time.time()
            self._exec_count  += 1

        logger.info("Command #%d: %r", self._exec_count, command)
        self._emit("command_received", {"command": command, "count": self._exec_count})

        # Parse
        try:
            parsed = self.parser.parse(command)
            logger.debug("Parsed → intent=%s slots=%s", parsed.intent, parsed.slots)
        except Exception as exc:
            logger.error("Parse error: %s", exc)
            return AutomationResult(False, "Failed to parse command.", error=str(exc))

        # Dispatch
        handler = self._handlers.get(parsed.intent, self._handle_unknown)
        try:
            result = handler(parsed)
        except Exception as exc:
            logger.exception("Handler error for intent=%s", parsed.intent)
            result = AutomationResult(
                False,
                f"Automation error: {exc}",
                intent=parsed.intent.name,
                error=str(exc),
            )

        result.intent   = parsed.intent.name
        result.duration = time.perf_counter() - t0
        logger.info("Result: %s (%.2fs)", result, result.duration)
        self._emit("command_done", {"result": str(result), "duration": result.duration})
        return result

    def execute_batch(self, commands: list[str], delay: float = 0.3) -> list[AutomationResult]:
        """Execute a list of commands sequentially with a delay between each."""
        results = []
        for cmd in commands:
            results.append(self.execute(cmd))
            time.sleep(delay)
        return results

    def shutdown(self):
        """Cleanly shut down all sub-systems."""
        logger.info("Shutting down AutomationEngine…")
        if self._browser:
            self._browser.quit()
        logger.info("AutomationEngine shut down.")

    # ── Lazy browser access ───────────────────────────────────────────────────

    @property
    def browser(self) -> BrowserManager:
        if self._browser is None:
            logger.info("Initialising browser (%s)…", self._browser_cfg["browser"])
            self._browser = BrowserManager(**self._browser_cfg)
        return self._browser

    # ── Intent handlers ───────────────────────────────────────────────────────

    def _handle_open_app(self, cmd: ParsedCommand) -> AutomationResult:
        app = cmd.slots.get("app_name", "")
        if not app:
            return AutomationResult(False, "No application name provided.")
        ok, msg = self.executor.open_application(app)
        return AutomationResult(ok, msg)

    def _handle_close_app(self, cmd: ParsedCommand) -> AutomationResult:
        app = cmd.slots.get("app_name", "")
        if not app:
            return AutomationResult(False, "No application name provided.")
        ok, msg = self.executor.close_application(app)
        return AutomationResult(ok, msg)

    def _handle_google_search(self, cmd: ParsedCommand) -> AutomationResult:
        query = cmd.slots.get("query", "")
        if not query:
            return AutomationResult(False, "No search query provided.")
        ok, msg = self.browser.google_search(query)
        return AutomationResult(ok, msg, data={"query": query})

    def _handle_youtube_search(self, cmd: ParsedCommand) -> AutomationResult:
        query = cmd.slots.get("query", "")
        if not query:
            return AutomationResult(False, "No YouTube query provided.")
        ok, msg = self.browser.youtube_search(query)
        return AutomationResult(ok, msg, data={"query": query})

    def _handle_open_url(self, cmd: ParsedCommand) -> AutomationResult:
        url = cmd.slots.get("url", "")
        if not url:
            return AutomationResult(False, "No URL provided.")
        ok, msg = self.browser.open_url(url)
        return AutomationResult(ok, msg, data={"url": url})

    def _handle_type_text(self, cmd: ParsedCommand) -> AutomationResult:
        text = cmd.slots.get("text", "")
        if not text:
            return AutomationResult(False, "No text to type.")
        ok, msg = self.desktop.type_text(text)
        return AutomationResult(ok, msg)

    def _handle_press_key(self, cmd: ParsedCommand) -> AutomationResult:
        key = cmd.slots.get("key", "")
        if not key:
            return AutomationResult(False, "No key specified.")
        ok, msg = self.desktop.press_key(key)
        return AutomationResult(ok, msg)

    def _handle_hotkey(self, cmd: ParsedCommand) -> AutomationResult:
        keys = cmd.slots.get("keys", [])
        if not keys:
            return AutomationResult(False, "No hotkey combination specified.")
        ok, msg = self.desktop.hotkey(*keys)
        return AutomationResult(ok, msg)

    def _handle_screenshot(self, cmd: ParsedCommand) -> AutomationResult:
        filename = cmd.slots.get("filename")
        ok, msg, path = self.desktop.take_screenshot(filename)
        return AutomationResult(ok, msg, data={"path": path})

    def _handle_set_volume(self, cmd: ParsedCommand) -> AutomationResult:
        level = cmd.slots.get("level")
        if level is None:
            return AutomationResult(False, "No volume level specified.")
        ok, msg = self.desktop.set_volume(int(level))
        return AutomationResult(ok, msg, data={"level": level})

    def _handle_get_volume(self, _cmd: ParsedCommand) -> AutomationResult:
        vol, msg = self.desktop.get_volume()
        return AutomationResult(True, msg, data={"volume": vol})

    def _handle_system_stats(self, _cmd: ParsedCommand) -> AutomationResult:
        stats = self.monitor.get_full_report()
        summary = (
            f"CPU {stats['cpu']['percent']}% | "
            f"RAM {stats['memory']['percent']}% | "
            f"Disk {stats['disk']['percent']}%"
        )
        return AutomationResult(True, summary, data=stats)

    def _handle_kill_process(self, cmd: ParsedCommand) -> AutomationResult:
        name = cmd.slots.get("process_name", "")
        if not name:
            return AutomationResult(False, "No process name specified.")
        ok, msg = self.executor.kill_process(name)
        return AutomationResult(ok, msg)

    def _handle_scroll(self, cmd: ParsedCommand) -> AutomationResult:
        direction = cmd.slots.get("direction", "down")
        amount    = int(cmd.slots.get("amount", 3))
        ok, msg   = self.desktop.scroll(direction, amount)
        return AutomationResult(ok, msg)

    def _handle_click(self, cmd: ParsedCommand) -> AutomationResult:
        x = cmd.slots.get("x")
        y = cmd.slots.get("y")
        button = cmd.slots.get("button", "left")
        ok, msg = self.desktop.click(x, y, button)
        return AutomationResult(ok, msg)

    def _handle_move_mouse(self, cmd: ParsedCommand) -> AutomationResult:
        x = cmd.slots.get("x", 0)
        y = cmd.slots.get("y", 0)
        ok, msg = self.desktop.move_mouse(int(x), int(y))
        return AutomationResult(ok, msg)

    def _handle_close_browser(self, _cmd: ParsedCommand) -> AutomationResult:
        if self._browser:
            self._browser.quit()
            self._browser = None
            return AutomationResult(True, "Browser closed.")
        return AutomationResult(True, "No browser was open.")

    def _handle_unknown(self, cmd: ParsedCommand) -> AutomationResult:
        return AutomationResult(
            False,
            f"I don't know how to handle: '{cmd.raw}'",
            error="UNKNOWN_INTENT",
        )

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _emit(self, event: str, data: dict):
        if self._on_event:
            try:
                self._on_event(event, data)
            except Exception as exc:
                logger.debug("Event callback error: %s", exc)

    def get_stats(self) -> dict:
        """Return engine execution statistics."""
        return {
            "commands_executed": self._exec_count,
            "browser_active":    self._browser is not None,
            "rate_limit_sec":    self._rate_limit,
        }


# ─────────────────────────────────────────────
# Convenience singleton
# ─────────────────────────────────────────────
_engine: Optional[AutomationEngine] = None

def get_engine(**kwargs) -> AutomationEngine:
    global _engine
    if _engine is None:
        _engine = AutomationEngine(**kwargs)
    return _engine
