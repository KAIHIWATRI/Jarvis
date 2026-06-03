"""
shutdown_coordinator.py — JARVIS Safe Shutdown Coordinator
Orchestrates a multi-phase, ordered teardown of all JARVIS sub-systems.
Handles timeouts, partial failures, and emergency force-kill as last resort.

Shutdown phases
───────────────
1. NOTIFY    — broadcast stop signal to all components
2. DRAIN     — wait for in-flight tasks to finish (graceful)
3. STOP      — call stop() on each registered component
4. CLEANUP   — flush logs, save state, release resources
5. FORCE     — terminate any remaining threads/tasks after timeout
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

logger = logging.getLogger("JARVIS.Shutdown")


# ─────────────────────────────────────────────
# Shutdown component descriptor
# ─────────────────────────────────────────────
@dataclass
class ShutdownComponent:
    name:          str
    stop_fn:       Callable
    timeout_sec:   float = 5.0
    priority:      int   = 50    # lower = shut down first
    critical:      bool  = False # if True, failure aborts graceful shutdown
    _done:         bool  = field(default=False, init=False, repr=False)
    _error:        Optional[str] = field(default=None, init=False, repr=False)


# ─────────────────────────────────────────────
# Shutdown Coordinator
# ─────────────────────────────────────────────
class ShutdownCoordinator:
    """
    Manages ordered, timed teardown of all JARVIS sub-systems.

    Usage
    -----
    coord = ShutdownCoordinator(service=svc)
    coord.register("tts",  tts_engine.stop,   priority=10, timeout_sec=3)
    coord.register("bg",   bg_manager.stop,   priority=20, timeout_sec=5)
    coord.register("tray", tray.stop,         priority=90, timeout_sec=2)
    coord.shutdown()
    """

    FORCE_KILL_TIMEOUT = 8.0    # seconds before os._exit
    DRAIN_TIMEOUT      = 3.0    # seconds to wait for in-flight work

    def __init__(self, service=None):
        self._service     = service
        self._components: List[ShutdownComponent] = []
        self._lock        = threading.Lock()
        self._started     = False
        self._done        = False

        # Pre-register service-level components if available
        if service:
            self._auto_register(service)

    # ── Registration ──────────────────────────

    def register(
        self,
        name:        str,
        stop_fn:     Callable,
        priority:    int   = 50,
        timeout_sec: float = 5.0,
        critical:    bool  = False,
    ):
        """Register a component for ordered shutdown."""
        with self._lock:
            # Prevent duplicate names
            self._components = [c for c in self._components if c.name != name]
            self._components.append(ShutdownComponent(
                name=name,
                stop_fn=stop_fn,
                timeout_sec=timeout_sec,
                priority=priority,
                critical=critical,
            ))
            self._components.sort(key=lambda c: c.priority)
        logger.debug("Registered shutdown component: %s (priority=%d)", name, priority)

    def _auto_register(self, service):
        """Auto-discover and register known sub-systems from the service object."""
        # Background service
        if hasattr(service, '_bg') and service._bg:
            import asyncio
            def _stop_bg():
                if service._loop and not service._loop.is_closed():
                    future = asyncio.run_coroutine_threadsafe(
                        service._bg.stop(), service._loop
                    )
                    try:
                        future.result(timeout=4)
                    except Exception:
                        pass
            self.register("background_service", _stop_bg, priority=10, timeout_sec=5)

        # UI window
        if hasattr(service, '_ui_window') and service._ui_window:
            def _stop_ui():
                try:
                    service._ui_window.after(0, service._ui_window.destroy)
                    time.sleep(0.5)
                except Exception:
                    pass
            self.register("ui_window", _stop_ui, priority=20, timeout_sec=3)

        # Tray icon
        if hasattr(service, '_tray') and service._tray:
            self.register("tray_icon", service._tray.stop, priority=90, timeout_sec=2)

    # ── Shutdown ──────────────────────────────

    def shutdown(self, reason: str = "requested") -> dict:
        """
        Execute the full shutdown sequence.
        Returns a report dict with per-component results.
        Thread-safe — only the first call executes; subsequent calls are no-ops.
        """
        with self._lock:
            if self._started:
                logger.debug("Shutdown already in progress.")
                return {}
            self._started = True

        t0 = time.perf_counter()
        logger.info("─── Shutdown initiated: %s ───", reason)

        # Arm force-kill watchdog
        watchdog = threading.Timer(
            self.FORCE_KILL_TIMEOUT,
            self._force_kill,
            args=(reason,),
        )
        watchdog.daemon = True
        watchdog.start()

        report = {}
        try:
            # Phase 1: Drain (brief wait for in-flight work)
            logger.info("[Phase 1] Draining in-flight tasks…")
            time.sleep(min(self.DRAIN_TIMEOUT, 1.0))

            # Phase 2: Stop components in priority order
            logger.info("[Phase 2] Stopping %d components…", len(self._components))
            for comp in self._components:
                result = self._stop_component(comp)
                report[comp.name] = result
                if not result["ok"] and comp.critical:
                    logger.error("Critical component %s failed — forcing.", comp.name)
                    break

            # Phase 3: Cleanup
            logger.info("[Phase 3] Cleanup…")
            self._flush_logs()

            elapsed = time.perf_counter() - t0
            logger.info("─── Shutdown complete in %.2fs ───", elapsed)
            report["_total_sec"] = round(elapsed, 3)
            report["_reason"]    = reason

        except Exception as exc:
            logger.exception("Shutdown error: %s", exc)
            report["_error"] = str(exc)
        finally:
            watchdog.cancel()
            self._done = True

        return report

    def _stop_component(self, comp: ShutdownComponent) -> dict:
        """Stop a single component with timeout enforcement."""
        logger.debug("Stopping: %s (timeout=%.1fs)…", comp.name, comp.timeout_sec)
        t0  = time.perf_counter()
        err = None
        ok  = False

        result_container = [None]

        def _run():
            try:
                comp.stop_fn()
                result_container[0] = True
            except Exception as exc:
                result_container[0] = exc

        t = threading.Thread(target=_run, name=f"stop-{comp.name}", daemon=True)
        t.start()
        t.join(timeout=comp.timeout_sec)

        elapsed = time.perf_counter() - t0

        if t.is_alive():
            err = f"Timed out after {comp.timeout_sec}s"
            logger.warning("Component %s timed out.", comp.name)
        elif isinstance(result_container[0], Exception):
            err = str(result_container[0])
            logger.warning("Component %s error: %s", comp.name, err)
        else:
            ok  = True
            logger.debug("Component %s stopped in %.2fs.", comp.name, elapsed)

        return {"ok": ok, "elapsed": round(elapsed, 3), "error": err}

    def _flush_logs(self):
        """Flush all log handlers."""
        root = logging.getLogger()
        for handler in root.handlers:
            try:
                handler.flush()
            except Exception:
                pass
        # Also flush JARVIS-specific loggers
        for name in logging.Logger.manager.loggerDict:
            if "JARVIS" in name:
                for h in logging.getLogger(name).handlers:
                    try:
                        h.flush()
                    except Exception:
                        pass

    def _force_kill(self, reason: str):
        """Emergency exit if graceful shutdown exceeds the watchdog timeout."""
        logger.critical(
            "Force-kill triggered after %.0fs — reason: %s",
            self.FORCE_KILL_TIMEOUT, reason,
        )
        self._flush_logs()
        os._exit(1)   # Hard exit, bypasses atexit handlers intentionally

    # ── Properties ────────────────────────────

    @property
    def is_done(self) -> bool:
        return self._done

    @property
    def component_names(self) -> list[str]:
        with self._lock:
            return [c.name for c in self._components]
