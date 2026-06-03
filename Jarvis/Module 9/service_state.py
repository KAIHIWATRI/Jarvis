"""
service_state.py — JARVIS Service State Machine
Thread-safe state machine with transition validation,
history tracking, and observer callbacks.
"""

from __future__ import annotations

import logging
import threading
import time
from enum import Enum, auto
from typing import Callable, List, Optional, Tuple

logger = logging.getLogger("JARVIS.State")


# ─────────────────────────────────────────────
# States
# ─────────────────────────────────────────────
class State(Enum):
    IDLE       = auto()   # Pre-start
    STARTING   = auto()   # Initialising sub-systems
    RUNNING    = auto()   # Fully operational
    LISTENING  = auto()   # Microphone active
    THINKING   = auto()   # LLM processing
    SPEAKING   = auto()   # TTS playback
    MUTED      = auto()   # Audio suppressed
    STOPPING   = auto()   # Teardown in progress
    STOPPED    = auto()   # Fully shut down
    ERROR      = auto()   # Unrecoverable fault


# ─────────────────────────────────────────────
# Valid transitions
# ─────────────────────────────────────────────
VALID_TRANSITIONS: dict[State, set[State]] = {
    State.IDLE:      {State.STARTING},
    State.STARTING:  {State.RUNNING, State.ERROR, State.STOPPED},
    State.RUNNING:   {State.LISTENING, State.THINKING, State.SPEAKING,
                      State.MUTED, State.STOPPING, State.ERROR},
    State.LISTENING: {State.THINKING, State.RUNNING, State.MUTED,
                      State.STOPPING, State.ERROR},
    State.THINKING:  {State.SPEAKING, State.RUNNING, State.STOPPING, State.ERROR},
    State.SPEAKING:  {State.RUNNING, State.LISTENING, State.STOPPING, State.ERROR},
    State.MUTED:     {State.RUNNING, State.STOPPING},
    State.STOPPING:  {State.STOPPED, State.ERROR},
    State.STOPPED:   set(),   # Terminal
    State.ERROR:     {State.STOPPING, State.STOPPED},
}

# Human-readable labels for UI / tray tooltip
STATE_LABELS: dict[State, str] = {
    State.IDLE:      "Idle",
    State.STARTING:  "Starting…",
    State.RUNNING:   "Online",
    State.LISTENING: "Listening",
    State.THINKING:  "Processing",
    State.SPEAKING:  "Speaking",
    State.MUTED:     "Muted",
    State.STOPPING:  "Shutting down…",
    State.STOPPED:   "Offline",
    State.ERROR:     "Error",
}

# Map state → tray icon animation name (links to TrayIconManager)
STATE_ICON: dict[State, str] = {
    State.IDLE:      "idle",
    State.STARTING:  "thinking",
    State.RUNNING:   "idle",
    State.LISTENING: "listening",
    State.THINKING:  "thinking",
    State.SPEAKING:  "speaking",
    State.MUTED:     "muted",
    State.STOPPING:  "idle",
    State.STOPPED:   "idle",
    State.ERROR:     "error",
}


# ─────────────────────────────────────────────
# Service State Machine
# ─────────────────────────────────────────────
class ServiceState:
    """
    Thread-safe finite state machine for the JARVIS service lifecycle.

    Features
    --------
    - Validated transitions (raises on illegal moves)
    - Full transition history (timestamp, from, to)
    - Observer callbacks notified on every transition
    - enter_state context manager for automatic revert on error
    """

    MAX_HISTORY = 200

    def __init__(self, initial: State = State.IDLE):
        self._state   = initial
        self._lock    = threading.RLock()
        self._history: List[Tuple[float, State, State]] = []
        self._observers: List[Callable[[State, State], None]] = []
        logger.debug("ServiceState initialised: %s", initial.name)

    # ── Core transition ───────────────────────

    def transition(self, new_state: State, force: bool = False) -> bool:
        """
        Attempt a state transition.
        Returns True on success, False if the transition is invalid.
        Raises ValueError only if force=False and transition is illegal.
        """
        with self._lock:
            current = self._state
            allowed = VALID_TRANSITIONS.get(current, set())

            if new_state not in allowed:
                if force:
                    logger.warning(
                        "Forced transition %s → %s (normally invalid).",
                        current.name, new_state.name,
                    )
                else:
                    logger.warning(
                        "Invalid transition %s → %s ignored.",
                        current.name, new_state.name,
                    )
                    return False

            self._state = new_state
            entry = (time.time(), current, new_state)
            self._history.append(entry)
            if len(self._history) > self.MAX_HISTORY:
                self._history = self._history[-self.MAX_HISTORY:]

            logger.info("State: %s → %s", current.name, new_state.name)

        # Notify observers outside the lock
        for cb in self._observers:
            try:
                cb(current, new_state)
            except Exception as exc:
                logger.debug("Observer error: %s", exc)

        return True

    # ── Properties ────────────────────────────

    @property
    def current(self) -> State:
        with self._lock:
            return self._state

    @property
    def label(self) -> str:
        return STATE_LABELS.get(self.current, "Unknown")

    @property
    def icon_state(self) -> str:
        return STATE_ICON.get(self.current, "idle")

    @property
    def is_active(self) -> bool:
        """True when the service is in an operational (non-terminal) state."""
        with self._lock:
            return self._state not in (State.STOPPED, State.ERROR, State.IDLE)

    @property
    def is_busy(self) -> bool:
        with self._lock:
            return self._state in (State.LISTENING, State.THINKING, State.SPEAKING)

    # ── Observers ─────────────────────────────

    def add_observer(self, cb: Callable[[State, State], None]):
        """Register a callback(from_state, to_state) called on every transition."""
        self._observers.append(cb)

    def remove_observer(self, cb: Callable[[State, State], None]):
        self._observers = [o for o in self._observers if o is not cb]

    # ── History ───────────────────────────────

    def get_history(self) -> List[dict]:
        with self._lock:
            return [
                {
                    "timestamp": ts,
                    "from":      frm.name,
                    "to":        to.name,
                    "elapsed":   round(ts - self._history[max(0, i-1)][0], 3) if i > 0 else 0,
                }
                for i, (ts, frm, to) in enumerate(self._history)
            ]

    def last_transition_ago(self) -> float:
        """Seconds since the last state transition."""
        with self._lock:
            if not self._history:
                return 0.0
            return time.time() - self._history[-1][0]

    def time_in_current_state(self) -> float:
        return self.last_transition_ago()

    # ── Context manager ───────────────────────

    def enter_state(self, target: State, revert_to: Optional[State] = None):
        """
        Context manager: transitions to target on enter,
        reverts to revert_to (or previous state) on exception.

        Usage
        -----
        with state.enter_state(State.THINKING):
            await llm_call()
        # auto-reverts to RUNNING on error
        """
        return _StateContext(self, target, revert_to)

    def __repr__(self) -> str:
        return f"<ServiceState current={self._state.name}>"


# ─────────────────────────────────────────────
# Context manager helper
# ─────────────────────────────────────────────
class _StateContext:
    def __init__(self, sm: ServiceState, target: State, revert_to: Optional[State]):
        self._sm       = sm
        self._target   = target
        self._revert   = revert_to
        self._previous = sm.current

    def __enter__(self):
        self._sm.transition(self._target)
        return self._sm

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            revert = self._revert or self._previous
            logger.debug("Context manager reverting to %s due to %s", revert.name, exc_type.__name__)
            self._sm.transition(revert, force=True)
        return False   # Don't suppress exceptions
