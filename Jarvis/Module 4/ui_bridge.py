"""
ui_bridge.py — JARVIS TTS ↔ CustomTkinter UI Bridge
Thread-safe callbacks and status hooks for CustomTkinter.
"""

import logging
import threading
from typing import Callable, Optional

from tts_engine import EngineState, Priority, TTSEngine, VoiceProfile, get_engine
from speech_manager import SpeechManager

logger = logging.getLogger("JARVIS.UIBridge")


class TTSUIBridge:
    """
    Bridges the TTS engine to a CustomTkinter UI.

    Provides:
    - on_speak_start / on_speak_end callbacks (scheduled on UI thread)
    - Volume / voice controls callable from UI widgets
    - Status polling property
    - Safe tk.after-based callback dispatch

    Example (CustomTkinter)
    -----------------------
    bridge = TTSUIBridge(root=app, engine=engine)
    bridge.on_speak_start = lambda item: status_label.configure(text="Speaking…")
    bridge.on_speak_end   = lambda item: status_label.configure(text="Idle")
    bridge.say("Hello from JARVIS!")
    """

    def __init__(
        self,
        root=None,           # tk/CTk root widget (for .after dispatch)
        engine: Optional[TTSEngine] = None,
        default_voice: VoiceProfile = VoiceProfile.JARVIS,
    ):
        self._root   = root
        self._engine = engine or get_engine(default_voice=default_voice)
        self._manager= SpeechManager(engine=self._engine, default_voice=default_voice)

        # UI event callbacks (set by the application)
        self.on_speak_start: Optional[Callable] = None
        self.on_speak_end:   Optional[Callable] = None
        self.on_speak_error: Optional[Callable] = None
        self.on_queue_empty: Optional[Callable] = None

        logger.info("TTSUIBridge initialised.")

    # ── Speech API ────────────────────────────

    def say(
        self,
        text: str,
        priority: Priority = Priority.NORMAL,
        voice: Optional[VoiceProfile] = None,
    ):
        """Enqueue speech, wiring UI callbacks automatically."""
        self._engine.speak(
            text,
            priority=priority,
            voice=voice or self._manager.default_voice,
            on_start=self._cb_start,
            on_done=self._cb_done,
            on_error=self._cb_error,
        )

    def say_urgent(self, text: str):
        self.say(text, priority=Priority.URGENT)

    def interrupt(self):
        self._engine.interrupt()
        self._dispatch(self.on_speak_end, None)

    def clear(self):
        self._engine.clear_queue()
        self._engine.interrupt()

    # ── Controls ──────────────────────────────

    def set_volume(self, value: float):
        """Bind directly to a CTkSlider command."""
        self._engine.set_volume(value)

    def set_voice(self, voice: VoiceProfile):
        self._manager.set_voice(voice)

    # ── Status ────────────────────────────────

    @property
    def is_speaking(self) -> bool:
        return self._engine.state in (EngineState.SYNTHESISING, EngineState.PLAYING)

    @property
    def status_text(self) -> str:
        state = self._engine.state
        labels = {
            EngineState.IDLE:         "Idle",
            EngineState.SYNTHESISING: "Synthesising…",
            EngineState.PLAYING:      "Speaking…",
            EngineState.INTERRUPTED:  "Interrupted",
            EngineState.STOPPING:     "Stopping…",
        }
        return labels.get(state, "Unknown")

    @property
    def queue_depth(self) -> int:
        return self._engine.queue_size

    # ── Internal Callbacks ────────────────────

    def _cb_start(self, item):
        self._dispatch(self.on_speak_start, item)

    def _cb_done(self, item):
        self._dispatch(self.on_speak_end, item)
        if self._engine.queue_size == 0:
            self._dispatch(self.on_queue_empty, None)

    def _cb_error(self, item, exc):
        logger.error("TTS error on [%s]: %s", item.item_id, exc)
        self._dispatch(self.on_speak_error, (item, exc))

    def _dispatch(self, cb: Optional[Callable], arg):
        """
        Schedule a callback on the Tk main thread (thread-safe).
        Falls back to direct call if no root widget is set.
        """
        if cb is None:
            return
        if self._root is not None:
            self._root.after(0, lambda: cb(arg))
        else:
            threading.Thread(target=lambda: cb(arg), daemon=True).start()


# ─────────────────────────────────────────────
# Example CustomTkinter snippet (not executed)
# ─────────────────────────────────────────────
_EXAMPLE = '''
import customtkinter as ctk
from tts_engine import VoiceProfile
from ui_bridge import TTSUIBridge

app = ctk.CTk()
app.title("JARVIS")

bridge = TTSUIBridge(root=app)

status_var = ctk.StringVar(value="Idle")
ctk.CTkLabel(app, textvariable=status_var).pack(pady=10)

bridge.on_speak_start = lambda _: status_var.set("🔊 Speaking…")
bridge.on_speak_end   = lambda _: status_var.set("✅ Idle")
bridge.on_queue_empty = lambda _: status_var.set("💤 Waiting")

# Volume slider
ctk.CTkSlider(app, from_=0, to=1, command=bridge.set_volume).pack()

# Voice selector
voice_menu = ctk.CTkOptionMenu(
    app,
    values=[v.name for v in VoiceProfile if v != VoiceProfile.CUSTOM],
    command=lambda name: bridge.set_voice(VoiceProfile[name]),
)
voice_menu.pack()

ctk.CTkButton(app, text="Speak", command=lambda: bridge.say("Hello, I am JARVIS.")).pack()
ctk.CTkButton(app, text="Interrupt", command=bridge.interrupt).pack(pady=5)

app.mainloop()
'''
