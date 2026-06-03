"""
tts_engine.py — JARVIS AI Voice Output Module
Edge-TTS powered, async, interruptible, queue-based speech engine.
"""

import asyncio
import logging
import os
import tempfile
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Callable, Optional

import edge_tts
import pygame

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logger = logging.getLogger("JARVIS.TTS")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    _ch = logging.StreamHandler()
    _ch.setLevel(logging.DEBUG)
    _fmt = logging.Formatter(
        "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    _ch.setFormatter(_fmt)
    logger.addHandler(_ch)


# ─────────────────────────────────────────────
# Voice Profiles
# ─────────────────────────────────────────────
class VoiceProfile(Enum):
    """Pre-configured voice personalities for JARVIS."""
    JARVIS    = "en-GB-RyanNeural"        # Default — calm British male
    FRIDAY    = "en-US-AriaNeural"        # Feminine US assistant
    EDWIN     = "en-US-GuyNeural"         # Authoritative US male
    KAREN     = "en-AU-NatashaNeural"     # Warm Australian female
    ATLAS     = "en-GB-SoniaNeural"       # British female
    CUSTOM    = None                      # Supply via TTSConfig.voice


VOICE_RATE: dict[VoiceProfile, str] = {
    VoiceProfile.JARVIS: "-5%",
    VoiceProfile.FRIDAY: "+0%",
    VoiceProfile.EDWIN:  "-8%",
    VoiceProfile.KAREN:  "+0%",
    VoiceProfile.ATLAS:  "-3%",
    VoiceProfile.CUSTOM: "+0%",
}

VOICE_PITCH: dict[VoiceProfile, str] = {
    VoiceProfile.JARVIS: "-5Hz",
    VoiceProfile.FRIDAY: "+0Hz",
    VoiceProfile.EDWIN:  "-10Hz",
    VoiceProfile.KAREN:  "+2Hz",
    VoiceProfile.ATLAS:  "+0Hz",
    VoiceProfile.CUSTOM: "+0Hz",
}


# ─────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────
class Priority(Enum):
    LOW    = 0
    NORMAL = 1
    HIGH   = 2
    URGENT = 3   # Interrupts current playback


@dataclass(order=True)
class SpeechItem:
    """A single unit of speech work."""
    priority:   int                         = field(compare=True)
    text:       str                         = field(compare=False)
    voice:      VoiceProfile                = field(compare=False, default=VoiceProfile.JARVIS)
    rate:       Optional[str]               = field(compare=False, default=None)
    pitch:      Optional[str]               = field(compare=False, default=None)
    on_start:   Optional[Callable]          = field(compare=False, default=None)
    on_done:    Optional[Callable]          = field(compare=False, default=None)
    on_error:   Optional[Callable]          = field(compare=False, default=None)
    item_id:    Optional[str]               = field(compare=False, default=None)

    # Invert for max-heap behaviour (highest priority first)
    def __lt__(self, other):
        return self.priority > other.priority


class EngineState(Enum):
    IDLE        = auto()
    SYNTHESISING= auto()
    PLAYING     = auto()
    INTERRUPTED = auto()
    STOPPING    = auto()


# ─────────────────────────────────────────────
# TTS Engine
# ─────────────────────────────────────────────
class TTSEngine:
    """
    Async, thread-safe, interruptible Edge-TTS speech engine.

    Usage
    -----
    engine = TTSEngine()
    engine.start()
    engine.speak("Hello, I am JARVIS.")
    engine.speak("Urgent message!", priority=Priority.URGENT)
    engine.stop()
    """

    def __init__(
        self,
        default_voice: VoiceProfile = VoiceProfile.JARVIS,
        custom_voice: Optional[str] = None,
        volume: float = 1.0,
        tmp_dir: Optional[str] = None,
    ):
        self.default_voice  = default_voice
        self.custom_voice   = custom_voice
        self.volume         = max(0.0, min(1.0, volume))
        self._tmp_dir       = Path(tmp_dir) if tmp_dir else Path(tempfile.gettempdir()) / "jarvis_tts"
        self._tmp_dir.mkdir(parents=True, exist_ok=True)

        # Queue (thread-safe priority queue via heapq + Lock)
        import heapq
        self._heap: list[SpeechItem]    = []
        self._heap_lock                 = threading.Lock()
        self._heap_event                = threading.Event()
        self._heapq                     = heapq

        # State
        self._state         = EngineState.IDLE
        self._state_lock    = threading.Lock()
        self._interrupt_flag= threading.Event()
        self._running       = False

        # Background worker
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread]         = None

        # Audio backend
        self._audio_ready   = False

        logger.info("TTSEngine created | voice=%s | vol=%.2f", default_voice.name, volume)

    # ── Public API ────────────────────────────

    def start(self):
        """Start the background speech worker thread."""
        if self._running:
            logger.warning("Engine already running.")
            return
        self._running = True
        self._thread = threading.Thread(target=self._worker_thread, name="TTS-Worker", daemon=True)
        self._thread.start()
        logger.info("TTS worker thread started.")

    def stop(self):
        """Gracefully stop the engine and flush the queue."""
        logger.info("Stopping TTS engine…")
        self._running = False
        self.interrupt()
        self._heap_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        self._cleanup_audio()
        logger.info("TTS engine stopped.")

    def speak(
        self,
        text: str,
        priority: Priority = Priority.NORMAL,
        voice: Optional[VoiceProfile] = None,
        rate: Optional[str] = None,
        pitch: Optional[str] = None,
        on_start: Optional[Callable] = None,
        on_done: Optional[Callable] = None,
        on_error: Optional[Callable] = None,
        item_id: Optional[str] = None,
    ):
        """
        Enqueue text for speech.
        URGENT priority automatically interrupts current playback.
        """
        if not text or not text.strip():
            return

        item = SpeechItem(
            priority=priority.value,
            text=text.strip(),
            voice=voice or self.default_voice,
            rate=rate,
            pitch=pitch,
            on_start=on_start,
            on_done=on_done,
            on_error=on_error,
            item_id=item_id or f"s_{int(time.time()*1000)}",
        )

        if priority == Priority.URGENT:
            self.interrupt()

        with self._heap_lock:
            self._heapq.heappush(self._heap, item)
        self._heap_event.set()
        logger.debug("Enqueued [%s] pri=%s text=%.40s…", item.item_id, priority.name, text)

    def interrupt(self):
        """Interrupt current playback immediately."""
        logger.debug("Interrupt requested.")
        self._interrupt_flag.set()
        self._stop_audio()

    def clear_queue(self):
        """Discard all pending speech items."""
        with self._heap_lock:
            dropped = len(self._heap)
            self._heap.clear()
        logger.info("Queue cleared (%d items dropped).", dropped)

    def set_volume(self, volume: float):
        self.volume = max(0.0, min(1.0, volume))
        if self._audio_ready:
            pygame.mixer.music.set_volume(self.volume)

    @property
    def state(self) -> EngineState:
        with self._state_lock:
            return self._state

    @property
    def queue_size(self) -> int:
        with self._heap_lock:
            return len(self._heap)

    # ── Internal Worker ───────────────────────

    def _worker_thread(self):
        """Dedicated thread running its own asyncio event loop."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._init_audio()
        try:
            self._loop.run_until_complete(self._dispatch_loop())
        finally:
            self._loop.close()
            logger.debug("Worker event loop closed.")

    async def _dispatch_loop(self):
        """Main async loop: pull items from queue and process them."""
        logger.debug("Dispatch loop running.")
        while self._running:
            item = self._dequeue()
            if item is None:
                # Wait for new items or stop signal
                await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self._heap_event.wait(timeout=0.2)
                )
                self._heap_event.clear()
                continue

            self._interrupt_flag.clear()
            await self._process(item)

        logger.debug("Dispatch loop exited.")

    async def _process(self, item: SpeechItem):
        """Synthesise and play a single SpeechItem."""
        self._set_state(EngineState.SYNTHESISING)
        tmp_file = self._tmp_dir / f"{item.item_id}.mp3"

        try:
            logger.debug("Synthesising [%s]…", item.item_id)
            if item.on_start:
                item.on_start(item)

            await self._synthesise(item, tmp_file)

            if self._interrupt_flag.is_set():
                logger.debug("Interrupted during synthesis [%s].", item.item_id)
                return

            self._set_state(EngineState.PLAYING)
            await self._play(tmp_file)

            if item.on_done:
                item.on_done(item)
            logger.debug("Finished [%s].", item.item_id)

        except asyncio.CancelledError:
            logger.debug("Task cancelled [%s].", item.item_id)
        except Exception as exc:
            logger.error("Error processing [%s]: %s", item.item_id, exc, exc_info=True)
            if item.on_error:
                item.on_error(item, exc)
        finally:
            self._set_state(EngineState.IDLE)
            try:
                tmp_file.unlink(missing_ok=True)
            except Exception:
                pass

    async def _synthesise(self, item: SpeechItem, out_path: Path):
        """Call Edge-TTS to produce an MP3 file."""
        voice_name = (
            self.custom_voice
            if item.voice == VoiceProfile.CUSTOM
            else item.voice.value
        )
        rate  = item.rate  or VOICE_RATE.get(item.voice, "+0%")
        pitch = item.pitch or VOICE_PITCH.get(item.voice, "+0Hz")

        communicate = edge_tts.Communicate(item.text, voice=voice_name, rate=rate, pitch=pitch)
        await communicate.save(str(out_path))
        logger.debug("Synthesis done → %s", out_path.name)

    async def _play(self, mp3_path: Path):
        """Play MP3 via pygame, polling for interrupts."""
        if not self._audio_ready:
            logger.warning("Audio not initialised, skipping playback.")
            return

        await asyncio.get_event_loop().run_in_executor(None, self._play_sync, mp3_path)

    def _play_sync(self, mp3_path: Path):
        """Blocking pygame playback (run in executor)."""
        try:
            pygame.mixer.music.load(str(mp3_path))
            pygame.mixer.music.set_volume(self.volume)
            pygame.mixer.music.play()

            while pygame.mixer.music.get_busy():
                if self._interrupt_flag.is_set():
                    pygame.mixer.music.stop()
                    logger.debug("Playback interrupted.")
                    return
                time.sleep(0.05)
        except Exception as exc:
            logger.error("Playback error: %s", exc)

    def _stop_audio(self):
        if self._audio_ready:
            try:
                pygame.mixer.music.stop()
            except Exception:
                pass

    # ── Queue Helpers ─────────────────────────

    def _dequeue(self) -> Optional[SpeechItem]:
        with self._heap_lock:
            if self._heap:
                return self._heapq.heappop(self._heap)
        return None

    # ── Audio Init ────────────────────────────

    def _init_audio(self):
        try:
            pygame.mixer.pre_init(frequency=24000, size=-16, channels=1, buffer=512)
            pygame.mixer.init()
            self._audio_ready = True
            logger.info("pygame.mixer initialised.")
        except Exception as exc:
            logger.error("Failed to init audio: %s", exc)
            self._audio_ready = False

    def _cleanup_audio(self):
        if self._audio_ready:
            try:
                pygame.mixer.quit()
            except Exception:
                pass

    # ── State ─────────────────────────────────

    def _set_state(self, state: EngineState):
        with self._state_lock:
            self._state = state
        logger.debug("State → %s", state.name)


# ─────────────────────────────────────────────
# Convenience singleton accessor
# ─────────────────────────────────────────────
_engine: Optional[TTSEngine] = None

def get_engine(**kwargs) -> TTSEngine:
    """Return (or create) the global TTSEngine singleton."""
    global _engine
    if _engine is None:
        _engine = TTSEngine(**kwargs)
        _engine.start()
    return _engine
