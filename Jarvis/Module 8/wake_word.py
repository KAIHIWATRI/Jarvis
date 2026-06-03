"""
JARVIS Wake-Word Detection Module
===================================
Continuous background listening for "Hey Jarvis" using openWakeWord.
Optimised for AMD Ryzen 7 3700U · 16 GB RAM (CPU-only inference).

Architecture
------------
  WakeWordConfig         – frozen settings dataclass
  WakeEvent              – value object emitted on every detection
  WakeWordCallback       – type alias for listener callbacks
  _AudioCaptureThread    – daemon thread: mic → raw audio queue
  _InferenceThread       – daemon thread: audio queue → OWW model → events
  WakeWordDetector       – public façade: owns threads, fires callbacks
  AsyncWakeWordBridge    – asyncio bridge: await next detection from any task

Threading model
---------------
  Main thread            → WakeWordDetector.start() / stop() / register()
  _AudioCaptureThread    → PyAudio blocking read → _audio_q (daemon)
  _InferenceThread       → _audio_q → OWW predict → callbacks + _event_q (daemon)
  asyncio event loop     → AsyncWakeWordBridge.wait_for_wake() polls _event_q

Quick start (sync)
------------------
    detector = WakeWordDetector()
    detector.register(lambda e: print("Activated!", e))
    detector.start()
    input("Press Enter to stop…")
    detector.stop()

Quick start (async)
-------------------
    async def main():
        detector = WakeWordDetector()
        bridge = AsyncWakeWordBridge(detector)
        detector.start()
        event = await bridge.wait_for_wake(timeout=30)
        if event:
            print("Activated!", event)
        detector.stop()

    asyncio.run(main())
"""

from __future__ import annotations

import asyncio
import logging
import logging.handlers
import os
import queue
import threading
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import openwakeword

# Suppress non-critical ONNX/CUDA provider warnings on CPU-only machines
warnings.filterwarnings(
    "ignore",
    message="Specified provider.*CUDAExecutionProvider",
    category=UserWarning,
)

from openwakeword.model import Model as _OWWModel

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def _build_logger(name: str) -> logging.Logger:
    """Rotating-file + console logger, module-scoped."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console (INFO and above only — keeps terminal clean)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # Rotating file (all levels)
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    fh = logging.handlers.RotatingFileHandler(
        log_dir / "wake_word.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


log = _build_logger("jarvis.wake_word")


# ─────────────────────────────────────────────────────────────────────────────
# WakeWordConfig
# ─────────────────────────────────────────────────────────────────────────────

def _default_model_path() -> Path:
    """Resolve the bundled hey_jarvis ONNX model path from openWakeWord."""
    oww_dir = Path(openwakeword.__file__).parent
    model = oww_dir / "resources" / "models" / "hey_jarvis_v0.1.onnx"
    if not model.exists():
        raise FileNotFoundError(
            f"Built-in hey_jarvis model not found at {model}. "
            "Reinstall openwakeword: pip install openwakeword"
        )
    return model


@dataclass(frozen=True)
class WakeWordConfig:
    """
    All tunables in one immutable place.

    Ryzen 7 3700U optimisation notes
    ----------------------------------
    chunk_ms        : 80 ms chunks → ~12.5 inferences/second, ~3 % CPU.
                      Lower = more responsive; higher = lower CPU.
                      Do NOT go below 32 ms (OWW needs enough context).
    detection_threshold: 0.5 is the default OWW recommendation.
                      Raise to 0.7 to reduce false positives (quieter rooms).
                      Lower to 0.4 for better sensitivity (noisier rooms).
    cooldown_s      : Seconds to ignore further detections after one fires.
                      Prevents double-firing when the word echoes.
    vad_threshold   : 0 = VAD disabled (OWW has its own built-in VAD).
                      Set 0.3–0.5 to skip inference on obvious silence frames,
                      saving CPU. Leave 0 for maximum accuracy.
    onnx_threads    : 1 is optimal for this model size on 3700U.
                      Increasing does NOT help — OWW is fast single-threaded.
    queue_maxsize   : Audio frame backpressure. Inference slower than capture
                      → oldest frames dropped rather than RAM growing.
    """
    # Audio capture
    sample_rate:        int   = 16_000     # Hz — OWW native rate
    channels:           int   = 1          # mono
    chunk_ms:           int   = 80         # milliseconds per inference frame
    device_index:       Optional[int] = None  # None → system default mic

    # Model
    model_path:         Optional[Path] = None  # None → auto-detect bundled model
    detection_threshold: float = 0.5
    cooldown_s:         float = 2.0        # seconds between successive activations
    vad_threshold:      float = 0.0        # 0 = disabled

    # ONNX runtime
    onnx_threads:       int   = 1          # inference threads (keep at 1 for 3700U)

    # Internal pipeline
    queue_maxsize:      int   = 30         # max buffered audio frames

    @property
    def chunk_samples(self) -> int:
        """Number of int16 samples per audio chunk."""
        return int(self.sample_rate * self.chunk_ms / 1000)

    @property
    def resolved_model_path(self) -> Path:
        return self.model_path or _default_model_path()


# ─────────────────────────────────────────────────────────────────────────────
# WakeEvent
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class WakeEvent:
    """
    Value object emitted every time the wake word is detected.

    Attributes
    ----------
    model_name  : internal OWW model key (e.g. 'hey_jarvis_v0.1')
    score       : detection confidence [0.0 – 1.0]
    timestamp   : monotonic clock at detection (seconds)
    wall_time   : human-readable UTC ISO-8601 timestamp
    """
    model_name: str
    score:      float
    timestamp:  float = field(default_factory=time.monotonic)
    wall_time:  str   = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    def __str__(self) -> str:
        return (
            f"WakeEvent(model={self.model_name!r}, "
            f"score={self.score:.3f}, "
            f"wall_time={self.wall_time})"
        )


# Type alias for registered callback functions
WakeWordCallback = Callable[[WakeEvent], None]


# ─────────────────────────────────────────────────────────────────────────────
# _AudioCaptureThread
# ─────────────────────────────────────────────────────────────────────────────

class _AudioCaptureThread(threading.Thread):
    """
    Daemon thread: reads raw int16 PCM audio from the microphone and
    puts numpy arrays into a shared queue for the inference thread.

    Uses PyAudio blocking-read mode (not callback mode) so the thread
    itself controls timing — simpler and more predictable on Linux.

    Error handling
    --------------
    Up to MAX_CONSECUTIVE_ERRORS mic read errors are tolerated (device
    glitches are common). Beyond that the thread sets the stop event so
    the WakeWordDetector can restart gracefully.
    """

    MAX_CONSECUTIVE_ERRORS = 20

    def __init__(
        self,
        config:    WakeWordConfig,
        audio_q:   queue.Queue,
        stop_evt:  threading.Event,
    ) -> None:
        super().__init__(name="jarvis-wake-capture", daemon=True)
        self._cfg      = config
        self._audio_q  = audio_q
        self._stop_evt = stop_evt
        self._pa       = None
        self._stream   = None

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def _open_mic(self) -> None:
        """Open the PyAudio input stream.  Raises on device error."""
        import pyaudio

        self._pa = pyaudio.PyAudio()
        device_idx = self._cfg.device_index

        if device_idx is None:
            device_idx = self._pa.get_default_input_device_info()["index"]

        info = self._pa.get_device_info_by_index(device_idx)
        log.info(
            "Mic opened: '%s' (idx=%d, %d Hz, %d-ms chunks / %d samples)",
            info["name"],
            device_idx,
            self._cfg.sample_rate,
            self._cfg.chunk_ms,
            self._cfg.chunk_samples,
        )

        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=self._cfg.channels,
            rate=self._cfg.sample_rate,
            input=True,
            input_device_index=device_idx,
            frames_per_buffer=self._cfg.chunk_samples,
        )

    def _close_mic(self) -> None:
        if self._stream:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception as exc:
                log.warning("Error closing audio stream: %s", exc)
            finally:
                self._stream = None

        if self._pa:
            try:
                self._pa.terminate()
            except Exception as exc:
                log.warning("Error terminating PyAudio: %s", exc)
            finally:
                self._pa = None

    # ── Main loop ─────────────────────────────────────────────────────────

    def run(self) -> None:
        log.debug("Audio capture thread started.")
        try:
            self._open_mic()
        except Exception as exc:
            log.error("Failed to open microphone: %s", exc)
            self._stop_evt.set()
            return

        errors = 0
        while not self._stop_evt.is_set():
            try:
                raw = self._stream.read(
                    self._cfg.chunk_samples,
                    exception_on_overflow=False,  # drop instead of crash
                )
                frame = np.frombuffer(raw, dtype=np.int16)
                errors = 0
            except OSError as exc:
                errors += 1
                log.warning("Mic read error #%d: %s", errors, exc)
                if errors >= self.MAX_CONSECUTIVE_ERRORS:
                    log.error(
                        "Too many consecutive mic errors — stopping capture."
                    )
                    self._stop_evt.set()
                    break
                time.sleep(0.01)
                continue

            # Enqueue; drop the oldest frame if the inference thread is behind
            try:
                self._audio_q.put_nowait(frame)
            except queue.Full:
                try:
                    self._audio_q.get_nowait()
                except queue.Empty:
                    pass
                self._audio_q.put_nowait(frame)
                log.debug("Audio queue full — oldest frame dropped.")

        self._close_mic()
        log.debug("Audio capture thread exited.")


# ─────────────────────────────────────────────────────────────────────────────
# _InferenceThread
# ─────────────────────────────────────────────────────────────────────────────

class _InferenceThread(threading.Thread):
    """
    Daemon thread: drains the audio queue, runs OWW inference, and
    dispatches WakeEvent objects to registered callbacks and the event queue.

    OWW model notes
    ---------------
    openWakeWord maintains internal state (mel-spectrogram feature buffer)
    across calls. Each call to model.predict(chunk) updates this buffer —
    the model implicitly sees a sliding window of recent audio.
    This means chunk order MUST be preserved, which our queue guarantees.

    Cooldown
    --------
    After a detection, a timestamp is recorded and subsequent detections
    within cooldown_s are silently dropped. This prevents double-firing
    when "Hey Jarvis" is spoken naturally (the word fades out slowly).
    """

    def __init__(
        self,
        config:      WakeWordConfig,
        audio_q:     queue.Queue,
        event_q:     queue.Queue,
        callbacks:   List[WakeWordCallback],
        callbacks_lock: threading.Lock,
        stop_evt:    threading.Event,
    ) -> None:
        super().__init__(name="jarvis-wake-inference", daemon=True)
        self._cfg            = config
        self._audio_q        = audio_q
        self._event_q        = event_q
        self._callbacks      = callbacks
        self._callbacks_lock = callbacks_lock
        self._stop_evt       = stop_evt
        self._model: Optional[_OWWModel] = None
        self._last_detection: float = 0.0     # monotonic timestamp

    # ── Model loading ─────────────────────────────────────────────────────

    def _load_model(self) -> None:
        """Load the OWW ONNX model.  Called once at thread start."""
        model_path = str(self._cfg.resolved_model_path)
        log.info("Loading OWW model: %s", model_path)
        t0 = time.monotonic()

        self._model = _OWWModel(
            wakeword_model_paths=[model_path],
            vad_threshold=self._cfg.vad_threshold,
            inference_framework="onnx",
        )

        elapsed = time.monotonic() - t0
        model_name = list(self._model.models.keys())[0]
        log.info(
            "OWW model loaded in %.2f s — key: '%s'",
            elapsed, model_name,
        )

    # ── Main loop ─────────────────────────────────────────────────────────

    def run(self) -> None:
        log.debug("Inference thread started.")
        try:
            self._load_model()
        except Exception as exc:
            log.error("Failed to load OWW model: %s", exc, exc_info=True)
            self._stop_evt.set()
            return

        frames_processed = 0
        while not self._stop_evt.is_set():
            try:
                frame = self._audio_q.get(timeout=0.5)
            except queue.Empty:
                continue

            # ── Run inference ─────────────────────────────────────────
            try:
                scores = self._model.predict(frame)
            except Exception as exc:
                log.error("OWW inference error: %s", exc, exc_info=True)
                continue

            frames_processed += 1
            if frames_processed % 500 == 0:
                log.debug("Inference: %d frames processed.", frames_processed)

            # ── Check scores against threshold ────────────────────────
            for model_key, score in scores.items():
                if score < self._cfg.detection_threshold:
                    continue  # below threshold — keep listening

                now = time.monotonic()
                if now - self._last_detection < self._cfg.cooldown_s:
                    log.debug(
                        "Wake word detected (score=%.3f) but in cooldown — skipped.",
                        score,
                    )
                    continue

                self._last_detection = now
                event = WakeEvent(model_name=model_key, score=score)
                log.info("🎙  Wake word detected! %s", event)
                self._dispatch(event)

        log.debug("Inference thread exited.")

    # ── Dispatch ──────────────────────────────────────────────────────────

    def _dispatch(self, event: WakeEvent) -> None:
        """Send event to the async queue and all registered sync callbacks."""
        # Async queue (non-blocking; newest event wins if full)
        try:
            self._event_q.put_nowait(event)
        except queue.Full:
            try:
                self._event_q.get_nowait()
            except queue.Empty:
                pass
            self._event_q.put_nowait(event)

        # Registered synchronous callbacks (called on inference thread —
        # keep them fast; spawn a thread for anything slow)
        with self._callbacks_lock:
            snapshot = list(self._callbacks)

        for cb in snapshot:
            try:
                cb(event)
            except Exception as exc:
                log.error("Wake-word callback raised: %s", exc, exc_info=True)


# ─────────────────────────────────────────────────────────────────────────────
# WakeWordDetector  (public façade)
# ─────────────────────────────────────────────────────────────────────────────

class WakeWordDetector:
    """
    Public interface for continuous wake-word detection.

    Manages two background daemon threads:
      - _AudioCaptureThread  : mic → audio queue
      - _InferenceThread     : audio queue → OWW model → events

    Sync usage
    ----------
        def on_wake(event: WakeEvent):
            print("Wake word!", event.score)

        detector = WakeWordDetector()
        detector.register(on_wake)
        detector.start()
        input("Press Enter to stop…")
        detector.stop()

    Context manager usage
    ---------------------
        with WakeWordDetector() as detector:
            detector.register(on_wake)
            time.sleep(30)

    Async usage  →  see AsyncWakeWordBridge below.
    """

    def __init__(self, config: Optional[WakeWordConfig] = None) -> None:
        self._cfg              = config or WakeWordConfig()
        self._callbacks:       List[WakeWordCallback] = []
        self._callbacks_lock   = threading.Lock()
        self._stop_evt         = threading.Event()
        self._audio_q:         queue.Queue  = queue.Queue(maxsize=self._cfg.queue_maxsize)
        self._event_q:         queue.Queue  = queue.Queue(maxsize=10)
        self._capture_thread:  Optional[_AudioCaptureThread] = None
        self._inference_thread: Optional[_InferenceThread]   = None
        self._running          = False
        self._start_lock       = threading.Lock()

    # ── Callback registration ─────────────────────────────────────────────

    def register(self, callback: WakeWordCallback) -> None:
        """
        Register a callable to be invoked on every wake-word detection.
        Thread-safe; can be called before or after start().

        The callback is called on the inference thread — keep it fast.
        For slow work (LLM calls, file I/O), spawn a thread or schedule
        a coroutine instead.
        """
        with self._callbacks_lock:
            if callback not in self._callbacks:
                self._callbacks.append(callback)
                log.debug("Callback registered: %s", getattr(callback, "__name__", repr(callback)))

    def unregister(self, callback: WakeWordCallback) -> bool:
        """Remove a previously registered callback.  Returns True if found."""
        with self._callbacks_lock:
            try:
                self._callbacks.remove(callback)
                log.debug("Callback unregistered: %s", getattr(callback, "__name__", repr(callback)))
                return True
            except ValueError:
                return False

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def start(self) -> None:
        """
        Open the microphone, load the OWW model, and begin listening.
        Returns immediately — all work runs on background daemon threads.
        """
        with self._start_lock:
            if self._running:
                log.warning("WakeWordDetector.start() called while already running.")
                return

            log.info(
                "Starting WakeWordDetector — model: %s, threshold: %.2f, "
                "cooldown: %.1f s, chunk: %d ms",
                self._cfg.resolved_model_path.name,
                self._cfg.detection_threshold,
                self._cfg.cooldown_s,
                self._cfg.chunk_ms,
            )

            self._stop_evt.clear()
            # Drain stale data from previous run
            while not self._audio_q.empty():
                self._audio_q.get_nowait()
            while not self._event_q.empty():
                self._event_q.get_nowait()

            self._capture_thread = _AudioCaptureThread(
                config=self._cfg,
                audio_q=self._audio_q,
                stop_evt=self._stop_evt,
            )
            self._inference_thread = _InferenceThread(
                config=self._cfg,
                audio_q=self._audio_q,
                event_q=self._event_q,
                callbacks=self._callbacks,
                callbacks_lock=self._callbacks_lock,
                stop_evt=self._stop_evt,
            )

            # Inference loads the model before it starts draining the queue,
            # so start capture thread first — it will block on the queue
            # until inference is ready.
            self._capture_thread.start()
            self._inference_thread.start()
            self._running = True
            log.info("WakeWordDetector running. Say 'Hey Jarvis' …")

    def stop(self) -> None:
        """Stop both background threads and release the microphone."""
        with self._start_lock:
            if not self._running:
                return

            log.info("Stopping WakeWordDetector …")
            self._stop_evt.set()

        if self._capture_thread:
            self._capture_thread.join(timeout=3.0)
            if self._capture_thread.is_alive():
                log.warning("Capture thread did not exit in time.")

        if self._inference_thread:
            self._inference_thread.join(timeout=5.0)
            if self._inference_thread.is_alive():
                log.warning("Inference thread did not exit in time.")

        self._running = False
        log.info("WakeWordDetector stopped.")

    def restart(self) -> None:
        """Stop and re-start the detector (useful after mic disconnection)."""
        log.info("Restarting WakeWordDetector …")
        self.stop()
        time.sleep(0.5)
        self.start()

    def __enter__(self) -> "WakeWordDetector":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()

    # ── Status ────────────────────────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def event_queue(self) -> queue.Queue:
        """Direct access to the WakeEvent queue (for custom polling)."""
        return self._event_q

    def status(self) -> dict:
        return {
            "running":              self._running,
            "model":                self._cfg.resolved_model_path.name,
            "detection_threshold":  self._cfg.detection_threshold,
            "cooldown_s":           self._cfg.cooldown_s,
            "chunk_ms":             self._cfg.chunk_ms,
            "audio_queue_size":     self._audio_q.qsize(),
            "event_queue_size":     self._event_q.qsize(),
            "registered_callbacks": len(self._callbacks),
        }

    def __repr__(self) -> str:
        return (
            f"WakeWordDetector("
            f"running={self._running}, "
            f"model={self._cfg.resolved_model_path.name!r}, "
            f"threshold={self._cfg.detection_threshold})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# AsyncWakeWordBridge
# ─────────────────────────────────────────────────────────────────────────────

class AsyncWakeWordBridge:
    """
    asyncio bridge for the WakeWordDetector.

    Allows any asyncio Task to await a wake-word event without blocking
    the event loop. Internally polls the detector's event queue using a
    small sleep interval.

    Usage
    -----
        detector = WakeWordDetector()
        bridge   = AsyncWakeWordBridge(detector)
        detector.start()

        async def handle():
            while True:
                event = await bridge.wait_for_wake(timeout=60)
                if event:
                    # Hand off to STT / orchestrator
                    asyncio.create_task(process_voice_command(event))
                else:
                    print("No wake word in the last 60 s.")
    """

    def __init__(
        self,
        detector:     WakeWordDetector,
        poll_interval: float = 0.05,   # seconds between queue polls (50 ms)
    ) -> None:
        self._detector     = detector
        self._poll_interval = poll_interval

    async def wait_for_wake(self, timeout: Optional[float] = None) -> Optional[WakeEvent]:
        """
        Await the next WakeEvent.

        Args:
            timeout: Maximum seconds to wait.  None = wait forever.

        Returns:
            WakeEvent on detection, or None on timeout.
        """
        deadline = (time.monotonic() + timeout) if timeout is not None else None
        q = self._detector.event_queue

        while True:
            try:
                event = q.get_nowait()
                return event
            except queue.Empty:
                pass

            if deadline is not None and time.monotonic() >= deadline:
                return None

            await asyncio.sleep(self._poll_interval)

    async def stream_events(self):
        """
        Async generator that yields WakeEvents indefinitely.

        Usage:
            async for event in bridge.stream_events():
                await handle_activation(event)
        """
        while True:
            event = await self.wait_for_wake(timeout=None)
            if event is not None:
                yield event

    async def wait_for_n_events(
        self, n: int, timeout: Optional[float] = None
    ) -> List[WakeEvent]:
        """Collect exactly n WakeEvents and return them as a list."""
        events: List[WakeEvent] = []
        deadline = (time.monotonic() + timeout) if timeout is not None else None

        while len(events) < n:
            remaining = (deadline - time.monotonic()) if deadline else None
            event = await self.wait_for_wake(timeout=remaining)
            if event is None:
                break
            events.append(event)

        return events


# ─────────────────────────────────────────────────────────────────────────────
# BackgroundWakeWordService
# ─────────────────────────────────────────────────────────────────────────────

class BackgroundWakeWordService:
    """
    Long-running service wrapper suitable for systemd or background app use.

    Features
    --------
    • Auto-restart on unrecoverable mic/model errors (with backoff)
    • Health-check via .is_healthy
    • Graceful shutdown on SIGINT / SIGTERM
    • Optional activity log (detection count, last seen time)

    Usage
    -----
        service = BackgroundWakeWordService(on_wake=my_callback)
        service.run_forever()     # blocks until SIGINT
    """

    def __init__(
        self,
        on_wake:        Optional[WakeWordCallback] = None,
        config:         Optional[WakeWordConfig]   = None,
        max_restarts:   int   = 10,
        restart_delay_s: float = 3.0,
    ) -> None:
        self._cfg            = config or WakeWordConfig()
        self._on_wake        = on_wake
        self._max_restarts   = max_restarts
        self._restart_delay  = restart_delay_s
        self._detector:      Optional[WakeWordDetector] = None
        self._restart_count  = 0
        self._total_detections = 0
        self._last_detection:  Optional[WakeEvent] = None
        self._service_stop   = threading.Event()

    # ── Public ────────────────────────────────────────────────────────────

    @property
    def is_healthy(self) -> bool:
        return (
            self._detector is not None
            and self._detector.is_running
            and not self._service_stop.is_set()
        )

    @property
    def stats(self) -> dict:
        return {
            "healthy":           self.is_healthy,
            "total_detections":  self._total_detections,
            "restart_count":     self._restart_count,
            "last_detection":    str(self._last_detection) if self._last_detection else None,
        }

    def run_forever(self) -> None:
        """
        Block until SIGINT or stop() is called, auto-restarting on failures.
        """
        import signal

        def _handle_signal(sig, frame):
            log.info("Signal %s received — shutting down service.", sig)
            self.stop()

        signal.signal(signal.SIGINT,  _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)

        log.info("BackgroundWakeWordService starting.")

        while not self._service_stop.is_set():
            self._detector = WakeWordDetector(config=self._cfg)
            self._detector.register(self._internal_callback)
            if self._on_wake:
                self._detector.register(self._on_wake)

            try:
                self._detector.start()
            except Exception as exc:
                log.error("Detector failed to start: %s", exc)
                self._handle_restart()
                continue

            # Wait until stop event fires or detector threads die
            while not self._service_stop.is_set():
                if not self._detector.is_running:
                    log.warning("Detector stopped unexpectedly.")
                    break
                time.sleep(1.0)

            if self._service_stop.is_set():
                break

            self._handle_restart()

        # Graceful shutdown
        if self._detector and self._detector.is_running:
            self._detector.stop()
        log.info("BackgroundWakeWordService exited. Total detections: %d",
                 self._total_detections)

    def stop(self) -> None:
        """Signal the service to shut down cleanly."""
        self._service_stop.set()

    # ── Internal ──────────────────────────────────────────────────────────

    def _internal_callback(self, event: WakeEvent) -> None:
        self._total_detections += 1
        self._last_detection = event

    def _handle_restart(self) -> None:
        self._restart_count += 1
        if self._restart_count > self._max_restarts:
            log.critical(
                "Max restarts (%d) exceeded — giving up.", self._max_restarts
            )
            self._service_stop.set()
            return

        log.warning(
            "Restarting detector in %.1f s (restart #%d / %d) …",
            self._restart_delay, self._restart_count, self._max_restarts,
        )
        time.sleep(self._restart_delay)


# ─────────────────────────────────────────────────────────────────────────────
# Module-level convenience helpers
# ─────────────────────────────────────────────────────────────────────────────

def list_microphones() -> List[dict]:
    """Return info dicts for all available input devices."""
    import pyaudio
    pa = pyaudio.PyAudio()
    devices = []
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info["maxInputChannels"] > 0:
            devices.append({
                "index":       i,
                "name":        info["name"],
                "sample_rate": int(info["defaultSampleRate"]),
                "channels":    info["maxInputChannels"],
            })
    pa.terminate()
    return devices


def available_models() -> List[str]:
    """Return the names of all OWW models bundled with openWakeWord."""
    return [Path(p).stem for p in openwakeword.get_pretrained_model_paths()]


# ─────────────────────────────────────────────────────────────────────────────
# CLI  (python wake_word.py)
# ─────────────────────────────────────────────────────────────────────────────

async def _async_demo() -> None:
    """Interactive async demo — awaits wake words and prints them."""
    print("\n── JARVIS Wake Word Demo ──")
    print(f"Available models : {available_models()}")
    print("Say 'Hey Jarvis' to activate. Press Ctrl+C to stop.\n")

    config   = WakeWordConfig(detection_threshold=0.5, cooldown_s=2.0)
    detector = WakeWordDetector(config=config)
    bridge   = AsyncWakeWordBridge(detector)

    activation_count = 0

    with detector:
        print("Listening …\n")
        try:
            async for event in bridge.stream_events():
                activation_count += 1
                print(f"[{activation_count}] ACTIVATED — {event}")
                print("     (Waiting for next wake word …)\n")
        except asyncio.CancelledError:
            pass

    print(f"\nTotal activations: {activation_count}")


if __name__ == "__main__":
    try:
        asyncio.run(_async_demo())
    except KeyboardInterrupt:
        print("\nBye.")
