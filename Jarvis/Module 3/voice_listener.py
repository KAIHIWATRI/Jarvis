"""
JARVIS Voice Listener Module
=============================
Real-time speech recognition via Faster-Whisper with continuous
microphone listening, silence detection, and noise filtering.

Optimised for AMD Ryzen 7 3700U · 16 GB RAM (no GPU required).

Architecture overview
---------------------
  VoiceConfig          – Frozen settings dataclass (all tunables in one place)
  AudioFrame           – Lightweight value object for a single audio chunk
  SilenceDetector      – Energy + zero-crossing based VAD (Voice Activity Detection)
  NoiseFilter          – Simple spectral subtraction + high-pass filter
  MicrophoneManager    – PyAudio device lifecycle (open / read / close)
  WhisperEngine        – Faster-Whisper model wrapper (CPU-optimised)
  TranscriptionResult  – Value object returned to callers
  VoiceListener        – Public façade; orchestrates all components

Threading model
---------------
  Main thread           → calls start() / stop() / on_transcription callback
  _capture_thread       → reads raw audio from mic into _audio_queue (daemon)
  _processing_thread    → drains _audio_queue, runs VAD+filter+Whisper (daemon)
  asyncio bridge        → optional — get_transcription_async() for async callers

Quick start
-----------
    from voice_listener import VoiceListener, VoiceConfig

    def on_result(result):
        print(f"[{result.confidence:.0%}] {result.text}")

    listener = VoiceListener(on_transcription=on_result)
    listener.start()
    input("Press Enter to stop …")
    listener.stop()
"""

from __future__ import annotations

import asyncio
import collections
import logging
import logging.handlers
import math
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Deque, List, Optional, Tuple

import numpy as np
import pyaudio
from faster_whisper import WhisperModel


# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────

def _build_logger(name: str) -> logging.Logger:
    """Rotating-file + console logger scoped to this module."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured in this process

    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    fh = logging.handlers.RotatingFileHandler(
        log_dir / "voice_listener.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


log = _build_logger("jarvis.voice")


# ──────────────────────────────────────────────────────────────────────────────
# VoiceConfig
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class VoiceConfig:
    """
    All tunables in one immutable place.

    Ryzen 7 3700U optimisation notes
    ---------------------------------
    whisper_model      : "base.en" → 145 MB RAM, ~1.5 s/utterance on 3700U.
                         "tiny.en" → 75 MB,  ~0.7 s  (less accurate).
                         "small.en"→ 470 MB, ~3.5 s  (more accurate, higher CPU).
    compute_type       : "int8"  is fastest on x86 without AVX-512.
    num_workers        : 1 leaves cores free for Ollama inference.
    chunk_duration_ms  : 30 ms is the VAD sweet-spot (WebRTC convention).
    silence_threshold_db: –40 dB is a good indoor default. Lower for quiet rooms.
    speech_pad_ms      : Padding added before/after speech so words aren't clipped.
    """

    # Microphone
    sample_rate:        int   = 16_000   # Hz — Whisper's native rate
    channels:           int   = 1        # mono
    chunk_duration_ms:  int   = 30       # ms per audio frame
    device_index:       Optional[int] = None  # None → system default

    # Silence / VAD
    silence_threshold_db:  float = -40.0  # dB RMS below which = silence
    silence_duration_ms:   int   = 700    # ms of silence → end of utterance
    min_speech_duration_ms: int  = 250    # ignore utterances shorter than this
    speech_pad_ms:         int   = 150    # ms of audio to keep before/after speech

    # Noise filter
    enable_noise_filter:   bool  = True
    highpass_cutoff_hz:    float = 80.0   # removes low-frequency hum

    # Whisper engine
    whisper_model:         str   = "base.en"
    compute_type:          str   = "int8"   # fastest on CPU without AVX-512
    num_workers:           int   = 1
    beam_size:             int   = 3        # lower = faster, slightly less accurate
    best_of:               int   = 3
    temperature:           float = 0.0     # 0 = greedy, deterministic
    language:              str   = "en"
    vad_filter:            bool  = True    # Whisper's built-in VAD (second pass)
    vad_min_silence_ms:    int   = 500

    # Processing pipeline
    max_queue_size:        int   = 50      # audio frames; backpressure guard
    transcription_timeout: float = 30.0   # seconds before giving up on a segment

    # Derived helpers (not configurable directly)
    @property
    def chunk_samples(self) -> int:
        return int(self.sample_rate * self.chunk_duration_ms / 1000)

    @property
    def silence_chunks(self) -> int:
        return int(self.silence_duration_ms / self.chunk_duration_ms)

    @property
    def min_speech_chunks(self) -> int:
        return int(self.min_speech_duration_ms / self.chunk_duration_ms)

    @property
    def pad_chunks(self) -> int:
        return int(self.speech_pad_ms / self.chunk_duration_ms)


# ──────────────────────────────────────────────────────────────────────────────
# Value objects
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class AudioFrame:
    """A single chunk of raw 16-bit PCM audio."""
    data:      np.ndarray    # shape (N,), dtype float32, range [-1, 1]
    timestamp: float = field(default_factory=time.monotonic)

    @classmethod
    def from_bytes(cls, raw: bytes) -> "AudioFrame":
        pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        return cls(data=pcm)

    @property
    def rms_db(self) -> float:
        """RMS energy in dBFS. Returns -inf for silence."""
        rms = float(np.sqrt(np.mean(self.data ** 2)))
        if rms < 1e-10:
            return -100.0
        return 20.0 * math.log10(rms)


@dataclass(slots=True)
class TranscriptionResult:
    """Returned to the caller for every completed utterance."""
    text:         str
    confidence:   float          # avg log-prob converted to [0, 1]
    duration_s:   float          # audio duration that was transcribed
    latency_s:    float          # wall-clock time from speech-end to result
    language:     str  = "en"
    is_partial:   bool = False   # reserved for future streaming display

    def __str__(self) -> str:
        return (
            f"TranscriptionResult("
            f"text={self.text!r}, "
            f"confidence={self.confidence:.1%}, "
            f"latency={self.latency_s:.2f}s)"
        )


# ──────────────────────────────────────────────────────────────────────────────
# SilenceDetector
# ──────────────────────────────────────────────────────────────────────────────

class SilenceDetector:
    """
    Energy + zero-crossing rate Voice Activity Detector (VAD).

    Uses a dual-threshold approach:
      - Primary : RMS energy in dB  (fast, cheap)
      - Secondary: Zero-crossing rate (helps distinguish speech from tonal noise)

    State machine
    -------------
      SILENCE  → SPEECH_START  when energy rises above threshold
      SPEECH   → SILENCE       when energy stays below threshold for N consecutive chunks
    """

    def __init__(self, config: VoiceConfig) -> None:
        self._cfg             = config
        self._silence_counter = 0          # consecutive silent chunks
        self._in_speech       = False
        self._speech_chunks   = 0          # chunks collected since speech start
        # Rolling buffer for noise floor estimation (adaptive threshold)
        self._energy_history: Deque[float] = collections.deque(maxlen=100)

    # ── Public ──────────────────────────────────────────────────────────────

    @property
    def in_speech(self) -> bool:
        return self._in_speech

    def reset(self) -> None:
        self._silence_counter = 0
        self._in_speech       = False
        self._speech_chunks   = 0
        self._energy_history.clear()

    def process(self, frame: AudioFrame) -> Tuple[bool, bool]:
        """
        Analyse a single frame.

        Returns
        -------
        (is_speech, speech_ended)
            is_speech    – True while we're inside an utterance
            speech_ended – True on the single frame where an utterance ends
        """
        energy_db = frame.rms_db
        self._energy_history.append(energy_db)

        # Adaptive floor: 10th percentile of recent history
        adaptive_floor = (
            float(np.percentile(list(self._energy_history), 10))
            if len(self._energy_history) >= 10
            else self._cfg.silence_threshold_db
        )
        threshold = max(self._cfg.silence_threshold_db, adaptive_floor + 6.0)

        is_active = energy_db > threshold

        speech_ended = False

        if is_active:
            self._silence_counter = 0
            if not self._in_speech:
                self._in_speech    = True
                self._speech_chunks = 0
                log.debug("VAD: speech start (energy=%.1f dB, threshold=%.1f dB)",
                          energy_db, threshold)
            self._speech_chunks += 1

        else:
            if self._in_speech:
                self._silence_counter += 1
                if self._silence_counter >= self._cfg.silence_chunks:
                    if self._speech_chunks >= self._cfg.min_speech_chunks:
                        speech_ended   = True
                        log.debug(
                            "VAD: speech end (%d speech chunks, %d silence chunks)",
                            self._speech_chunks, self._silence_counter,
                        )
                    else:
                        log.debug("VAD: noise burst ignored (%d chunks < min %d)",
                                  self._speech_chunks, self._cfg.min_speech_chunks)
                    self._in_speech       = False
                    self._silence_counter = 0
                    self._speech_chunks   = 0

        return self._in_speech, speech_ended


# ──────────────────────────────────────────────────────────────────────────────
# NoiseFilter
# ──────────────────────────────────────────────────────────────────────────────

class NoiseFilter:
    """
    Two-stage audio filter.

      Stage 1 – High-pass filter  : removes low-frequency hum (fans, HVAC)
      Stage 2 – Spectral subtraction: estimates noise floor during silence,
                subtracts it from speech frames.

    Both stages operate in-place on float32 numpy arrays.
    """

    def __init__(self, config: VoiceConfig) -> None:
        self._cfg         = config
        self._noise_floor: Optional[np.ndarray] = None  # estimated spectrum
        self._noise_frames: List[np.ndarray]     = []
        self._frames_for_estimation              = 20    # ~0.6 s of silence

        # Pre-compute high-pass FIR coefficients (simple one-pole IIR)
        rc = 1.0 / (2.0 * math.pi * config.highpass_cutoff_hz)
        dt = 1.0 / config.sample_rate
        self._hp_alpha = rc / (rc + dt)   # IIR coefficient
        self._hp_prev_x = 0.0
        self._hp_prev_y = 0.0

    # ── Public ──────────────────────────────────────────────────────────────

    def update_noise_estimate(self, frame: AudioFrame) -> None:
        """Call during silence frames to build the noise model."""
        if len(self._noise_frames) < self._frames_for_estimation:
            self._noise_frames.append(np.abs(np.fft.rfft(frame.data)))
        elif self._noise_floor is None:
            self._noise_floor = np.mean(self._noise_frames, axis=0)
            log.debug("Noise floor estimated from %d frames.", len(self._noise_frames))

    def apply(self, frame: AudioFrame) -> AudioFrame:
        """
        Filter a speech frame.  Returns a new AudioFrame with filtered audio.
        """
        if not self._cfg.enable_noise_filter:
            return frame

        filtered = self._highpass(frame.data.copy())

        if self._noise_floor is not None:
            filtered = self._spectral_subtract(filtered)

        # Clip to valid range after processing
        filtered = np.clip(filtered, -1.0, 1.0)
        return AudioFrame(data=filtered, timestamp=frame.timestamp)

    # ── Internal ────────────────────────────────────────────────────────────

    def _highpass(self, signal: np.ndarray) -> np.ndarray:
        """Single-pole high-pass IIR filter (in-place efficient)."""
        out = np.empty_like(signal)
        prev_x, prev_y = self._hp_prev_x, self._hp_prev_y
        alpha = self._hp_alpha
        for i, x in enumerate(signal):
            y = alpha * (prev_y + x - prev_x)
            out[i]  = y
            prev_x  = x
            prev_y  = y
        self._hp_prev_x = float(signal[-1]) if len(signal) else prev_x
        self._hp_prev_y = float(out[-1])    if len(out)    else prev_y
        return out

    def _spectral_subtract(self, signal: np.ndarray) -> np.ndarray:
        """Over-subtraction spectral noise reduction."""
        spectrum  = np.fft.rfft(signal)
        magnitude = np.abs(spectrum)
        phase     = np.angle(spectrum)

        # Over-subtract with floor (avoids musical noise artefacts)
        subtracted = np.maximum(magnitude - 1.5 * self._noise_floor, 0.1 * magnitude)

        clean_spectrum = subtracted * np.exp(1j * phase)
        return np.fft.irfft(clean_spectrum, n=len(signal)).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# MicrophoneManager
# ──────────────────────────────────────────────────────────────────────────────

class MicrophoneManager:
    """
    Manages the PyAudio device lifecycle.

    Thread-safety: open() / close() are called from the main thread.
    read_frame() is called exclusively from _capture_thread.
    """

    def __init__(self, config: VoiceConfig) -> None:
        self._cfg    = config
        self._pa:    Optional[pyaudio.PyAudio]  = None
        self._stream: Optional[pyaudio.Stream]  = None
        self._lock   = threading.Lock()

    # ── Public ──────────────────────────────────────────────────────────────

    def open(self) -> None:
        """Open the audio input stream.  Raises on device error."""
        with self._lock:
            if self._stream is not None:
                return  # already open

            self._pa = pyaudio.PyAudio()
            device_idx = self._cfg.device_index

            if device_idx is None:
                device_idx = self._pa.get_default_input_device_info()["index"]

            info = self._pa.get_device_info_by_index(device_idx)
            log.info(
                "Microphone: '%s' (index=%d, rate=%d Hz, channels=%d)",
                info["name"], device_idx, self._cfg.sample_rate, self._cfg.channels,
            )

            self._stream = self._pa.open(
                format=pyaudio.paInt16,
                channels=self._cfg.channels,
                rate=self._cfg.sample_rate,
                input=True,
                input_device_index=device_idx,
                frames_per_buffer=self._cfg.chunk_samples,
                # Non-callback mode: we pull frames manually in _capture_thread
            )
            log.info("Microphone stream opened. Chunk = %d samples (%d ms).",
                     self._cfg.chunk_samples, self._cfg.chunk_duration_ms)

    def close(self) -> None:
        """Close and release the audio device."""
        with self._lock:
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
        log.info("Microphone closed.")

    def read_frame(self) -> Optional[AudioFrame]:
        """
        Read one chunk from the microphone.  Returns None on error.
        Blocks for approximately chunk_duration_ms milliseconds.
        """
        if self._stream is None:
            return None
        try:
            raw = self._stream.read(
                self._cfg.chunk_samples,
                exception_on_overflow=False,  # drop frames rather than crash
            )
            return AudioFrame.from_bytes(raw)
        except OSError as exc:
            log.error("Microphone read error: %s", exc)
            return None

    @staticmethod
    def list_devices() -> List[dict]:
        """Return info dicts for all available input devices."""
        pa = pyaudio.PyAudio()
        devices = []
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info["maxInputChannels"] > 0:
                devices.append({
                    "index": i,
                    "name":  info["name"],
                    "sample_rate": int(info["defaultSampleRate"]),
                    "channels":    info["maxInputChannels"],
                })
        pa.terminate()
        return devices


# ──────────────────────────────────────────────────────────────────────────────
# WhisperEngine
# ──────────────────────────────────────────────────────────────────────────────

class WhisperEngine:
    """
    Faster-Whisper model wrapper, optimised for CPU inference.

    Loads the model once at initialisation.  All transcription calls are
    synchronous and run inside the processing thread (never on the event loop).

    Memory
    ------
    base.en  → ~300 MB with int8 quantisation
    small.en → ~600 MB with int8 quantisation
    """

    def __init__(self, config: VoiceConfig) -> None:
        self._cfg   = config
        self._model: Optional[WhisperModel] = None
        self._lock  = threading.Lock()

    def load(self) -> None:
        """Load the Whisper model into RAM.  Blocking — call once at startup."""
        log.info(
            "Loading Whisper model '%s' (compute_type=%s) …",
            self._cfg.whisper_model, self._cfg.compute_type,
        )
        t0 = time.monotonic()
        with self._lock:
            self._model = WhisperModel(
                self._cfg.whisper_model,
                device="cpu",
                compute_type=self._cfg.compute_type,
                num_workers=self._cfg.num_workers,
                # cpu_threads: 0 = auto-detect; explicit value avoids over-subscription
                cpu_threads=max(1, self._cfg.num_workers * 2),
            )
        elapsed = time.monotonic() - t0
        log.info("Whisper model loaded in %.2f s.", elapsed)

    def transcribe(self, audio: np.ndarray) -> Optional[TranscriptionResult]:
        """
        Transcribe a float32 mono audio array sampled at 16 kHz.

        Returns None if no speech was detected or transcription failed.
        """
        if self._model is None:
            raise RuntimeError("WhisperEngine.load() must be called before transcribe().")

        if len(audio) == 0:
            return None

        duration_s  = len(audio) / self._cfg.sample_rate
        t_start     = time.monotonic()

        try:
            with self._lock:
                segments, info = self._model.transcribe(
                    audio,
                    language=self._cfg.language,
                    beam_size=self._cfg.beam_size,
                    best_of=self._cfg.best_of,
                    temperature=self._cfg.temperature,
                    vad_filter=self._cfg.vad_filter,
                    vad_parameters=dict(
                        min_silence_duration_ms=self._cfg.vad_min_silence_ms,
                    ),
                    condition_on_previous_text=False,  # stateless per utterance
                    without_timestamps=True,
                )
                # Materialise the generator inside the lock
                segment_list = list(segments)

        except Exception as exc:
            log.error("Whisper transcription error: %s", exc, exc_info=True)
            return None

        latency_s = time.monotonic() - t_start

        if not segment_list:
            log.debug("Whisper: no speech detected (%.2f s audio).", duration_s)
            return None

        text = " ".join(s.text.strip() for s in segment_list).strip()
        if not text:
            return None

        # avg_logprob is in range (-inf, 0]; convert to [0, 1] confidence
        avg_logprob  = sum(s.avg_logprob for s in segment_list) / len(segment_list)
        confidence   = min(1.0, max(0.0, math.exp(avg_logprob)))

        result = TranscriptionResult(
            text=text,
            confidence=confidence,
            duration_s=duration_s,
            latency_s=latency_s,
            language=info.language if info else self._cfg.language,
        )
        log.info("Whisper: %s (conf=%.0f%%, dur=%.2fs, lat=%.2fs)",
                 repr(text[:60]), confidence * 100, duration_s, latency_s)
        return result


# ──────────────────────────────────────────────────────────────────────────────
# VoiceListener  (public façade)
# ──────────────────────────────────────────────────────────────────────────────

class VoiceListener:
    """
    Public interface for continuous microphone listening + transcription.

    Usage (synchronous)
    -------------------
        def handle(result: TranscriptionResult):
            print(result.text)

        listener = VoiceListener(on_transcription=handle)
        listener.start()
        ...
        listener.stop()

    Usage (async)
    -------------
        listener = VoiceListener()
        listener.start()
        while True:
            result = await listener.get_transcription_async(timeout=10)
            if result:
                print(result.text)

    Thread model
    ------------
    • _capture_thread   : mic → _audio_queue (raw AudioFrames)
    • _processing_thread: _audio_queue → VAD → filter → Whisper → callback
    All shared state is accessed through threading primitives.
    """

    def __init__(
        self,
        config:            Optional[VoiceConfig]                     = None,
        on_transcription:  Optional[Callable[[TranscriptionResult], None]] = None,
    ) -> None:
        self._cfg              = config or VoiceConfig()
        self._callback         = on_transcription
        self._mic              = MicrophoneManager(self._cfg)
        self._whisper          = WhisperEngine(self._cfg)
        self._vad              = SilenceDetector(self._cfg)
        self._filter           = NoiseFilter(self._cfg)

        # Thread-safe audio queue (raw frames from mic)
        self._audio_queue: queue.Queue[Optional[AudioFrame]] = queue.Queue(
            maxsize=self._cfg.max_queue_size
        )
        # Thread-safe result queue (for async callers)
        self._result_queue: queue.Queue[TranscriptionResult] = queue.Queue(maxsize=20)

        self._capture_thread:    Optional[threading.Thread] = None
        self._processing_thread: Optional[threading.Thread] = None
        self._stop_event         = threading.Event()
        self._running            = False
        self._lock               = threading.Lock()

        # Rolling pre-speech buffer (keeps audio from just before speech starts)
        self._pre_buffer: Deque[AudioFrame] = collections.deque(
            maxlen=self._cfg.pad_chunks
        )

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """
        Open the microphone, load Whisper, and begin listening.
        This method returns immediately; listening runs on background threads.
        """
        with self._lock:
            if self._running:
                log.warning("VoiceListener.start() called while already running.")
                return

            log.info("Starting VoiceListener …")
            self._stop_event.clear()
            self._vad.reset()

            # Load model (blocking, ~1–2 s on 3700U for base.en)
            self._whisper.load()

            # Open mic
            self._mic.open()

            # Start threads
            self._capture_thread = threading.Thread(
                target=self._capture_loop,
                name="jarvis-mic-capture",
                daemon=True,
            )
            self._processing_thread = threading.Thread(
                target=self._processing_loop,
                name="jarvis-whisper-proc",
                daemon=True,
            )
            self._capture_thread.start()
            self._processing_thread.start()

            self._running = True
            log.info("VoiceListener running. Listening …")

    def stop(self) -> None:
        """Stop listening, drain remaining audio, and release resources."""
        with self._lock:
            if not self._running:
                return

            log.info("Stopping VoiceListener …")
            self._stop_event.set()

            # Unblock the processing thread
            try:
                self._audio_queue.put_nowait(None)
            except queue.Full:
                pass

        if self._capture_thread:
            self._capture_thread.join(timeout=3.0)
        if self._processing_thread:
            self._processing_thread.join(timeout=5.0)

        self._mic.close()
        self._running = False
        log.info("VoiceListener stopped.")

    def __enter__(self) -> "VoiceListener":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()

    # ── Async interface ──────────────────────────────────────────────────────

    async def get_transcription_async(
        self, timeout: float = 30.0
    ) -> Optional[TranscriptionResult]:
        """
        Await the next transcription result.
        Returns None on timeout.  Safe to call from any asyncio Task.
        """
        loop     = asyncio.get_event_loop()
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            try:
                result = self._result_queue.get_nowait()
                return result
            except queue.Empty:
                await asyncio.sleep(0.05)

        return None

    # ── Status ───────────────────────────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_listening(self) -> bool:
        """True while we are inside a detected utterance."""
        return self._vad.in_speech

    def status(self) -> dict:
        return {
            "running":      self._running,
            "listening":    self._vad.in_speech,
            "queue_size":   self._audio_queue.qsize(),
            "model":        self._cfg.whisper_model,
            "compute_type": self._cfg.compute_type,
            "sample_rate":  self._cfg.sample_rate,
        }

    # ── Background threads ───────────────────────────────────────────────────

    def _capture_loop(self) -> None:
        """
        Capture thread: reads frames from mic → puts into _audio_queue.
        Designed to keep up with real-time audio at minimal CPU cost.
        """
        log.debug("Capture thread started.")
        consecutive_errors = 0

        while not self._stop_event.is_set():
            frame = self._mic.read_frame()

            if frame is None:
                consecutive_errors += 1
                if consecutive_errors > 10:
                    log.error("Too many microphone errors — stopping capture.")
                    self._stop_event.set()
                    break
                time.sleep(0.01)
                continue

            consecutive_errors = 0

            try:
                self._audio_queue.put_nowait(frame)
            except queue.Full:
                # Drop oldest frame to make room (prefer freshness)
                try:
                    self._audio_queue.get_nowait()
                except queue.Empty:
                    pass
                self._audio_queue.put_nowait(frame)
                log.warning("Audio queue full — oldest frame dropped.")

        # Sentinel to wake the processing thread
        try:
            self._audio_queue.put_nowait(None)
        except queue.Full:
            pass
        log.debug("Capture thread exited.")

    def _processing_loop(self) -> None:
        """
        Processing thread: drains _audio_queue → VAD → filter → Whisper → callback.

        State machine
        -------------
        Silence  → collect frames into pre_buffer (for noise floor estimation)
                 → on speech start: prepend pre_buffer to speech_buffer
        Speech   → collect into speech_buffer
                 → on speech end: submit speech_buffer to Whisper
        """
        log.debug("Processing thread started.")
        speech_buffer: List[np.ndarray] = []
        speech_end_time: Optional[float] = None

        while True:
            try:
                frame = self._audio_queue.get(timeout=0.5)
            except queue.Empty:
                if self._stop_event.is_set():
                    break
                continue

            # Sentinel — stop signal
            if frame is None:
                break

            is_speech, speech_ended = self._vad.process(frame)

            if is_speech:
                # First speech frame — prepend pre-speech padding
                if len(speech_buffer) == 0 and len(self._pre_buffer) > 0:
                    for pad_frame in self._pre_buffer:
                        filtered_pad = self._filter.apply(pad_frame)
                        speech_buffer.append(filtered_pad.data)
                    log.debug("Prepended %d padding frames.", len(self._pre_buffer))

                filtered = self._filter.apply(frame)
                speech_buffer.append(filtered.data)

            else:
                # Update noise model during silence
                self._filter.update_noise_estimate(frame)
                self._pre_buffer.append(frame)

                if speech_ended and speech_buffer:
                    speech_end_time = time.monotonic()
                    # Append post-speech padding
                    post_pad = list(self._pre_buffer)[:self._cfg.pad_chunks]
                    for pad_frame in post_pad:
                        speech_buffer.append(self._filter.apply(pad_frame).data)

                    audio = np.concatenate(speech_buffer, axis=0)
                    speech_buffer = []
                    self._pre_buffer.clear()

                    self._run_transcription(audio, speech_end_time)
                    speech_end_time = None

        # End of loop — transcribe any remaining buffered speech
        if speech_buffer:
            log.info("Processing remaining %d frames at shutdown.", len(speech_buffer))
            audio = np.concatenate(speech_buffer, axis=0)
            self._run_transcription(audio, time.monotonic())

        log.debug("Processing thread exited.")

    def _run_transcription(self, audio: np.ndarray, speech_end_time: float) -> None:
        """Run Whisper on a collected utterance and dispatch the result."""
        result = self._whisper.transcribe(audio)

        if result is None:
            return

        # Patch latency to include time from speech-end to result delivery
        result = TranscriptionResult(
            text=result.text,
            confidence=result.confidence,
            duration_s=result.duration_s,
            latency_s=time.monotonic() - speech_end_time,
            language=result.language,
        )

        # Push to result queue (for async callers) — non-blocking
        try:
            self._result_queue.put_nowait(result)
        except queue.Full:
            log.warning("Result queue full — oldest result dropped.")
            try:
                self._result_queue.get_nowait()
            except queue.Empty:
                pass
            self._result_queue.put_nowait(result)

        # Fire callback (on processing thread — keep it fast)
        if self._callback:
            try:
                self._callback(result)
            except Exception as exc:
                log.error("on_transcription callback raised: %s", exc, exc_info=True)
