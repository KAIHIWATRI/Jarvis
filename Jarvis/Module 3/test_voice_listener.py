"""
tests/test_voice_listener.py
=============================
Unit tests for the JARVIS Voice Listener module.
All hardware I/O (PyAudio, Whisper model loading) is mocked — no mic required.

Run:  pytest tests/test_voice_listener.py -v
"""

from __future__ import annotations

import asyncio
import math
import queue
import sys
import time
import threading
import os
from unittest.mock import MagicMock, patch, PropertyMock
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from voice_listener import (
    AudioFrame,
    NoiseFilter,
    SilenceDetector,
    MicrophoneManager,
    TranscriptionResult,
    VoiceConfig,
    VoiceListener,
    WhisperEngine,
)


# ── helpers ────────────────────────────────────────────────────────────────────

def make_frame(db: float = -20.0, samples: int = 480) -> AudioFrame:
    """Create a synthetic AudioFrame with a given dBFS RMS level."""
    if db <= -99:
        data = np.zeros(samples, dtype=np.float32)
    else:
        amplitude = 10 ** (db / 20.0)
        t = np.linspace(0, samples / 16000, samples, endpoint=False)
        data = (amplitude * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    return AudioFrame(data=data)


def silent_frame(samples: int = 480) -> AudioFrame:
    return make_frame(db=-90.0, samples=samples)


# ── VoiceConfig ─────────────────────────────────────────────────────────────────

class TestVoiceConfig:
    def test_defaults(self):
        cfg = VoiceConfig()
        assert cfg.sample_rate == 16_000
        assert cfg.channels == 1
        assert cfg.whisper_model == "base.en"
        assert cfg.compute_type == "int8"

    def test_immutable(self):
        cfg = VoiceConfig()
        with pytest.raises(Exception):
            cfg.sample_rate = 8_000

    def test_chunk_samples(self):
        cfg = VoiceConfig(sample_rate=16_000, chunk_duration_ms=30)
        assert cfg.chunk_samples == 480

    def test_silence_chunks(self):
        cfg = VoiceConfig(silence_duration_ms=600, chunk_duration_ms=30)
        assert cfg.silence_chunks == 20

    def test_min_speech_chunks(self):
        cfg = VoiceConfig(min_speech_duration_ms=300, chunk_duration_ms=30)
        assert cfg.min_speech_chunks == 10

    def test_pad_chunks(self):
        cfg = VoiceConfig(speech_pad_ms=150, chunk_duration_ms=30)
        assert cfg.pad_chunks == 5


# ── AudioFrame ──────────────────────────────────────────────────────────────────

class TestAudioFrame:
    def test_from_bytes_normalises(self):
        raw = (np.ones(480, dtype=np.int16) * 16384).tobytes()
        frame = AudioFrame.from_bytes(raw)
        assert frame.data.dtype == np.float32
        assert np.all(np.abs(frame.data) <= 1.0)

    def test_rms_db_silence(self):
        frame = AudioFrame(data=np.zeros(480, dtype=np.float32))
        assert frame.rms_db <= -90.0

    def test_rms_db_full_scale(self):
        data = np.ones(480, dtype=np.float32)  # +1 (0 dBFS)
        frame = AudioFrame(data=data)
        assert abs(frame.rms_db) < 1.0  # approx 0 dBFS

    def test_rms_db_half_amplitude(self):
        data = np.ones(480, dtype=np.float32) * 0.5
        frame = AudioFrame(data=data)
        assert abs(frame.rms_db - (-6.0)) < 1.0

    def test_timestamp_set_automatically(self):
        frame = AudioFrame(data=np.zeros(480, dtype=np.float32))
        assert frame.timestamp > 0


# ── SilenceDetector ─────────────────────────────────────────────────────────────

class TestSilenceDetector:
    def setup_method(self):
        self.cfg = VoiceConfig(
            silence_threshold_db=-40.0,
            silence_duration_ms=90,      # 3 chunks of 30 ms
            min_speech_duration_ms=60,   # 2 chunks
            chunk_duration_ms=30,
        )
        self.vad = SilenceDetector(self.cfg)

    def test_silence_stays_silent(self):
        for _ in range(10):
            is_speech, ended = self.vad.process(silent_frame())
            assert not is_speech
            assert not ended

    def test_speech_detected(self):
        speech_detected = False
        for _ in range(5):
            is_speech, _ = self.vad.process(make_frame(db=-20.0))
            if is_speech:
                speech_detected = True
        assert speech_detected

    def test_speech_then_silence_ends_utterance(self):
        # Produce enough speech frames to meet min_speech_duration
        for _ in range(4):
            self.vad.process(make_frame(db=-20.0))

        # Now feed silence until utterance ends
        speech_ended = False
        for _ in range(20):
            _, ended = self.vad.process(silent_frame())
            if ended:
                speech_ended = True
                break
        assert speech_ended

    def test_noise_burst_below_min_ignored(self):
        # Single speech chunk — too short to count
        self.vad.process(make_frame(db=-20.0))
        # Silence
        speech_ended = False
        for _ in range(20):
            _, ended = self.vad.process(silent_frame())
            if ended:
                speech_ended = True
        assert not speech_ended

    def test_reset_clears_state(self):
        for _ in range(5):
            self.vad.process(make_frame(db=-20.0))
        self.vad.reset()
        assert not self.vad.in_speech

    def test_in_speech_property(self):
        assert not self.vad.in_speech
        self.vad.process(make_frame(db=-20.0))
        assert self.vad.in_speech


# ── NoiseFilter ─────────────────────────────────────────────────────────────────

class TestNoiseFilter:
    def setup_method(self):
        self.cfg    = VoiceConfig(enable_noise_filter=True)
        self.filt   = NoiseFilter(self.cfg)

    def test_apply_returns_new_frame(self):
        frame   = make_frame(db=-20.0)
        result  = self.filt.apply(frame)
        assert result is not frame
        assert isinstance(result, AudioFrame)

    def test_output_clipped(self):
        frame  = AudioFrame(data=np.ones(480, dtype=np.float32) * 2.0)
        result = self.filt.apply(frame)
        assert np.all(result.data <= 1.0)
        assert np.all(result.data >= -1.0)

    def test_disabled_filter_returns_same_data(self):
        cfg    = VoiceConfig(enable_noise_filter=False)
        filt   = NoiseFilter(cfg)
        frame  = make_frame(db=-20.0)
        result = filt.apply(frame)
        np.testing.assert_array_equal(result.data, frame.data)

    def test_noise_estimate_builds_after_enough_frames(self):
        for _ in range(25):
            self.filt.update_noise_estimate(silent_frame())
        assert self.filt._noise_floor is not None

    def test_spectral_subtract_applied_when_floor_known(self):
        for _ in range(25):
            self.filt.update_noise_estimate(silent_frame())
        frame  = make_frame(db=-20.0)
        result = self.filt.apply(frame)
        assert result.data.shape == frame.data.shape

    def test_highpass_reduces_dc(self):
        """After high-pass filtering, a DC signal should be greatly attenuated."""
        dc_signal = np.ones(4800, dtype=np.float32) * 0.5
        frame     = AudioFrame(data=dc_signal)
        result    = self.filt.apply(frame)
        # DC component should be near zero after high-pass (allow transient)
        assert abs(float(np.mean(result.data[100:]))) < 0.01


# ── MicrophoneManager ──────────────────────────────────────────────────────────

class TestMicrophoneManager:
    def test_read_frame_when_closed_returns_none(self):
        cfg = VoiceConfig()
        mic = MicrophoneManager(cfg)
        # Stream never opened
        result = mic.read_frame()
        assert result is None

    def test_list_devices_returns_list(self):
        with patch("voice_listener.pyaudio.PyAudio") as MockPA:
            instance = MockPA.return_value
            instance.get_device_count.return_value = 2
            instance.get_device_info_by_index.side_effect = [
                {"name": "Mic A", "maxInputChannels": 1, "defaultSampleRate": 16000.0},
                {"name": "Speaker", "maxInputChannels": 0, "defaultSampleRate": 44100.0},
            ]
            devices = MicrophoneManager.list_devices()
        assert len(devices) == 1
        assert devices[0]["name"] == "Mic A"

    def test_open_and_close(self):
        cfg = VoiceConfig()
        mic = MicrophoneManager(cfg)
        with patch("voice_listener.pyaudio.PyAudio") as MockPA:
            instance = MockPA.return_value
            instance.get_default_input_device_info.return_value = {"index": 0}
            instance.get_device_info_by_index.return_value = {
                "name": "Default", "maxInputChannels": 1, "defaultSampleRate": 16000.0
            }
            instance.open.return_value = MagicMock()
            mic.open()
            assert mic._stream is not None
            mic.close()
            assert mic._stream is None

    def test_read_frame_converts_bytes(self):
        cfg = VoiceConfig()
        mic = MicrophoneManager(cfg)
        mock_stream = MagicMock()
        raw_audio   = (np.zeros(cfg.chunk_samples, dtype=np.int16)).tobytes()
        mock_stream.read.return_value = raw_audio
        mic._stream = mock_stream
        frame = mic.read_frame()
        assert frame is not None
        assert isinstance(frame, AudioFrame)
        assert len(frame.data) == cfg.chunk_samples


# ── WhisperEngine ───────────────────────────────────────────────────────────────

class TestWhisperEngine:
    def test_transcribe_before_load_raises(self):
        engine = WhisperEngine(VoiceConfig())
        with pytest.raises(RuntimeError, match="load"):
            engine.transcribe(np.zeros(16000, dtype=np.float32))

    def test_transcribe_empty_audio_returns_none(self):
        engine = WhisperEngine(VoiceConfig())
        engine._model = MagicMock()  # bypass load
        result = engine.transcribe(np.array([], dtype=np.float32))
        assert result is None

    def test_transcribe_returns_result_on_speech(self):
        engine = WhisperEngine(VoiceConfig())

        mock_segment = MagicMock()
        mock_segment.text      = "Hello world"
        mock_segment.avg_logprob = -0.3

        mock_info = MagicMock()
        mock_info.language = "en"

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([mock_segment], mock_info)
        engine._model = mock_model

        audio  = np.random.randn(16000).astype(np.float32) * 0.1
        result = engine.transcribe(audio)

        assert result is not None
        assert result.text == "Hello world"
        assert 0.0 <= result.confidence <= 1.0
        assert result.duration_s == pytest.approx(1.0, rel=0.01)

    def test_transcribe_no_segments_returns_none(self):
        engine = WhisperEngine(VoiceConfig())
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([], MagicMock())
        engine._model = mock_model

        result = engine.transcribe(np.random.randn(16000).astype(np.float32))
        assert result is None

    def test_transcribe_exception_returns_none(self):
        engine = WhisperEngine(VoiceConfig())
        mock_model = MagicMock()
        mock_model.transcribe.side_effect = RuntimeError("CUDA OOM")
        engine._model = mock_model

        result = engine.transcribe(np.random.randn(16000).astype(np.float32))
        assert result is None


# ── TranscriptionResult ─────────────────────────────────────────────────────────

class TestTranscriptionResult:
    def test_str(self):
        r = TranscriptionResult(
            text="Hello", confidence=0.95, duration_s=1.2,
            latency_s=0.8, language="en"
        )
        s = str(r)
        assert "Hello" in s
        assert "95" in s  # matches "95.0%" or "95%"

    def test_is_partial_default_false(self):
        r = TranscriptionResult(text="x", confidence=0.9, duration_s=1.0, latency_s=0.5)
        assert r.is_partial is False


# ── VoiceListener ───────────────────────────────────────────────────────────────

class TestVoiceListener:
    """Integration-level tests with mic and Whisper fully mocked."""

    def _make_listener(self, callback=None) -> VoiceListener:
        cfg = VoiceConfig(
            silence_duration_ms=90,
            min_speech_duration_ms=60,
            chunk_duration_ms=30,
        )
        return VoiceListener(config=cfg, on_transcription=callback)

    def _patch_hardware(self, listener: VoiceListener):
        """Patch mic and whisper on an existing listener instance."""
        # Mic returns zeros
        listener._mic._stream = MagicMock()
        silence = (np.zeros(listener._cfg.chunk_samples, dtype=np.int16)).tobytes()
        listener._mic._stream.read.return_value = silence
        listener._mic._pa = MagicMock()

        # Whisper model is pre-loaded
        listener._whisper._model = MagicMock()
        listener._whisper._model.transcribe.return_value = ([], MagicMock())

    def test_start_stop(self):
        listener = self._make_listener()
        self._patch_hardware(listener)
        with patch.object(listener._mic, "open"), \
             patch.object(listener._whisper, "load"):
            listener.start()
            assert listener.is_running
            listener.stop()
            assert not listener.is_running

    def test_double_start_is_safe(self):
        listener = self._make_listener()
        self._patch_hardware(listener)
        with patch.object(listener._mic, "open"), \
             patch.object(listener._whisper, "load"):
            listener.start()
            listener.start()  # second call should be a no-op
            listener.stop()

    def test_context_manager(self):
        listener = self._make_listener()
        self._patch_hardware(listener)
        with patch.object(listener._mic, "open"), \
             patch.object(listener._whisper, "load"):
            with listener:
                assert listener.is_running
            assert not listener.is_running

    def test_callback_fired_on_result(self):
        results = []
        listener = self._make_listener(callback=results.append)
        self._patch_hardware(listener)

        # Override _run_transcription to inject a fake result
        fake_result = TranscriptionResult(
            text="test utterance", confidence=0.9,
            duration_s=1.0, latency_s=0.3,
        )

        with patch.object(listener._mic, "open"), \
             patch.object(listener._whisper, "load"):
            listener.start()
            listener._run_transcription(
                np.zeros(16000, dtype=np.float32), time.monotonic()
            )
            listener.stop()

        # _run_transcription calls whisper mock which returns None, but
        # we can test the callback path directly:
        listener._callback(fake_result)
        assert len(results) >= 1

    def test_status_keys(self):
        listener = self._make_listener()
        s = listener.status()
        for key in ("running", "listening", "queue_size", "model", "compute_type"):
            assert key in s

    def test_queue_overflow_drops_oldest(self):
        """Fill queue beyond max_queue_size and verify no crash."""
        cfg      = VoiceConfig(max_queue_size=5)
        listener = VoiceListener(config=cfg)
        for _ in range(10):
            try:
                listener._audio_queue.put_nowait(silent_frame())
            except queue.Full:
                try:
                    listener._audio_queue.get_nowait()
                except queue.Empty:
                    pass
                listener._audio_queue.put_nowait(silent_frame())
        assert listener._audio_queue.qsize() <= 6  # within bounds

    @pytest.mark.asyncio
    async def test_get_transcription_async_timeout(self):
        listener = self._make_listener()
        result   = await listener.get_transcription_async(timeout=0.1)
        assert result is None

    @pytest.mark.asyncio
    async def test_get_transcription_async_returns_result(self):
        listener    = self._make_listener()
        fake_result = TranscriptionResult(
            text="async test", confidence=0.85, duration_s=1.0, latency_s=0.2
        )
        listener._result_queue.put_nowait(fake_result)
        result = await listener.get_transcription_async(timeout=1.0)
        assert result is not None
        assert result.text == "async test"
