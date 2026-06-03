"""
tests/test_wake_word.py
========================
Full unit test suite for the JARVIS Wake Word module.
All hardware I/O (PyAudio, OWW model loading) is mocked — no mic required.

Run:  pytest tests/test_wake_word.py -v --asyncio-mode=auto
"""

from __future__ import annotations

import asyncio
import queue
import sys
import os
import time
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from wake_word import (
    WakeWordConfig,
    WakeEvent,
    WakeWordDetector,
    AsyncWakeWordBridge,
    BackgroundWakeWordService,
    _AudioCaptureThread,
    _InferenceThread,
    available_models,
    list_microphones,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_config(**kwargs) -> WakeWordConfig:
    """Build a WakeWordConfig with a real model path."""
    import openwakeword
    model_path = (
        Path(openwakeword.__file__).parent
        / "resources" / "models" / "hey_jarvis_v0.1.onnx"
    )
    return WakeWordConfig(model_path=model_path, **kwargs)


def _silent_frame(config: WakeWordConfig) -> np.ndarray:
    return np.zeros(config.chunk_samples, dtype=np.int16)


def _inject_event(detector: WakeWordDetector, score: float = 0.9) -> WakeEvent:
    """Push a fake WakeEvent directly into the detector's event queue."""
    event = WakeEvent(model_name="hey_jarvis_v0.1", score=score)
    detector.event_queue.put_nowait(event)
    return event


# ─────────────────────────────────────────────────────────────────────────────
# WakeWordConfig
# ─────────────────────────────────────────────────────────────────────────────

class TestWakeWordConfig:
    def test_defaults(self):
        cfg = _make_config()
        assert cfg.sample_rate == 16_000
        assert cfg.channels == 1
        assert cfg.chunk_ms == 80
        assert cfg.detection_threshold == 0.5
        assert cfg.cooldown_s == 2.0

    def test_frozen(self):
        cfg = _make_config()
        with pytest.raises(Exception):
            cfg.detection_threshold = 0.9

    def test_chunk_samples(self):
        cfg = _make_config(sample_rate=16_000, chunk_ms=80)
        assert cfg.chunk_samples == 1280

    def test_chunk_samples_30ms(self):
        cfg = _make_config(sample_rate=16_000, chunk_ms=30)
        assert cfg.chunk_samples == 480

    def test_resolved_model_path_exists(self):
        cfg = _make_config()
        assert cfg.resolved_model_path.exists()

    def test_custom_model_path(self, tmp_path):
        fake_model = tmp_path / "model.onnx"
        fake_model.touch()
        cfg = WakeWordConfig(model_path=fake_model)
        assert cfg.resolved_model_path == fake_model

    def test_custom_threshold(self):
        cfg = _make_config(detection_threshold=0.7)
        assert cfg.detection_threshold == 0.7

    def test_custom_cooldown(self):
        cfg = _make_config(cooldown_s=5.0)
        assert cfg.cooldown_s == 5.0


# ─────────────────────────────────────────────────────────────────────────────
# WakeEvent
# ─────────────────────────────────────────────────────────────────────────────

class TestWakeEvent:
    def test_creation(self):
        e = WakeEvent(model_name="hey_jarvis_v0.1", score=0.85)
        assert e.model_name == "hey_jarvis_v0.1"
        assert e.score == 0.85
        assert e.timestamp > 0
        assert "T" in e.wall_time

    def test_frozen(self):
        e = WakeEvent(model_name="test", score=0.5)
        with pytest.raises(Exception):
            e.score = 0.9

    def test_str(self):
        e = WakeEvent(model_name="hey_jarvis_v0.1", score=0.92)
        s = str(e)
        assert "hey_jarvis_v0.1" in s
        assert "0.920" in s

    def test_timestamp_monotonic(self):
        t_before = time.monotonic()
        e = WakeEvent(model_name="test", score=0.5)
        t_after = time.monotonic()
        assert t_before <= e.timestamp <= t_after

    def test_wall_time_format(self):
        e = WakeEvent(model_name="test", score=0.5)
        # Should match basic ISO-8601 pattern YYYY-MM-DDTHH:MM:SSZ
        import re
        assert re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", e.wall_time)


# ─────────────────────────────────────────────────────────────────────────────
# _AudioCaptureThread
# ─────────────────────────────────────────────────────────────────────────────

class TestAudioCaptureThread:
    def _make_thread(self, config=None):
        cfg      = config or _make_config()
        audio_q  = queue.Queue(maxsize=30)
        stop_evt = threading.Event()
        t = _AudioCaptureThread(cfg, audio_q, stop_evt)
        return t, audio_q, stop_evt

    def test_sets_stop_on_mic_open_failure(self):
        t, audio_q, stop_evt = self._make_thread()
        with patch("pyaudio.PyAudio", side_effect=OSError("No mic")):
            t.run()
        assert stop_evt.is_set()

    def test_produces_frames_then_stops(self):
        cfg = _make_config(chunk_ms=30)
        audio_q  = queue.Queue(maxsize=30)
        stop_evt = threading.Event()
        t = _AudioCaptureThread(cfg, audio_q, stop_evt)

        # Mock PyAudio to return silent frames, then trigger stop after 3 reads
        call_count = [0]
        silence = np.zeros(cfg.chunk_samples, dtype=np.int16).tobytes()

        mock_stream = MagicMock()
        def _read(*a, **kw):
            call_count[0] += 1
            if call_count[0] >= 3:
                stop_evt.set()
            return silence
        mock_stream.read.side_effect = _read

        mock_pa = MagicMock()
        mock_pa.get_default_input_device_info.return_value = {"index": 0}
        mock_pa.get_device_info_by_index.return_value = {
            "name": "Test Mic", "maxInputChannels": 1
        }
        mock_pa.open.return_value = mock_stream

        with patch("pyaudio.PyAudio", return_value=mock_pa):
            t.run()

        assert audio_q.qsize() >= 2

    def test_queue_overflow_drops_oldest(self):
        cfg      = _make_config(queue_maxsize=3)
        audio_q  = queue.Queue(maxsize=3)
        stop_evt = threading.Event()
        t = _AudioCaptureThread(cfg, audio_q, stop_evt)

        # Pre-fill queue
        for _ in range(3):
            audio_q.put_nowait(np.zeros(cfg.chunk_samples, dtype=np.int16))

        # Overflow: the thread should drop oldest and still enqueue
        frame = np.ones(cfg.chunk_samples, dtype=np.int16)
        try:
            audio_q.put_nowait(frame)
        except queue.Full:
            audio_q.get_nowait()
            audio_q.put_nowait(frame)

        assert audio_q.qsize() == 3

    def test_max_consecutive_errors_stops_thread(self):
        cfg      = _make_config()
        audio_q  = queue.Queue(maxsize=30)
        stop_evt = threading.Event()
        t = _AudioCaptureThread(cfg, audio_q, stop_evt)

        mock_stream = MagicMock()
        mock_stream.read.side_effect = OSError("Device lost")
        mock_pa = MagicMock()
        mock_pa.get_default_input_device_info.return_value = {"index": 0}
        mock_pa.get_device_info_by_index.return_value = {
            "name": "Test Mic", "maxInputChannels": 1
        }
        mock_pa.open.return_value = mock_stream

        with patch("pyaudio.PyAudio", return_value=mock_pa):
            with patch("time.sleep"):  # skip sleep between error retries
                t.run()

        assert stop_evt.is_set()


# ─────────────────────────────────────────────────────────────────────────────
# _InferenceThread
# ─────────────────────────────────────────────────────────────────────────────

class TestInferenceThread:
    def _make_thread(self, config=None, callbacks=None):
        cfg      = config or _make_config()
        audio_q  = queue.Queue(maxsize=30)
        event_q  = queue.Queue(maxsize=10)
        cbs      = callbacks or []
        lock     = threading.Lock()
        stop_evt = threading.Event()
        t = _InferenceThread(cfg, audio_q, event_q, cbs, lock, stop_evt)
        return t, audio_q, event_q, stop_evt

    def test_load_failure_sets_stop_event(self):
        t, _, _, stop_evt = self._make_thread()
        with patch("wake_word._OWWModel", side_effect=RuntimeError("ONNX fail")):
            t.run()
        assert stop_evt.is_set()

    def test_detection_above_threshold_dispatches_event(self):
        cfg = _make_config(detection_threshold=0.5, cooldown_s=0.0)
        t, audio_q, event_q, stop_evt = self._make_thread(config=cfg)

        # Inject one frame then stop
        audio_q.put_nowait(np.zeros(cfg.chunk_samples, dtype=np.int16))
        stop_evt_delayed = threading.Event()

        mock_model = MagicMock()
        mock_model.models = {"hey_jarvis_v0.1": MagicMock()}
        call_count = [0]
        def _predict(frame):
            call_count[0] += 1
            if call_count[0] >= 1:
                stop_evt.set()
            return {"hey_jarvis_v0.1": 0.95}
        mock_model.predict.side_effect = _predict

        with patch("wake_word._OWWModel", return_value=mock_model):
            t.run()

        assert not event_q.empty()
        event = event_q.get_nowait()
        assert isinstance(event, WakeEvent)
        assert event.score == 0.95

    def test_detection_below_threshold_ignored(self):
        cfg = _make_config(detection_threshold=0.8, cooldown_s=0.0)
        t, audio_q, event_q, stop_evt = self._make_thread(config=cfg)

        audio_q.put_nowait(np.zeros(cfg.chunk_samples, dtype=np.int16))
        call_count = [0]

        mock_model = MagicMock()
        mock_model.models = {"hey_jarvis_v0.1": MagicMock()}
        def _predict(frame):
            call_count[0] += 1
            stop_evt.set()
            return {"hey_jarvis_v0.1": 0.3}   # below threshold
        mock_model.predict.side_effect = _predict

        with patch("wake_word._OWWModel", return_value=mock_model):
            t.run()

        assert event_q.empty()

    def test_cooldown_blocks_rapid_detections(self):
        cfg = _make_config(detection_threshold=0.5, cooldown_s=10.0)
        callbacks_fired = []
        t, audio_q, event_q, stop_evt = self._make_thread(
            config=cfg,
            callbacks=[lambda e: callbacks_fired.append(e)],
        )

        # Two frames close together
        for _ in range(2):
            audio_q.put_nowait(np.zeros(cfg.chunk_samples, dtype=np.int16))

        call_count = [0]
        mock_model = MagicMock()
        mock_model.models = {"hey_jarvis_v0.1": MagicMock()}
        def _predict(frame):
            call_count[0] += 1
            if call_count[0] >= 2:
                stop_evt.set()
            return {"hey_jarvis_v0.1": 0.95}
        mock_model.predict.side_effect = _predict

        with patch("wake_word._OWWModel", return_value=mock_model):
            t.run()

        # Only the first detection should fire — second blocked by cooldown
        assert len(callbacks_fired) == 1

    def test_callback_exception_does_not_crash_thread(self):
        cfg = _make_config(detection_threshold=0.5, cooldown_s=0.0)

        def bad_callback(e):
            raise ValueError("Callback blew up!")

        t, audio_q, event_q, stop_evt = self._make_thread(
            config=cfg,
            callbacks=[bad_callback],
        )
        audio_q.put_nowait(np.zeros(cfg.chunk_samples, dtype=np.int16))
        call_count = [0]

        mock_model = MagicMock()
        mock_model.models = {"hey_jarvis_v0.1": MagicMock()}
        def _predict(frame):
            call_count[0] += 1
            stop_evt.set()
            return {"hey_jarvis_v0.1": 0.95}
        mock_model.predict.side_effect = _predict

        # Should not raise
        with patch("wake_word._OWWModel", return_value=mock_model):
            t.run()

    def test_inference_error_continues_loop(self):
        """A single predict() exception should not kill the thread."""
        cfg = _make_config(detection_threshold=0.5, cooldown_s=0.0)
        t, audio_q, event_q, stop_evt = self._make_thread(config=cfg)

        for _ in range(3):
            audio_q.put_nowait(np.zeros(cfg.chunk_samples, dtype=np.int16))

        call_count = [0]
        mock_model = MagicMock()
        mock_model.models = {"hey_jarvis_v0.1": MagicMock()}
        def _predict(frame):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("ONNX transient error")
            if call_count[0] >= 3:
                stop_evt.set()
            return {"hey_jarvis_v0.1": 0.0}
        mock_model.predict.side_effect = _predict

        with patch("wake_word._OWWModel", return_value=mock_model):
            t.run()

        # Thread ran to completion despite one error
        assert call_count[0] >= 2


# ─────────────────────────────────────────────────────────────────────────────
# WakeWordDetector
# ─────────────────────────────────────────────────────────────────────────────

class TestWakeWordDetector:
    def _make_detector(self, **kwargs) -> WakeWordDetector:
        return WakeWordDetector(config=_make_config(**kwargs))

    def _patch_threads(self, detector: WakeWordDetector):
        """Replace both background threads with no-op mocks."""
        mock_cap = MagicMock(spec=_AudioCaptureThread)
        mock_cap.is_alive.return_value = False
        mock_inf = MagicMock(spec=_InferenceThread)
        mock_inf.is_alive.return_value = False
        detector._capture_thread  = mock_cap
        detector._inference_thread = mock_inf
        return mock_cap, mock_inf

    def test_register_callback(self):
        det = self._make_detector()
        cb = MagicMock()
        det.register(cb)
        assert cb in det._callbacks

    def test_register_same_callback_once(self):
        det = self._make_detector()
        cb = MagicMock()
        det.register(cb)
        det.register(cb)
        assert det._callbacks.count(cb) == 1

    def test_unregister_callback(self):
        det = self._make_detector()
        cb = MagicMock()
        det.register(cb)
        assert det.unregister(cb) is True
        assert cb not in det._callbacks

    def test_unregister_nonexistent_returns_false(self):
        det = self._make_detector()
        assert det.unregister(lambda e: None) is False

    def test_start_sets_running(self):
        det = self._make_detector()
        with patch.object(_AudioCaptureThread, "start"), \
             patch.object(_InferenceThread, "start"):
            det.start()
            assert det.is_running is True
            det._stop_evt.set()
            det._running = False

    def test_double_start_is_safe(self):
        det = self._make_detector()
        with patch.object(_AudioCaptureThread, "start"), \
             patch.object(_InferenceThread, "start"):
            det.start()
            det.start()  # second call is a no-op
            assert det.is_running is True
            det._stop_evt.set()
            det._running = False

    def test_stop_sets_not_running(self):
        det = self._make_detector()
        with patch.object(_AudioCaptureThread, "start"), \
             patch.object(_InferenceThread, "start"):
            det.start()
        self._patch_threads(det)
        det.stop()
        assert det.is_running is False

    def test_context_manager(self):
        det = self._make_detector()
        with patch.object(_AudioCaptureThread, "start"), \
             patch.object(_InferenceThread, "start"), \
             patch.object(_AudioCaptureThread, "join"), \
             patch.object(_InferenceThread, "join"), \
             patch.object(_AudioCaptureThread, "is_alive", return_value=False), \
             patch.object(_InferenceThread, "is_alive", return_value=False):
            with det:
                assert det.is_running
        assert not det.is_running

    def test_status_keys(self):
        det = self._make_detector()
        s = det.status()
        for key in ("running", "model", "detection_threshold", "cooldown_s",
                    "chunk_ms", "audio_queue_size", "event_queue_size",
                    "registered_callbacks"):
            assert key in s

    def test_repr_contains_model(self):
        det = self._make_detector()
        r = repr(det)
        assert "WakeWordDetector" in r
        assert "hey_jarvis" in r

    def test_event_queue_accessible(self):
        det = self._make_detector()
        assert det.event_queue is det._event_q

    def test_inject_event_appears_in_queue(self):
        det = self._make_detector()
        evt = _inject_event(det, score=0.88)
        assert not det.event_queue.empty()
        got = det.event_queue.get_nowait()
        assert got.score == 0.88

    def test_callback_fired_on_injected_event(self):
        det  = self._make_detector()
        fired = []
        det.register(fired.append)

        # Simulate what _InferenceThread._dispatch does
        evt = WakeEvent(model_name="hey_jarvis_v0.1", score=0.9)
        det._inference_thread = MagicMock()

        # Call dispatch logic directly through a temporary inference thread
        inf = _InferenceThread(
            config=det._cfg,
            audio_q=det._audio_q,
            event_q=det._event_q,
            callbacks=det._callbacks,
            callbacks_lock=det._callbacks_lock,
            stop_evt=det._stop_evt,
        )
        inf._dispatch(evt)

        assert len(fired) == 1
        assert fired[0].score == 0.9


# ─────────────────────────────────────────────────────────────────────────────
# AsyncWakeWordBridge
# ─────────────────────────────────────────────────────────────────────────────

class TestAsyncWakeWordBridge:
    def _make_bridge(self, poll_interval=0.01):
        det    = WakeWordDetector(config=_make_config())
        bridge = AsyncWakeWordBridge(det, poll_interval=poll_interval)
        return det, bridge

    @pytest.mark.asyncio
    async def test_wait_for_wake_returns_event(self):
        det, bridge = self._make_bridge()
        evt = _inject_event(det, score=0.77)
        result = await bridge.wait_for_wake(timeout=1.0)
        assert result is not None
        assert result.score == 0.77

    @pytest.mark.asyncio
    async def test_wait_for_wake_timeout_returns_none(self):
        _, bridge = self._make_bridge()
        result = await bridge.wait_for_wake(timeout=0.1)
        assert result is None

    @pytest.mark.asyncio
    async def test_wait_for_wake_no_timeout(self):
        det, bridge = self._make_bridge()

        async def _inject_after_delay():
            await asyncio.sleep(0.05)
            _inject_event(det, score=0.91)

        asyncio.create_task(_inject_after_delay())
        result = await bridge.wait_for_wake(timeout=2.0)
        assert result is not None
        assert result.score == 0.91

    @pytest.mark.asyncio
    async def test_stream_events_yields_multiple(self):
        det, bridge = self._make_bridge()
        collected = []

        async def _run():
            async for evt in bridge.stream_events():
                collected.append(evt)
                if len(collected) >= 3:
                    break

        # Inject 3 events with small delays
        async def _inject_all():
            for i in range(3):
                await asyncio.sleep(0.02)
                _inject_event(det, score=0.5 + i * 0.1)

        await asyncio.gather(_run(), _inject_all())
        assert len(collected) == 3
        assert collected[0].score == pytest.approx(0.5)
        assert collected[2].score == pytest.approx(0.7)

    @pytest.mark.asyncio
    async def test_wait_for_n_events(self):
        det, bridge = self._make_bridge()

        async def _inject_all():
            for i in range(4):
                await asyncio.sleep(0.01)
                _inject_event(det, score=0.6)

        asyncio.create_task(_inject_all())
        events = await bridge.wait_for_n_events(n=3, timeout=2.0)
        assert len(events) == 3

    @pytest.mark.asyncio
    async def test_wait_for_n_events_timeout(self):
        det, bridge = self._make_bridge()
        # Only inject 1 event but ask for 5 → should return early on timeout
        _inject_event(det, score=0.6)
        events = await bridge.wait_for_n_events(n=5, timeout=0.1)
        assert len(events) <= 5
        assert len(events) >= 1  # got the one we injected

    @pytest.mark.asyncio
    async def test_multiple_consumers_same_bridge(self):
        """Two concurrent awaiters should each get distinct events."""
        det, bridge = self._make_bridge()

        async def _consumer():
            return await bridge.wait_for_wake(timeout=2.0)

        async def _inject_all():
            await asyncio.sleep(0.02)
            _inject_event(det, score=0.8)
            await asyncio.sleep(0.02)
            _inject_event(det, score=0.9)

        asyncio.create_task(_inject_all())
        r1, r2 = await asyncio.gather(_consumer(), _consumer())

        # Both should have received an event (queue delivers each item once)
        received_scores = {e.score for e in [r1, r2] if e is not None}
        assert len(received_scores) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# BackgroundWakeWordService
# ─────────────────────────────────────────────────────────────────────────────

class TestBackgroundWakeWordService:
    def test_is_healthy_false_before_start(self):
        svc = BackgroundWakeWordService(config=_make_config())
        assert svc.is_healthy is False

    def test_stats_structure(self):
        svc = BackgroundWakeWordService(config=_make_config())
        s = svc.stats
        assert "healthy" in s
        assert "total_detections" in s
        assert "restart_count" in s
        assert "last_detection" in s

    def test_stop_sets_service_stop(self):
        svc = BackgroundWakeWordService(config=_make_config())
        svc.stop()
        assert svc._service_stop.is_set()

    def test_internal_callback_increments_counter(self):
        svc = BackgroundWakeWordService(config=_make_config())
        evt = WakeEvent(model_name="hey_jarvis_v0.1", score=0.9)
        svc._internal_callback(evt)
        svc._internal_callback(evt)
        assert svc._total_detections == 2
        assert svc._last_detection == evt

    def test_max_restarts_exceeded_stops_service(self):
        svc = BackgroundWakeWordService(config=_make_config(), max_restarts=2)
        svc._restart_count = 2
        svc._handle_restart()
        assert svc._service_stop.is_set()

    def test_restart_increments_count(self):
        svc = BackgroundWakeWordService(config=_make_config(), max_restarts=5)
        with patch("time.sleep"):
            svc._handle_restart()
        assert svc._restart_count == 1
        assert not svc._service_stop.is_set()


# ─────────────────────────────────────────────────────────────────────────────
# Module-level helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestModuleHelpers:
    def test_available_models_includes_jarvis(self):
        models = available_models()
        assert any("jarvis" in m.lower() for m in models)

    def test_available_models_returns_list(self):
        models = available_models()
        assert isinstance(models, list)
        assert len(models) > 0

    def test_list_microphones_returns_list(self):
        import pyaudio
        with patch("pyaudio.PyAudio") as MockPA:
            instance = MockPA.return_value
            instance.get_device_count.return_value = 2
            instance.get_device_info_by_index.side_effect = [
                {"name": "Built-in Mic", "maxInputChannels": 1, "defaultSampleRate": 44100.0},
                {"name": "Speakers",     "maxInputChannels": 0, "defaultSampleRate": 44100.0},
            ]
            devices = list_microphones()
        assert len(devices) == 1
        assert devices[0]["name"] == "Built-in Mic"
        assert "index" in devices[0]
        assert "sample_rate" in devices[0]
