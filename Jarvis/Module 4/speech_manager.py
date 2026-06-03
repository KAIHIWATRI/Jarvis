"""
speech_manager.py — JARVIS Async Speech Manager
High-level async API wrapping TTSEngine.
Integrates cleanly with Faster-Whisper, Ollama, and CustomTkinter.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator, Callable, Optional

from tts_engine import Priority, TTSEngine, VoiceProfile, get_engine

logger = logging.getLogger("JARVIS.SpeechManager")


class SpeechManager:
    """
    High-level async speech manager.

    Designed to sit between Ollama/Whisper and the TTS engine.
    Supports streaming text (token-by-token), sentence batching,
    and clean lifecycle management.

    Example
    -------
    async with SpeechManager() as sm:
        await sm.say("Hello, how can I help?")
        # Stream Ollama tokens:
        async for token in ollama_stream:
            await sm.feed_token(token)
        await sm.flush_tokens()
    """

    # Punctuation that signals a sentence boundary for streaming
    SENTENCE_ENDINGS = {'.', '!', '?', ':', '\n'}

    def __init__(
        self,
        engine: Optional[TTSEngine] = None,
        default_voice: VoiceProfile = VoiceProfile.JARVIS,
        stream_min_chars: int = 60,
    ):
        self._engine        = engine or get_engine(default_voice=default_voice)
        self.default_voice  = default_voice
        self._stream_buffer = ""
        self._stream_min    = stream_min_chars
        self._lock          = asyncio.Lock()
        logger.info("SpeechManager ready (voice=%s).", default_voice.name)

    # ── Lifecycle ─────────────────────────────

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        await self.flush_tokens()

    # ── Core API ──────────────────────────────

    async def say(
        self,
        text: str,
        priority: Priority = Priority.NORMAL,
        voice: Optional[VoiceProfile] = None,
        on_done: Optional[Callable] = None,
    ):
        """Enqueue a complete utterance (non-blocking)."""
        if not text.strip():
            return
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: self._engine.speak(
                text,
                priority=priority,
                voice=voice or self.default_voice,
                on_done=on_done,
            ),
        )
        logger.debug("say() enqueued: %.50s…", text)

    async def say_urgent(self, text: str, voice: Optional[VoiceProfile] = None):
        """Interrupt current speech and say this immediately."""
        await self.say(text, priority=Priority.URGENT, voice=voice)

    async def interrupt(self):
        """Interrupt current playback."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._engine.interrupt)

    async def clear(self):
        """Clear queue and stop current speech."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._engine.clear_queue)
        await self.interrupt()

    # ── Streaming (Ollama token feed) ─────────

    async def feed_token(self, token: str):
        """
        Feed a single LLM token into the stream buffer.
        Flushes automatically at sentence boundaries or min_chars threshold.
        """
        async with self._lock:
            self._stream_buffer += token
            should_flush = (
                any(self._stream_buffer.rstrip().endswith(p) for p in self.SENTENCE_ENDINGS)
                and len(self._stream_buffer) >= self._stream_min
            )
            if should_flush:
                await self._flush_locked()

    async def flush_tokens(self):
        """Force-flush any remaining buffered tokens."""
        async with self._lock:
            await self._flush_locked()

    async def _flush_locked(self):
        """Internal flush — must be called under self._lock."""
        text = self._stream_buffer.strip()
        self._stream_buffer = ""
        if text:
            await self.say(text)
            logger.debug("Stream flush: %.50s…", text)

    # ── Convenience wrappers ──────────────────

    async def speak_response(self, ollama_stream: AsyncIterator[str]):
        """
        Consume an Ollama async token stream and speak it sentence-by-sentence.

        Usage:
            stream = ollama.chat(model='mistral', messages=[...], stream=True)
            await sm.speak_response(token['message']['content'] for token in stream)
        """
        async for token in ollama_stream:
            await self.feed_token(token)
        await self.flush_tokens()

    @property
    def is_speaking(self) -> bool:
        from tts_engine import EngineState
        return self._engine.state in (EngineState.SYNTHESISING, EngineState.PLAYING)

    @property
    def queue_size(self) -> int:
        return self._engine.queue_size

    def set_volume(self, volume: float):
        self._engine.set_volume(volume)

    def set_voice(self, voice: VoiceProfile):
        self.default_voice = voice
        logger.info("Default voice changed to %s.", voice.name)
