"""
jarvis_integration.py — Full JARVIS Pipeline Example
Faster-Whisper → Ollama → TTS Engine
"""

import asyncio
import logging
import sys
from typing import AsyncIterator

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("JARVIS")


# ─────────────────────────────────────────────────────────────────────────────
# Whisper STT  (faster-whisper)
# ─────────────────────────────────────────────────────────────────────────────
class WhisperSTT:
    """
    Thin async wrapper around faster-whisper.
    Runs transcription in an executor to stay non-blocking.
    """

    def __init__(self, model_size: str = "base", device: str = "cpu", compute_type: str = "int8"):
        from faster_whisper import WhisperModel
        self._model = WhisperModel(model_size, device=device, compute_type=compute_type)
        logger.info("WhisperSTT ready (model=%s, device=%s).", model_size, device)

    async def transcribe(self, audio_path: str) -> str:
        loop = asyncio.get_event_loop()
        text = await loop.run_in_executor(None, self._transcribe_sync, audio_path)
        return text

    def _transcribe_sync(self, audio_path: str) -> str:
        segments, _ = self._model.transcribe(audio_path, beam_size=5)
        result = " ".join(seg.text.strip() for seg in segments)
        logger.info("Transcribed: %s", result)
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Ollama LLM
# ─────────────────────────────────────────────────────────────────────────────
class OllamaLLM:
    """
    Async Ollama client that yields tokens for streaming TTS.
    """

    def __init__(self, model: str = "mistral", host: str = "http://localhost:11434"):
        self._model = model
        self._host  = host
        logger.info("OllamaLLM ready (model=%s).", model)

    async def stream(self, prompt: str) -> AsyncIterator[str]:
        """Yield tokens from Ollama streaming response."""
        import httpx, json
        url = f"{self._host}/api/generate"
        payload = {"model": self._model, "prompt": prompt, "stream": True}

        async with httpx.AsyncClient(timeout=60) as client:
            async with client.stream("POST", url, json=payload) as response:
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                        token = chunk.get("response", "")
                        if token:
                            yield token
                        if chunk.get("done"):
                            break
                    except json.JSONDecodeError:
                        continue


# ─────────────────────────────────────────────────────────────────────────────
# JARVIS Core
# ─────────────────────────────────────────────────────────────────────────────
class JARVIS:
    """
    Main JARVIS assistant pipeline.
    STT → LLM → TTS, all async, all interruptible.
    """

    SYSTEM_PROMPT = (
        "You are JARVIS, a sophisticated AI assistant. "
        "Respond concisely and naturally. Avoid markdown formatting."
    )

    def __init__(
        self,
        whisper_model:  str = "base",
        ollama_model:   str = "mistral",
        ollama_host:    str = "http://localhost:11434",
        tts_voice:      str = "JARVIS",
    ):
        from tts_engine import VoiceProfile, get_engine
        from speech_manager import SpeechManager

        voice = VoiceProfile[tts_voice.upper()]
        self._engine  = get_engine(default_voice=voice)
        self._speech  = SpeechManager(engine=self._engine, default_voice=voice)
        self._stt     = WhisperSTT(model_size=whisper_model)
        self._llm     = OllamaLLM(model=ollama_model, host=ollama_host)
        self._history = [{"role": "system", "content": self.SYSTEM_PROMPT}]
        logger.info("JARVIS initialised.")

    async def greet(self):
        await self._speech.say("JARVIS online. All systems nominal. How may I assist?")

    async def process_audio(self, audio_path: str):
        """Full pipeline: audio file → speech output."""
        # 1. Transcribe
        user_text = await self._stt.transcribe(audio_path)
        if not user_text.strip():
            await self._speech.say("I didn't catch that. Could you repeat?")
            return

        logger.info("User: %s", user_text)

        # 2. Build prompt with history
        self._history.append({"role": "user", "content": user_text})
        full_prompt = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in self._history
        ) + "\nASSISTANT:"

        # 3. Stream LLM → TTS
        collected = []
        async with self._speech:
            async for token in self._llm.stream(full_prompt):
                collected.append(token)
                await self._speech.feed_token(token)

        response = "".join(collected).strip()
        self._history.append({"role": "assistant", "content": response})
        logger.info("JARVIS: %s", response)

    async def process_text(self, user_text: str):
        """Direct text input pipeline (skip STT)."""
        self._history.append({"role": "user", "content": user_text})
        full_prompt = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in self._history
        ) + "\nASSISTANT:"

        collected = []
        async with self._speech:
            async for token in self._llm.stream(full_prompt):
                collected.append(token)
                await self._speech.feed_token(token)

        response = "".join(collected).strip()
        self._history.append({"role": "assistant", "content": response})

    async def interrupt(self):
        await self._speech.interrupt()

    async def shutdown(self):
        await self._speech.clear()
        self._engine.stop()
        logger.info("JARVIS shutdown complete.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI demo
# ─────────────────────────────────────────────────────────────────────────────
async def _demo():
    jarvis = JARVIS(whisper_model="base", ollama_model="mistral")
    await jarvis.greet()

    print("\nType a message (or 'quit' to exit, 'interrupt' to stop speech):")
    while True:
        try:
            user_input = await asyncio.get_event_loop().run_in_executor(None, input, "> ")
        except (EOFError, KeyboardInterrupt):
            break

        if user_input.lower() in ("quit", "exit"):
            break
        elif user_input.lower() == "interrupt":
            await jarvis.interrupt()
        else:
            await jarvis.process_text(user_input)

    await jarvis.shutdown()


if __name__ == "__main__":
    asyncio.run(_demo())
