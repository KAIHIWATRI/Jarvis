"""
JARVIS Brain Module
===================
CPU-optimised async LLM interface for Ollama / Llama 3 8B.
Designed for AMD Ryzen 7 3700U · 16 GB RAM.

Architecture
------------
  BrainConfig        – immutable runtime settings
  PromptManager      – system prompt + message assembly
  ContextMemory      – rolling conversation window with token budgeting
  OllamaClient       – low-level async HTTP wrapper around the Ollama API
  Brain              – public façade; orchestrates all components

Usage (quickstart)
------------------
    import asyncio
    from brain import Brain

    async def main():
        brain = Brain()
        await brain.initialise()

        async for token in brain.think("What is the speed of light?"):
            print(token, end="", flush=True)

    asyncio.run(main())
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator, Optional

import httpx

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _build_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    """Return a module-scoped logger with console + rotating-file handlers."""
    import logging.handlers, os

    logger = logging.getLogger(name)
    if logger.handlers:          # already configured in this process
        return logger

    logger.setLevel(level)
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File (rotates at 5 MB, keeps 3 backups)
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)
    fh = logging.handlers.RotatingFileHandler(
        filename=os.path.join(log_dir, "brain.log"),
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


log = _build_logger("jarvis.brain")


# ---------------------------------------------------------------------------
# Enums & constants
# ---------------------------------------------------------------------------

class Role(str, Enum):
    SYSTEM    = "system"
    USER      = "user"
    ASSISTANT = "assistant"


# ---------------------------------------------------------------------------
# BrainConfig
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BrainConfig:
    """
    All tunables in one place.  Frozen so accidental mutation raises immediately.

    Ryzen 7 3700U optimisation notes
    ---------------------------------
    - model: Q4_K_M quantisation → ~5 GB RSS, ~10 tok/s on 3700U
    - num_ctx: 2048 keeps KV-cache small; raise to 4096 only if you need
      long documents (costs ~400 MB extra RAM per 2 k tokens)
    - num_thread: 4 leaves 4 logical cores free for the OS and GUI
    - temperature: 0.7 is a good creative/factual balance
    """

    # Ollama server
    ollama_base_url: str  = "http://localhost:11434"
    model_name:      str  = "llama3:8b-instruct-q4_K_M"

    # Inference — Ryzen 7 3700U sweet-spot values
    num_ctx:         int   = 2048        # context window (tokens)
    num_thread:      int   = 4           # CPU threads for inference
    num_predict:     int   = 512         # max tokens per response
    temperature:     float = 0.7
    top_p:           float = 0.9
    top_k:           int   = 40
    repeat_penalty:  float = 1.1

    # Context memory
    max_history_turns: int = 10          # user+assistant turn pairs kept
    system_prompt:     str = (
        "You are JARVIS, an intelligent personal AI assistant running locally "
        "on the user's machine. You are fast, concise, and helpful. "
        "You never fabricate facts. When unsure, say so briefly. "
        "Respond in plain, natural language unless the user asks for code or lists."
    )

    # HTTP client
    request_timeout:  float = 120.0      # seconds — generous for first-token latency
    connect_timeout:  float = 10.0


# ---------------------------------------------------------------------------
# PromptManager
# ---------------------------------------------------------------------------

class PromptManager:
    """
    Builds the messages array that Ollama's /api/chat endpoint expects.

    The Llama 3 instruct template is applied automatically by Ollama when
    you use the /api/chat route, so we just produce a clean messages list.
    """

    def __init__(self, config: BrainConfig) -> None:
        self._config = config

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def build_messages(self, history: list[dict]) -> list[dict]:
        """
        Prepend the system prompt to a history list and return the full
        messages array ready for the API call.

        Args:
            history: list of {"role": str, "content": str} dicts

        Returns:
            Full messages array with system message at index 0.
        """
        system_msg = {
            "role":    Role.SYSTEM.value,
            "content": self._config.system_prompt,
        }
        return [system_msg, *history]

    def build_options(self) -> dict:
        """Return the Ollama model-options dict from config."""
        return {
            "num_ctx":        self._config.num_ctx,
            "num_thread":     self._config.num_thread,
            "num_predict":    self._config.num_predict,
            "temperature":    self._config.temperature,
            "top_p":          self._config.top_p,
            "top_k":          self._config.top_k,
            "repeat_penalty": self._config.repeat_penalty,
        }

    @staticmethod
    def user_message(text: str) -> dict:
        return {"role": Role.USER.value, "content": text.strip()}

    @staticmethod
    def assistant_message(text: str) -> dict:
        return {"role": Role.ASSISTANT.value, "content": text.strip()}


# ---------------------------------------------------------------------------
# ContextMemory
# ---------------------------------------------------------------------------

class ContextMemory:
    """
    Rolling conversation window.

    Keeps up to `max_turns` user+assistant pairs.  The system prompt is
    NOT stored here — PromptManager injects it at call time.

    Thread-safety: this class is not thread-safe by design; it is always
    accessed from the asyncio event loop (single-threaded for this module).
    """

    def __init__(self, max_turns: int = 10) -> None:
        self._max_turns = max_turns
        self._history:   list[dict] = []
        self._turn_count: int        = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def history(self) -> list[dict]:
        """Read-only view of the current history."""
        return list(self._history)

    @property
    def turn_count(self) -> int:
        return self._turn_count

    @property
    def is_empty(self) -> bool:
        return len(self._history) == 0

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_user(self, text: str) -> None:
        self._history.append(PromptManager.user_message(text))

    def add_assistant(self, text: str) -> None:
        self._history.append(PromptManager.assistant_message(text))
        self._turn_count += 1
        self._trim()

    def clear(self) -> None:
        self._history.clear()
        self._turn_count = 0
        log.info("Context memory cleared.")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _trim(self) -> None:
        """
        Remove oldest turn-pairs when history exceeds the limit.
        Each pair = 2 messages (user + assistant).
        """
        max_messages = self._max_turns * 2
        if len(self._history) > max_messages:
            excess = len(self._history) - max_messages
            self._history = self._history[excess:]
            log.debug("Context trimmed: removed %d messages, %d remain.",
                      excess, len(self._history))

    def __repr__(self) -> str:
        return (f"ContextMemory(turns={self._turn_count}, "
                f"messages={len(self._history)}, max={self._max_turns})")


# ---------------------------------------------------------------------------
# OllamaClient
# ---------------------------------------------------------------------------

class OllamaClient:
    """
    Thin async wrapper around the Ollama REST API.

    Endpoints used
    --------------
    GET  /api/tags              → list available models
    POST /api/pull              → download a model
    POST /api/chat              → streaming chat completions
    GET  /api/version           → server version / health check

    The client owns its httpx.AsyncClient lifecycle.  Call `aclose()` when
    done, or use as an async context manager.
    """

    def __init__(self, config: BrainConfig) -> None:
        self._config = config
        self._client: Optional[httpx.AsyncClient] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Create the underlying HTTP client."""
        if self._client and not self._client.is_closed:
            return
        timeout = httpx.Timeout(
            connect=self._config.connect_timeout,
            read=self._config.request_timeout,
            write=30.0,
            pool=5.0,
        )
        self._client = httpx.AsyncClient(
            base_url=self._config.ollama_base_url,
            timeout=timeout,
            headers={"Content-Type": "application/json"},
        )
        log.debug("OllamaClient connected to %s", self._config.ollama_base_url)

    async def aclose(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            log.debug("OllamaClient closed.")

    async def __aenter__(self) -> "OllamaClient":
        await self.connect()
        return self

    async def __aexit__(self, *_) -> None:
        await self.aclose()

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    async def health_check(self) -> bool:
        """Return True if the Ollama server is reachable."""
        try:
            r = await self._client.get("/api/version")
            r.raise_for_status()
            version = r.json().get("version", "unknown")
            log.info("Ollama server healthy — version %s", version)
            return True
        except Exception as exc:
            log.error("Ollama health check failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------

    async def list_models(self) -> list[str]:
        """Return the names of all locally available models."""
        r = await self._client.get("/api/tags")
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]

    async def model_is_available(self, model_name: str) -> bool:
        models = await self.list_models()
        return any(m.startswith(model_name.split(":")[0]) for m in models)

    async def pull_model(self, model_name: str) -> None:
        """Pull a model if not already cached.  Streams progress to the log."""
        log.info("Pulling model '%s' — this may take several minutes …", model_name)
        async with self._client.stream(
            "POST", "/api/pull",
            json={"name": model_name, "stream": True}
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line.strip():
                    try:
                        data = json.loads(line)
                        status = data.get("status", "")
                        if "total" in data and "completed" in data:
                            pct = data["completed"] / data["total"] * 100
                            log.debug("  pull %s: %.1f %%", status, pct)
                        else:
                            log.debug("  pull: %s", status)
                    except json.JSONDecodeError:
                        pass
        log.info("Model '%s' ready.", model_name)

    async def ensure_model(self, model_name: str) -> None:
        """Pull model only if it is not already available."""
        if not await self.model_is_available(model_name):
            log.warning("Model '%s' not found locally. Pulling …", model_name)
            await self.pull_model(model_name)
        else:
            log.info("Model '%s' already available.", model_name)

    # ------------------------------------------------------------------
    # Inference — streaming
    # ------------------------------------------------------------------

    async def chat_stream(
        self,
        messages: list[dict],
        options:  dict,
        model:    str,
    ) -> AsyncIterator[str]:
        """
        Yield individual response tokens as they arrive from Ollama.

        Args:
            messages: full messages array (system + history + user turn)
            options:  Ollama model options dict
            model:    model tag string

        Yields:
            str: one token (or sub-token chunk) at a time
        """
        payload = {
            "model":    model,
            "messages": messages,
            "options":  options,
            "stream":   True,
        }

        log.debug("chat_stream → model=%s ctx=%s", model, options.get("num_ctx", "?"))
        t_start = time.perf_counter()
        token_count = 0

        async with self._client.stream("POST", "/api/chat", json=payload) as resp:
            resp.raise_for_status()
            async for raw_line in resp.aiter_lines():
                if not raw_line.strip():
                    continue
                try:
                    chunk = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    log.warning("Malformed JSON chunk skipped: %s — %s", raw_line[:80], exc)
                    continue

                if chunk.get("done"):
                    elapsed = time.perf_counter() - t_start
                    tps = token_count / elapsed if elapsed > 0 else 0
                    log.info(
                        "Generation complete — %d tokens in %.2f s (%.1f tok/s)",
                        token_count, elapsed, tps,
                    )
                    break

                token = chunk.get("message", {}).get("content", "")
                if token:
                    token_count += 1
                    yield token


# ---------------------------------------------------------------------------
# Brain  (public façade)
# ---------------------------------------------------------------------------

class Brain:
    """
    Public interface to JARVIS intelligence.

    Typical lifecycle
    -----------------
        brain = Brain()                         # optionally: Brain(config=BrainConfig(...))
        await brain.initialise()                # connects, health-checks, ensures model
        async for tok in brain.think("Hi"):
            print(tok, end="", flush=True)
        await brain.shutdown()

    The `think()` method is the primary entry point.  It:
      1. Adds the user message to context memory.
      2. Assembles the full messages + options payload.
      3. Streams tokens from Ollama, yielding each immediately.
      4. Collects the full response and commits it to context memory.
    """

    def __init__(self, config: Optional[BrainConfig] = None) -> None:
        self._config  = config or BrainConfig()
        self._prompts = PromptManager(self._config)
        self._memory  = ContextMemory(self._config.max_history_turns)
        self._client  = OllamaClient(self._config)
        self._ready   = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialise(self) -> None:
        """
        Must be called once before `think()`.
        Connects the HTTP client, verifies the server, and pulls the model
        if needed.
        """
        log.info("Initialising JARVIS Brain …")
        await self._client.connect()

        if not await self._client.health_check():
            raise RuntimeError(
                "Cannot reach Ollama server at "
                f"{self._config.ollama_base_url}. "
                "Is `ollama serve` running?"
            )

        await self._client.ensure_model(self._config.model_name)
        self._ready = True
        log.info("Brain ready. Model: %s", self._config.model_name)

    async def shutdown(self) -> None:
        """Gracefully close the HTTP client."""
        await self._client.aclose()
        self._ready = False
        log.info("Brain shut down.")

    async def __aenter__(self) -> "Brain":
        await self.initialise()
        return self

    async def __aexit__(self, *_) -> None:
        await self.shutdown()

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    async def think(self, user_input: str) -> AsyncIterator[str]:
        """
        Stream a response to `user_input`.

        Args:
            user_input: the raw text from the user (or STT transcription)

        Yields:
            str: response tokens as they arrive from the model

        Raises:
            RuntimeError:   if `initialise()` has not been called
            httpx.HTTPError: on network or server errors
        """
        if not self._ready:
            raise RuntimeError("Brain.initialise() must be called before think().")

        user_input = user_input.strip()
        if not user_input:
            log.warning("think() called with empty input — ignoring.")
            return

        log.info("User: %s", user_input[:120])

        # Build payload
        self._memory.add_user(user_input)
        messages = self._prompts.build_messages(self._memory.history)
        options  = self._prompts.build_options()

        # Stream tokens
        full_response: list[str] = []
        try:
            async for token in self._client.chat_stream(
                messages=messages,
                options=options,
                model=self._config.model_name,
            ):
                full_response.append(token)
                yield token

        except httpx.HTTPStatusError as exc:
            log.error("Ollama HTTP error %d: %s", exc.response.status_code, exc)
            self._memory._history.pop()          # remove the un-answered user turn
            raise

        except httpx.RequestError as exc:
            log.error("Network error during generation: %s", exc)
            self._memory._history.pop()
            raise

        # Commit full response to memory
        if full_response:
            response_text = "".join(full_response)
            self._memory.add_assistant(response_text)
            log.debug("Assistant: %s …", response_text[:80])

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    async def think_complete(self, user_input: str) -> str:
        """
        Non-streaming convenience wrapper.  Waits for the full response
        and returns it as a single string.  Useful for skill integrations
        that need the complete answer before acting.
        """
        parts: list[str] = []
        async for token in self.think(user_input):
            parts.append(token)
        return "".join(parts)

    def clear_memory(self) -> None:
        """Reset the conversation context (start a fresh session)."""
        self._memory.clear()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def config(self) -> BrainConfig:
        return self._config

    @property
    def memory(self) -> ContextMemory:
        return self._memory

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def model_name(self) -> str:
        return self._config.model_name

    def status(self) -> dict:
        """Return a snapshot of the brain's current state."""
        return {
            "ready":       self._ready,
            "model":       self._config.model_name,
            "base_url":    self._config.ollama_base_url,
            "turn_count":  self._memory.turn_count,
            "history_len": len(self._memory.history),
            "max_turns":   self._config.max_history_turns,
            "num_ctx":     self._config.num_ctx,
            "num_thread":  self._config.num_thread,
        }

    def __repr__(self) -> str:
        return (
            f"Brain(model={self._config.model_name!r}, "
            f"ready={self._ready}, "
            f"turns={self._memory.turn_count})"
        )


# ---------------------------------------------------------------------------
# CLI demo  (python brain.py)
# ---------------------------------------------------------------------------

async def _demo() -> None:
    """Interactive REPL for quick local testing."""
    print("\n── JARVIS Brain Demo ──  (type 'quit' to exit, 'clear' to reset)\n")

    async with Brain() as brain:
        print(f"Model  : {brain.model_name}")
        print(f"Status : {brain.status()}\n")

        while True:
            try:
                user_input = input("You: ").strip()
            except (KeyboardInterrupt, EOFError):
                break

            if not user_input:
                continue
            if user_input.lower() == "quit":
                break
            if user_input.lower() == "clear":
                brain.clear_memory()
                print("[context cleared]\n")
                continue
            if user_input.lower() == "status":
                print(brain.status(), "\n")
                continue

            print("JARVIS: ", end="", flush=True)
            try:
                async for token in brain.think(user_input):
                    print(token, end="", flush=True)
                print("\n")
            except Exception as exc:
                print(f"\n[Error: {exc}]\n")

    print("\nGoodbye.")


if __name__ == "__main__":
    asyncio.run(_demo())
