"""
tests/test_brain.py
===================
Unit tests for the JARVIS Brain module.
Uses unittest.mock so no live Ollama server is required.

Run: pytest tests/test_brain.py -v
"""

from __future__ import annotations

import asyncio
import json
import sys
import os
import pytest

# Make brain importable from the parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from unittest.mock import AsyncMock, MagicMock, patch
from brain import (
    Brain,
    BrainConfig,
    ContextMemory,
    OllamaClient,
    PromptManager,
    Role,
)


# ---------------------------------------------------------------------------
# BrainConfig
# ---------------------------------------------------------------------------

class TestBrainConfig:
    def test_defaults(self):
        cfg = BrainConfig()
        assert cfg.model_name == "llama3:8b-instruct-q4_K_M"
        assert cfg.num_ctx == 2048
        assert cfg.num_thread == 4
        assert cfg.max_history_turns == 10

    def test_immutable(self):
        cfg = BrainConfig()
        with pytest.raises(Exception):
            cfg.model_name = "other-model"   # frozen dataclass

    def test_custom_values(self):
        cfg = BrainConfig(num_ctx=4096, num_thread=6)
        assert cfg.num_ctx == 4096
        assert cfg.num_thread == 6


# ---------------------------------------------------------------------------
# PromptManager
# ---------------------------------------------------------------------------

class TestPromptManager:
    def setup_method(self):
        self.cfg = BrainConfig()
        self.pm = PromptManager(self.cfg)

    def test_build_messages_prepends_system(self):
        history = [{"role": "user", "content": "hello"}]
        msgs = self.pm.build_messages(history)
        assert msgs[0]["role"] == Role.SYSTEM.value
        assert msgs[1]["role"] == "user"
        assert len(msgs) == 2

    def test_build_messages_empty_history(self):
        msgs = self.pm.build_messages([])
        assert len(msgs) == 1
        assert msgs[0]["role"] == "system"

    def test_build_options_keys(self):
        opts = self.pm.build_options()
        for key in ("num_ctx", "num_thread", "num_predict",
                    "temperature", "top_p", "top_k", "repeat_penalty"):
            assert key in opts

    def test_user_message_strips_whitespace(self):
        msg = PromptManager.user_message("  hello  ")
        assert msg["content"] == "hello"
        assert msg["role"] == "user"

    def test_assistant_message(self):
        msg = PromptManager.assistant_message("I am JARVIS")
        assert msg["role"] == "assistant"
        assert msg["content"] == "I am JARVIS"


# ---------------------------------------------------------------------------
# ContextMemory
# ---------------------------------------------------------------------------

class TestContextMemory:
    def test_empty_on_creation(self):
        mem = ContextMemory()
        assert mem.is_empty
        assert mem.turn_count == 0

    def test_add_user_and_assistant(self):
        mem = ContextMemory()
        mem.add_user("Hello")
        assert len(mem.history) == 1
        mem.add_assistant("Hi there")
        assert len(mem.history) == 2
        assert mem.turn_count == 1

    def test_trim_enforced(self):
        mem = ContextMemory(max_turns=2)
        for i in range(5):
            mem.add_user(f"msg {i}")
            mem.add_assistant(f"reply {i}")
        # max_turns=2 → at most 4 messages
        assert len(mem.history) <= 4

    def test_clear_resets_state(self):
        mem = ContextMemory()
        mem.add_user("hi")
        mem.add_assistant("hello")
        mem.clear()
        assert mem.is_empty
        assert mem.turn_count == 0

    def test_history_returns_copy(self):
        mem = ContextMemory()
        mem.add_user("hi")
        h = mem.history
        h.append({"role": "user", "content": "injected"})
        assert len(mem.history) == 1  # original unchanged


# ---------------------------------------------------------------------------
# OllamaClient (mocked)
# ---------------------------------------------------------------------------

class TestOllamaClient:
    def setup_method(self):
        self.cfg = BrainConfig()

    @pytest.mark.asyncio
    async def test_health_check_success(self):
        client = OllamaClient(self.cfg)
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"version": "0.3.0"}

        mock_http = AsyncMock()
        mock_http.get = AsyncMock(return_value=mock_response)
        client._client = mock_http

        result = await client.health_check()
        assert result is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self):
        import httpx
        client = OllamaClient(self.cfg)
        mock_http = AsyncMock()
        mock_http.get = AsyncMock(side_effect=httpx.RequestError("refused"))
        client._client = mock_http

        result = await client.health_check()
        assert result is False

    @pytest.mark.asyncio
    async def test_list_models(self):
        client = OllamaClient(self.cfg)
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "models": [{"name": "llama3:8b"}, {"name": "mistral:7b"}]
        }
        mock_http = AsyncMock()
        mock_http.get = AsyncMock(return_value=mock_response)
        client._client = mock_http

        names = await client.list_models()
        assert "llama3:8b" in names
        assert "mistral:7b" in names

    @pytest.mark.asyncio
    async def test_model_is_available_true(self):
        client = OllamaClient(self.cfg)
        with patch.object(client, "list_models", AsyncMock(return_value=["llama3:8b-instruct-q4_K_M"])):
            assert await client.model_is_available("llama3") is True

    @pytest.mark.asyncio
    async def test_model_is_available_false(self):
        client = OllamaClient(self.cfg)
        with patch.object(client, "list_models", AsyncMock(return_value=["mistral:7b"])):
            assert await client.model_is_available("llama3") is False

    @pytest.mark.asyncio
    async def test_chat_stream_yields_tokens(self):
        client = OllamaClient(self.cfg)

        # Simulate a 3-chunk streamed response
        chunks = [
            json.dumps({"message": {"content": "Hello"}, "done": False}),
            json.dumps({"message": {"content": " world"}, "done": False}),
            json.dumps({"done": True}),
        ]

        async def fake_aiter_lines():
            for c in chunks:
                yield c

        mock_stream_ctx = AsyncMock()
        mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_stream_ctx)
        mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_stream_ctx.raise_for_status = MagicMock()
        mock_stream_ctx.aiter_lines = fake_aiter_lines

        mock_http = MagicMock()
        mock_http.stream = MagicMock(return_value=mock_stream_ctx)
        client._client = mock_http

        tokens = []
        async for tok in client.chat_stream([], {}, "llama3:8b"):
            tokens.append(tok)

        assert tokens == ["Hello", " world"]


# ---------------------------------------------------------------------------
# Brain (integration-level, all IO mocked)
# ---------------------------------------------------------------------------

class TestBrain:
    def _make_ready_brain(self) -> Brain:
        brain = Brain()
        brain._ready = True
        return brain

    @pytest.mark.asyncio
    async def test_think_empty_input_ignored(self):
        brain = self._make_ready_brain()
        tokens = []
        async for tok in brain.think("   "):
            tokens.append(tok)
        assert tokens == []

    @pytest.mark.asyncio
    async def test_think_not_initialised_raises(self):
        brain = Brain()
        with pytest.raises(RuntimeError, match="initialise"):
            async for _ in brain.think("hello"):
                pass

    @pytest.mark.asyncio
    async def test_think_streams_tokens_and_updates_memory(self):
        brain = self._make_ready_brain()

        async def fake_stream(*_, **__):
            for tok in ["I", " am", " JARVIS"]:
                yield tok

        brain._client.chat_stream = fake_stream

        tokens = []
        async for tok in brain.think("Who are you?"):
            tokens.append(tok)

        assert "".join(tokens) == "I am JARVIS"
        assert brain.memory.turn_count == 1

    @pytest.mark.asyncio
    async def test_think_complete_returns_string(self):
        brain = self._make_ready_brain()

        async def fake_stream(*_, **__):
            for tok in ["Done", "."]:
                yield tok

        brain._client.chat_stream = fake_stream
        result = await brain.think_complete("test")
        assert result == "Done."

    def test_clear_memory(self):
        brain = self._make_ready_brain()
        brain._memory.add_user("hi")
        brain._memory.add_assistant("hello")
        brain.clear_memory()
        assert brain.memory.is_empty

    def test_status_shape(self):
        brain = Brain()
        s = brain.status()
        assert "ready" in s
        assert "model" in s
        assert "turn_count" in s

    def test_repr(self):
        brain = Brain()
        r = repr(brain)
        assert "Brain" in r
        assert "ready=False" in r
