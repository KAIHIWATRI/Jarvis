"""
tests/test_memory_manager.py
==============================
Full unit + integration test suite for the JARVIS Memory Manager.
Uses an in-memory SQLite database — no files written during tests.

Run:  pytest tests/test_memory_manager.py -v
"""

from __future__ import annotations

import asyncio
import json
import sys
import os
import tempfile
import pytest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from memory_manager import (
    Command,
    CommandStore,
    ConversationStore,
    MemoryConfig,
    MemoryManager,
    Message,
    PreferenceStore,
    Session,
    Task,
    TaskStore,
    VectorStoreInterface,
    _ConnectionPool,
    _NullVectorStore,
    _SchemaManager,
    _now_iso,
    _new_id,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_db(tmp_path):
    """Provide a fresh MemoryConfig pointing to a temp file."""
    return MemoryConfig(db_path=tmp_path / "test.db")


@pytest.fixture
def pool(tmp_db):
    """Open a _ConnectionPool and apply schema; tear down after test."""
    p = _ConnectionPool(tmp_db)
    p.open()
    _SchemaManager(p).apply_migrations()
    yield p
    p.close()


@pytest.fixture
def conversations(pool, tmp_db):
    return ConversationStore(pool, tmp_db)


@pytest.fixture
def preferences(pool):
    return PreferenceStore(pool)


@pytest.fixture
def tasks(pool):
    return TaskStore(pool)


@pytest.fixture
def commands(pool, tmp_db):
    return CommandStore(pool, tmp_db)


@pytest.fixture
def mem(tmp_db):
    """A fully initialised MemoryManager backed by a temp file."""
    manager = MemoryManager(config=tmp_db)
    manager.initialise_sync()
    yield manager
    manager.close()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _now_iso():
    from memory_manager import _now_iso as _n
    return _n()


# ─────────────────────────────────────────────────────────────────────────────
# MemoryConfig
# ─────────────────────────────────────────────────────────────────────────────

class TestMemoryConfig:
    def test_defaults(self, tmp_path):
        # Just verify frozen dataclass works
        cfg = MemoryConfig()
        assert cfg.wal_mode is True
        assert cfg.max_conversations == 500
        assert cfg.max_commands == 10_000

    def test_custom(self, tmp_path):
        cfg = MemoryConfig(db_path=tmp_path / "x.db", max_conversations=10)
        assert cfg.max_conversations == 10

    def test_immutable(self):
        cfg = MemoryConfig()
        with pytest.raises(Exception):
            cfg.wal_mode = False


# ─────────────────────────────────────────────────────────────────────────────
# _ConnectionPool
# ─────────────────────────────────────────────────────────────────────────────

class TestConnectionPool:
    def test_open_creates_file(self, tmp_db, tmp_path):
        p = _ConnectionPool(tmp_db)
        p.open()
        assert (tmp_path / "test.db").exists()
        p.close()

    def test_execute_and_fetchone(self, pool):
        pool.execute("CREATE TEMP TABLE t (x INTEGER);", commit=True)
        pool.execute("INSERT INTO t VALUES (42);", commit=True)
        row = pool.fetchone("SELECT x FROM t;")
        assert row["x"] == 42

    def test_fetchall(self, pool):
        pool.execute("CREATE TEMP TABLE t2 (v TEXT);", commit=True)
        pool.executemany("INSERT INTO t2 VALUES (?);", [("a",), ("b",), ("c",)])
        rows = pool.fetchall("SELECT v FROM t2 ORDER BY v;")
        assert [r["v"] for r in rows] == ["a", "b", "c"]

    def test_transaction_rollback_on_error(self, pool):
        pool.execute("CREATE TEMP TABLE t3 (id INTEGER PRIMARY KEY);", commit=True)
        try:
            pool.transaction([
                ("INSERT INTO t3 VALUES (1);", ()),
                ("INSERT INTO t3 VALUES (1);", ()),   # duplicate — causes error
            ])
        except Exception:
            pass
        row = pool.fetchone("SELECT COUNT(*) AS n FROM t3;")
        assert row["n"] == 0

    def test_double_open_is_safe(self, tmp_db):
        p = _ConnectionPool(tmp_db)
        p.open()
        p.open()   # second call should be a no-op
        p.close()

    def test_close_is_idempotent(self, tmp_db):
        p = _ConnectionPool(tmp_db)
        p.open()
        p.close()
        p.close()  # second close should not raise


# ─────────────────────────────────────────────────────────────────────────────
# _SchemaManager
# ─────────────────────────────────────────────────────────────────────────────

class TestSchemaManager:
    def test_tables_exist(self, pool):
        tables = {r["name"] for r in pool.fetchall(
            "SELECT name FROM sqlite_master WHERE type='table';"
        )}
        for expected in ("sessions", "messages", "preferences", "tasks", "commands"):
            assert expected in tables, f"Missing table: {expected}"

    def test_migrations_idempotent(self, pool, tmp_db):
        # Running migrations again should not error
        _SchemaManager(pool).apply_migrations()
        rows = pool.fetchall("SELECT id FROM schema_migrations;")
        assert len(rows) == 1   # still only one migration applied


# ─────────────────────────────────────────────────────────────────────────────
# ConversationStore — Sessions
# ─────────────────────────────────────────────────────────────────────────────

class TestConversationStoreSessions:
    def test_new_session_returns_id(self, conversations):
        sid = conversations.new_session("Test")
        assert isinstance(sid, str) and len(sid) == 32

    def test_get_session(self, conversations):
        sid = conversations.new_session("Hello")
        session = conversations.get_session(sid)
        assert isinstance(session, Session)
        assert session.title == "Hello"
        assert session.id == sid

    def test_get_session_not_found(self, conversations):
        assert conversations.get_session("nonexistent") is None

    def test_end_session(self, conversations):
        sid = conversations.new_session()
        conversations.end_session(sid, summary="It went well.")
        session = conversations.get_session(sid)
        assert session.ended_at is not None
        assert session.summary == "It went well."

    def test_list_sessions_newest_first(self, conversations):
        s1 = conversations.new_session("A")
        s2 = conversations.new_session("B")
        sessions = conversations.list_sessions()
        ids = [s.id for s in sessions]
        assert ids.index(s2) < ids.index(s1)   # B (newer) comes first

    def test_update_session(self, conversations):
        sid = conversations.new_session("Old title")
        conversations.update_session(sid, title="New title", summary="Updated")
        s = conversations.get_session(sid)
        assert s.title == "New title"
        assert s.summary == "Updated"

    def test_update_session_missing_raises(self, conversations):
        with pytest.raises(KeyError):
            conversations.update_session("bad_id", title="x")

    def test_delete_session(self, conversations):
        sid = conversations.new_session()
        assert conversations.delete_session(sid) is True
        assert conversations.get_session(sid) is None

    def test_delete_session_cascades_messages(self, conversations):
        sid = conversations.new_session()
        conversations.add_message(sid, "user", "hi")
        conversations.delete_session(sid)
        # Messages should be gone too
        msgs = conversations.get_messages(sid)
        assert msgs == []

    def test_export_session(self, conversations):
        sid = conversations.new_session("Export test")
        conversations.add_message(sid, "user", "hello")
        conversations.add_message(sid, "assistant", "world")
        export = conversations.export_session(sid)
        assert export["session"]["id"] == sid
        assert len(export["messages"]) == 2

    def test_export_session_not_found_raises(self, conversations):
        with pytest.raises(KeyError):
            conversations.export_session("bad_id")


# ─────────────────────────────────────────────────────────────────────────────
# ConversationStore — Messages
# ─────────────────────────────────────────────────────────────────────────────

class TestConversationStoreMessages:
    def test_add_and_get_message(self, conversations):
        sid = conversations.new_session()
        mid = conversations.add_message(sid, "user", "Hello there")
        msgs = conversations.get_messages(sid)
        assert len(msgs) == 1
        assert msgs[0].content == "Hello there"
        assert msgs[0].role == "user"
        assert msgs[0].id == mid

    def test_messages_ordered_oldest_first(self, conversations):
        sid = conversations.new_session()
        conversations.add_message(sid, "user", "first")
        conversations.add_message(sid, "assistant", "second")
        msgs = conversations.get_messages(sid)
        assert msgs[0].content == "first"
        assert msgs[1].content == "second"

    def test_filter_by_role(self, conversations):
        sid = conversations.new_session()
        conversations.add_message(sid, "user", "u1")
        conversations.add_message(sid, "assistant", "a1")
        conversations.add_message(sid, "user", "u2")
        user_msgs = conversations.get_messages(sid, role="user")
        assert all(m.role == "user" for m in user_msgs)
        assert len(user_msgs) == 2

    def test_get_last_n_messages(self, conversations):
        sid = conversations.new_session()
        for i in range(10):
            conversations.add_message(sid, "user", f"msg {i}")
        last3 = conversations.get_last_n_messages(sid, n=3)
        assert len(last3) == 3
        assert last3[-1].content == "msg 9"  # most recent last, chronological order

    def test_search_messages(self, conversations):
        sid = conversations.new_session()
        conversations.add_message(sid, "user", "What is the speed of light?")
        conversations.add_message(sid, "user", "Tell me about Python.")
        results = conversations.search_messages("speed")
        assert len(results) == 1
        assert "speed" in results[0].content.lower()

    def test_count_messages(self, conversations):
        sid = conversations.new_session()
        for _ in range(5):
            conversations.add_message(sid, "user", "x")
        assert conversations.count_messages(sid) == 5

    def test_delete_message(self, conversations):
        sid = conversations.new_session()
        mid = conversations.add_message(sid, "user", "delete me")
        assert conversations.delete_message(mid) is True
        assert conversations.count_messages(sid) == 0

    def test_message_metadata(self, conversations):
        sid = conversations.new_session()
        mid = conversations.add_message(sid, "user", "hi", metadata={"source": "voice"})
        msgs = conversations.get_messages(sid)
        assert msgs[0].metadata["source"] == "voice"

    def test_message_tokens_stored(self, conversations):
        sid = conversations.new_session()
        conversations.add_message(sid, "assistant", "response", tokens=42)
        msgs = conversations.get_messages(sid)
        assert msgs[0].tokens == 42


# ─────────────────────────────────────────────────────────────────────────────
# PreferenceStore
# ─────────────────────────────────────────────────────────────────────────────

class TestPreferenceStore:
    def test_set_and_get_str(self, preferences):
        preferences.set("voice", "en-US")
        assert preferences.get("voice") == "en-US"

    def test_set_and_get_int(self, preferences):
        preferences.set("volume", 75)
        val = preferences.get("volume")
        assert val == 75
        assert isinstance(val, int)

    def test_set_and_get_float(self, preferences):
        preferences.set("speed", 1.25)
        val = preferences.get("speed")
        assert abs(val - 1.25) < 1e-9
        assert isinstance(val, float)

    def test_set_and_get_bool_true(self, preferences):
        preferences.set("dark_mode", True)
        val = preferences.get("dark_mode")
        assert val is True
        assert isinstance(val, bool)

    def test_set_and_get_bool_false(self, preferences):
        preferences.set("notifications", False)
        val = preferences.get("notifications")
        assert val is False

    def test_set_and_get_json(self, preferences):
        data = {"wake_word": "hey jarvis", "sensitivity": 0.7}
        preferences.set("wake_config", data)
        val = preferences.get("wake_config")
        assert val["wake_word"] == "hey jarvis"

    def test_set_and_get_list(self, preferences):
        preferences.set("languages", ["en", "fr", "de"])
        val = preferences.get("languages")
        assert val == ["en", "fr", "de"]

    def test_upsert_overwrites(self, preferences):
        preferences.set("theme", "dark")
        preferences.set("theme", "light")
        assert preferences.get("theme") == "light"

    def test_get_default_when_missing(self, preferences):
        assert preferences.get("missing_key", default="fallback") == "fallback"

    def test_get_all(self, preferences):
        preferences.set("a", 1)
        preferences.set("b", "two")
        all_prefs = preferences.get_all()
        assert all_prefs["a"] == 1
        assert all_prefs["b"] == "two"

    def test_set_many(self, preferences):
        preferences.set_many({"x": 1, "y": 2.0, "z": "three"})
        assert preferences.get("x") == 1
        assert preferences.get("y") == 2.0
        assert preferences.get("z") == "three"

    def test_exists(self, preferences):
        preferences.set("present", True)
        assert preferences.exists("present") is True
        assert preferences.exists("absent")  is False

    def test_delete(self, preferences):
        preferences.set("to_delete", "bye")
        assert preferences.delete("to_delete") is True
        assert preferences.get("to_delete") is None

    def test_delete_nonexistent(self, preferences):
        assert preferences.delete("ghost") is False

    def test_reset_all(self, preferences):
        preferences.set("k1", 1)
        preferences.set("k2", 2)
        preferences.reset_all()
        assert preferences.get_all() == {}

    def test_get_meta(self, preferences):
        preferences.set("tts_voice", "en-GB", description="Text-to-speech voice ID")
        meta = preferences.get_meta("tts_voice")
        assert meta.key == "tts_voice"
        assert meta.description == "Text-to-speech voice ID"
        assert meta.value == "en-GB"


# ─────────────────────────────────────────────────────────────────────────────
# TaskStore
# ─────────────────────────────────────────────────────────────────────────────

class TestTaskStore:
    def test_add_returns_id(self, tasks):
        tid = tasks.add("Buy milk")
        assert isinstance(tid, str) and len(tid) == 32

    def test_get_task(self, tasks):
        tid = tasks.add("Buy milk", description="Full fat", priority=1)
        t = tasks.get(tid)
        assert isinstance(t, Task)
        assert t.title == "Buy milk"
        assert t.priority == 1
        assert t.status == "pending"

    def test_get_task_not_found(self, tasks):
        assert tasks.get("nonexistent") is None

    def test_list_pending(self, tasks):
        tasks.add("A", priority=2)
        tasks.add("B", priority=1)
        tid_done = tasks.add("C", priority=3)
        tasks.update_status(tid_done, "done")
        pending = tasks.list_pending()
        assert all(t.status == "pending" for t in pending)
        assert pending[0].priority <= pending[-1].priority   # ordered by priority

    def test_list_by_status(self, tasks):
        tid = tasks.add("Task X")
        tasks.update_status(tid, "in_progress")
        in_progress = tasks.list_by_status("in_progress")
        assert any(t.id == tid for t in in_progress)

    def test_list_by_invalid_status_raises(self, tasks):
        with pytest.raises(ValueError):
            tasks.list_by_status("flying")

    def test_update_status_done_sets_completed_at(self, tasks):
        tid = tasks.add("Finish report")
        tasks.update_status(tid, "done")
        t = tasks.get(tid)
        assert t.status == "done"
        assert t.completed_at is not None

    def test_update_status_nonexistent_returns_false(self, tasks):
        assert tasks.update_status("bad_id", "done") is False

    def test_update_task_fields(self, tasks):
        tid = tasks.add("Original")
        tasks.update(tid, title="Updated", priority=1)
        t = tasks.get(tid)
        assert t.title == "Updated"
        assert t.priority == 1

    def test_update_invalid_priority_raises(self, tasks):
        tid = tasks.add("Task")
        with pytest.raises(ValueError):
            tasks.update(tid, priority=99)

    def test_add_invalid_priority_raises(self, tasks):
        with pytest.raises(ValueError):
            tasks.add("Bad task", priority=0)

    def test_delete_task(self, tasks):
        tid = tasks.add("Temp task")
        assert tasks.delete(tid) is True
        assert tasks.get(tid) is None

    def test_delete_done_tasks(self, tasks):
        t1 = tasks.add("Done task")
        t2 = tasks.add("Keep task")
        tasks.update_status(t1, "done")
        deleted = tasks.delete_done()
        assert deleted == 1
        assert tasks.get(t1) is None
        assert tasks.get(t2) is not None

    def test_search_tasks(self, tasks):
        tasks.add("Buy groceries", description="Weekly shop")
        tasks.add("Call dentist")
        results = tasks.search("groceries")
        assert len(results) == 1

    def test_overdue_tasks(self, tasks):
        past = "2000-01-01T00:00:00+00:00"
        future = "2099-01-01T00:00:00+00:00"
        t1 = tasks.add("Old task", due_iso=past)
        t2 = tasks.add("Future task", due_iso=future)
        overdue = tasks.overdue()
        ids = [t.id for t in overdue]
        assert t1 in ids
        assert t2 not in ids

    def test_count_by_status(self, tasks):
        tasks.add("T1")
        t2 = tasks.add("T2")
        tasks.update_status(t2, "done")
        counts = tasks.count_by_status()
        assert counts.get("pending", 0) >= 1
        assert counts.get("done", 0) >= 1

    def test_task_tags_stored(self, tasks):
        tid = tasks.add("Tagged", tags=["work", "urgent"])
        t = tasks.get(tid)
        assert "work" in t.tags
        assert "urgent" in t.tags

    def test_list_due_before(self, tasks):
        past   = "2000-01-01T00:00:00+00:00"
        future = "2099-01-01T00:00:00+00:00"
        mid    = "2050-01-01T00:00:00+00:00"
        t1 = tasks.add("Past",   due_iso=past)
        t2 = tasks.add("Future", due_iso=future)
        results = tasks.list_due_before(mid)
        ids = [t.id for t in results]
        assert t1 in ids
        assert t2 not in ids


# ─────────────────────────────────────────────────────────────────────────────
# CommandStore
# ─────────────────────────────────────────────────────────────────────────────

class TestCommandStore:
    def test_log_returns_id(self, commands):
        cid = commands.log("browser_skill", input_text="open youtube")
        assert isinstance(cid, str)

    def test_get_command(self, commands):
        cid = commands.log("browser_skill", input_text="search")
        cmd = commands.get(cid)
        assert isinstance(cmd, Command)
        assert cmd.skill_name == "browser_skill"
        assert cmd.success is True

    def test_get_nonexistent(self, commands):
        assert commands.get("bad") is None

    def test_recent(self, commands):
        for i in range(5):
            commands.log(f"skill_{i}")
        recent = commands.recent(limit=3)
        assert len(recent) == 3

    def test_for_skill(self, commands):
        commands.log("weather_skill")
        commands.log("weather_skill")
        commands.log("browser_skill")
        results = commands.for_skill("weather_skill")
        assert all(c.skill_name == "weather_skill" for c in results)
        assert len(results) == 2

    def test_failures(self, commands):
        commands.log("skill_a", success=True)
        commands.log("skill_b", success=False)
        fails = commands.failures()
        assert all(not c.success for c in fails)
        assert len(fails) == 1

    def test_skill_stats(self, commands):
        commands.log("calc", success=True,  latency_ms=100)
        commands.log("calc", success=False, latency_ms=200)
        stats = commands.skill_stats()
        calc_stats = next(s for s in stats if s["skill_name"] == "calc")
        assert calc_stats["total_calls"] == 2
        assert calc_stats["success_pct"] == 50.0

    def test_for_session(self, pool, commands, tmp_db):
        # Must create real sessions first (FK constraint enforced)
        conv = ConversationStore(pool, tmp_db)
        sid1 = conv.new_session("S1")
        sid2 = conv.new_session("S2")
        commands.log("skill_x", session_id=sid1)
        commands.log("skill_y", session_id=sid1)
        commands.log("skill_z", session_id=sid2)
        results = commands.for_session(sid1)
        assert len(results) == 2

    def test_search(self, commands):
        commands.log("weather", input_text="What is the weather in Paris?")
        commands.log("timer",   input_text="Set a 5 minute timer")
        results = commands.search("Paris")
        assert len(results) == 1

    def test_delete(self, commands):
        cid = commands.log("skill")
        assert commands.delete(cid) is True
        assert commands.get(cid) is None

    def test_clear_for_skill(self, commands):
        commands.log("tmp_skill")
        commands.log("tmp_skill")
        deleted = commands.clear_for_skill("tmp_skill")
        assert deleted == 2

    def test_latency_stored(self, commands):
        cid = commands.log("fast_skill", latency_ms=42)
        cmd = commands.get(cid)
        assert cmd.latency_ms == 42


# ─────────────────────────────────────────────────────────────────────────────
# VectorStoreInterface
# ─────────────────────────────────────────────────────────────────────────────

class TestNullVectorStore:
    def test_add_is_noop(self):
        vs = _NullVectorStore()
        vs.add("id1", "some text")   # should not raise

    def test_search_returns_empty(self):
        vs = _NullVectorStore()
        assert vs.search("anything") == []

    def test_delete_is_noop(self):
        vs = _NullVectorStore()
        vs.delete("id1")             # should not raise

    def test_count_returns_zero(self):
        vs = _NullVectorStore()
        assert vs.count() == 0


class _FakeVectorStore(VectorStoreInterface):
    """Minimal in-memory implementation for testing the interface contract."""
    def __init__(self):
        self._docs = {}

    def add(self, doc_id, text, metadata=None):
        self._docs[doc_id] = text

    def search(self, query, n_results=5):
        return [v for v in self._docs.values() if query.lower() in v.lower()][:n_results]

    def delete(self, doc_id):
        self._docs.pop(doc_id, None)

    def count(self):
        return len(self._docs)


class TestFakeVectorStore:
    def test_add_and_search(self):
        vs = _FakeVectorStore()
        vs.add("1", "Python programming")
        vs.add("2", "Weather in Tokyo")
        results = vs.search("python")
        assert len(results) == 1
        assert "Python" in results[0]

    def test_delete(self):
        vs = _FakeVectorStore()
        vs.add("1", "test")
        vs.delete("1")
        assert vs.count() == 0


# ─────────────────────────────────────────────────────────────────────────────
# MemoryManager (integration)
# ─────────────────────────────────────────────────────────────────────────────

class TestMemoryManager:
    def test_initialise_sync(self, tmp_db):
        mem = MemoryManager(config=tmp_db)
        mem.initialise_sync()
        mem.close()

    def test_context_manager_sync(self, tmp_db):
        # MemoryManager doesn't have a sync context manager,
        # so we test the init/close pattern
        mem = MemoryManager(config=tmp_db)
        mem.initialise_sync()
        sid = mem.conversations.new_session("ctx test")
        assert mem.conversations.get_session(sid) is not None
        mem.close()

    @pytest.mark.asyncio
    async def test_async_context_manager(self, tmp_db):
        async with MemoryManager(config=tmp_db) as mem:
            sid = await mem.run(mem.conversations.new_session, "async session")
            assert sid is not None

    def test_save_turn(self, mem):
        sid = mem.conversations.new_session("Turn test")
        uid, aid = mem.save_turn(
            session_id=sid,
            user_text="Hello",
            assistant_text="Hi there!",
            skill_name="chat_skill",
            latency_ms=300,
        )
        assert uid is not None
        assert aid is not None
        msgs = mem.conversations.get_messages(sid)
        assert len(msgs) == 2
        cmds = mem.commands.for_session(sid)
        assert len(cmds) == 1
        assert cmds[0].skill_name == "chat_skill"

    def test_stats(self, mem):
        s = mem.stats()
        assert "sessions" in s
        assert "messages" in s
        assert "tasks" in s
        assert "commands" in s
        assert "preferences" in s
        assert "db_path" in s

    def test_semantic_search_falls_back_to_sqlite(self, mem):
        sid = mem.conversations.new_session()
        mem.conversations.add_message(sid, "user", "I love Python programming")
        results = mem.semantic_search("Python")
        assert len(results) >= 1
        assert any("Python" in r for r in results)

    def test_semantic_search_uses_vector_store(self, tmp_db):
        vs  = _FakeVectorStore()
        mem = MemoryManager(config=tmp_db, vector_store=vs)
        mem.initialise_sync()
        vs.add("doc1", "Machine learning with Python")
        results = mem.semantic_search("machine learning")
        assert len(results) == 1
        mem.close()

    def test_repr(self, mem):
        r = repr(mem)
        assert "MemoryManager" in r
        assert "sessions=" in r

    @pytest.mark.asyncio
    async def test_run_helper(self, mem):
        result = await mem.run(mem.conversations.new_session, "run test")
        assert isinstance(result, str)

    def test_full_workflow(self, mem):
        """End-to-end: session → messages → preferences → tasks → commands."""
        # Conversation
        sid = mem.conversations.new_session("Full workflow")
        mem.conversations.add_message(sid, "user", "Set an alarm for 8 AM")
        mem.conversations.add_message(sid, "assistant", "Alarm set for 8 AM.")

        # Preferences
        mem.preferences.set("timezone", "UTC+5:30")

        # Task
        tid = mem.tasks.add("Morning alarm", due_iso="2025-01-01T08:00:00")

        # Command
        mem.commands.log("alarm_skill", session_id=sid, latency_ms=50)

        # Verify
        assert mem.conversations.count_messages(sid) == 2
        assert mem.preferences.get("timezone") == "UTC+5:30"
        assert mem.tasks.get(tid).title == "Morning alarm"
        assert len(mem.commands.for_session(sid)) == 1

        stats = mem.stats()
        assert stats["sessions"] >= 1
        assert stats["messages"] >= 2
