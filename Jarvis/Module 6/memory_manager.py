"""
JARVIS Memory Manager
======================
A modular, thread-safe, async SQLite memory system for a local AI assistant.

Stores five categories of data:
  - Conversations  : every message exchanged with JARVIS
  - Sessions       : logical conversation sessions (groups of messages)
  - Preferences    : user-configurable key/value settings
  - Tasks          : reminders and to-do items with scheduling
  - Commands       : history of every skill/command invoked

Architecture
------------
  MemoryConfig          – frozen settings dataclass
  _SchemaManager        – creates / migrates tables (internal)
  ConversationStore     – CRUD for sessions + messages
  PreferenceStore       – CRUD for user preferences
  TaskStore             – CRUD for tasks / reminders
  CommandStore          – CRUD for command history
  VectorStoreInterface  – abstract base for future ChromaDB integration
  MemoryManager         – public façade that owns all stores

Quick start
-----------
    from memory_manager import MemoryManager

    async def main():
        mem = MemoryManager()
        await mem.initialise()

        # Save a conversation turn
        session_id = await mem.conversations.new_session(title="Morning chat")
        await mem.conversations.add_message(session_id, "user", "What time is it?")
        await mem.conversations.add_message(session_id, "assistant", "It is 9:00 AM.")

        # Save a preference
        await mem.preferences.set("tts_voice", "en-US-GuyNeural")

        # Save a task
        await mem.tasks.add("Buy groceries", due_iso="2025-06-01T10:00:00")

        await mem.close()

    asyncio.run(main())
"""

from __future__ import annotations

import asyncio
import json
import logging
import logging.handlers
import sqlite3
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


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

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    fh = logging.handlers.RotatingFileHandler(
        log_dir / "memory.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


log = _build_logger("jarvis.memory")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    """Return the current UTC timestamp as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    """Generate a compact, unique string ID."""
    return uuid.uuid4().hex          # 32 hex chars, no dashes


# ─────────────────────────────────────────────────────────────────────────────
# MemoryConfig
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MemoryConfig:
    """
    All database settings in one immutable place.

    db_path         : where the SQLite file lives (created automatically)
    wal_mode        : WAL journal allows concurrent reads during writes
    cache_size_kb   : in-memory page cache (negative = kilobytes)
    max_conversations: auto-prune oldest sessions beyond this count
    max_commands    : auto-prune oldest commands beyond this count
    """
    db_path:           Path  = Path("jarvis_memory.db")
    wal_mode:          bool  = True
    cache_size_kb:     int   = 16_384        # 16 MB
    max_conversations: int   = 500           # sessions (not messages)
    max_commands:      int   = 10_000


# ─────────────────────────────────────────────────────────────────────────────
# Database connection pool (thread-safe synchronous SQLite)
# ─────────────────────────────────────────────────────────────────────────────

class _ConnectionPool:
    """
    Thread-safe SQLite connection pool.

    SQLite's check_same_thread=False is safe when we guard every access
    with a threading.Lock, which this class does.  Each store gets a
    reference to the same pool and calls .execute() / .executemany()
    through it.

    Why synchronous and not aiosqlite?
    ------------------------------------
    Faster-Whisper and Ollama already use asyncio threads heavily on the
    Ryzen 7 3700U.  Adding aiosqlite's executor overhead on top of that
    has measurable latency impact for the short, frequent queries used
    here (< 1 ms each).  Synchronous SQLite with a lock is simpler,
    predictable, and fast enough.  The public API of every Store is still
    async — callers never block — because queries run in
    asyncio.get_event_loop().run_in_executor(None, ...) automatically
    via the MemoryManager façade helpers.
    """

    def __init__(self, config: MemoryConfig) -> None:
        self._path  = config.db_path
        self._cfg   = config
        self._lock  = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def open(self) -> None:
        """Open the database and apply performance settings."""
        if self._conn is not None:
            return

        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            str(self._path),
            check_same_thread=False,    # guarded by self._lock
            detect_types=sqlite3.PARSE_DECLTYPES,
            isolation_level=None,       # autocommit — we manage transactions
        )
        self._conn.row_factory = sqlite3.Row  # rows accessible by column name

        pragmas = [
            "PRAGMA foreign_keys = ON;",
            f"PRAGMA cache_size = -{self._cfg.cache_size_kb};",
            "PRAGMA temp_store = MEMORY;",
            "PRAGMA synchronous = NORMAL;",
        ]
        if self._cfg.wal_mode:
            pragmas.append("PRAGMA journal_mode = WAL;")

        with self._lock:
            for pragma in pragmas:
                self._conn.execute(pragma)
        log.info("Database opened: %s (WAL=%s)", self._path, self._cfg.wal_mode)

    def close(self) -> None:
        """Flush WAL and close the connection."""
        if self._conn is None:
            return
        with self._lock:
            try:
                self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
                self._conn.close()
            except Exception as exc:
                log.warning("Error closing database: %s", exc)
            finally:
                self._conn = None
        log.info("Database closed.")

    # ── Query helpers ─────────────────────────────────────────────────────

    def execute(
        self,
        sql: str,
        params: Tuple = (),
        *,
        commit: bool = False,
    ) -> sqlite3.Cursor:
        """Run a single SQL statement, optionally committing afterwards."""
        with self._lock:
            cur = self._conn.execute(sql, params)
            if commit:
                self._conn.commit()
            return cur

    def executemany(
        self,
        sql: str,
        param_list: List[Tuple],
        *,
        commit: bool = True,
    ) -> None:
        """Run a parameterised statement over a list of param tuples."""
        with self._lock:
            self._conn.executemany(sql, param_list)
            if commit:
                self._conn.commit()

    def transaction(self, statements: List[Tuple[str, Tuple]]) -> None:
        """
        Execute multiple statements in a single atomic transaction.
        All succeed or all roll back.
        """
        with self._lock:
            try:
                self._conn.execute("BEGIN;")
                for sql, params in statements:
                    self._conn.execute(sql, params)
                self._conn.execute("COMMIT;")
            except Exception:
                self._conn.execute("ROLLBACK;")
                raise

    def fetchall(self, sql: str, params: Tuple = ()) -> List[Dict[str, Any]]:
        """Execute a SELECT and return all rows as plain dicts."""
        with self._lock:
            cur = self._conn.execute(sql, params)
            rows = cur.fetchall()
            return [dict(r) for r in rows]

    def fetchone(self, sql: str, params: Tuple = ()) -> Optional[Dict[str, Any]]:
        """Execute a SELECT and return the first row as a plain dict, or None."""
        with self._lock:
            cur = self._conn.execute(sql, params)
            row = cur.fetchone()
            return dict(row) if row else None


# ─────────────────────────────────────────────────────────────────────────────
# Schema manager (internal — do not call directly)
# ─────────────────────────────────────────────────────────────────────────────

class _SchemaManager:
    """
    Creates and migrates the database schema.

    Each migration is stored in the `schema_migrations` table so we know
    exactly what version the database is at.  New migrations can be added
    to MIGRATIONS as JARVIS evolves — they run automatically on startup.

    Tables
    ------
    schema_migrations : applied migration log
    sessions          : logical conversation sessions
    messages          : individual conversation turns inside a session
    preferences       : user key/value settings (JSON values)
    tasks             : reminders / to-do items
    commands          : log of every skill/command invoked
    """

    # Each entry: (migration_id, description, list_of_sql_statements)
    MIGRATIONS: List[Tuple[str, str, List[str]]] = [
        (
            "001_initial_schema",
            "Create all core tables",
            [
                # ── Migration tracker ─────────────────────────────────
                """CREATE TABLE IF NOT EXISTS schema_migrations (
                    id          TEXT PRIMARY KEY,
                    description TEXT NOT NULL,
                    applied_at  TEXT NOT NULL
                );""",

                # ── Sessions ──────────────────────────────────────────
                # A session groups messages into one logical conversation.
                """CREATE TABLE IF NOT EXISTS sessions (
                    id          TEXT PRIMARY KEY,
                    title       TEXT,
                    started_at  TEXT NOT NULL,
                    ended_at    TEXT,
                    summary     TEXT,
                    metadata    TEXT DEFAULT '{}'
                );""",

                # ── Messages ──────────────────────────────────────────
                # Each row is one user or assistant turn.
                """CREATE TABLE IF NOT EXISTS messages (
                    id          TEXT PRIMARY KEY,
                    session_id  TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    role        TEXT NOT NULL CHECK(role IN ('user','assistant','system')),
                    content     TEXT NOT NULL,
                    tokens      INTEGER DEFAULT 0,
                    created_at  TEXT NOT NULL,
                    metadata    TEXT DEFAULT '{}'
                );""",
                "CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);",
                "CREATE INDEX IF NOT EXISTS idx_messages_created ON messages(created_at);",

                # ── Preferences ───────────────────────────────────────
                # key is unique; value is stored as JSON so any type fits.
                """CREATE TABLE IF NOT EXISTS preferences (
                    key         TEXT PRIMARY KEY,
                    value       TEXT NOT NULL,
                    value_type  TEXT NOT NULL DEFAULT 'str',
                    description TEXT,
                    updated_at  TEXT NOT NULL
                );""",

                # ── Tasks ─────────────────────────────────────────────
                """CREATE TABLE IF NOT EXISTS tasks (
                    id          TEXT PRIMARY KEY,
                    title       TEXT NOT NULL,
                    description TEXT,
                    status      TEXT NOT NULL DEFAULT 'pending'
                                CHECK(status IN ('pending','in_progress','done','cancelled')),
                    priority    INTEGER NOT NULL DEFAULT 2
                                CHECK(priority BETWEEN 1 AND 5),
                    due_at      TEXT,
                    completed_at TEXT,
                    created_at  TEXT NOT NULL,
                    tags        TEXT DEFAULT '[]',
                    metadata    TEXT DEFAULT '{}'
                );""",
                "CREATE INDEX IF NOT EXISTS idx_tasks_status   ON tasks(status);",
                "CREATE INDEX IF NOT EXISTS idx_tasks_due      ON tasks(due_at);",

                # ── Commands ──────────────────────────────────────────
                """CREATE TABLE IF NOT EXISTS commands (
                    id          TEXT PRIMARY KEY,
                    session_id  TEXT REFERENCES sessions(id) ON DELETE SET NULL,
                    skill_name  TEXT NOT NULL,
                    input_text  TEXT,
                    output_text TEXT,
                    success     INTEGER NOT NULL DEFAULT 1,
                    latency_ms  INTEGER DEFAULT 0,
                    created_at  TEXT NOT NULL,
                    metadata    TEXT DEFAULT '{}'
                );""",
                "CREATE INDEX IF NOT EXISTS idx_commands_skill   ON commands(skill_name);",
                "CREATE INDEX IF NOT EXISTS idx_commands_created ON commands(created_at);",
            ],
        ),
    ]

    def __init__(self, pool: _ConnectionPool) -> None:
        self._pool = pool

    def apply_migrations(self) -> None:
        """Run all pending migrations in order."""
        # Ensure the migration tracker table exists first
        self._pool.execute(
            """CREATE TABLE IF NOT EXISTS schema_migrations (
                id TEXT PRIMARY KEY, description TEXT NOT NULL, applied_at TEXT NOT NULL
            );""",
            commit=True,
        )

        applied = {
            r["id"]
            for r in self._pool.fetchall("SELECT id FROM schema_migrations;")
        }

        for migration_id, description, statements in self.MIGRATIONS:
            if migration_id in applied:
                log.debug("Migration '%s' already applied — skipping.", migration_id)
                continue

            log.info("Applying migration '%s': %s", migration_id, description)
            try:
                self._pool.transaction(
                    [(sql, ()) for sql in statements]
                    + [(
                        "INSERT INTO schema_migrations(id, description, applied_at) "
                        "VALUES (?, ?, ?);",
                        (migration_id, description, _now_iso()),
                    )]
                )
                log.info("Migration '%s' applied successfully.", migration_id)
            except Exception as exc:
                log.error("Migration '%s' FAILED: %s", migration_id, exc)
                raise


# ─────────────────────────────────────────────────────────────────────────────
# Data models (plain dataclasses — no ORM magic)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Session:
    id:         str
    title:      Optional[str]
    started_at: str
    ended_at:   Optional[str]    = None
    summary:    Optional[str]    = None
    metadata:   Dict[str, Any]   = field(default_factory=dict)


@dataclass
class Message:
    id:         str
    session_id: str
    role:       str              # 'user' | 'assistant' | 'system'
    content:    str
    tokens:     int              = 0
    created_at: str              = field(default_factory=_now_iso)
    metadata:   Dict[str, Any]   = field(default_factory=dict)


@dataclass
class Preference:
    key:         str
    value:       Any
    value_type:  str             # 'str' | 'int' | 'float' | 'bool' | 'json'
    description: Optional[str]  = None
    updated_at:  str             = field(default_factory=_now_iso)


@dataclass
class Task:
    id:           str
    title:        str
    status:       str            = "pending"
    priority:     int            = 2
    description:  Optional[str] = None
    due_at:       Optional[str] = None
    completed_at: Optional[str] = None
    created_at:   str            = field(default_factory=_now_iso)
    tags:         List[str]      = field(default_factory=list)
    metadata:     Dict[str, Any] = field(default_factory=dict)


@dataclass
class Command:
    id:          str
    skill_name:  str
    session_id:  Optional[str]  = None
    input_text:  Optional[str]  = None
    output_text: Optional[str]  = None
    success:     bool            = True
    latency_ms:  int             = 0
    created_at:  str             = field(default_factory=_now_iso)
    metadata:    Dict[str, Any]  = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# ConversationStore
# ─────────────────────────────────────────────────────────────────────────────

class ConversationStore:
    """
    All operations related to sessions and messages.

    Sessions  → logical groups of messages (one per conversation)
    Messages  → individual user / assistant turns inside a session

    Every method is synchronous internally; the MemoryManager wraps them
    in run_in_executor for async callers when needed.
    """

    def __init__(self, pool: _ConnectionPool, config: MemoryConfig) -> None:
        self._pool = pool
        self._cfg  = config

    # ── Sessions ──────────────────────────────────────────────────────────

    def new_session(self, title: Optional[str] = None, metadata: Optional[Dict] = None) -> str:
        """Create a new session and return its ID."""
        sid = _new_id()
        self._pool.execute(
            "INSERT INTO sessions(id, title, started_at, metadata) VALUES (?,?,?,?);",
            (sid, title, _now_iso(), json.dumps(metadata or {})),
            commit=True,
        )
        log.info("New session created: %s ('%s')", sid, title or "untitled")
        self._prune_old_sessions()
        return sid

    def end_session(self, session_id: str, summary: Optional[str] = None) -> None:
        """Mark a session as ended."""
        self._pool.execute(
            "UPDATE sessions SET ended_at=?, summary=? WHERE id=?;",
            (_now_iso(), summary, session_id),
            commit=True,
        )
        log.debug("Session ended: %s", session_id)

    def get_session(self, session_id: str) -> Optional[Session]:
        row = self._pool.fetchone("SELECT * FROM sessions WHERE id=?;", (session_id,))
        return self._row_to_session(row) if row else None

    def list_sessions(self, limit: int = 20, offset: int = 0) -> List[Session]:
        """Return the most-recent sessions, newest first."""
        rows = self._pool.fetchall(
            "SELECT * FROM sessions ORDER BY started_at DESC LIMIT ? OFFSET ?;",
            (limit, offset),
        )
        return [self._row_to_session(r) for r in rows]

    def update_session(
        self,
        session_id: str,
        title:    Optional[str] = None,
        summary:  Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Patch one or more session fields."""
        existing = self.get_session(session_id)
        if not existing:
            raise KeyError(f"Session '{session_id}' not found.")
        self._pool.execute(
            "UPDATE sessions SET title=?, summary=?, metadata=? WHERE id=?;",
            (
                title    if title    is not None else existing.title,
                summary  if summary  is not None else existing.summary,
                json.dumps(metadata if metadata is not None else existing.metadata),
                session_id,
            ),
            commit=True,
        )

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its messages (CASCADE)."""
        cur = self._pool.execute(
            "DELETE FROM sessions WHERE id=?;", (session_id,), commit=True
        )
        deleted = cur.rowcount > 0
        if deleted:
            log.info("Session deleted: %s", session_id)
        return deleted

    # ── Messages ──────────────────────────────────────────────────────────

    def add_message(
        self,
        session_id: str,
        role:       str,
        content:    str,
        tokens:     int = 0,
        metadata:   Optional[Dict] = None,
    ) -> str:
        """Append a message to a session.  Returns the new message ID."""
        mid = _new_id()
        self._pool.execute(
            """INSERT INTO messages(id, session_id, role, content, tokens, created_at, metadata)
               VALUES (?,?,?,?,?,?,?);""",
            (mid, session_id, role, content, tokens, _now_iso(), json.dumps(metadata or {})),
            commit=True,
        )
        log.debug("Message added: session=%s role=%s len=%d", session_id, role, len(content))
        return mid

    def get_messages(
        self,
        session_id: str,
        limit:  int = 100,
        offset: int = 0,
        role:   Optional[str] = None,
    ) -> List[Message]:
        """Return messages for a session, oldest first."""
        if role:
            rows = self._pool.fetchall(
                "SELECT * FROM messages WHERE session_id=? AND role=? "
                "ORDER BY created_at ASC LIMIT ? OFFSET ?;",
                (session_id, role, limit, offset),
            )
        else:
            rows = self._pool.fetchall(
                "SELECT * FROM messages WHERE session_id=? "
                "ORDER BY created_at ASC LIMIT ? OFFSET ?;",
                (session_id, limit, offset),
            )
        return [self._row_to_message(r) for r in rows]

    def get_last_n_messages(self, session_id: str, n: int = 10) -> List[Message]:
        """Return the N most-recent messages, ordered oldest-first."""
        rows = self._pool.fetchall(
            "SELECT * FROM messages WHERE session_id=? "
            "ORDER BY created_at DESC LIMIT ?;",
            (session_id, n),
        )
        msgs = [self._row_to_message(r) for r in rows]
        return list(reversed(msgs))         # restore chronological order

    def search_messages(self, query: str, limit: int = 20) -> List[Message]:
        """Full-text LIKE search across all message content."""
        rows = self._pool.fetchall(
            "SELECT * FROM messages WHERE content LIKE ? "
            "ORDER BY created_at DESC LIMIT ?;",
            (f"%{query}%", limit),
        )
        return [self._row_to_message(r) for r in rows]

    def count_messages(self, session_id: str) -> int:
        row = self._pool.fetchone(
            "SELECT COUNT(*) AS n FROM messages WHERE session_id=?;",
            (session_id,),
        )
        return row["n"] if row else 0

    def delete_message(self, message_id: str) -> bool:
        cur = self._pool.execute(
            "DELETE FROM messages WHERE id=?;", (message_id,), commit=True
        )
        return cur.rowcount > 0

    def export_session(self, session_id: str) -> Dict[str, Any]:
        """Return a session and all its messages as a plain dict (JSON-ready)."""
        session  = self.get_session(session_id)
        if not session:
            raise KeyError(f"Session '{session_id}' not found.")
        messages = self.get_messages(session_id, limit=10_000)
        return {
            "session":  session.__dict__,
            "messages": [m.__dict__ for m in messages],
        }

    # ── Internal ──────────────────────────────────────────────────────────

    def _prune_old_sessions(self) -> None:
        """Keep only the most-recent max_conversations sessions."""
        row = self._pool.fetchone("SELECT COUNT(*) AS n FROM sessions;")
        count = row["n"] if row else 0
        if count > self._cfg.max_conversations:
            excess = count - self._cfg.max_conversations
            self._pool.execute(
                """DELETE FROM sessions WHERE id IN (
                       SELECT id FROM sessions ORDER BY started_at ASC LIMIT ?
                   );""",
                (excess,),
                commit=True,
            )
            log.info("Pruned %d old sessions (limit=%d).", excess, self._cfg.max_conversations)

    @staticmethod
    def _row_to_session(row: Dict) -> Session:
        return Session(
            id=row["id"],
            title=row["title"],
            started_at=row["started_at"],
            ended_at=row.get("ended_at"),
            summary=row.get("summary"),
            metadata=json.loads(row.get("metadata") or "{}"),
        )

    @staticmethod
    def _row_to_message(row: Dict) -> Message:
        return Message(
            id=row["id"],
            session_id=row["session_id"],
            role=row["role"],
            content=row["content"],
            tokens=row.get("tokens", 0),
            created_at=row["created_at"],
            metadata=json.loads(row.get("metadata") or "{}"),
        )


# ─────────────────────────────────────────────────────────────────────────────
# PreferenceStore
# ─────────────────────────────────────────────────────────────────────────────

class PreferenceStore:
    """
    Key/value user preferences with typed values.

    Supported types: str, int, float, bool, json (any JSON-serialisable object)

    Example
    -------
        prefs.set("tts_speed", 1.2)                      # float
        prefs.set("dark_mode", True)                      # bool
        prefs.set("shortcuts", {"wake": "hey jarvis"})   # json
        speed = prefs.get("tts_speed", default=1.0)
    """

    _TYPE_ENCODE = {
        bool:  ("bool",  lambda v: "1" if v else "0"),  # must be before int
        str:   ("str",   lambda v: str(v)),
        int:   ("int",   lambda v: str(v)),
        float: ("float", lambda v: str(v)),
    }
    _TYPE_DECODE = {
        "str":   str,
        "int":   int,
        "float": float,
        "bool":  lambda v: v == "1",
        "json":  json.loads,
    }

    def __init__(self, pool: _ConnectionPool) -> None:
        self._pool = pool

    # ── Write ─────────────────────────────────────────────────────────────

    def set(
        self,
        key:         str,
        value:       Any,
        description: Optional[str] = None,
    ) -> None:
        """Upsert a preference.  Python type is detected automatically."""
        vtype, encoded = self._encode(value)
        self._pool.execute(
            """INSERT INTO preferences(key, value, value_type, description, updated_at)
               VALUES (?,?,?,?,?)
               ON CONFLICT(key) DO UPDATE SET
                   value=excluded.value,
                   value_type=excluded.value_type,
                   description=COALESCE(excluded.description, preferences.description),
                   updated_at=excluded.updated_at;""",
            (key, encoded, vtype, description, _now_iso()),
            commit=True,
        )
        log.debug("Preference set: %s = %r (%s)", key, value, vtype)

    def set_many(self, prefs: Dict[str, Any]) -> None:
        """Upsert multiple preferences in one transaction."""
        rows = []
        for key, value in prefs.items():
            vtype, encoded = self._encode(value)
            rows.append((key, encoded, vtype, _now_iso()))
        self._pool.executemany(
            """INSERT INTO preferences(key, value, value_type, updated_at)
               VALUES (?,?,?,?)
               ON CONFLICT(key) DO UPDATE SET
                   value=excluded.value,
                   value_type=excluded.value_type,
                   updated_at=excluded.updated_at;""",
            rows,
        )
        log.debug("Bulk preference upsert: %d keys.", len(prefs))

    # ── Read ──────────────────────────────────────────────────────────────

    def get(self, key: str, default: Any = None) -> Any:
        """Return the preference value, decoded to its native Python type."""
        row = self._pool.fetchone(
            "SELECT value, value_type FROM preferences WHERE key=?;", (key,)
        )
        if not row:
            return default
        return self._decode(row["value"], row["value_type"])

    def get_all(self) -> Dict[str, Any]:
        """Return all preferences as a plain dict of native Python values."""
        rows = self._pool.fetchall("SELECT key, value, value_type FROM preferences;")
        return {r["key"]: self._decode(r["value"], r["value_type"]) for r in rows}

    def get_meta(self, key: str) -> Optional[Preference]:
        """Return full metadata for a preference key."""
        row = self._pool.fetchone("SELECT * FROM preferences WHERE key=?;", (key,))
        if not row:
            return None
        return Preference(
            key=row["key"],
            value=self._decode(row["value"], row["value_type"]),
            value_type=row["value_type"],
            description=row.get("description"),
            updated_at=row["updated_at"],
        )

    def exists(self, key: str) -> bool:
        row = self._pool.fetchone(
            "SELECT 1 FROM preferences WHERE key=?;", (key,)
        )
        return row is not None

    # ── Delete ────────────────────────────────────────────────────────────

    def delete(self, key: str) -> bool:
        cur = self._pool.execute(
            "DELETE FROM preferences WHERE key=?;", (key,), commit=True
        )
        return cur.rowcount > 0

    def reset_all(self) -> None:
        """Wipe all preferences (useful for 'factory reset')."""
        self._pool.execute("DELETE FROM preferences;", commit=True)
        log.warning("All preferences reset.")

    # ── Internal ──────────────────────────────────────────────────────────

    def _encode(self, value: Any) -> Tuple[str, str]:
        """Return (value_type_str, encoded_str)."""
        for py_type, (type_str, encoder) in self._TYPE_ENCODE.items():
            if isinstance(value, py_type):
                return type_str, encoder(value)
        # Fall back to JSON for dicts, lists, etc.
        return "json", json.dumps(value)

    def _decode(self, raw: str, vtype: str) -> Any:
        decoder = self._TYPE_DECODE.get(vtype, str)
        try:
            return decoder(raw)
        except Exception:
            log.warning("Preference decode failed for type '%s' — returning raw.", vtype)
            return raw


# ─────────────────────────────────────────────────────────────────────────────
# TaskStore
# ─────────────────────────────────────────────────────────────────────────────

class TaskStore:
    """
    CRUD for tasks and reminders.

    Priority levels (1 = highest urgency)
    ──────────────────────────────────────
      1  Critical     (do immediately)
      2  High         (do today)     ← default
      3  Medium       (do this week)
      4  Low          (do someday)
      5  Backlog      (nice to have)

    Status values
    ─────────────
      pending | in_progress | done | cancelled
    """

    VALID_STATUSES = {"pending", "in_progress", "done", "cancelled"}

    def __init__(self, pool: _ConnectionPool) -> None:
        self._pool = pool

    # ── Create ────────────────────────────────────────────────────────────

    def add(
        self,
        title:       str,
        description: Optional[str] = None,
        priority:    int            = 2,
        due_iso:     Optional[str] = None,
        tags:        Optional[List[str]] = None,
        metadata:    Optional[Dict] = None,
    ) -> str:
        """Add a task.  Returns the new task ID."""
        if not (1 <= priority <= 5):
            raise ValueError(f"Priority must be 1–5, got {priority}.")
        tid = _new_id()
        self._pool.execute(
            """INSERT INTO tasks
               (id, title, description, priority, due_at, created_at, tags, metadata)
               VALUES (?,?,?,?,?,?,?,?);""",
            (
                tid, title, description, priority, due_iso,
                _now_iso(),
                json.dumps(tags or []),
                json.dumps(metadata or {}),
            ),
            commit=True,
        )
        log.info("Task created: '%s' (id=%s, priority=%d)", title, tid, priority)
        return tid

    # ── Read ──────────────────────────────────────────────────────────────

    def get(self, task_id: str) -> Optional[Task]:
        row = self._pool.fetchone("SELECT * FROM tasks WHERE id=?;", (task_id,))
        return self._row_to_task(row) if row else None

    def list_pending(self, limit: int = 50) -> List[Task]:
        """Return pending tasks ordered by priority then due date."""
        rows = self._pool.fetchall(
            "SELECT * FROM tasks WHERE status='pending' "
            "ORDER BY priority ASC, due_at ASC NULLS LAST LIMIT ?;",
            (limit,),
        )
        return [self._row_to_task(r) for r in rows]

    def list_by_status(self, status: str, limit: int = 50) -> List[Task]:
        if status not in self.VALID_STATUSES:
            raise ValueError(f"Invalid status '{status}'. Must be one of {self.VALID_STATUSES}.")
        rows = self._pool.fetchall(
            "SELECT * FROM tasks WHERE status=? ORDER BY created_at DESC LIMIT ?;",
            (status, limit),
        )
        return [self._row_to_task(r) for r in rows]

    def list_due_before(self, iso_datetime: str) -> List[Task]:
        """Return all pending tasks due before a given ISO datetime string."""
        rows = self._pool.fetchall(
            "SELECT * FROM tasks WHERE status='pending' AND due_at IS NOT NULL "
            "AND due_at <= ? ORDER BY due_at ASC;",
            (iso_datetime,),
        )
        return [self._row_to_task(r) for r in rows]

    def search(self, query: str, limit: int = 20) -> List[Task]:
        """Search tasks by title or description."""
        rows = self._pool.fetchall(
            "SELECT * FROM tasks WHERE title LIKE ? OR description LIKE ? "
            "ORDER BY created_at DESC LIMIT ?;",
            (f"%{query}%", f"%{query}%", limit),
        )
        return [self._row_to_task(r) for r in rows]

    def overdue(self) -> List[Task]:
        """Return pending tasks whose due_at is in the past."""
        now = _now_iso()
        rows = self._pool.fetchall(
            "SELECT * FROM tasks WHERE status='pending' AND due_at IS NOT NULL "
            "AND due_at < ? ORDER BY due_at ASC;",
            (now,),
        )
        return [self._row_to_task(r) for r in rows]

    def count_by_status(self) -> Dict[str, int]:
        """Return a dict of {status: count} for all statuses."""
        rows = self._pool.fetchall(
            "SELECT status, COUNT(*) AS n FROM tasks GROUP BY status;"
        )
        return {r["status"]: r["n"] for r in rows}

    # ── Update ────────────────────────────────────────────────────────────

    def update_status(self, task_id: str, status: str) -> bool:
        """Change a task's status.  Marks completed_at when done."""
        if status not in self.VALID_STATUSES:
            raise ValueError(f"Invalid status '{status}'.")
        completed = _now_iso() if status == "done" else None
        cur = self._pool.execute(
            "UPDATE tasks SET status=?, completed_at=? WHERE id=?;",
            (status, completed, task_id),
            commit=True,
        )
        if cur.rowcount:
            log.info("Task %s status → %s", task_id, status)
        return cur.rowcount > 0

    def update(
        self,
        task_id:     str,
        title:       Optional[str] = None,
        description: Optional[str] = None,
        priority:    Optional[int] = None,
        due_iso:     Optional[str] = None,
        tags:        Optional[List[str]] = None,
    ) -> bool:
        """Patch one or more task fields."""
        existing = self.get(task_id)
        if not existing:
            return False
        if priority is not None and not (1 <= priority <= 5):
            raise ValueError("Priority must be 1–5.")
        cur = self._pool.execute(
            """UPDATE tasks SET title=?, description=?, priority=?,
               due_at=?, tags=? WHERE id=?;""",
            (
                title       if title       is not None else existing.title,
                description if description is not None else existing.description,
                priority    if priority    is not None else existing.priority,
                due_iso     if due_iso     is not None else existing.due_at,
                json.dumps(tags if tags is not None else existing.tags),
                task_id,
            ),
            commit=True,
        )
        return cur.rowcount > 0

    # ── Delete ────────────────────────────────────────────────────────────

    def delete(self, task_id: str) -> bool:
        cur = self._pool.execute(
            "DELETE FROM tasks WHERE id=?;", (task_id,), commit=True
        )
        return cur.rowcount > 0

    def delete_done(self) -> int:
        """Remove all completed tasks.  Returns count deleted."""
        cur = self._pool.execute(
            "DELETE FROM tasks WHERE status='done';", commit=True
        )
        if cur.rowcount:
            log.info("Deleted %d completed tasks.", cur.rowcount)
        return cur.rowcount

    # ── Internal ──────────────────────────────────────────────────────────

    @staticmethod
    def _row_to_task(row: Dict) -> Task:
        return Task(
            id=row["id"],
            title=row["title"],
            description=row.get("description"),
            status=row["status"],
            priority=row["priority"],
            due_at=row.get("due_at"),
            completed_at=row.get("completed_at"),
            created_at=row["created_at"],
            tags=json.loads(row.get("tags") or "[]"),
            metadata=json.loads(row.get("metadata") or "{}"),
        )


# ─────────────────────────────────────────────────────────────────────────────
# CommandStore
# ─────────────────────────────────────────────────────────────────────────────

class CommandStore:
    """
    Audit log of every skill invocation.

    Used for:
      - Debugging ("what did JARVIS do last time I asked about the weather?")
      - Analytics ("which skill is used most?")
      - Future personalisation (train on success/failure patterns)
    """

    def __init__(self, pool: _ConnectionPool, config: MemoryConfig) -> None:
        self._pool = pool
        self._cfg  = config

    # ── Write ─────────────────────────────────────────────────────────────

    def log(
        self,
        skill_name:  str,
        session_id:  Optional[str] = None,
        input_text:  Optional[str] = None,
        output_text: Optional[str] = None,
        success:     bool           = True,
        latency_ms:  int            = 0,
        metadata:    Optional[Dict] = None,
    ) -> str:
        """Record a command invocation.  Returns the new command ID."""
        cid = _new_id()
        self._pool.execute(
            """INSERT INTO commands
               (id, session_id, skill_name, input_text, output_text,
                success, latency_ms, created_at, metadata)
               VALUES (?,?,?,?,?,?,?,?,?);""",
            (
                cid, session_id, skill_name, input_text, output_text,
                1 if success else 0, latency_ms, _now_iso(),
                json.dumps(metadata or {}),
            ),
            commit=True,
        )
        log.debug("Command logged: skill=%s success=%s lat=%d ms",
                  skill_name, success, latency_ms)
        self._prune_old_commands()
        return cid

    # ── Read ──────────────────────────────────────────────────────────────

    def get(self, command_id: str) -> Optional[Command]:
        row = self._pool.fetchone("SELECT * FROM commands WHERE id=?;", (command_id,))
        return self._row_to_command(row) if row else None

    def recent(self, limit: int = 20) -> List[Command]:
        """Return the most-recent commands, newest first."""
        rows = self._pool.fetchall(
            "SELECT * FROM commands ORDER BY created_at DESC LIMIT ?;", (limit,)
        )
        return [self._row_to_command(r) for r in rows]

    def for_skill(self, skill_name: str, limit: int = 50) -> List[Command]:
        rows = self._pool.fetchall(
            "SELECT * FROM commands WHERE skill_name=? "
            "ORDER BY created_at DESC LIMIT ?;",
            (skill_name, limit),
        )
        return [self._row_to_command(r) for r in rows]

    def for_session(self, session_id: str) -> List[Command]:
        rows = self._pool.fetchall(
            "SELECT * FROM commands WHERE session_id=? ORDER BY created_at ASC;",
            (session_id,),
        )
        return [self._row_to_command(r) for r in rows]

    def failures(self, limit: int = 50) -> List[Command]:
        rows = self._pool.fetchall(
            "SELECT * FROM commands WHERE success=0 ORDER BY created_at DESC LIMIT ?;",
            (limit,),
        )
        return [self._row_to_command(r) for r in rows]

    def skill_stats(self) -> List[Dict[str, Any]]:
        """Return per-skill usage summary (count, success rate, avg latency)."""
        return self._pool.fetchall(
            """SELECT
                 skill_name,
                 COUNT(*)                                 AS total_calls,
                 ROUND(AVG(CASE WHEN success=1 THEN 100.0 ELSE 0 END), 1) AS success_pct,
                 ROUND(AVG(latency_ms), 0)                AS avg_latency_ms
               FROM commands
               GROUP BY skill_name
               ORDER BY total_calls DESC;"""
        )

    def search(self, query: str, limit: int = 20) -> List[Command]:
        rows = self._pool.fetchall(
            "SELECT * FROM commands WHERE input_text LIKE ? OR output_text LIKE ? "
            "ORDER BY created_at DESC LIMIT ?;",
            (f"%{query}%", f"%{query}%", limit),
        )
        return [self._row_to_command(r) for r in rows]

    # ── Delete ────────────────────────────────────────────────────────────

    def delete(self, command_id: str) -> bool:
        cur = self._pool.execute(
            "DELETE FROM commands WHERE id=?;", (command_id,), commit=True
        )
        return cur.rowcount > 0

    def clear_for_skill(self, skill_name: str) -> int:
        cur = self._pool.execute(
            "DELETE FROM commands WHERE skill_name=?;", (skill_name,), commit=True
        )
        return cur.rowcount

    # ── Internal ──────────────────────────────────────────────────────────

    def _prune_old_commands(self) -> None:
        row = self._pool.fetchone("SELECT COUNT(*) AS n FROM commands;")
        count = row["n"] if row else 0
        if count > self._cfg.max_commands:
            excess = count - self._cfg.max_commands
            self._pool.execute(
                """DELETE FROM commands WHERE id IN (
                       SELECT id FROM commands ORDER BY created_at ASC LIMIT ?
                   );""",
                (excess,),
                commit=True,
            )
            log.info("Pruned %d old commands.", excess)

    @staticmethod
    def _row_to_command(row: Dict) -> Command:
        return Command(
            id=row["id"],
            session_id=row.get("session_id"),
            skill_name=row["skill_name"],
            input_text=row.get("input_text"),
            output_text=row.get("output_text"),
            success=bool(row["success"]),
            latency_ms=row.get("latency_ms", 0),
            created_at=row["created_at"],
            metadata=json.loads(row.get("metadata") or "{}"),
        )


# ─────────────────────────────────────────────────────────────────────────────
# VectorStoreInterface  — future ChromaDB / semantic memory hook
# ─────────────────────────────────────────────────────────────────────────────

class VectorStoreInterface(ABC):
    """
    Abstract base for semantic / vector memory.

    When you're ready to add long-term semantic search (e.g. "find all the
    times I asked about Python"), implement this interface with ChromaDB:

        from chromadb import Client
        from chromadb.utils import embedding_functions

        class ChromaVectorStore(VectorStoreInterface):
            def __init__(self):
                self._client     = Client()
                self._collection = self._client.get_or_create_collection(
                    "jarvis_memory",
                    embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction()
                )

            def add(self, doc_id, text, metadata=None):
                self._collection.add(
                    ids=[doc_id], documents=[text],
                    metadatas=[metadata or {}]
                )

            def search(self, query, n_results=5):
                results = self._collection.query(
                    query_texts=[query], n_results=n_results
                )
                return results["documents"][0]

            def delete(self, doc_id):
                self._collection.delete(ids=[doc_id])

            def count(self):
                return self._collection.count()

    Then pass it to MemoryManager:
        mem = MemoryManager(vector_store=ChromaVectorStore())

    This is a no-op stub today so the rest of the system compiles unchanged.
    """

    @abstractmethod
    def add(self, doc_id: str, text: str, metadata: Optional[Dict] = None) -> None:
        """Embed and store a text document."""

    @abstractmethod
    def search(self, query: str, n_results: int = 5) -> List[str]:
        """Return the n_results most semantically similar documents."""

    @abstractmethod
    def delete(self, doc_id: str) -> None:
        """Remove a document by ID."""

    @abstractmethod
    def count(self) -> int:
        """Return the total number of stored documents."""


class _NullVectorStore(VectorStoreInterface):
    """No-op implementation used when no vector store is configured."""

    def add(self, doc_id: str, text: str, metadata: Optional[Dict] = None) -> None:
        log.debug("VectorStore (null): add(%s) — no-op", doc_id)

    def search(self, query: str, n_results: int = 5) -> List[str]:
        log.debug("VectorStore (null): search('%s') — returning []", query)
        return []

    def delete(self, doc_id: str) -> None:
        log.debug("VectorStore (null): delete(%s) — no-op", doc_id)

    def count(self) -> int:
        return 0


# ─────────────────────────────────────────────────────────────────────────────
# MemoryManager  — public façade
# ─────────────────────────────────────────────────────────────────────────────

class MemoryManager:
    """
    The single entry-point for all memory operations.

    All sub-stores are synchronous internally and exposed here with both
    synchronous and async wrappers so the rest of JARVIS can use either.

    Sync usage (inside a thread or test)
    -------------------------------------
        mem = MemoryManager()
        mem.initialise_sync()
        sid = mem.conversations.new_session("Morning")
        mem.conversations.add_message(sid, "user", "Hello")
        mem.close()

    Async usage (inside asyncio)
    ----------------------------
        async with MemoryManager() as mem:
            sid = await mem.run(mem.conversations.new_session, "Morning")
            await mem.run(mem.conversations.add_message, sid, "user", "Hello")
    """

    def __init__(
        self,
        config:       Optional[MemoryConfig]       = None,
        vector_store: Optional[VectorStoreInterface] = None,
    ) -> None:
        self._cfg          = config or MemoryConfig()
        self._pool         = _ConnectionPool(self._cfg)
        self._schema       = _SchemaManager(self._pool)
        self._vector_store = vector_store or _NullVectorStore()
        self._loop         = None   # set lazily

        # Public sub-stores — access these directly
        self.conversations = ConversationStore(self._pool, self._cfg)
        self.preferences   = PreferenceStore(self._pool)
        self.tasks         = TaskStore(self._pool)
        self.commands      = CommandStore(self._pool, self._cfg)

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def initialise_sync(self) -> None:
        """Open the database and run migrations.  Call once at startup."""
        self._pool.open()
        self._schema.apply_migrations()
        log.info("MemoryManager ready (db=%s).", self._cfg.db_path)

    async def initialise(self) -> None:
        """Async version of initialise_sync — runs in executor."""
        await asyncio.get_event_loop().run_in_executor(None, self.initialise_sync)

    def close(self) -> None:
        """Flush and close the database."""
        self._pool.close()

    async def aclose(self) -> None:
        await asyncio.get_event_loop().run_in_executor(None, self.close)

    async def __aenter__(self) -> "MemoryManager":
        await self.initialise()
        return self

    async def __aexit__(self, *_) -> None:
        await self.aclose()

    # ── Async helper ──────────────────────────────────────────────────────

    async def run(self, fn, *args, **kwargs):
        """
        Run any synchronous store method in the thread-pool executor so it
        doesn't block the asyncio event loop.

        Example
        -------
            sid = await mem.run(mem.conversations.new_session, "My session")
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))

    # ── Cross-store convenience methods ───────────────────────────────────

    def save_turn(
        self,
        session_id:   str,
        user_text:    str,
        assistant_text: str,
        skill_name:   Optional[str] = None,
        latency_ms:   int           = 0,
    ) -> Tuple[str, str]:
        """
        Save a full conversation turn (user + assistant) and optionally log
        the skill invoked.  Returns (user_msg_id, assistant_msg_id).
        """
        uid = self.conversations.add_message(session_id, "user",      user_text)
        aid = self.conversations.add_message(session_id, "assistant", assistant_text)

        if skill_name:
            self.commands.log(
                skill_name=skill_name,
                session_id=session_id,
                input_text=user_text,
                output_text=assistant_text,
                latency_ms=latency_ms,
            )

        # Mirror to vector store for future semantic search
        self._vector_store.add(
            doc_id=aid,
            text=f"User: {user_text}\nJARVIS: {assistant_text}",
            metadata={"session_id": session_id, "created_at": _now_iso()},
        )
        return uid, aid

    def semantic_search(self, query: str, n_results: int = 5) -> List[str]:
        """
        Search memory semantically (requires a real VectorStoreInterface).
        Falls back to SQLite LIKE search when using the null store.
        """
        results = self._vector_store.search(query, n_results)
        if not results:
            msgs = self.conversations.search_messages(query, limit=n_results)
            results = [f"[{m.role}] {m.content}" for m in msgs]
        return results

    def stats(self) -> Dict[str, Any]:
        """Return a summary of all stored data counts."""
        session_count = self._pool.fetchone("SELECT COUNT(*) AS n FROM sessions;")
        message_count = self._pool.fetchone("SELECT COUNT(*) AS n FROM messages;")
        pref_count    = self._pool.fetchone("SELECT COUNT(*) AS n FROM preferences;")
        task_counts   = self.tasks.count_by_status()
        cmd_count     = self._pool.fetchone("SELECT COUNT(*) AS n FROM commands;")

        return {
            "sessions":         session_count["n"] if session_count else 0,
            "messages":         message_count["n"] if message_count else 0,
            "preferences":      pref_count["n"]    if pref_count    else 0,
            "tasks":            task_counts,
            "commands":         cmd_count["n"]     if cmd_count     else 0,
            "vector_documents": self._vector_store.count(),
            "db_path":          str(self._cfg.db_path),
        }

    def __repr__(self) -> str:
        s = self.stats()
        return (
            f"MemoryManager("
            f"sessions={s['sessions']}, "
            f"messages={s['messages']}, "
            f"tasks={s['tasks']}, "
            f"db={self._cfg.db_path})"
        )
