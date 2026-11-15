"""
memory.py — Conversation memory manager for EduBot.

Wraps LangChain's ``ConversationBufferWindowMemory`` to provide per-session
sliding-window memory (default 20 turns). Supports multiple concurrent
sessions keyed by ``session_id``.
"""

from __future__ import annotations

import logging
from datetime import datetime
from threading import Lock
from typing import Any

from langchain.memory import ConversationBufferWindowMemory

import config

logger = logging.getLogger(__name__)


class SessionRecord:
    """Internal bookkeeping for a single conversation session."""

    def __init__(self, session_id: str, window_size: int = config.MEMORY_WINDOW_SIZE):
        self.session_id = session_id
        self.memory = ConversationBufferWindowMemory(
            k=window_size,
            return_messages=True,
            memory_key="history",
        )
        self.created_at = datetime.utcnow()
        self.last_active = datetime.utcnow()
        self.turn_count: int = 0

    def touch(self) -> None:
        """Update the *last_active* timestamp."""
        self.last_active = datetime.utcnow()


class MemoryManager:
    """Thread-safe manager for multiple conversation sessions.

    Each session maintains its own ``ConversationBufferWindowMemory`` with a
    configurable sliding window (default: 20 turns).

    Usage::

        mgr = MemoryManager()
        mgr.add_message("sess-1", user_input="Hello", ai_output="Hi there!")
        history = mgr.get_history("sess-1")
    """

    def __init__(self, window_size: int = config.MEMORY_WINDOW_SIZE):
        self._sessions: dict[str, SessionRecord] = {}
        self._lock = Lock()
        self._window_size = window_size
        logger.info("MemoryManager initialised (window=%d)", window_size)

    # ── Session lifecycle ────────────────────

    def _get_or_create(self, session_id: str) -> SessionRecord:
        """Return an existing session or create a new one (caller holds lock)."""
        if session_id not in self._sessions:
            if len(self._sessions) >= config.MAX_SESSIONS:
                self._evict_oldest()
            self._sessions[session_id] = SessionRecord(session_id, self._window_size)
            logger.info("Created new session: %s", session_id)
        return self._sessions[session_id]

    def _evict_oldest(self) -> None:
        """Remove the least-recently-active session to stay under the cap."""
        if not self._sessions:
            return
        oldest_id = min(self._sessions, key=lambda k: self._sessions[k].last_active)
        del self._sessions[oldest_id]
        logger.info("Evicted oldest session: %s", oldest_id)

    # ── Public API ───────────────────────────

    def add_message(self, session_id: str, *, user_input: str, ai_output: str) -> None:
        """Append a user–assistant exchange to the session memory.

        Args:
            session_id: Conversation session identifier.
            user_input: The user's message.
            ai_output:  The assistant's reply.
        """
        with self._lock:
            rec = self._get_or_create(session_id)
            rec.memory.save_context(
                {"input": user_input},
                {"output": ai_output},
            )
            rec.turn_count += 1
            rec.touch()
            logger.debug(
                "Session %s: saved turn %d", session_id, rec.turn_count,
            )

    def get_history(self, session_id: str) -> list[dict[str, str]]:
        """Return the conversation history as a list of dicts.

        Each dict has keys ``"role"`` (``"user"`` or ``"assistant"``)
        and ``"content"``.

        Args:
            session_id: Conversation session identifier.

        Returns:
            Ordered list of messages within the memory window.
        """
        with self._lock:
            rec = self._get_or_create(session_id)
            messages = rec.memory.load_memory_variables({}).get("history", [])
            history: list[dict[str, str]] = []
            for msg in messages:
                role = "user" if msg.type == "human" else "assistant"
                history.append({"role": role, "content": msg.content})
            rec.touch()
            return history

    def get_history_as_string(self, session_id: str) -> str:
        """Return conversation history formatted as a readable string.

        Args:
            session_id: Conversation session identifier.

        Returns:
            Multi-line string of the conversation.
        """
        history = self.get_history(session_id)
        lines: list[str] = []
        for msg in history:
            prefix = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{prefix}: {msg['content']}")
        return "\n".join(lines)

    def clear_session(self, session_id: str) -> bool:
        """Delete a session and its memory.

        Args:
            session_id: Conversation session identifier.

        Returns:
            ``True`` if the session existed and was removed.
        """
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                logger.info("Cleared session: %s", session_id)
                return True
            return False

    def get_all_sessions(self) -> list[dict[str, Any]]:
        """List metadata for every active session.

        Returns:
            List of dicts with ``session_id``, ``turn_count``,
            ``created_at``, and ``last_active``.
        """
        with self._lock:
            return [
                {
                    "session_id": rec.session_id,
                    "turn_count": rec.turn_count,
                    "created_at": rec.created_at.isoformat(),
                    "last_active": rec.last_active.isoformat(),
                }
                for rec in self._sessions.values()
            ]

    @property
    def active_session_count(self) -> int:
        """Number of sessions currently held in memory."""
        return len(self._sessions)
