from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Any


DEFAULT_SESSION_ID = "default"


@dataclass
class ChatTurn:
    role: str
    content: str
    sources: list[dict[str, Any]] | None = None


class ConversationMemoryStore:
    def __init__(self, max_stored_turns: int = 20) -> None:
        self.max_stored_turns = max_stored_turns
        self._sessions: dict[str, list[ChatTurn]] = {}
        self._lock = RLock()

    def add_user_message(self, session_id: str | None, content: str) -> None:
        self._add_turn(session_id, ChatTurn(role="user", content=content))

    def add_assistant_message(
        self,
        session_id: str | None,
        content: str,
        sources: list[dict[str, Any]] | None = None,
    ) -> None:
        self._add_turn(session_id, ChatTurn(role="assistant", content=content, sources=sources))

    def get_recent_history(self, session_id: str | None, max_turns: int = 6) -> list[ChatTurn]:
        normalized = normalize_session_id(session_id)
        with self._lock:
            return list(self._sessions.get(normalized, [])[-max_turns:])

    def clear_session(self, session_id: str | None) -> dict[str, str]:
        normalized = normalize_session_id(session_id)
        with self._lock:
            self._sessions.pop(normalized, None)
        return {"session_id": normalized, "status": "cleared"}

    def build_history_text(self, session_id: str | None, max_turns: int = 6) -> str:
        turns = self.get_recent_history(session_id, max_turns=max_turns)
        lines = []
        for turn in turns:
            content = _truncate(turn.content, 600)
            role = "User" if turn.role == "user" else "Assistant"
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def _add_turn(self, session_id: str | None, turn: ChatTurn) -> None:
        normalized = normalize_session_id(session_id)
        with self._lock:
            turns = self._sessions.setdefault(normalized, [])
            turns.append(turn)
            if len(turns) > self.max_stored_turns:
                del turns[: len(turns) - self.max_stored_turns]


def normalize_session_id(session_id: str | None) -> str:
    normalized = (session_id or "").strip()
    return normalized or DEFAULT_SESSION_ID


def _truncate(text: str, max_chars: int) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3].rstrip() + "..."


memory_store = ConversationMemoryStore()
