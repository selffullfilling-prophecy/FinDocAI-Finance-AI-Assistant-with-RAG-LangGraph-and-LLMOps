from __future__ import annotations

import re
from typing import Any

from app.rag.conversation_memory import memory_store
from app.rag.llm_client import rewrite_query_with_llm


FOLLOW_UP_PHRASES = [
    "how about",
    "what about",
    "what of",
    "compare that",
    "compare it",
    "last year",
    "previous year",
]
FOLLOW_UP_PRONOUNS = {"that", "it", "those", "them", "this", "same"}
SHORT_ENTITIES = {
    "services",
    "products",
    "iphone",
    "mac",
    "ipad",
    "wearables",
}
METRIC_PHRASES = [
    "gross margin percentage",
    "gross margin",
    "total net sales and net income",
    "net sales",
    "net income",
    "weighted-average interest rate",
    "weighted average interest rate",
    "commercial paper",
    "cash flow",
    "risk factors",
    "revenue",
]


def build_retrieval_query(
    question: str,
    session_id: str,
    use_memory: bool = True,
    use_memory_for_retrieval: bool = False,
) -> tuple[str, dict[str, Any]]:
    """Build a standalone query for retrieval without blindly appending history."""

    question = question.strip()
    base_debug = {
        "original_question": question,
        "rewritten_query": question,
        "retrieval_query": question,
        "is_follow_up": False,
        "used_history_turns": 0,
    }

    if not use_memory or not use_memory_for_retrieval:
        return question, {**base_debug, "rewrite_strategy": "disabled"}

    turns = memory_store.get_recent_history(session_id, max_turns=6)
    if not turns:
        return question, {**base_debug, "rewrite_strategy": "none"}

    is_follow_up = is_follow_up_question(question)
    if not is_follow_up:
        return question, {**base_debug, "rewrite_strategy": "none", "used_history_turns": len(turns)}

    history_text = "\n".join(f"{turn.role}: {_preview(turn.content, 300)}" for turn in turns)
    rewritten_query = rule_based_rewrite(question, history_text)
    if rewritten_query:
        return rewritten_query, {
            **base_debug,
            "rewrite_strategy": "rule_based",
            "rewritten_query": rewritten_query,
            "retrieval_query": rewritten_query,
            "is_follow_up": True,
            "used_history_turns": len(turns),
        }

    try:
        llm_rewrite = rewrite_query_with_llm(question, history_text).strip()
    except Exception as exc:
        return question, {
            **base_debug,
            "rewrite_strategy": "none",
            "is_follow_up": True,
            "used_history_turns": len(turns),
            "rewrite_error": str(exc),
        }

    if not llm_rewrite:
        return question, {
            **base_debug,
            "rewrite_strategy": "none",
            "is_follow_up": True,
            "used_history_turns": len(turns),
        }

    return llm_rewrite, {
        **base_debug,
        "rewrite_strategy": "llm",
        "rewritten_query": llm_rewrite,
        "retrieval_query": llm_rewrite,
        "is_follow_up": True,
        "used_history_turns": len(turns),
    }


def is_follow_up_question(question: str) -> bool:
    question_lower = question.lower().strip()
    if not question_lower:
        return False

    has_year = bool(re.search(r"\b20\d{2}\b", question_lower))
    has_metric = any(metric in question_lower for metric in METRIC_PHRASES)
    if has_year and has_metric:
        return False

    if any(phrase in question_lower for phrase in FOLLOW_UP_PHRASES):
        return True

    tokens = re.findall(r"[a-zA-Z0-9']+", question_lower)
    token_set = set(tokens)
    if question_lower.startswith("and "):
        return True
    if token_set & FOLLOW_UP_PRONOUNS and len(tokens) <= 12:
        return True
    if len(tokens) <= 4 and (token_set & SHORT_ENTITIES or has_year):
        return True
    if len(tokens) <= 5 and not has_metric:
        return True

    return False


def rule_based_rewrite(question: str, history_text: str) -> str | None:
    previous_question = _latest_user_question(history_text)
    if not previous_question:
        return None

    company = _extract_company(previous_question) or "Apple"
    previous_year = _extract_year(previous_question)
    current_year = _extract_year(question)
    year = current_year or previous_year
    metric = _extract_metric(previous_question)
    dimension = _extract_dimension(question)

    if not metric or not year:
        return None

    if metric == "gross margin percentage":
        if dimension in {"Services", "Products"}:
            return f"What was {company}'s {dimension} gross margin percentage in {year}?"
        if current_year:
            return f"What was {company}'s gross margin percentage in {year}?"

    if metric == "total net sales and net income":
        if current_year:
            return f"What were {company}'s total net sales and net income in {year}?"

    if current_year:
        return f"What was {company}'s {metric} in {year}?"

    if dimension:
        return f"What was {company}'s {dimension} {metric} in {year}?"

    return None


def _latest_user_question(history_text: str) -> str:
    questions: list[str] = []
    for line in history_text.splitlines():
        if line.lower().startswith("user:"):
            questions.append(line.split(":", 1)[1].strip())
    return questions[-1] if questions else ""


def _extract_company(text: str) -> str | None:
    if re.search(r"\bapple(?:'s|’s)?\b", text, flags=re.IGNORECASE):
        return "Apple"
    return None


def _extract_year(text: str) -> str | None:
    match = re.search(r"\b(20\d{2})\b", text)
    return match.group(1) if match else None


def _extract_metric(text: str) -> str | None:
    text_lower = text.lower()
    if "total net sales" in text_lower and "net income" in text_lower:
        return "total net sales and net income"
    for metric in METRIC_PHRASES:
        if metric in text_lower:
            if metric == "weighted average interest rate":
                return "weighted-average interest rate"
            return metric
    return None


def _extract_dimension(text: str) -> str | None:
    for entity in ["Services", "Products", "iPhone", "Mac", "iPad", "Wearables"]:
        if re.search(rf"\b{re.escape(entity)}\b", text, flags=re.IGNORECASE):
            return entity
    return None


def _preview(text: str, max_length: int = 300) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    if len(normalized) <= max_length:
        return normalized
    return normalized[: max_length - 3].rstrip() + "..."
