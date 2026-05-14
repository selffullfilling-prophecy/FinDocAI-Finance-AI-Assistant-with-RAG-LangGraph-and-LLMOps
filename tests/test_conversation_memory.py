from app.rag.conversation_memory import ConversationMemoryStore, normalize_session_id


def test_add_get_recent_history_and_max_turns_limit():
    store = ConversationMemoryStore(max_stored_turns=4)
    for index in range(6):
        store.add_user_message("s1", f"question {index}")

    turns = store.get_recent_history("s1", max_turns=3)

    assert [turn.content for turn in turns] == ["question 3", "question 4", "question 5"]


def test_clear_session():
    store = ConversationMemoryStore()
    store.add_user_message("s1", "hello")

    result = store.clear_session("s1")

    assert result == {"session_id": "s1", "status": "cleared"}
    assert store.get_recent_history("s1") == []


def test_build_history_text_and_default_session():
    store = ConversationMemoryStore()
    store.add_user_message(None, "What drove net sales?")
    store.add_assistant_message(None, "Services revenue. [Source 1]")

    history = store.build_history_text(None)

    assert normalize_session_id(None) == "default"
    assert "User: What drove net sales?" in history
    assert "Assistant: Services revenue. [Source 1]" in history
