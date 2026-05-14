from app.rag.conversation_memory import memory_store
from app.rag import query_rewriter


def test_standalone_question_is_not_follow_up():
    assert query_rewriter.is_follow_up_question("What was Apple's gross margin percentage in 2023?") is False
    assert query_rewriter.is_follow_up_question("What were Apple's total net sales and net income in 2023?") is False
    assert (
        query_rewriter.is_follow_up_question(
            "Why did Apple's total net sales decrease in 2023 compared to 2022?"
        )
        is False
    )


def test_follow_up_services_is_detected():
    assert query_rewriter.is_follow_up_question("How about Services?") is True
    assert query_rewriter.is_follow_up_question("What about Products?") is True
    assert query_rewriter.is_follow_up_question("And in 2022?") is True


def test_rule_based_rewrite_services_gross_margin():
    history = (
        "User: What was Apple's gross margin percentage in 2023?\n"
        "assistant: Apple's total gross margin percentage in 2023 was 44.1%."
    )

    rewritten = query_rewriter.rule_based_rewrite("How about Services?", history)

    assert rewritten == "What was Apple's Services gross margin percentage in 2023?"


def test_rule_based_rewrite_products_gross_margin():
    history = "User: What was Apple's gross margin percentage in 2023?"

    rewritten = query_rewriter.rule_based_rewrite("What about Products?", history)

    assert rewritten == "What was Apple's Products gross margin percentage in 2023?"


def test_rule_based_rewrite_year_for_net_sales_and_net_income():
    history = "User: What were Apple's total net sales and net income in 2023?"

    rewritten = query_rewriter.rule_based_rewrite("How about 2022?", history)

    assert rewritten == "What were Apple's total net sales and net income in 2022?"


def test_build_retrieval_query_disabled_returns_original_question():
    session_id = "unit-test-query-rewrite-disabled"
    memory_store.clear_session(session_id)
    memory_store.add_user_message(session_id, "What was Apple's gross margin percentage in 2023?")

    query, debug = query_rewriter.build_retrieval_query(
        "How about Services?",
        session_id,
        use_memory=True,
        use_memory_for_retrieval=False,
    )

    assert query == "How about Services?"
    assert debug["rewrite_strategy"] == "disabled"
    memory_store.clear_session(session_id)


def test_build_retrieval_query_rule_based_for_follow_up():
    session_id = "unit-test-query-rewrite-rule"
    memory_store.clear_session(session_id)
    memory_store.add_user_message(session_id, "What was Apple's gross margin percentage in 2023?")

    query, debug = query_rewriter.build_retrieval_query(
        "How about Services?",
        session_id,
        use_memory=True,
        use_memory_for_retrieval=True,
    )

    assert query == "What was Apple's Services gross margin percentage in 2023?"
    assert debug["rewrite_strategy"] == "rule_based"
    assert debug["is_follow_up"] is True
    memory_store.clear_session(session_id)


def test_build_retrieval_query_uses_llm_fallback(monkeypatch):
    session_id = "unit-test-query-rewrite-llm"
    memory_store.clear_session(session_id)
    memory_store.add_user_message(session_id, "Tell me about Apple's cash flow in 2023.")

    monkeypatch.setattr(
        query_rewriter,
        "rule_based_rewrite",
        lambda question, history_text: None,
    )
    monkeypatch.setattr(
        query_rewriter,
        "rewrite_query_with_llm",
        lambda question, history_text: "What was Apple's operating cash flow in 2023?",
    )

    query, debug = query_rewriter.build_retrieval_query(
        "How about operating activities?",
        session_id,
        use_memory=True,
        use_memory_for_retrieval=True,
    )

    assert query == "What was Apple's operating cash flow in 2023?"
    assert debug["rewrite_strategy"] == "llm"
    memory_store.clear_session(session_id)
