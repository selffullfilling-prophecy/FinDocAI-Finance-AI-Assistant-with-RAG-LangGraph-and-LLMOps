from app.rag import llm_client


class DummySettings:
    def __init__(
        self,
        llm_provider="nvidia",
        nvidia_api_key="key",
        nvidia_reasoning_enabled=False,
    ) -> None:
        self.llm_provider = llm_provider
        self.nvidia_api_key = nvidia_api_key
        self.nvidia_base_url = "https://example.test/v1"
        self.nvidia_model = "model"
        self.nvidia_temperature = 0.2
        self.nvidia_top_p = 0.95
        self.nvidia_max_tokens = 128
        self.nvidia_reasoning_enabled = nvidia_reasoning_enabled
        self.nvidia_reasoning_effort = "high"


class FakeMessage:
    content = "answer"


class FakeChoice:
    def __init__(self, content="answer") -> None:
        self.message = FakeMessage()
        self.delta = type("Delta", (), {"content": content})()


class FakeCompletion:
    choices = [FakeChoice()]


class FakeCompletions:
    def create(self, **kwargs):
        if kwargs.get("stream"):
            return [type("Chunk", (), {"choices": [FakeChoice("A")]})(), type("Chunk", (), {"choices": [FakeChoice("B")]})()]
        return FakeCompletion()


class FakeClient:
    chat = type("Chat", (), {"completions": FakeCompletions()})()


def test_generate_answer_unsupported_provider_raises(monkeypatch):
    monkeypatch.setattr(llm_client, "get_settings", lambda: DummySettings(llm_provider="openai"))

    try:
        llm_client.generate_answer("prompt")
    except ValueError as exc:
        assert str(exc) == "Unsupported LLM provider: openai"
    else:
        raise AssertionError("Expected unsupported provider ValueError")


def test_generate_answer_missing_api_key_raises(monkeypatch):
    monkeypatch.setattr(llm_client, "get_settings", lambda: DummySettings(nvidia_api_key=None))

    try:
        llm_client.generate_answer("prompt")
    except ValueError as exc:
        assert str(exc) == "NVIDIA_API_KEY is not configured."
    else:
        raise AssertionError("Expected missing API key ValueError")


def test_generate_answer_blank_prompt_raises():
    try:
        llm_client.generate_answer("   ")
    except ValueError as exc:
        assert str(exc) == "Prompt is required."
    else:
        raise AssertionError("Expected blank prompt ValueError")


def test_generate_answer_uses_fake_openai_client(monkeypatch):
    monkeypatch.setattr(llm_client, "get_settings", lambda: DummySettings())
    monkeypatch.setattr(llm_client, "_build_client", lambda base_url, api_key: FakeClient())

    assert llm_client.generate_answer("prompt") == "answer"


def test_stream_answer_yields_tokens(monkeypatch):
    monkeypatch.setattr(llm_client, "get_settings", lambda: DummySettings())
    monkeypatch.setattr(llm_client, "_build_client", lambda base_url, api_key: FakeClient())

    assert list(llm_client.stream_answer("prompt")) == ["A", "B"]


def test_rewrite_query_with_llm_uses_fake_openai_client(monkeypatch):
    monkeypatch.setattr(llm_client, "get_settings", lambda: DummySettings())
    monkeypatch.setattr(llm_client, "_build_client", lambda base_url, api_key: FakeClient())

    assert llm_client.rewrite_query_with_llm("How about Services?", "User: gross margin") == "answer"
