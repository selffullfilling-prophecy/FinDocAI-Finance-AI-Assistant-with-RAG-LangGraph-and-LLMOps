from __future__ import annotations

from collections.abc import Iterator

from openai import OpenAI

from app.core.config import get_settings
from app.rag.prompt_templates import RAG_SYSTEM_PROMPT


def generate_answer(prompt: str) -> str:
    """Generate an answer through NVIDIA's OpenAI-compatible API."""

    prompt = _validate_prompt(prompt)
    settings = get_settings()
    _validate_provider(settings.llm_provider)
    _validate_api_key(settings.nvidia_api_key)

    client = _build_client(settings.nvidia_base_url, settings.nvidia_api_key)
    completion = client.chat.completions.create(**_completion_kwargs(prompt, stream=False))

    return completion.choices[0].message.content or ""


def stream_answer(prompt: str) -> Iterator[str]:
    """Stream an answer through NVIDIA's OpenAI-compatible API."""

    prompt = _validate_prompt(prompt)
    settings = get_settings()
    _validate_provider(settings.llm_provider)
    _validate_api_key(settings.nvidia_api_key)

    client = _build_client(settings.nvidia_base_url, settings.nvidia_api_key)
    stream = client.chat.completions.create(**_completion_kwargs(prompt, stream=True))
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        content = getattr(delta, "content", None)
        if content:
            yield content


def _validate_prompt(prompt: str) -> str:
    prompt = prompt.strip()
    if not prompt:
        raise ValueError("Prompt is required.")
    return prompt


def _validate_provider(provider: str) -> None:
    if provider.lower() != "nvidia":
        raise ValueError(f"Unsupported LLM provider: {provider}")


def _validate_api_key(api_key: str | None) -> None:
    if not api_key:
        raise ValueError("NVIDIA_API_KEY is not configured.")


def _build_client(base_url: str, api_key: str | None) -> OpenAI:
    return OpenAI(base_url=base_url, api_key=api_key)


def _completion_kwargs(prompt: str, stream: bool) -> dict:
    settings = get_settings()
    extra_body = None
    if settings.nvidia_reasoning_enabled:
        extra_body = {
            "chat_template_kwargs": {
                "thinking": True,
                "reasoning_effort": settings.nvidia_reasoning_effort,
            }
        }

    kwargs = {
        "model": settings.nvidia_model,
        "messages": [
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": settings.nvidia_temperature,
        "top_p": settings.nvidia_top_p,
        "max_tokens": settings.nvidia_max_tokens,
        "stream": stream,
    }
    if extra_body is not None:
        kwargs["extra_body"] = extra_body

    return kwargs
