from __future__ import annotations

from openai import OpenAI

from app.core.config import get_settings


SYSTEM_MESSAGE = (
    "You are a financial document assistant. "
    "Answer only using the provided context. "
    "If the context is insufficient, say that the provided documents do not contain enough information. "
    "Do not invent financial numbers, dates, metrics, or claims."
)


def generate_answer(prompt: str) -> str:
    """Generate an answer through NVIDIA's OpenAI-compatible API."""

    settings = get_settings()
    if not settings.nvidia_api_key:
        raise ValueError("NVIDIA_API_KEY is not configured.")

    prompt = prompt.strip()
    if not prompt:
        raise ValueError("Prompt is required.")

    client = OpenAI(
        base_url=settings.nvidia_base_url,
        api_key=settings.nvidia_api_key,
    )

    extra_body = None
    if settings.nvidia_reasoning_enabled:
        extra_body = {
            "chat_template_kwargs": {
                "thinking": True,
                "reasoning_effort": settings.nvidia_reasoning_effort,
            }
        }

    completion_kwargs = {
        "model": settings.nvidia_model,
        "messages": [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": prompt},
        ],
        "temperature": settings.nvidia_temperature,
        "top_p": settings.nvidia_top_p,
        "max_tokens": settings.nvidia_max_tokens,
    }
    if extra_body is not None:
        completion_kwargs["extra_body"] = extra_body

    completion = client.chat.completions.create(**completion_kwargs)

    return completion.choices[0].message.content or ""
