"""Тесты выбора провайдера и OpenAI-совместимого клиента (без сети, через httpx.MockTransport)."""
import asyncio
import json

import httpx
import pytest

from app.config import Settings
from app.llm import LLMClient, LLMError


def settings(**kw) -> Settings:
    base = dict(
        provider="auto", model="", gemini_api_key="", groq_api_key="",
        xai_api_key="", anthropic_api_key="", queue_timeout=1,
    )
    base.update(kw)
    return Settings(**base)


MSGS = [{"role": "user", "content": "привет"}]


def test_auto_provider_selection():
    assert settings().active_provider == "mock"
    assert settings(groq_api_key="k").active_provider == "groq"
    assert settings(gemini_api_key="k", groq_api_key="k").active_provider == "gemini"  # Gemini в приоритете
    assert settings(anthropic_api_key="k").active_provider == "anthropic"


def test_explicit_provider_without_key_falls_back_to_mock():
    assert settings(provider="groq").active_provider == "mock"
    assert settings(provider="groq", groq_api_key="k", gemini_api_key="g").active_provider == "groq"


def test_default_models_and_override():
    assert settings(groq_api_key="k").active_model == "openai/gpt-oss-20b"
    assert settings(gemini_api_key="k").base_url.endswith("/v1beta/openai")
    assert settings(groq_api_key="k", model="my-model").active_model == "my-model"


def test_openai_compat_complete():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "  Привет!  "}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2},
        })

    client = LLMClient(settings(groq_api_key="secret"), transport=httpx.MockTransport(handler))
    result = asyncio.run(client.complete(MSGS))

    assert result["text"] == "Привет!"
    assert result["usage"] == {"input_tokens": 5, "output_tokens": 2}
    assert seen["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert seen["auth"] == "Bearer secret"
    assert seen["body"]["messages"][0]["role"] == "system"
    assert seen["body"]["messages"][-1] == MSGS[0]
    assert seen["body"]["model"] == "openai/gpt-oss-20b"


def test_openai_compat_stream():
    sse = (
        'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"При"}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"вет"}}]}\n\n'
        "data: [DONE]\n\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["stream"] is True
        return httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"})

    client = LLMClient(settings(gemini_api_key="k"), transport=httpx.MockTransport(handler))

    async def collect():
        return [chunk async for chunk in client.stream(MSGS)]

    assert "".join(asyncio.run(collect())) == "Привет"


def test_auth_error_is_not_retried_and_mapped():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(401, json={"error": {"message": "API key not valid"}})

    client = LLMClient(settings(gemini_api_key="bad"), transport=httpx.MockTransport(handler))
    with pytest.raises(LLMError) as exc:
        asyncio.run(client.complete(MSGS))
    assert exc.value.status_code == 401
    assert "API key not valid" in str(exc.value)
    assert calls["n"] == 1  # 401 не повторяем


def test_stream_error_status_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "bad model"}})

    client = LLMClient(settings(groq_api_key="k"), transport=httpx.MockTransport(handler))

    async def collect():
        return [c async for c in client.stream(MSGS)]

    with pytest.raises(LLMError) as exc:
        asyncio.run(collect())
    assert exc.value.status_code == 400


def test_stream_retries_once_after_timeout_before_first_chunk():
    calls = {"n": 0}
    sse = 'data: {"choices":[{"delta":{"content":"ок"}}]}\n\ndata: [DONE]\n\n'

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ReadTimeout("slow", request=request)
        return httpx.Response(200, text=sse)

    client = LLMClient(settings(gemini_api_key="k"), transport=httpx.MockTransport(handler))

    async def collect():
        return [c async for c in client.stream(MSGS)]

    assert "".join(asyncio.run(collect())) == "ок"
    assert calls["n"] == 2


def test_stream_timeout_twice_gives_clear_message():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    client = LLMClient(settings(gemini_api_key="k"), transport=httpx.MockTransport(handler))

    async def collect():
        return [c async for c in client.stream(MSGS)]

    with pytest.raises(LLMError) as exc:
        asyncio.run(collect())
    assert "слишком долго не отвечает" in str(exc.value)
