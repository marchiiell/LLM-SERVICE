"""Клиент LLM: Gemini / Groq / xAI (OpenAI-совместимый API через httpx), Anthropic и режим-заглушка.

Единый интерфейс: complete() — ответ целиком, stream() — по кусочкам. Ретраи с паузой,
семафор как очередь к модели.
"""
import asyncio
import json
import logging
from typing import AsyncIterator

import httpx

from .config import Settings

log = logging.getLogger("llm")

RETRY_STATUSES = {429, 500, 502, 503, 504}


class LLMError(Exception):
    def __init__(self, message: str, status_code: int | None = None, retryable: bool = False):
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


class QueueTimeout(Exception):
    pass


def _error_text(resp_text: str) -> str:
    """Достаём короткое сообщение об ошибке из тела ответа провайдера."""
    try:
        data = json.loads(resp_text)
        if isinstance(data, list) and data:
            data = data[0]
        err = data.get("error", data)
        return str(err.get("message", err))[:300]
    except Exception:
        return resp_text[:300]


def _transport_error(e: httpx.TransportError) -> LLMError:
    """Превращаем сетевую ошибку в понятное сообщение."""
    if isinstance(e, httpx.TimeoutException):
        msg = f"Модель слишком долго не отвечает ({type(e).__name__}). Попробуйте ещё раз."
    else:
        msg = f"Нет связи с провайдером ({type(e).__name__}). Проверьте интернет."
    return LLMError(msg, retryable=True)


class LLMClient:
    def __init__(self, settings: Settings, transport: httpx.AsyncBaseTransport | None = None):
        self.s = settings
        self.provider = settings.active_provider
        self._sem = asyncio.Semaphore(settings.max_concurrent_llm)
        self._http = None
        self._anthropic = None

        if self.provider in ("gemini", "groq", "xai"):
            self._http = httpx.AsyncClient(
                base_url=settings.base_url,
                headers={"Authorization": f"Bearer {settings.active_key}"},
                timeout=httpx.Timeout(settings.llm_timeout, connect=10.0),
                transport=transport,
            )
        elif self.provider == "anthropic":
            from anthropic import AsyncAnthropic  # ленивый импорт: пакет нужен только для этого провайдера

            self._anthropic = AsyncAnthropic(api_key=settings.active_key)

    # ------------------------------------------------------------------ очередь
    async def _acquire(self) -> None:
        try:
            await asyncio.wait_for(self._sem.acquire(), timeout=self.s.queue_timeout)
        except asyncio.TimeoutError:
            raise QueueTimeout("Сервис перегружен, попробуйте позже")

    # ------------------------------------------------------------------ ответ целиком
    async def complete(self, messages: list[dict], system: str | None = None) -> dict:
        system = system or self.s.system_prompt
        if self.provider == "mock":
            return self._mock(messages)

        await self._acquire()
        try:
            if self.provider == "anthropic":
                return await self._complete_anthropic(messages, system)
            return await self._complete_openai(messages, system)
        finally:
            self._sem.release()

    def _payload(self, messages: list[dict], system: str, stream: bool) -> dict:
        return {
            "model": self.s.active_model,
            "messages": [{"role": "system", "content": system}, *messages],
            "max_tokens": self.s.max_tokens,
            "stream": stream,
        }

    async def _complete_openai(self, messages: list[dict], system: str) -> dict:
        last_error: LLMError | None = None
        for attempt in range(3):
            try:
                resp = await self._http.post("/chat/completions", json=self._payload(messages, system, False))
            except httpx.TransportError as e:
                last_error = _transport_error(e)
            else:
                if resp.status_code == 200:
                    data = resp.json()
                    text = (data["choices"][0]["message"].get("content") or "").strip()
                    usage = data.get("usage") or {}
                    return {
                        "text": text,
                        "usage": {
                            "input_tokens": usage.get("prompt_tokens", 0),
                            "output_tokens": usage.get("completion_tokens", 0),
                        },
                    }
                last_error = LLMError(
                    _error_text(resp.text), resp.status_code, retryable=resp.status_code in RETRY_STATUSES
                )
                if not last_error.retryable:
                    raise last_error
            if attempt < 2:
                delay = 2**attempt
                log.warning("retry %s in %ss: %s", attempt + 1, delay, last_error)
                await asyncio.sleep(delay)
        raise last_error

    async def _complete_anthropic(self, messages: list[dict], system: str) -> dict:
        from anthropic import APIConnectionError, APIStatusError, RateLimitError

        for attempt in range(3):
            try:
                resp = await self._anthropic.messages.create(
                    model=self.s.active_model,
                    max_tokens=self.s.max_tokens,
                    system=system,
                    messages=messages,
                )
                text = "".join(b.text for b in resp.content if b.type == "text")
                return {
                    "text": text,
                    "usage": {
                        "input_tokens": resp.usage.input_tokens,
                        "output_tokens": resp.usage.output_tokens,
                    },
                }
            except (APIConnectionError, RateLimitError) as e:
                if attempt == 2:
                    raise LLMError(type(e).__name__, getattr(e, "status_code", None))
                await asyncio.sleep(2**attempt)
            except APIStatusError as e:
                raise LLMError(str(e)[:300], e.status_code)

    # ------------------------------------------------------------------ стриминг
    async def stream(self, messages: list[dict], system: str | None = None) -> AsyncIterator[str]:
        system = system or self.s.system_prompt
        if self.provider == "mock":
            for word in self._mock(messages)["text"].split(" "):
                await asyncio.sleep(0.03)
                yield word + " "
            return

        await self._acquire()
        try:
            if self.provider == "anthropic":
                async with self._anthropic.messages.stream(
                    model=self.s.active_model,
                    max_tokens=self.s.max_tokens,
                    system=system,
                    messages=messages,
                ) as stream:
                    async for chunk in stream.text_stream:
                        yield chunk
                return

            # Одна повторная попытка, если сбой случился ДО первого кусочка текста
            # (после начала вывода повторять нельзя: пользователь уже видит часть ответа).
            for attempt in range(2):
                started = False
                try:
                    async for chunk in self._stream_openai_once(messages, system):
                        started = True
                        yield chunk
                    return
                except LLMError as e:
                    if started or attempt == 1 or not e.retryable:
                        raise
                    log.warning("stream failed before first chunk, retrying: %s", e)
                    await asyncio.sleep(1)
        finally:
            self._sem.release()

    async def _stream_openai_once(self, messages: list[dict], system: str) -> AsyncIterator[str]:
        try:
            async with self._http.stream(
                "POST", "/chat/completions", json=self._payload(messages, system, True)
            ) as resp:
                if resp.status_code != 200:
                    body = (await resp.aread()).decode(errors="replace")
                    raise LLMError(
                        _error_text(body), resp.status_code, retryable=resp.status_code in RETRY_STATUSES
                    )
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        choices = json.loads(data).get("choices") or []
                    except json.JSONDecodeError:
                        continue
                    if choices:
                        delta = (choices[0].get("delta") or {}).get("content")
                        if delta:
                            yield delta
        except httpx.TransportError as e:
            log.warning("stream transport error: %r", e)
            raise _transport_error(e)

    # ------------------------------------------------------------------ заглушка
    @staticmethod
    def _mock(messages: list[dict]) -> dict:
        last = messages[-1]["content"]
        text = f"[mock] Вы написали: {last}. Задайте ключ провайдера (например GEMINI_API_KEY), чтобы получать настоящие ответы."
        return {"text": text, "usage": {"input_tokens": len(last.split()), "output_tokens": len(text.split())}}
