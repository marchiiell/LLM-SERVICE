"""Клиент эмбеддингов Gemini (родной API): превращает текст в числовой вектор.

Тексты, близкие по смыслу, получают близкие векторы, даже если они на разных языках.
Для документов и для вопросов используются разные режимы (RETRIEVAL_DOCUMENT / RETRIEVAL_QUERY).
"""
import asyncio
import logging

import httpx

log = logging.getLogger("embeddings")

BASE_URL = "https://generativelanguage.googleapis.com/v1beta/"
BATCH_SIZE = 20
RETRY_STATUSES = {429, 500, 502, 503, 504}


class EmbeddingError(Exception):
    pass


def _error_text(resp: httpx.Response) -> str:
    try:
        return str(resp.json()["error"]["message"])[:300]
    except Exception:
        return resp.text[:300]


class Embedder:
    def __init__(
        self,
        api_key: str,
        model: str = "gemini-embedding-001",
        dimensions: int = 768,
        timeout: float = 60.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.model = model
        self.dimensions = dimensions
        self._http = httpx.AsyncClient(
            base_url=BASE_URL,
            headers={"x-goog-api-key": api_key},
            timeout=httpx.Timeout(timeout, connect=10.0),
            transport=transport,
        )

    def _request(self, text: str, task_type: str) -> dict:
        config: dict = {"taskType": task_type}
        if self.dimensions:
            config["outputDimensionality"] = self.dimensions
        return {
            "model": f"models/{self.model}",
            "content": {"parts": [{"text": text}]},
            "embedContentConfig": config,
        }

    async def _post(self, path: str, body: dict) -> dict:
        last = "неизвестная ошибка"
        for attempt in range(3):
            try:
                resp = await self._http.post(path, json=body)
            except httpx.TransportError as e:
                last = f"нет связи ({type(e).__name__})"
            else:
                if resp.status_code == 200:
                    return resp.json()
                last = f"{resp.status_code}: {_error_text(resp)}"
                if resp.status_code not in RETRY_STATUSES:
                    raise EmbeddingError(last)
            if attempt < 2:
                delay = 2**attempt
                log.warning("embeddings retry %s in %ss: %s", attempt + 1, delay, last)
                await asyncio.sleep(delay)
        raise EmbeddingError(last)

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            data = await self._post(
                f"models/{self.model}:batchEmbedContents",
                {"requests": [self._request(t, "RETRIEVAL_DOCUMENT") for t in batch]},
            )
            items = data.get("embeddings") or []
            if len(items) != len(batch):
                raise EmbeddingError("провайдер вернул неверное число векторов")
            vectors.extend(item["values"] for item in items)
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        data = await self._post(f"models/{self.model}:embedContent", self._request(text, "RETRIEVAL_QUERY"))
        try:
            return data["embedding"]["values"]
        except (KeyError, TypeError):
            raise EmbeddingError("провайдер не вернул вектор")
