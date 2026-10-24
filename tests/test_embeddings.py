"""Тесты клиента эмбеддингов Gemini на подставном сервере (httpx.MockTransport), без сети."""
import asyncio
import json

import httpx
import pytest

from app.embeddings import BATCH_SIZE, Embedder, EmbeddingError


def make(handler) -> Embedder:
    return Embedder("secret", "gemini-embedding-001", 2, transport=httpx.MockTransport(handler))


def test_documents_are_sent_in_batches_with_correct_shape():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append((str(request.url), request.headers["x-goog-api-key"], body))
        return httpx.Response(200, json={"embeddings": [{"values": [0.1, 0.2]} for _ in body["requests"]]})

    texts = [f"текст {i}" for i in range(BATCH_SIZE * 2 + 5)]
    vectors = asyncio.run(make(handler).embed_documents(texts))

    assert len(vectors) == len(texts) and vectors[0] == [0.1, 0.2]
    assert [len(b["requests"]) for _, _, b in seen] == [BATCH_SIZE, BATCH_SIZE, 5]

    url, key, body = seen[0]
    assert url.endswith(":batchEmbedContents") and "gemini-embedding-001" in url
    assert key == "secret"
    first = body["requests"][0]
    assert first["model"] == "models/gemini-embedding-001"
    assert first["content"]["parts"][0]["text"] == "текст 0"
    assert first["embedContentConfig"] == {"taskType": "RETRIEVAL_DOCUMENT", "outputDimensionality": 2}


def test_query_uses_retrieval_query_task():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"embedding": {"values": [0.3, 0.4]}})

    assert asyncio.run(make(handler).embed_query("что такое матрица?")) == [0.3, 0.4]
    assert seen["url"].endswith(":embedContent")
    assert seen["body"]["embedContentConfig"]["taskType"] == "RETRIEVAL_QUERY"


def test_client_error_is_not_retried():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(400, json={"error": {"message": "API key not valid"}})

    with pytest.raises(EmbeddingError) as exc:
        asyncio.run(make(handler).embed_query("x"))
    assert "400" in str(exc.value) and "API key not valid" in str(exc.value)
    assert calls["n"] == 1


def test_rate_limit_is_retried():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, json={"error": {"message": "quota"}})
        return httpx.Response(200, json={"embedding": {"values": [1.0, 0.0]}})

    assert asyncio.run(make(handler).embed_query("x")) == [1.0, 0.0]
    assert calls["n"] == 2


def test_wrong_number_of_vectors_is_an_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"embeddings": [{"values": [0.1, 0.2]}]})

    with pytest.raises(EmbeddingError):
        asyncio.run(make(handler).embed_documents(["a", "b", "c"]))


def test_network_error_becomes_embedding_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no network", request=request)

    with pytest.raises(EmbeddingError) as exc:
        asyncio.run(make(handler).embed_query("x"))
    assert "нет связи" in str(exc.value)
