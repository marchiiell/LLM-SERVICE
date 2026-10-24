import pytest
from fastapi.testclient import TestClient

from app.cache import TTLCache
from app.config import Settings
from app.main import create_app
from app.ratelimit import RateLimiter


def make_client(**overrides) -> TestClient:
    # Все ключи пустые, чтобы тесты не зависели от переменных окружения на вашей машине
    base = dict(
        provider="auto",
        model="",
        gemini_api_key="",
        groq_api_key="",
        xai_api_key="",
        anthropic_api_key="",
        rate_limit_requests=100,
        service_api_key="",
        data_dir="",  # документы только в памяти, на диск тесты не пишут
    )
    base.update(overrides)
    return TestClient(create_app(Settings(**base)))


BODY = {"messages": [{"role": "user", "content": "привет"}]}


def test_health_mock_mode():
    r = make_client().get("/health")
    assert r.status_code == 200
    assert r.json()["mock_mode"] is True


def test_chat_and_cache():
    c = make_client()
    first = c.post("/api/chat", json=BODY).json()
    second = c.post("/api/chat", json=BODY).json()
    assert first["cached"] is False
    assert second["cached"] is True
    assert first["text"] == second["text"]
    stats = c.get("/api/stats").json()
    assert stats["cache"]["hits"] == 1
    assert stats["metrics"]["llm_calls"] == 1


def test_validation_last_message_must_be_user():
    c = make_client()
    r = c.post("/api/chat", json={"messages": [{"role": "assistant", "content": "x"}]})
    assert r.status_code == 422


def test_rate_limit_returns_429():
    c = make_client(rate_limit_requests=2, rate_limit_window=60)
    codes = [c.post("/api/chat", json=BODY).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def test_service_api_key():
    c = make_client(service_api_key="secret")
    assert c.post("/api/chat", json=BODY).status_code == 401
    ok = c.post("/api/chat", json=BODY, headers={"X-API-Key": "secret"})
    assert ok.status_code == 200


def test_stream_sse():
    c = make_client()
    r = c.post("/api/chat/stream", json=BODY)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    assert "data: [DONE]" in r.text


def test_request_id_header():
    assert "x-request-id" in make_client().get("/health").headers


def test_index_page():
    r = make_client().get("/")
    assert r.status_code == 200
    assert "LLM Service" in r.text


def test_cache_ttl_and_eviction():
    cache = TTLCache(ttl=60, max_items=2)
    cache.set("a", {"v": 1})
    cache.set("b", {"v": 2})
    cache.set("c", {"v": 3})
    assert cache.get("a") is None  # вытеснен
    assert cache.get("c") == {"v": 3}

    expired = TTLCache(ttl=-1, max_items=2)
    expired.set("x", {"v": 1})
    assert expired.get("x") is None


def test_rate_limiter_is_per_client():
    rl = RateLimiter(1, 60)
    assert rl.check("a")[0] is True
    assert rl.check("a")[0] is False
    assert rl.check("b")[0] is True


# ----------------------------------------------------------------- RAG по документам

DOC = (
    "Сервис ограничивает частоту запросов: не больше 10 запросов в минуту с одного клиента.\n\n"
    "Кэш хранит одинаковые ответы 300 секунд."
)


def upload(c, name="notes.md", body=DOC):
    return c.post("/api/docs", params={"name": name}, content=body.encode("utf-8"))


def test_docs_upload_list_delete():
    c = make_client()
    r = upload(c)
    assert r.status_code == 200
    doc = r.json()
    assert doc["name"] == "notes.md" and doc["chunks"] >= 1

    listed = c.get("/api/docs").json()
    assert [d["name"] for d in listed] == ["notes.md"]
    assert c.get("/health").json()["docs"] == 1

    assert c.delete(f"/api/docs/{doc['id']}").status_code == 200
    assert c.delete(f"/api/docs/{doc['id']}").status_code == 404
    assert c.get("/api/docs").json() == []


def test_docs_upload_validation():
    c = make_client()
    assert c.post("/api/docs", params={"name": "a.txt"}, content=b"").status_code == 400
    assert upload(c, "virus.exe").status_code == 415
    assert upload(c, "blank.txt", "   \n  ").status_code == 422
    big = make_client(max_upload_mb=1)
    assert upload(big, "big.txt", "а" * 1_100_000).status_code == 413


def test_ask_returns_sources():
    c = make_client()
    upload(c)
    r = c.post("/api/ask", json={"question": "сколько запросов в минуту разрешено?"})
    assert r.status_code == 200
    data = r.json()
    assert data["sources"] and data["sources"][0]["doc"] == "notes.md"
    assert "[mock]" in data["text"]  # заглушка получила вопрос вместе с источниками


def test_ask_without_documents_does_not_call_model():
    c = make_client()
    data = c.post("/api/ask", json={"question": "что угодно"}).json()
    assert "загрузите документы" in data["text"]
    assert data["sources"] == []
    assert c.get("/api/stats").json()["metrics"]["llm_calls"] == 0


def test_ask_unrelated_question_switches_to_overview_mode():
    c = make_client()
    upload(c)
    data = c.post("/api/ask", json={"question": "квантовая хромодинамика"}).json()
    assert data["sources"] and all(s["overview"] for s in data["sources"])
    assert "[mock]" in data["text"]
    assert c.get("/api/stats").json()["metrics"]["llm_calls"] == 1


# ----------------------------------------------------------------- поиск по смыслу (с подставным эмбеддером)

from app.embeddings import EmbeddingError


class FakeEmbedder:
    """Вместо обращения к Gemini: «матрица» и matrix получают один вектор, всё остальное другой."""

    @staticmethod
    def _vec(text):
        t = text.lower()
        return [1.0, 0.0] if ("matrix" in t or "матриц" in t) else [0.0, 1.0]

    async def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    async def embed_query(self, text):
        return self._vec(text)


class FailingEmbedder:
    async def embed_documents(self, texts):
        raise EmbeddingError("429: quota exceeded")

    async def embed_query(self, text):
        raise EmbeddingError("429: quota exceeded")


def semantic_client(embedder):
    # provider="mock": чат отвечает заглушкой, но ключ Gemini включает поиск по смыслу
    c = make_client(provider="mock", gemini_api_key="k", embeddings="auto", embedding_dimensions=2, min_similarity=0.5)
    c.app.state.embedder = embedder
    return c


def test_semantic_search_finds_english_text_by_russian_question():
    c = semantic_client(FakeEmbedder())
    lecture = upload(c, "lecture.md", "The inverse of a matrix A multiplies with A to give the identity.").json()
    upload(c, "recipe.md", "Boil oats in water for five minutes and add salt.")
    assert lecture["semantic"] is True

    data = c.post("/api/ask", json={"question": "Что такое обратная матрица?"}).json()
    assert {s["doc"] for s in data["sources"]} == {"lecture.md"}  # рецепт отсеян по смыслу
    assert not any(s["overview"] for s in data["sources"])


def test_upload_survives_embedding_failure_and_reindex_fixes_it():
    c = semantic_client(FailingEmbedder())
    doc = upload(c, "lecture.md", "The inverse of a matrix.").json()
    assert doc["semantic"] is False and "429" in doc["semantic_error"]
    assert c.get("/api/docs").json()[0]["semantic"] is False

    c.app.state.embedder = FakeEmbedder()
    assert c.post("/api/docs/reindex").json() == {"indexed": 1, "failed": []}
    assert c.get("/api/docs").json()[0]["semantic"] is True


def test_reindex_requires_embeddings():
    assert make_client().post("/api/docs/reindex").status_code == 400


def test_query_embedding_failure_falls_back_to_keywords():
    c = semantic_client(FakeEmbedder())
    upload(c, "notes.md", DOC)
    c.app.state.embedder = FailingEmbedder()  # вопрос уже не удастся превратить в вектор
    data = c.post("/api/ask", json={"question": "сколько запросов в минуту?"}).json()
    assert data["sources"] and data["sources"][0]["doc"] == "notes.md"


def test_ask_stream_sends_sources_first():
    c = make_client()
    upload(c)
    r = c.post("/api/ask/stream", json={"question": "сколько секунд хранит кэш?"})
    assert r.status_code == 200
    first = r.text.split("\n\n")[0]
    assert first.startswith("data: ") and '"sources"' in first
    assert "data: [DONE]" in r.text


def test_docs_endpoints_require_service_key():
    c = make_client(service_api_key="secret")
    assert c.get("/api/docs").status_code == 401
    assert c.get("/api/docs", headers={"X-API-Key": "secret"}).status_code == 200
