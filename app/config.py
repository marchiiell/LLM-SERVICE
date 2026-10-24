"""Настройки сервиса. Всё берётся из переменных окружения (см. .env.example)."""
import os
from dataclasses import dataclass


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


# Провайдеры с OpenAI-совместимым API: (base_url, модель по умолчанию)
OPENAI_COMPAT = {
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "gemini-3.8-flash"),
    "groq": ("https://api.groq.com/openai/v1", "openai/gpt-oss-20b"),
    "xai": ("https://api.x.ai/v1", ""),  # модель для xAI задайте сами через MODEL
}
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5-5"

# Порядок автоопределения провайдера: первый, для которого задан ключ
AUTO_ORDER = ["gemini", "groq", "xai", "anthropic"]


@dataclass(frozen=True)
class Settings:
    # "auto" — выбрать по тому, какой ключ задан; либо gemini | groq | xai | anthropic | mock
    provider: str = os.getenv("PROVIDER", "auto").lower()
    model: str = os.getenv("MODEL", "")  # пусто = модель по умолчанию для провайдера

    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    xai_api_key: str = os.getenv("XAI_API_KEY", "")
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")

    # Для моделей с «рассуждением» лимит включает скрытые токены, поэтому по умолчанию побольше
    max_tokens: int = _int("MAX_TOKENS", 2048)
    system_prompt: str = os.getenv(
        "SYSTEM_PROMPT", "Ты полезный ассистент. Отвечай кратко и по делу."
    )

    # Rate limit: сколько запросов с одного клиента за окно (секунды)
    rate_limit_requests: int = _int("RATE_LIMIT_REQUESTS", 10)
    rate_limit_window: int = _int("RATE_LIMIT_WINDOW", 60)

    # Кэш ответов
    cache_ttl: int = _int("CACHE_TTL", 300)
    cache_max_items: int = _int("CACHE_MAX_ITEMS", 500)

    # Сколько секунд ждать данных от модели, прежде чем считать запрос зависшим
    llm_timeout: float = float(os.getenv("LLM_TIMEOUT", 120))

    # Очередь: максимум одновременных вызовов LLM и сколько ждать своей очереди
    max_concurrent_llm: int = _int("MAX_CONCURRENT_LLM", 4)
    queue_timeout: float = float(os.getenv("QUEUE_TIMEOUT", 15))

    # RAG: где хранить загруженные документы (пусто = только в памяти), размер чанка и лимит файла
    data_dir: str = os.getenv("DATA_DIR", "data")
    rag_chunk_size: int = _int("RAG_CHUNK_SIZE", 800)
    rag_overlap: int = _int("RAG_OVERLAP", 100)
    max_upload_mb: int = _int("MAX_UPLOAD_MB", 5)

    # Поиск по смыслу (эмбеддинги Gemini). Работает при заданном GEMINI_API_KEY, даже если
    # чат идёт через другого провайдера (например, Groq). EMBEDDINGS=off отключает.
    embeddings: str = os.getenv("EMBEDDINGS", "auto").lower()
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
    embedding_dimensions: int = _int("EMBEDDING_DIMENSIONS", 768)
    # Минимальное косинусное сходство, чтобы фрагмент считался подходящим по смыслу
    min_similarity: float = float(os.getenv("RAG_MIN_SIMILARITY", 0.5))

    # Необязательный ключ доступа к вашему API (заголовок X-API-Key)
    service_api_key: str = os.getenv("SERVICE_API_KEY", "")

    def _key_for(self, provider: str) -> str:
        return {
            "gemini": self.gemini_api_key,
            "groq": self.groq_api_key,
            "xai": self.xai_api_key,
            "anthropic": self.anthropic_api_key,
        }.get(provider, "")

    @property
    def active_provider(self) -> str:
        if self.provider in ("mock", ""):
            return "mock"
        if self.provider != "auto":
            return self.provider if self._key_for(self.provider) else "mock"
        for p in AUTO_ORDER:
            if self._key_for(p):
                return p
        return "mock"

    @property
    def active_key(self) -> str:
        return self._key_for(self.active_provider)

    @property
    def active_model(self) -> str:
        if self.model:
            return self.model
        p = self.active_provider
        if p == "anthropic":
            return DEFAULT_ANTHROPIC_MODEL
        if p in OPENAI_COMPAT:
            return OPENAI_COMPAT[p][1]
        return "mock"

    @property
    def base_url(self) -> str:
        return OPENAI_COMPAT.get(self.active_provider, ("", ""))[0]

    @property
    def embeddings_enabled(self) -> bool:
        return self.embeddings != "off" and bool(self.gemini_api_key)

    @property
    def embedding_signature(self) -> str | None:
        """Метка текущих настроек эмбеддингов: векторы с другой меткой считаются устаревшими."""
        if not self.embeddings_enabled:
            return None
        return f"{self.embedding_model}:{self.embedding_dimensions}"

    @property
    def mock_mode(self) -> bool:
        """Без ключа сервис работает на заглушке: удобно для разработки и тестов."""
        return self.active_provider == "mock"


def get_settings() -> Settings:
    return Settings()
