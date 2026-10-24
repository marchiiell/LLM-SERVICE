"""RAG без внешних зависимостей: извлечение текста, нарезка на чанки, гибридный поиск, хранилище.

Поиск состоит из двух частей:
  1. по словам (BM25): точные термины, названия, цифры; работает всегда и без интернета;
  2. по смыслу (векторы-эмбеддинги): находит похожее по значению, даже на другом языке.
Результаты объединяются методом RRF (Reciprocal Rank Fusion). Если векторов нет, работает только BM25.
"""
import json
import math
import re
import time
import uuid
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------- токенизация

_TOKEN_RE = re.compile(r"[a-zа-яё0-9]+", re.IGNORECASE)
_SUFFIX_RE = re.compile(
    r"(ами|ями|ого|его|ому|ему|ыми|ими|ах|ях|ов|ев|ей|ам|ям|ой|ий|ый|ая|яя|ое|ее|ые|ие|ию|ью|ом|ем|ы|и|а|я|о|е|у|ю|ь|й)$"
)
STOPWORDS = set(
    """и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по от из для это этот эта эти
    или если при о об до чем какой какая какие какое где когда есть быть был была были будет ли их мы они
    the a an and or of to in on is are for with that this""".split()
)


def stem(word: str) -> str:
    """Очень простой «стеммер»: срезает типичное русское окончание, чтобы «каша»/«каши»/«кашу» совпадали."""
    if len(word) <= 3 or not re.match(r"[а-я]", word):
        return word
    s = _SUFFIX_RE.sub("", word, count=1)
    return s if len(s) >= 3 else word


def tokenize(text: str) -> list[str]:
    words = _TOKEN_RE.findall(text.lower().replace("ё", "е"))
    return [stem(w) for w in words if w not in STOPWORDS]


# ---------------------------------------------------------------- извлечение текста

TEXT_EXT = {".txt", ".md", ".markdown", ".csv", ".log", ".rst"}


def extract_text(name: str, data: bytes) -> str:
    ext = Path(name).suffix.lower()
    if ext in TEXT_EXT:
        for enc in ("utf-8-sig", "cp1251"):
            try:
                return data.decode(enc)
            except UnicodeDecodeError:
                continue
        return data.decode("utf-8", errors="replace")
    if ext == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ValueError("Для PDF нужен пакет pypdf: pip install pypdf")
        import io

        reader = PdfReader(io.BytesIO(data))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)
    raise ValueError("Поддерживаются файлы .txt, .md и .pdf")


# ---------------------------------------------------------------- чанкинг


def _split_long(paragraph: str, size: int) -> list[str]:
    """Режем длинный абзац по предложениям, а слишком длинные предложения — жёстко."""
    pieces, cur = [], ""
    for sent in re.split(r"(?<=[.!?])\s+", paragraph):
        while len(sent) > size:
            if cur:
                pieces.append(cur)
                cur = ""
            pieces.append(sent[:size])
            sent = sent[size:]
        if cur and len(cur) + len(sent) + 1 > size:
            pieces.append(cur)
            cur = sent
        else:
            cur = f"{cur} {sent}".strip()
    if cur:
        pieces.append(cur)
    return pieces


def chunk_text(text: str, size: int = 800, overlap: int = 100) -> list[str]:
    """Нарезка на куски около `size` символов по границам абзацев, с небольшим перекрытием."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    cur = ""
    for para in paragraphs:
        pieces = [para] if len(para) <= size else _split_long(para, size)
        for piece in pieces:
            if cur and len(cur) + len(piece) + 2 > size:
                chunks.append(cur)
                tail = cur[-overlap:] if overlap else ""
                if " " in tail:
                    tail = tail[tail.find(" ") + 1 :]  # не начинаем с середины слова
                cur = f"{tail}\n\n{piece}" if tail else piece
            else:
                cur = f"{cur}\n\n{piece}" if cur else piece
    if cur:
        chunks.append(cur)
    return chunks


# ---------------------------------------------------------------- BM25


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self._tf: list[Counter] = []
        self._len: list[int] = []
        self._df: Counter = Counter()
        self._avgdl = 0.0

    def build(self, token_lists: list[list[str]]) -> None:
        self._tf = [Counter(t) for t in token_lists]
        self._len = [len(t) for t in token_lists]
        self._df = Counter()
        for tf in self._tf:
            self._df.update(tf.keys())
        self._avgdl = sum(self._len) / len(self._len) if self._len else 0.0

    def scores(self, query_tokens: list[str]) -> list[float]:
        n = len(self._tf)
        result = [0.0] * n
        if not n:
            return result
        for term in set(query_tokens):
            df = self._df.get(term, 0)
            if not df:
                continue
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            for i, tf in enumerate(self._tf):
                f = tf.get(term, 0)
                if not f:
                    continue
                norm = 1 - self.b + self.b * self._len[i] / (self._avgdl or 1)
                result[i] += idf * f * (self.k1 + 1) / (f + self.k1 * norm)
        return result


# ---------------------------------------------------------------- хранилище

RRF_K = 60  # стандартная константа RRF: чем больше, тем мягче вклад позиции в рейтинге


def _normalize(vec: list[float]) -> list[float] | None:
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm else None


class DocStore:
    """Документы, их чанки, векторы и поисковый индекс. Если задан путь, хранится в JSON-файле."""

    def __init__(
        self,
        path: str | Path | None = None,
        chunk_size: int = 800,
        overlap: int = 100,
        emb_signature: str | None = None,
        min_similarity: float = 0.5,
    ):
        self.path = Path(path) if path else None
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.emb_signature = emb_signature  # метка актуальных настроек эмбеддингов (None = векторы выключены)
        self.min_similarity = min_similarity
        self.docs: dict[str, dict] = {}
        self._flat: list[tuple[str, int]] = []  # (doc_id, номер чанка) в порядке индекса
        self._vecs: list[list[float] | None] = []  # нормированные векторы в том же порядке
        self._index = BM25Index()
        self._load()
        self._rebuild()

    # --- публичный интерфейс
    def split(self, text: str) -> list[str]:
        return chunk_text(text, self.chunk_size, self.overlap)

    def add(self, name: str, text: str, vectors: list[list[float]] | None = None) -> dict:
        return self.add_chunks(name, self.split(text), vectors)

    def add_chunks(self, name: str, chunks: list[str], vectors: list[list[float]] | None = None) -> dict:
        doc_id = uuid.uuid4().hex[:8]
        doc = {"id": doc_id, "name": name, "chunks": chunks, "added": int(time.time()), "vectors": None, "emb": None}
        self.docs[doc_id] = doc
        if vectors is not None:
            self._attach_vectors(doc, vectors)
        self._rebuild()
        self._save()
        return self._summary(doc)

    def set_vectors(self, doc_id: str, vectors: list[list[float]]) -> dict:
        doc = self.docs[doc_id]
        self._attach_vectors(doc, vectors)
        self._rebuild()
        self._save()
        return self._summary(doc)

    def delete(self, doc_id: str) -> bool:
        if doc_id not in self.docs:
            return False
        del self.docs[doc_id]
        self._rebuild()
        self._save()
        return True

    def list_docs(self) -> list[dict]:
        return [self._summary(d) for d in sorted(self.docs.values(), key=lambda d: d["added"])]

    def docs_without_vectors(self) -> list[dict]:
        return [d for d in self.docs.values() if not self._is_semantic(d)]

    @property
    def has_vectors(self) -> bool:
        return any(self._is_semantic(d) for d in self.docs.values())

    def search(self, query: str, top_k: int = 4, query_vector: list[float] | None = None) -> list[dict]:
        n = len(self._flat)
        if not n:
            return []

        # 1) по словам
        tokens = tokenize(query)
        kw_scores = self._index.scores(tokens) if tokens else [0.0] * n
        kw_rank = [i for i in sorted(range(n), key=lambda i: kw_scores[i], reverse=True) if kw_scores[i] > 0]

        # 2) по смыслу: косинусное сходство нормированных векторов
        sims: dict[int, float] = {}
        qv = _normalize(query_vector) if query_vector else None
        if qv:
            for i, v in enumerate(self._vecs):
                if v is not None and len(v) == len(qv):
                    s = sum(a * b for a, b in zip(qv, v))
                    if s >= self.min_similarity:
                        sims[i] = s
        vec_rank = sorted(sims, key=sims.get, reverse=True)

        # 3) объединение рейтингов (RRF)
        fused: dict[int, float] = {}
        for ranking in (kw_rank, vec_rank):
            for pos, i in enumerate(ranking):
                fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + pos + 1)
        top = sorted(fused, key=fused.get, reverse=True)[:top_k]

        kw_set = set(kw_rank)
        hits = []
        for i in top:
            doc_id, chunk_idx = self._flat[i]
            doc = self.docs[doc_id]
            hits.append(
                {
                    "doc_id": doc_id,
                    "doc_name": doc["name"],
                    "chunk_index": chunk_idx,
                    "text": doc["chunks"][chunk_idx],
                    "score": round(fused[i], 4),
                    "similarity": round(sims[i], 3) if i in sims else None,
                    "keyword": i in kw_set,
                }
            )
        return hits

    def overview(self, max_chunks: int = 4, per_doc: int = 2) -> list[dict]:
        """Начало документов: используется, когда по вопросу ничего конкретного не нашлось."""
        hits: list[dict] = []
        for doc in sorted(self.docs.values(), key=lambda d: d["added"]):
            for idx in range(min(per_doc, len(doc["chunks"]))):
                if len(hits) >= max_chunks:
                    return hits
                hits.append(
                    {
                        "doc_id": doc["id"],
                        "doc_name": doc["name"],
                        "chunk_index": idx,
                        "text": doc["chunks"][idx],
                        "score": 0.0,
                        "similarity": None,
                        "keyword": False,
                    }
                )
        return hits

    # --- внутреннее
    def _is_semantic(self, doc: dict) -> bool:
        return bool(doc.get("vectors")) and self.emb_signature is not None and doc.get("emb") == self.emb_signature

    def _attach_vectors(self, doc: dict, vectors: list[list[float]]) -> None:
        if len(vectors) != len(doc["chunks"]):
            raise ValueError("число векторов не совпадает с числом фрагментов")
        doc["vectors"] = [[round(x, 5) for x in v] for v in vectors]
        doc["emb"] = self.emb_signature

    def _summary(self, doc: dict) -> dict:
        return {
            "id": doc["id"],
            "name": doc["name"],
            "chunks": len(doc["chunks"]),
            "chars": sum(len(c) for c in doc["chunks"]),
            "semantic": self._is_semantic(doc),
        }

    def _rebuild(self) -> None:
        self._flat = [(d["id"], i) for d in self.docs.values() for i in range(len(d["chunks"]))]
        self._index = BM25Index()
        self._index.build([tokenize(self.docs[d]["chunks"][i]) for d, i in self._flat])
        self._vecs = [
            _normalize(self.docs[d]["vectors"][i]) if self._is_semantic(self.docs[d]) else None
            for d, i in self._flat
        ]

    def _save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"docs": list(self.docs.values())}, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)

    def _load(self) -> None:
        if not self.path or not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.docs = {d["id"]: d for d in data.get("docs", [])}
        except (json.JSONDecodeError, KeyError, OSError):
            self.docs = {}  # повреждённый файл: начинаем с чистого индекса
