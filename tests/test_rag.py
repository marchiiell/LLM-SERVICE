"""Тесты RAG-модуля: токенизация, нарезка, поиск, хранилище. Без сети и внешних пакетов."""
import pytest

from app.rag import DocStore, chunk_text, extract_text, stem, tokenize

DOC = """Сервис ограничивает частоту запросов: не больше 10 запросов в минуту с одного клиента.

Кэш хранит одинаковые ответы 300 секунд, поэтому повторный вопрос приходит мгновенно.

Рецепт овсяной каши: залейте хлопья водой и варите пять минут, добавьте соль и масло."""


def test_stemming_matches_russian_forms():
    assert stem("каша") == stem("каши") == stem("кашу") == stem("каше")
    assert tokenize("Овсяной каши") == tokenize("овсяная каша")


def test_tokenize_drops_stopwords_and_lowercases():
    assert "и" not in tokenize("Каша и масло")
    assert tokenize("ЁЖ") == tokenize("еж")


def test_chunk_text_respects_size_and_keeps_all_content():
    text = "\n\n".join(f"Абзац номер {i}. " + "слово " * 40 for i in range(12))
    chunks = chunk_text(text, size=500, overlap=60)
    assert len(chunks) > 1
    assert all(len(c) <= 500 + 60 + 2 for c in chunks)
    joined = "\n".join(chunks)
    assert all(f"Абзац номер {i}." in joined for i in range(12))


def test_chunk_text_splits_one_huge_paragraph():
    chunks = chunk_text("Предложение. " * 200, size=300, overlap=0)
    assert len(chunks) > 1
    assert all(len(c) <= 300 for c in chunks)


def test_chunk_text_empty():
    assert chunk_text("   \n\n  ") == []


def test_search_finds_relevant_chunk():
    store = DocStore(chunk_size=120, overlap=0)
    store.add("notes.md", DOC)
    hits = store.search("сколько запросов в минуту разрешено", top_k=2)
    assert hits and "10 запросов" in hits[0]["text"]
    assert hits[0]["doc_name"] == "notes.md"

    hits = store.search("как сварить овсяную кашу")
    assert hits and "хлопья" in hits[0]["text"]


def test_search_unrelated_returns_nothing():
    store = DocStore()
    store.add("notes.md", DOC)
    assert store.search("квантовая хромодинамика") == []
    assert store.search("и в на") == []  # одни стоп-слова


def test_search_empty_store():
    assert DocStore().search("что угодно") == []


def test_add_list_delete():
    store = DocStore()
    d = store.add("a.txt", "Первый документ про яблоки.")
    store.add("b.txt", "Второй документ про груши.")
    assert [x["name"] for x in store.list_docs()] == ["a.txt", "b.txt"]
    assert store.search("яблоки")[0]["doc_name"] == "a.txt"
    assert store.delete(d["id"]) is True
    assert store.delete(d["id"]) is False
    assert store.search("яблоки") == []
    assert store.search("груши")[0]["doc_name"] == "b.txt"


def test_persistence_roundtrip(tmp_path):
    path = tmp_path / "data" / "index.json"
    DocStore(path).add("notes.md", DOC)
    reloaded = DocStore(path)
    assert [d["name"] for d in reloaded.list_docs()] == ["notes.md"]
    assert reloaded.search("кэш ответы")


def test_corrupt_index_file_is_ignored(tmp_path):
    path = tmp_path / "index.json"
    path.write_text("{ не json", encoding="utf-8")
    assert DocStore(path).list_docs() == []


def test_extract_text_encodings():
    assert extract_text("a.txt", "Привет".encode("utf-8")) == "Привет"
    assert extract_text("a.txt", "Привет".encode("utf-8-sig")) == "Привет"
    assert extract_text("a.md", "Привет".encode("cp1251")) == "Привет"  # файлы из Блокнота Windows


def test_extract_text_rejects_unknown_type():
    with pytest.raises(ValueError):
        extract_text("virus.exe", b"MZ")


# ----------------------------------------------------------------- векторы, гибридный поиск, обзор


def test_semantic_search_finds_text_in_other_language():
    store = DocStore(emb_signature="m:2", min_similarity=0.5)
    store.add_chunks("a.txt", ["The inverse of a matrix", "Porridge recipe with oats"], vectors=[[1, 0], [0, 1]])
    # у вопроса и текста нет общих слов, поэтому находит только поиск по смыслу
    hits = store.search("обратная матрица", top_k=3, query_vector=[0.9, 0.1])
    assert [h["text"] for h in hits] == ["The inverse of a matrix"]
    assert hits[0]["similarity"] > 0.9
    assert hits[0]["keyword"] is False


def test_hybrid_combines_keyword_and_vector_rankings():
    store = DocStore(emb_signature="m:2", min_similarity=0.0)
    store.add_chunks("a.txt", ["alpha beta", "gamma delta", "alpha gamma"], vectors=[[1, 0], [0, 1], [0.7, 0.7]])
    hits = store.search("alpha", top_k=3, query_vector=[1, 0])
    assert hits[0]["text"] == "alpha beta"  # лучший и по словам, и по смыслу
    assert hits[0]["keyword"] is True and hits[0]["similarity"] == 1.0


def test_low_similarity_is_filtered_out():
    store = DocStore(emb_signature="m:2", min_similarity=0.5)
    store.add_chunks("a.txt", ["Porridge recipe"], vectors=[[0, 1]])
    assert store.search("matrix", query_vector=[1, 0]) == []


def test_vectors_ignored_when_embedding_settings_change(tmp_path):
    path = tmp_path / "index.json"
    DocStore(path, emb_signature="m:2").add_chunks("a.txt", ["some text"], vectors=[[1, 0]])

    same = DocStore(path, emb_signature="m:2")
    assert same.list_docs()[0]["semantic"] is True and same.has_vectors

    other = DocStore(path, emb_signature="m:3")  # другая модель или размерность
    assert other.list_docs()[0]["semantic"] is False and not other.has_vectors
    assert len(other.docs_without_vectors()) == 1
    assert other.search("text")  # по словам по-прежнему работает

    assert DocStore(path).list_docs()[0]["semantic"] is False  # эмбеддинги выключены


def test_set_vectors_checks_length():
    store = DocStore(emb_signature="m:2")
    d = store.add_chunks("a.txt", ["x y", "z"])
    assert d["semantic"] is False
    with pytest.raises(ValueError):
        store.set_vectors(d["id"], [[1, 0]])
    assert store.set_vectors(d["id"], [[1, 0], [0, 1]])["semantic"] is True


def test_overview_returns_beginning_of_documents():
    store = DocStore(chunk_size=50, overlap=0)
    store.add("a.txt", "Первый абзац документа А.\n\nВторой абзац документа А.\n\nТретий абзац документа А.")
    store.add("b.txt", "Документ Б.")
    ov = store.overview(max_chunks=3, per_doc=2)
    assert [h["doc_name"] for h in ov] == ["a.txt", "a.txt", "b.txt"]
    assert [h["chunk_index"] for h in ov] == [0, 1, 0]
    assert DocStore().overview() == []
