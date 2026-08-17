"""
Step 6: tests.

Deliberately fast and deterministic — no LLM calls anywhere. Model behaviour is
measured by evaluate.py, which is a different job: evaluation asks "how good is
it", these tests ask "did something break". Mixing them would give you a suite
too slow to run often and too fuzzy to trust when it fails.

Every test that touches storage is redirected to a temporary directory, so
running the suite never disturbs the real chroma_db.

    python -m pytest test_rag.py -v
"""

import fitz
import pytest

import embed_store
from answer import (
    NO_ANSWER,
    build_prompt,
    dedupe_sources,
    normalise_refusal,
    validate_citations,
)
from ingest import chunk_text, load_pdf, process_all_pdfs


@pytest.fixture
def temp_store(tmp_path, monkeypatch):
    """Point the vector store at a throwaway directory for one test."""
    monkeypatch.setattr(embed_store, "CHROMA_DIR", str(tmp_path / "chroma"))
    return tmp_path


def make_pdf(path, text, pages=1):
    """Write a small text PDF for tests to ingest."""
    document = fitz.open()
    for _ in range(pages):
        page = document.new_page()
        page.insert_textbox(fitz.Rect(50, 50, 550, 750), text, fontsize=11)
    document.save(str(path))


# --------------------------------------------------------------- chunking


def test_chunks_overlap_so_boundary_sentences_survive():
    text = "".join(str(i % 10) for i in range(1200))
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    assert chunks[0][-50:] == chunks[1][:50]


def test_short_fragments_are_dropped():
    """Running headers are short and made of the document's own keywords, so
    they match vague queries structurally. They must not reach the index."""
    assert chunk_text("The economic potential of generative AI") == []
    assert len(chunk_text("x" * 300)) == 1


def test_overlap_larger_than_chunk_is_rejected():
    with pytest.raises(ValueError):
        chunk_text("some text " * 100, chunk_size=100, overlap=100)


def test_blank_pages_are_skipped(tmp_path):
    path = tmp_path / "mixed.pdf"
    document = fitz.open()
    document.new_page()  # blank
    page = document.new_page()
    page.insert_textbox(fitz.Rect(50, 50, 550, 750), "Real content. " * 40, fontsize=11)
    document.save(str(path))

    pages = load_pdf(str(path))
    assert [p["page_number"] for p in pages] == [2]


# --------------------------------------------------------------- citations


def test_invalid_citation_is_caught():
    """The model has been observed citing a page that was never retrieved."""
    chunks = [{"source": "r.pdf", "page": 5}, {"source": "r.pdf", "page": 10}]
    assert validate_citations("Claim [r.pdf, p.3].", chunks) == ["[r.pdf, p.3]"]


def test_valid_citations_pass_untouched():
    chunks = [{"source": "r.pdf", "page": 5}, {"source": "r.pdf", "page": 10}]
    assert validate_citations("A [r.pdf, p.5] and B [r.pdf, p.10].", chunks) == []


def test_citation_from_a_different_document_is_caught():
    chunks = [{"source": "a.pdf", "page": 5}]
    assert validate_citations("Claim [b.pdf, p.5].", chunks) == ["[b.pdf, p.5]"]


def test_prompt_labels_every_chunk_with_its_tag():
    """Chunks are pre-labelled so the model copies a tag rather than building
    one from digits in the text — which is how invented page numbers appear."""
    chunks = [{"rank": 1, "source": "r.pdf", "page": 7, "text": "Revenue grew."}]
    _, user_prompt = build_prompt("How much?", chunks)
    assert "[r.pdf, p.7]" in user_prompt
    assert "How much?" in user_prompt


# --------------------------------------------------------------- refusal


def test_refusal_with_a_citation_is_normalised():
    """Rule 2 tells the model to cite every sentence, and it sometimes applies
    that to the refusal too. Later steps need this string to match exactly."""
    assert normalise_refusal(f"{NO_ANSWER} [r.pdf, p.28]") == NO_ANSWER


def test_real_answers_are_never_rewritten():
    answer = "$2.6 trillion to $4.4 trillion [r.pdf, p.11]"
    assert normalise_refusal(answer) == answer


# --------------------------------------------------------------- sources


def test_sources_are_deduplicated_in_retrieval_order():
    chunks = [
        {"source": "a.pdf", "page": 5},
        {"source": "a.pdf", "page": 5},
        {"source": "a.pdf", "page": 9},
    ]
    assert dedupe_sources(chunks) == [
        {"source": "a.pdf", "page": 5},
        {"source": "a.pdf", "page": 9},
    ]


# --------------------------------------------------------------- isolation


def test_one_users_document_is_invisible_to_another(temp_store):
    """The most expensive thing in this project to break silently.

    Two people sharing an instance must not be able to reach each other's
    uploads. This is checked end to end - real embeddings, real vector search -
    because the guarantee depends on storage and query agreeing.
    """
    from retrieve import retrieve

    alice_dir = temp_store / "alice"
    alice_dir.mkdir()
    make_pdf(alice_dir / "salary.pdf", "Senior engineers earn 180000 dollars. " * 10)
    embed_store.embed_and_store(process_all_pdfs(str(alice_dir)), user_id="alice")

    question = "What do senior engineers earn?"
    assert retrieve(question, 3, user_id="alice"), "owner must see her own document"
    assert retrieve(question, 3, user_id="bob") == [], "another user must see nothing"


def test_chunk_ids_are_namespaced_per_user(temp_store):
    """Without a per-user ID prefix, two people uploading the same filename
    collide and the second upload is skipped as a duplicate — leaving them
    querying someone else's document."""
    for owner in ("alice", "bob"):
        folder = temp_store / owner
        folder.mkdir()
        make_pdf(folder / "policy.pdf", f"Policy for {owner}. " * 40)
        embed_store.embed_and_store(process_all_pdfs(str(folder)), user_id=owner)

    collection = embed_store.get_or_create_collection()
    ids = collection.get()["ids"]

    assert any(i.startswith("alice::") for i in ids)
    assert any(i.startswith("bob::") for i in ids)
    # Same filename, both stored: the namespace prevented the collision.
    assert sum(1 for i in ids if "policy.pdf" in i) == len(ids)


def test_reindexing_the_same_document_adds_nothing(temp_store):
    folder = temp_store / "docs"
    folder.mkdir()
    make_pdf(folder / "a.pdf", "Stable content for idempotency. " * 40)
    chunks = process_all_pdfs(str(folder))

    embed_store.embed_and_store(chunks, user_id="u1")
    first = embed_store.get_or_create_collection().count()
    embed_store.embed_and_store(chunks, user_id="u1")
    second = embed_store.get_or_create_collection().count()

    assert first == second and first > 0
