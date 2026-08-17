"""
Step 8: Streamlit UI.

The whole design is shaped by one number: on CPU this model produces a few
tokens a second, so a complete answer takes 30-60 seconds. Rather than hide that
behind a spinner, the page fills in as work completes — sources appear the
moment retrieval finishes (about seven seconds), then the answer types itself
underneath while the reader is already looking at where it came from.
"""

import time
import uuid

import streamlit as st

st.set_page_config(page_title="RAG Document Assistant", page_icon="📄", layout="wide")

# Distances measured over the Step 7 eval sets: in-domain questions scored
# 0.32-1.09, out-of-domain 1.51-1.68. These cutoffs come from that data rather
# than from taste.
STRONG_MATCH = 0.6
WEAK_MATCH = 1.2


@st.cache_resource(show_spinner="Loading embedding model...")
def load_backend():
    """Load the model and open the collection exactly once.

    Streamlit re-runs this whole script on every interaction. Without caching,
    each click would re-import torch and reload the embedding model — around a
    minute of work per keystroke. cache_resource keeps one copy alive for the
    life of the server, so only the LLM call is ever slow.
    """
    from embed_store import COLLECTION_NAME, get_embedder, get_or_create_collection

    return get_embedder(), get_or_create_collection(), COLLECTION_NAME


def ollama_models() -> list[str] | None:
    """Model names from Ollama, or None when the server cannot be reached."""
    import json
    import urllib.request

    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3) as r:
            return [m.get("name", "") for m in json.load(r).get("models", [])]
    except Exception:
        return None


def confidence(distance: float) -> tuple[str, str]:
    if distance < STRONG_MATCH:
        return "🟢", "strong match"
    if distance < WEAK_MATCH:
        return "🟡", "moderate match"
    return "🔴", "weak match — the documents may not cover this"


embedder, collection, collection_name = load_backend()
models = ollama_models()

# One identity per browser session. Everything this person uploads is tagged
# with it, and every search filters on it, so two people using the same server
# cannot see each other's documents.
#
# Known limitation: a session ends when the tab closes, and a refresh may issue
# a new id — so uploads are effectively per-visit. Making documents outlive a
# session needs real accounts, which is the next piece of work, not this one.
if "user_id" not in st.session_state:
    st.session_state.user_id = f"session-{uuid.uuid4().hex[:12]}"
user_id = st.session_state.user_id


def user_sources() -> set[str]:
    """Filenames indexed by this session only."""
    if collection.count() == 0:
        return set()
    rows = collection.get(where={"user_id": user_id}, include=["metadatas"])
    return {m["source"] for m in rows["metadatas"]}

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Status")

    # Scoped to this session, not collection.count(). A global count would tell
    # a visitor how much everyone else has uploaded, and reads as a
    # contradiction next to an empty document list.
    session_rows = collection.get(where={"user_id": user_id}, include=[])
    vector_count = len(session_rows["ids"])
    st.markdown(f"{'🟢' if vector_count else '⚪'} **{vector_count}** vectors in this session")

    if models is None:
        st.markdown("🔴 Ollama unreachable")
        st.caption("Start Ollama, then reload this page.")
    else:
        st.markdown(f"🟢 Ollama ready — {len(models)} model(s)")

    st.divider()
    st.subheader("Settings")

    model = st.selectbox("Model", models or ["llama3.2:3b"], index=0)
    k = st.slider("Chunks retrieved", 1, 10, 4)
    # Capped at 0.3 deliberately. Citation behaviour was measured at 0.1, and a
    # high temperature would produce hallucinations that look like a bug in the
    # system rather than a setting the user chose.
    temperature = st.slider("Temperature", 0.0, 0.3, 0.1, 0.05)

    st.divider()
    st.subheader("Documents")

    indexed = user_sources()
    if indexed:
        for source in sorted(indexed):
            st.caption(f"📄 {source}")
    else:
        st.caption("No documents in this session yet — upload one below.")
    st.caption(f"Session: `{user_id}`")

    uploaded = st.file_uploader(
        "Add PDFs", type=["pdf"], accept_multiple_files=True,
        help="Files are saved to documents/ and shared with the CLI.",
    )

    if uploaded:
        # Comparing against what is already indexed lets us say "0 new" instead
        # of making someone wait half a minute for a no-op. Re-indexing an
        # existing file is harmless either way — embed_and_store skips by ID.
        new_files = [f for f in uploaded if f.name not in indexed]
        for f in uploaded:
            state = "new" if f.name not in indexed else "already indexed"
            st.caption(f"{'✓' if f.name not in indexed else '•'} {f.name} — {state}")

        if not new_files:
            st.caption("Nothing new to index.")
        elif st.button(f"Index {len(new_files)} new document(s)", type="primary"):
            import os

            from embed_store import embed_and_store
            from ingest import BASE_DIR, process_all_pdfs

            # Each session gets its own folder. process_all_pdfs() reads a whole
            # directory, so a shared one would sweep up every other user's files
            # and re-tag them with this user's id — turning the isolation
            # mechanism into the leak it was meant to prevent.
            user_dir = os.path.join(BASE_DIR, "documents", "sessions", user_id)
            os.makedirs(user_dir, exist_ok=True)
            for f in new_files:
                with open(os.path.join(user_dir, f.name), "wb") as handle:
                    handle.write(f.getbuffer())

            # Embedding is the slow part — roughly a second per ten chunks — so
            # say so rather than letting the page look frozen.
            with st.spinner("Reading PDFs and embedding new chunks..."):
                chunks = process_all_pdfs(user_dir)
                embed_and_store(chunks, user_id=user_id)

            st.success(f"Indexed {len(new_files)} document(s).")
            st.rerun()

# ---------------------------------------------------------------- main
st.title("📄 Enterprise RAG Document Assistant")
st.caption("Answers come only from your indexed documents, with page citations.")

question = st.chat_input("Ask a question about your documents...")

if question:
    if not indexed:
        st.error("No documents in this session. Upload a PDF in the sidebar first.")
        st.stop()
    if models is None:
        st.error("Ollama is not reachable at localhost:11434.")
        st.stop()

    st.markdown(f"**Question:** {question}")

    from answer import finalise_answer, stream_answer
    from retrieve import retrieve

    # Phase 1 — retrieval. Fast enough to wait for, and it gives the reader
    # something to look at while the slow part runs.
    retrieval_start = time.perf_counter()
    with st.spinner("Searching documents..."):
        chunks = retrieve(question, k, user_id=user_id)
    retrieval_seconds = time.perf_counter() - retrieval_start

    if not chunks:
        st.warning("Nothing retrieved for that question.")
        st.stop()

    icon, label = confidence(chunks[0]["distance"])
    st.markdown(f"{icon} **{label}** · closest distance {chunks[0]['distance']:.3f}")

    # Phase 2 — show the sources immediately, before generation starts.
    with st.expander(f"📚 Retrieved passages ({len(chunks)})", expanded=False):
        for chunk in chunks:
            st.markdown(
                f"**{chunk['rank']}. {chunk['source']}, p.{chunk['page']}** "
                f"· distance {chunk['distance']:.3f}"
            )
            st.text(chunk["text"][:600])
            st.divider()

    # Phase 3 — stream the answer.
    st.subheader("Answer")
    generation_start = time.perf_counter()
    answer_text = st.write_stream(stream_answer(question, chunks, model, temperature))
    generation_seconds = time.perf_counter() - generation_start

    # Phase 4 — citations can only be checked once the text is complete, so the
    # verdict necessarily arrives after the answer is already on screen.
    result = finalise_answer(
        question, answer_text, chunks, model, retrieval_seconds, generation_seconds
    )

    if result["invalid_citations"]:
        st.warning(
            "**Unverified citations:** "
            + ", ".join(f"`{c}`" for c in result["invalid_citations"])
            + " — these pages were not among the passages retrieved, so the "
            "claim cannot be checked against them."
        )

    st.subheader("Sources")
    for position, source in enumerate(result["sources"], start=1):
        st.markdown(f"{position}. `{source['source']}` — page {source['page']}")

    st.caption(
        f"Retrieval {retrieval_seconds:.1f}s · generation {generation_seconds:.1f}s "
        f"· model {model}"
    )
