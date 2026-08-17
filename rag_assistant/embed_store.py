"""
Step 2 of the Enterprise RAG Document Assistant: embed chunks and store them.

This lives in its own file rather than inside ingest.py on purpose. Step 1
(reading PDFs into chunks) is fast, has no model to load, and is easy to test
on its own. Step 2 loads a neural network and writes to a database. Keeping
them apart means you can re-run either one without dragging the other along.

Nothing here searches, ranks, or answers anything — this file only fills the
database. Retrieval comes in a later step.
"""

import os

import chromadb
from sentence_transformers import SentenceTransformer

from ingest import BASE_DIR, process_all_pdfs

MODEL_NAME = "all-MiniLM-L6-v2"

# Every chunk is tagged with the user who uploaded it, and every search filters
# on that tag. Without it a single shared collection means one person's upload
# is searchable by everyone else — fine for a personal tool, a data leak the
# moment two people use the same instance.
DEFAULT_USER = "default"

# Anchored to the script's folder for the same reason as the documents folder:
# so the database is always the same database, no matter which directory you
# happened to launch from. A relative path here would silently create a second,
# empty chroma_db/ the first time you ran the script from somewhere else — and
# an empty database looks exactly like a broken one.
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "documents"

# Chroma rejects a single add() that is too large, and embedding thousands of
# chunks in one go is memory-hungry, so we work through the list in batches.
BATCH_SIZE = 200

# Module-level cache for the model. See get_embedder() for why.
_embedder = None


def get_embedder():
    """Load the sentence-transformers model once and reuse it.

    WHY EMBED AT ALL?
    Plain keyword search only finds documents containing the words you typed.
    Someone asking "how much time off do I get?" would never match a policy
    that says "annual leave entitlement" — no shared words. An embedding model
    converts text into a list of numbers (a vector) that represents its
    *meaning*, so passages about the same idea land near each other in that
    number-space even when the wording is completely different. That is what
    makes a RAG assistant feel like it understands the question.

    WHY THIS MODEL?
    'all-MiniLM-L6-v2' is small, free, and runs on your own machine with no API
    key and no per-request cost. Its 384-dimensional vectors are more than
    accurate enough at this scale, and staying local means document contents
    never leave the machine — which matters for enterprise documents.

    WHY CACHE IT?
    Loading the model reads weights from disk and takes a second or two. Doing
    that once per call would dominate the runtime of the whole pipeline, so we
    keep the loaded model in a module-level variable and hand back the same
    object every time.
    """
    global _embedder

    if _embedder is None:
        print(f"Loading embedding model '{MODEL_NAME}' (first run downloads it)...")
        _embedder = SentenceTransformer(MODEL_NAME)

    return _embedder


def get_or_create_collection():
    """Return the Chroma collection, creating it on first use.

    WHY A PERSISTENT CLIENT?
    Embedding is the slowest part of this pipeline. An in-memory client would
    throw all that work away the moment the script exits, forcing a full
    re-embed on every single run. PersistentClient writes the vectors to
    'chroma_db/' on disk, so the database is simply there the next time — you
    embed a document once and query it forever.

    A "collection" is Chroma's equivalent of a table: it holds the vectors,
    the original text, and the metadata together as one unit.
    """
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_or_create_collection(name=COLLECTION_NAME)


def embed_and_store(chunks, user_id: str = DEFAULT_USER):
    """Embed chunk dicts from process_all_pdfs() and store them in Chroma.

    WHY STORE METADATA NEXT TO THE VECTOR?
    A vector on its own is just 384 numbers — it has no memory of where it came
    from. When retrieval later pulls back the best-matching chunks, the answer
    is only trustworthy if we can say "this came from handbook.pdf, page 12".
    Chroma keeps the text and the metadata attached to each vector, so the
    citation travels with the match automatically. If we skipped this now, the
    information would be gone for good — you cannot reverse-engineer a filename
    out of a list of numbers.

    WHY IDEMPOTENCY?
    You will run this script many times: after adding one new PDF, after a
    crash, after changing something unrelated. Re-embedding everything each
    time would be slow, and adding the same chunk twice would let one passage
    win several retrieval slots and crowd out other documents. So every chunk
    gets a stable ID built from its filename and position, we ask Chroma which
    of those IDs it already has, and we only embed the genuinely new ones.
    Running this twice in a row is therefore safe and nearly instant.
    """
    collection = get_or_create_collection()

    # Build a stable ID per chunk: filename + its index within that file.
    # The index restarts per file so that adding a new PDF never renumbers the
    # chunks of an existing one — IDs stay stable across runs, which is exactly
    # what makes the duplicate check reliable.
    # The user id is part of the ID, not just the metadata. Two people
    # uploading a file called policy.pdf would otherwise generate identical
    # IDs, and the second upload would be silently skipped as a duplicate —
    # leaving that person querying someone else's document.
    per_source_index = {}
    ids = []
    for chunk in chunks:
        source = chunk["source"]
        index = per_source_index.get(source, 0)
        per_source_index[source] = index + 1
        ids.append(f"{user_id}::{source}_chunk_{index}")

    # Ask the database which of these IDs it already holds, in one call.
    existing_ids = set(collection.get(ids=ids)["ids"])

    new_ids = []
    new_texts = []
    new_metadatas = []

    for chunk_id, chunk in zip(ids, chunks):
        if chunk_id in existing_ids:
            continue

        new_ids.append(chunk_id)
        new_texts.append(chunk["text"])
        new_metadatas.append({
            "source": chunk["source"],
            "page": chunk["page"],
            "user_id": user_id,
        })

    skipped = len(chunks) - len(new_ids)

    if new_ids:
        embedder = get_embedder()
        print(f"Embedding {len(new_ids)} new chunk(s)...")

        for start in range(0, len(new_ids), BATCH_SIZE):
            end = start + BATCH_SIZE
            batch_texts = new_texts[start:end]

            # We compute the vectors ourselves rather than letting Chroma pick a
            # default model, so the exact same model is used here and at query
            # time later. Mixing models would put questions and documents in
            # different number-spaces and retrieval would return nonsense.
            embeddings = embedder.encode(batch_texts).tolist()

            collection.add(
                ids=new_ids[start:end],
                documents=batch_texts,
                metadatas=new_metadatas[start:end],
                embeddings=embeddings,
            )

    print(f"Embedded and stored: {len(new_ids)} chunk(s)")
    print(f"Skipped as already stored: {skipped} chunk(s)")
    print(f"Total vectors in collection '{COLLECTION_NAME}': {collection.count()}")

    return collection


if __name__ == "__main__":
    chunks = process_all_pdfs()

    # No chunks is a normal situation (empty folder, or scanned PDFs with no
    # text layer), not a bug — so say what happened and stop, rather than
    # loading a model and opening a database for no reason.
    if not chunks:
        print(
            "No chunks were produced, so there is nothing to embed.\n"
            "Put a text-based PDF in the 'documents' folder and run "
            "'python ingest.py' first to check it extracts correctly."
        )
        raise SystemExit(0)

    collection = embed_and_store(chunks)

    # Read one item straight back out of the database. This is the real check:
    # it proves the text and the source/page metadata actually survived storage,
    # which is what every citation later depends on.
    sample = collection.get(limit=1, include=["documents", "metadatas"])

    if sample["ids"]:
        print("\nSample stored item:")
        print(f"  id:     {sample['ids'][0]}")
        print(f"  source: {sample['metadatas'][0]['source']}")
        print(f"  page:   {sample['metadatas'][0]['page']}")
        print(f"  text:   {sample['documents'][0][:200]}")
