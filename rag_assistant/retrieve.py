"""
Step 3 of the Enterprise RAG Document Assistant: retrieval.

Given a question, find the chunks in the database whose meaning is closest to
it. That is all this file does — it does not write a prompt, call an LLM, or
format a citation.

WHY IS RETRIEVAL ITS OWN STEP?
Two reasons, one practical and one that matters more than it first appears.

The practical one: retrieval is testable on its own. You can run a question and
read the chunks that come back, and judge them yourself in a second. Once an
LLM sits on top, a bad answer could be caused by bad retrieval, a bad prompt,
or the model itself — three suspects instead of one.

The one that matters more: retrieval quality is a ceiling on answer quality.
The LLM can only work with the passages it is handed. If the right passage is
not in the top k, no amount of prompt engineering will produce a correct
answer — the model will either say it doesn't know or, worse, invent something
plausible. So it is worth getting this step visibly right before building
anything on top of it.
"""

import sys

# Imported, never re-implemented. This is the single most important line in the
# file. The stored document vectors were produced by this exact model, and a
# question can only be compared against them if it is turned into a vector the
# same way. Two different models produce two different "meaning-spaces", and
# distances measured between them are arithmetic on unrelated numbers.
#
# The dangerous part is that this failure is silent. There is no error, no
# warning, no dimension mismatch to catch it — Chroma happily computes
# distances and returns its k nearest neighbours, which are simply the wrong
# chunks. Retrieval looks like it is working and quietly returns noise.
from embed_store import get_embedder, get_or_create_collection


def retrieve(query: str, k: int = 4, user_id: str | None = None) -> list[dict]:
    """Return the k chunks whose meaning is closest to `query`.

    Each result is a dict with 'text', 'source', 'page', 'distance' and 'rank'.

    WHY RETURN THE DISTANCE?
    Because "the closest chunk" and "a chunk that actually answers the question"
    are not the same thing. Nearest-neighbour search always returns something —
    ask about a topic the documents never mention and you still get k results,
    just distant ones. The distance is what lets a caller tell those apart, so
    a later step can decline to answer instead of confidently citing an
    irrelevant passage. It is also what you need for evaluation: to measure
    whether a change to chunking or the model helped, you need a number, not an
    impression.

    How to read it: this collection uses L2 distance and the model emits
    unit-length vectors, so distance relates to cosine similarity as
    similarity ~= 1 - distance / 2. Lower is closer. Roughly, 0.3-0.6 is a
    strong match, around 1.0 is weak, and 2.0 means unrelated.
    """
    if not query or not query.strip():
        print("Warning: empty query — nothing to search for.")
        return []

    collection = get_or_create_collection()
    count = collection.count()

    # An empty database is the most common cause of confusing results, and it
    # looks identical to "no good matches" if we just return an empty list
    # without saying why.
    if count == 0:
        print(
            "Warning: the 'documents' collection is empty, so there is nothing "
            "to retrieve.\nPut a PDF in the documents folder and run "
            "'python embed_store.py' first."
        )
        return []

    # Asking for more results than exist is not an error — it just means "give
    # me everything". Chroma already caps this internally, but clamping here
    # keeps the printed ranks honest and makes the behaviour obvious to anyone
    # reading the code.
    k = min(k, count)

    embedder = get_embedder()

    # encode() must be given a list. Passing a bare string returns a 1-D array
    # of shape (384,), while query_embeddings expects a list of vectors —
    # shape (1, 384) — because Chroma supports searching several questions at
    # once.
    query_embedding = embedder.encode([query]).tolist()

    # We pass query_embeddings, never query_texts. With query_texts, Chroma
    # would embed the question using its own bundled default model instead of
    # ours — the exact vector-space mismatch described at the top of this file.
    # With a user_id, the search is confined to that user's own chunks. Passing
    # None searches everything, which is right for a single-user CLI and wrong
    # for anything serving more than one person — so a hosted caller must always
    # supply it.
    response = collection.query(
        query_embeddings=query_embedding,
        n_results=k,
        where={"user_id": user_id} if user_id else None,
        include=["documents", "metadatas", "distances"],
    )

    # Chroma answers in batch form: one list of results per query. We sent a
    # single question, so everything we want is at index 0.
    documents = response["documents"][0]
    metadatas = response["metadatas"][0]
    distances = response["distances"][0]

    results = []
    for position, (text, metadata, distance) in enumerate(
        zip(documents, metadatas, distances)
    ):
        results.append({
            "text": text,
            # Carried through from ingest, untouched. This is what makes a
            # citation possible later — the answer can name the file and page.
            "source": metadata["source"],
            "page": metadata["page"],
            "distance": distance,
            "rank": position + 1,  # 1-indexed: humans count from one
        })

    return results


def format_results(results: list[dict]) -> str:
    """Render results as readable text for checking things at the command line.

    Returns a string rather than printing, so the caller decides what to do
    with it. Debugging aid only — not how answers will be shown to users.
    """
    if not results:
        return "No results."

    lines = [f"Retrieved {len(results)} chunk(s):"]

    for result in results:
        preview = result["text"][:200].replace("\n", " ")
        lines.append("")
        lines.append(
            f"[{result['rank']}] distance={result['distance']:.4f}  "
            f"source={result['source']}  page={result['page']}"
        )
        lines.append(f"    {preview}")

    return "\n".join(lines)


if __name__ == "__main__":
    # Everything after the script name is the question, so it can be typed
    # without quotes. Falls back to a prompt when no arguments are given.
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = input("Ask: ")

    print(f"\nQuestion: {question}")
    print(format_results(retrieve(question, k=4)))
