"""
Step 4: grounded answer generation. Takes the chunks Step 3 retrieved and asks
a local LLM to answer using nothing else. Adds no search of its own — it
consumes retrieve() and does not modify it.
"""

import os
import re
import sys
import time

import ollama

from retrieve import retrieve

# Matches a citation like "[report.pdf, p.26]" inside the model's answer.
CITATION_PATTERN = re.compile(r"\[([^\]]+?),\s*p\.(\d+)\]")

# The refusal wording, as a constant so later steps can check for it exactly.
NO_ANSWER = "I don't have that information in the provided documents."


def build_prompt(query: str, chunks: list[dict]) -> tuple[str, str]:
    """Turn a question and its retrieved chunks into (system, user) prompts.

    WHY THE SYSTEM PROMPT FORBIDS OUTSIDE KNOWLEDGE
    Left to itself, an LLM tops up the retrieved passages with whatever it
    remembers from training and blends both into one fluent answer. The result
    reads exactly like a grounded answer while containing claims the documents
    never made, and nothing marks which is which. That is the classic RAG
    failure. Forbidding outside knowledge, and giving the model an explicit way
    to say "not in here", turns an invisible hallucination into a visible
    refusal you can act on.

    WHY CITATIONS ARE MANDATORY AND FORMATTED EXACTLY
    A citation lets a reader check the answer against the page it came from.
    Without one, a correct answer and a fabricated one look identical. Note
    that each chunk below is pre-labelled with its own "[source, p.N]" tag: the
    model only has to copy that tag, not build one from metadata. Small models
    copy far more reliably than they construct, which is what keeps page
    numbers real rather than invented.
    """
    system_prompt = (
        "You answer questions using ONLY the context provided by the user.\n"
        "\n"
        "Rules:\n"
        "1. Use only the context. Never use outside or prior knowledge.\n"
        "2. Every factual sentence MUST end with a citation tag. Use only the "
        "tags printed above the chunks, copied character for character. Do not "
        "invent a tag from numbers inside the chunk text.\n"
        f"3. If the context does not contain the answer, reply with exactly "
        f"this sentence and nothing else, with NO citation tag: {NO_ANSWER}\n"
        "4. Do not speculate, infer beyond the text, or fill gaps.\n"
        "5. Be concise and start with the answer itself. Never open with "
        "'According to', 'Based on the context', or any similar preamble.\n"
        "\n"
        "Example of a correct answer:\n"
        "Context:\n"
        "[report.pdf, p.7]\n"
        "Revenue grew 12 percent in 2023.\n"
        "Question: How much did revenue grow?\n"
        "Answer: Revenue grew 12 percent in 2023. [report.pdf, p.7]"
    )

    # The tag sits alone on its own line above the text. An earlier version
    # prefixed each chunk with its rank as "1. [tag]: text", which collided
    # with the report's own numbered lists — the model read a "3." out of the
    # body text and cited a page 3 that was never retrieved.
    blocks = [f"[{c['source']}, p.{c['page']}]\n{c['text']}" for c in chunks]
    context = "\n\n".join(blocks) if blocks else "(no context)"
    return system_prompt, f"Context:\n{context}\n\nQuestion: {query}"


def validate_citations(answer: str, chunks: list[dict]) -> list[str]:
    """Return citations in `answer` that point outside the retrieved chunks.

    A citation the reader cannot check is worse than no citation at all: it
    looks verifiable, so nobody verifies it. The model has been seen inventing
    a page number out of digits inside a chunk's text, producing a tag that
    reads correctly and points nowhere.

    This check is plain string matching, not another model call, so it cannot
    hallucinate in turn. It reports rather than rewrites — silently editing the
    answer would change its meaning with no record of what changed.
    """
    valid = {(chunk["source"], chunk["page"]) for chunk in chunks}
    invalid = []
    for source, page in CITATION_PATTERN.findall(answer):
        if (source.strip(), int(page)) not in valid:
            invalid.append(f"[{source.strip()}, p.{page}]")
    return invalid


def dedupe_sources(chunks: list[dict]) -> list[dict]:
    """Collapse chunks to unique (source, page) pairs, keeping retrieval order.

    Several chunks often come from one page, and a reader wants a list of places
    to look, not one entry per fragment.
    """
    sources: list[dict] = []
    seen = set()
    for chunk in chunks:
        key = (chunk["source"], chunk["page"])
        if key not in seen:
            seen.add(key)
            sources.append({"source": chunk["source"], "page": chunk["page"]})
    return sources


def normalise_refusal(answer: str) -> str:
    """Reduce a decorated refusal to the exact sentence, leaving answers alone."""
    if CITATION_PATTERN.sub("", answer).strip().rstrip(".") == NO_ANSWER.rstrip("."):
        return NO_ANSWER
    return answer


def stream_answer(
    query: str,
    chunks: list[dict],
    model: str = "llama3.2:3b",
    temperature: float = 0.1,
):
    """Yield the answer token by token, for a UI that shows text as it arrives.

    On CPU the model produces a few tokens a second, so waiting for a complete
    reply means half a minute of blank screen. Streaming puts the first words on
    screen in a few seconds. Retrieval is done by the caller and passed in, so
    the UI can display the sources before generation even starts.
    """
    system_prompt, user_prompt = build_prompt(query, chunks)
    stream = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        options={"temperature": temperature},
        stream=True,
    )
    for part in stream:
        yield part["message"]["content"]


def finalise_answer(
    query: str,
    answer: str,
    chunks: list[dict],
    model: str,
    retrieval_seconds: float,
    generation_seconds: float,
) -> dict:
    """Build the same result dict as generate_answer, after streaming finishes.

    Citations can only be checked once the whole answer exists — you cannot
    validate half a sentence. So a streamed answer is on screen before its
    citations are verified, and the caller must make that pending check visible
    rather than letting silence read as approval.
    """
    answer = normalise_refusal(answer.strip())
    return {
        "query": query,
        "answer": answer,
        "sources": dedupe_sources(chunks),
        "chunks_used": chunks,
        "invalid_citations": validate_citations(answer, chunks),
        "model": model,
        "retrieval_seconds": retrieval_seconds,
        "generation_seconds": generation_seconds,
    }


def generate_answer(
    query: str,
    k: int = 4,
    model: str = "llama3.2:3b",
    temperature: float = 0.1,
    user_id: str | None = None,
) -> dict:
    """Retrieve context for `query`, then have the LLM answer from it.

    WHY TEMPERATURE IS LOW
    Temperature controls how far the model wanders from its most likely next
    word. That freedom helps when you want writing; here the job is to extract
    what a document already says. Every degree of creativity is a chance to
    paraphrase a number wrong or smooth over a gap with invention, so we keep
    it near zero and stay close to the retrieved text.

    WHY THE RETRIEVED CHUNKS COME BACK WITH THE ANSWER
    Step 7 measures faithfulness: does the answer actually follow from the
    passages the system found? That needs both halves side by side. Returning
    only the text would throw the evidence away, leaving no way to tell a
    well-grounded answer from a lucky one.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    if not 0.0 <= temperature <= 1.0:
        raise ValueError(f"temperature must be between 0.0 and 1.0, got {temperature}")

    query = query.strip()
    retrieval_start = time.perf_counter()
    chunks = retrieve(query, k, user_id=user_id)
    retrieval_seconds = time.perf_counter() - retrieval_start

    sources = dedupe_sources(chunks)

    # With no context at all the reply is already decided, so calling the model
    # would spend half a minute of CPU confirming what we know. Delete this
    # block if you would rather every answer come from the LLM.
    if not chunks:
        return {
            "query": query,
            "answer": NO_ANSWER,
            "sources": [],
            "chunks_used": [],
            "invalid_citations": [],
            "model": model,
            "retrieval_seconds": retrieval_seconds,
            "generation_seconds": 0.0,
        }

    system_prompt, user_prompt = build_prompt(query, chunks)

    generation_start = time.perf_counter()
    try:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            options={"temperature": temperature},
        )
    except Exception as error:
        # The package raises different exception types depending on how the
        # connection failed, so matching the message is steadier than guessing
        # the class. The error is re-raised either way — the caller decides.
        text = str(error).lower()
        if "connect" in text or "refused" in text or "10061" in text:
            print(
                "Ollama server not reachable at localhost:11434 — is Ollama running?",
                file=sys.stderr,
            )
        raise
    generation_seconds = time.perf_counter() - generation_start
    answer = response["message"]["content"].strip()

    # The model sometimes bolts a citation onto the refusal, because rule 2
    # tells it to cite every sentence. Rule 3 forbids that, but a 3B does not
    # always comply, so we normalise here too. Safe to rewrite where an ordinary
    # answer would not be: no factual content changes, and later steps need this
    # string to match exactly.
    answer = normalise_refusal(answer)

    return {
        "query": query,
        "answer": answer,
        "sources": sources,
        "chunks_used": chunks,
        "invalid_citations": validate_citations(answer, chunks),
        "model": model,
        "retrieval_seconds": retrieval_seconds,
        "generation_seconds": generation_seconds,
    }


def format_answer(result: dict) -> str:
    """Render a result dict for the terminal. Returns a string, prints nothing."""
    lines = [f"Question: {result['query']}", "", "Answer:", result["answer"], "", "Sources:"]

    lines += [
        f"  [{n}] {s['source']}, p.{s['page']}"
        for n, s in enumerate(result["sources"], start=1)
    ] or ["  (none)"]

    lines += ["", "Retrieved chunks (for verification):"]
    lines += [
        f"  rank {c['rank']} [d={c['distance']:.3f}] {c['source']} p.{c['page']}"
        for c in result["chunks_used"]
    ] or ["  (none)"]

    if result["invalid_citations"]:
        lines += ["", "WARNING - citations not found in retrieved chunks:"]
        lines += [f"  {c}" for c in result["invalid_citations"]]

    lines += ["", f"Timing: retrieval {result['retrieval_seconds']:.1f}s, "
                  f"generation {result['generation_seconds']:.1f}s"]
    return "\n".join(lines)


if __name__ == "__main__":
    try:
        k, model, words = 4, "llama3.2:3b", []
        for argument in sys.argv[1:]:
            if argument.startswith("--k="):
                k = int(argument[len("--k="):])
            elif argument.startswith("--model="):
                model = argument[len("--model="):]
            else:
                words.append(argument)

        question = " ".join(words).strip() or input("Ask: ").strip()
        if not question:
            print("No question given.", file=sys.stderr)
            raise SystemExit(1)

        print(format_answer(generate_answer(question, k=k, model=model)))

    except SystemExit:
        raise
    except Exception as error:
        # RAG_DEBUG=1 shows the full traceback instead of a one-line message.
        if os.environ.get("RAG_DEBUG") == "1":
            raise
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
