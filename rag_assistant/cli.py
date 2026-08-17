"""
Step 5: one entry point for the whole pipeline.

A thin wrapper over the four modules — it imports them and prints their output,
and contains no retrieval, embedding or prompting logic of its own. Each script
keeps its own __main__ block; this just saves you remembering four of them.
"""

import argparse
import json
import os
import sys
import urllib.request

OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
DEFAULT_MODEL = "llama3.2:3b"
DEFAULT_USER = "default"

# Every import below happens inside a handler rather than at the top of the
# file. Importing answer/retrieve pulls in embed_store, which pulls in
# sentence-transformers and torch — around seven seconds. At module level that
# cost would be paid by "cli.py --help" and by every mistyped subcommand.
# Deferring it means each command loads only what it actually uses.


def _require_query(query: str) -> str:
    """Reject an empty question before doing any expensive work."""
    if not query:
        raise ValueError("No question given. Put your question after the subcommand.")
    return query


def _ollama_models() -> list[str] | None:
    """Return Ollama's model names, or None if the server is unreachable.

    One request answers both status questions — is it up, and does the default
    model exist. A short timeout matters here: a hung server should report as
    unreachable in seconds, not block the status check indefinitely.
    """
    try:
        with urllib.request.urlopen(OLLAMA_TAGS_URL, timeout=3) as response:
            payload = json.load(response)
        return [model.get("name", "") for model in payload.get("models", [])]
    except Exception:
        # Down, refusing connections, timing out, or returning nonsense are all
        # the same answer to the caller: we cannot talk to it right now.
        return None


def cmd_ingest(args: argparse.Namespace) -> None:
    """Load every PDF in documents/ and embed anything not already stored."""
    from embed_store import embed_and_store
    from ingest import process_all_pdfs

    chunks = process_all_pdfs()
    print(f"Chunks discovered: {len(chunks)}")

    if not chunks:
        print("Nothing to embed. Add a text-based PDF to documents/ and re-run.")
        return

    print(f"Indexing as user: {args.user}")
    # embed_and_store prints embedded / skipped / total itself, so we do not
    # repeat those numbers here.
    embed_and_store(chunks, user_id=args.user)


def cmd_retrieve(args: argparse.Namespace) -> None:
    """Show which chunks match a question, without involving the LLM."""
    from retrieve import format_results, retrieve

    query = _require_query(" ".join(args.query).strip())
    print(format_results(retrieve(query, args.k, user_id=args.user)))


def cmd_ask(args: argparse.Namespace) -> None:
    """Retrieve context and have the LLM answer from it, with citations."""
    from answer import format_answer, generate_answer

    query = _require_query(" ".join(args.query).strip())
    print(format_answer(
        generate_answer(query, k=args.k, model=args.model, user_id=args.user)
    ))


def cmd_status(args: argparse.Namespace) -> None:
    """Report whether the pieces are ready, before a demo or a long session.

    An empty collection and an unreachable Ollama are both normal states you
    can recover from, not failures — so this reports them and carries on rather
    than raising.
    """
    from embed_store import COLLECTION_NAME, get_or_create_collection

    collection = get_or_create_collection()
    count = collection.count()
    print(f"Collection '{COLLECTION_NAME}': {count} vector(s)")

    if count:
        # Grouped by owner, so it is obvious at a glance whether isolation is
        # doing anything and whose documents are present.
        metadatas = collection.get(include=["metadatas"])["metadatas"]
        by_user: dict[str, set[str]] = {}
        for metadata in metadatas:
            owner = metadata.get("user_id", "(untagged)")
            by_user.setdefault(owner, set()).add(metadata["source"])
        for owner in sorted(by_user):
            print(f"  user '{owner}':")
            for source in sorted(by_user[owner]):
                print(f"    - {source}")
    else:
        print("  (no documents indexed - run: python cli.py ingest)")

    models = _ollama_models()
    if models is None:
        print("Ollama: unreachable at localhost:11434")
        print(f"Model '{DEFAULT_MODEL}': unknown (server unreachable)")
        return

    print(f"Ollama: reachable ({len(models)} model(s) installed)")
    present = DEFAULT_MODEL in models
    print(f"Model '{DEFAULT_MODEL}': {'available' if present else 'NOT FOUND'}")
    if not present:
        print(f"  pull it with: ollama pull {DEFAULT_MODEL}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cli.py", description="Enterprise RAG Document Assistant"
    )
    # --user scopes both indexing and search to one owner. Documents indexed
    # under one user are invisible to every other.
    parser.add_argument(
        "--user", default=DEFAULT_USER, help="document owner (default: %(default)s)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="load and embed PDFs from documents/")
    ingest.set_defaults(func=cmd_ingest)

    # nargs="+" so questions work with or without quotes, matching how
    # retrieve.py and answer.py already behave on the command line.
    retrieve_parser = subparsers.add_parser("retrieve", help="show matching chunks")
    retrieve_parser.add_argument("query", nargs="+", help="the question")
    retrieve_parser.add_argument("--k", type=int, default=4, help="chunks to return")
    retrieve_parser.set_defaults(func=cmd_retrieve)

    ask = subparsers.add_parser("ask", help="get a cited answer from the LLM")
    ask.add_argument("query", nargs="+", help="the question")
    ask.add_argument("--k", type=int, default=4, help="chunks to retrieve")
    ask.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name")
    ask.set_defaults(func=cmd_ask)

    status = subparsers.add_parser("status", help="check the collection and Ollama")
    status.set_defaults(func=cmd_status)

    return parser


if __name__ == "__main__":
    try:
        arguments = build_parser().parse_args()
        arguments.func(arguments)
    except SystemExit:
        raise
    except Exception as error:
        # RAG_DEBUG=1 shows the full traceback instead of a one-line message.
        if os.environ.get("RAG_DEBUG") == "1":
            raise
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
