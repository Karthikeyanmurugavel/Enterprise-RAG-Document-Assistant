"""
Step 7: evaluation.

Measures the pipeline against fixed question sets with known-correct answers,
so a change to chunking, k, or the model produces a number rather than an
impression.

Two modes, deliberately separate:

  --retrieval (default)  no LLM. Seconds to run. This is where tuning happens
                         - chunk size, overlap, k - and none of it involves
                         generation, so paying for generation would be waste.

  --full                 adds the LLM. Minutes to run on CPU. Measures whether
                         the answer cites honestly and refuses when it should.

There is deliberately no LLM-as-judge here. The only judge available locally is
the same 3B model being evaluated, and a judge no more reliable than its subject
produces numbers that look rigorous while meaning very little. Every metric
below is exact string or set matching, so the evaluation itself cannot
hallucinate.
"""

import argparse
import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SETS_DIR = os.path.join(BASE_DIR, "eval_sets")

# The evaluation corpus is owned by a reserved user, so the fixture stays
# queryable by this harness while being invisible to real users. Ground truth
# only means something against a corpus that does not move.
EVAL_USER = "__eval__"


def load_set(name: str) -> dict:
    """Read one question set by name."""
    path = os.path.join(SETS_DIR, f"{name}.json")
    if not os.path.isfile(path):
        available = ", ".join(sorted(available_sets())) or "none"
        raise FileNotFoundError(f"No eval set '{name}'. Available: {available}")
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def available_sets() -> list[str]:
    if not os.path.isdir(SETS_DIR):
        return []
    return [f[:-5] for f in os.listdir(SETS_DIR) if f.endswith(".json")]


def evaluate_retrieval(question_set: dict, k: int) -> dict:
    """Score retrieval only: did a correct page reach the top-k, and how high?

    hit@k is the headline: if the right passage never reaches the model, no
    amount of prompting can produce a correct answer. MRR adds where it landed
    - rank 1 and rank 4 both count as hits, but rank 1 is a better system.
    """
    from retrieve import retrieve

    rows = []
    for item in question_set["questions"]:
        chunks = retrieve(item["question"], k, user_id=EVAL_USER)
        pages = [chunk["page"] for chunk in chunks]
        expected = set(item.get("expected_pages", []))

        # Reciprocal rank of the first correct page, 0.0 if none appear.
        reciprocal_rank = 0.0
        for position, page in enumerate(pages, start=1):
            if page in expected:
                reciprocal_rank = 1.0 / position
                break

        rows.append({
            "question": item["question"],
            "expected": sorted(expected),
            "retrieved": pages,
            "hit": reciprocal_rank > 0,
            "rr": reciprocal_rank,
            "best_distance": chunks[0]["distance"] if chunks else None,
            "refusal_case": bool(item.get("must_refuse")),
        })

    return {"mode": "retrieval", "rows": rows}


def evaluate_full(question_set: dict, k: int, model: str) -> dict:
    """Score generation as well: citation honesty, keywords, refusal behaviour."""
    from answer import NO_ANSWER, generate_answer

    rows = []
    for item in question_set["questions"]:
        result = generate_answer(item["question"], k=k, model=model, user_id=EVAL_USER)
        answer_text = result["answer"]
        lowered = answer_text.lower()
        refused = answer_text.strip() == NO_ANSWER

        keywords = item.get("expected_keywords", [])
        found = [word for word in keywords if word.lower() in lowered]

        must_refuse = bool(item.get("must_refuse"))
        # An out-of-domain question passes only by refusing; an in-domain one
        # passes by producing at least one expected keyword with clean
        # citations. Refusing an answerable question is a failure too.
        if must_refuse:
            correct = refused
        else:
            correct = (not refused) and bool(found or not keywords)

        rows.append({
            "question": item["question"],
            "answer": answer_text,
            "refused": refused,
            "must_refuse": must_refuse,
            "keywords_found": f"{len(found)}/{len(keywords)}" if keywords else "-",
            "invalid_citations": result["invalid_citations"],
            "correct": correct,
            "seconds": result["retrieval_seconds"] + result["generation_seconds"],
        })

    return {"mode": "full", "rows": rows}


def format_report(name: str, report: dict) -> str:
    """Per-question rows first, aggregates second.

    An aggregate says a problem exists; only the rows say which question caused
    it and what it did. Both are printed, rows first, because the rows are what
    you act on.
    """
    lines = [f"=== eval set: {name} ({report['mode']}) ===", ""]
    rows = report["rows"]

    if report["mode"] == "retrieval":
        scored = [r for r in rows if not r["refusal_case"]]
        for row in rows:
            if row["refusal_case"]:
                lines.append(f"  [ood ] d={row['best_distance']:.3f}  {row['question'][:52]}")
            else:
                mark = "HIT " if row["hit"] else "MISS"
                lines.append(
                    f"  [{mark}] rr={row['rr']:.2f} d={row['best_distance']:.3f}  "
                    f"{row['question'][:52]}"
                )
                if not row["hit"]:
                    lines.append(f"         expected {row['expected']}, got {row['retrieved']}")
        if scored:
            hits = sum(1 for r in scored if r["hit"])
            mrr = sum(r["rr"] for r in scored) / len(scored)
            lines += ["", f"  hit@k: {hits}/{len(scored)} ({100*hits/len(scored):.0f}%)",
                      f"  MRR:   {mrr:.3f}"]
        ood = [r for r in rows if r["refusal_case"]]
        if ood:
            mean_distance = sum(r["best_distance"] for r in ood) / len(ood)
            lines.append(f"  mean best distance on out-of-domain: {mean_distance:.3f}")
    else:
        for row in rows:
            mark = "PASS" if row["correct"] else "FAIL"
            flag = " CITE!" if row["invalid_citations"] else ""
            lines.append(
                f"  [{mark}] kw={row['keywords_found']} {row['seconds']:.0f}s{flag}  "
                f"{row['question'][:48]}"
            )
            if row["invalid_citations"]:
                lines.append(f"         invalid: {row['invalid_citations']}")
        passed = sum(1 for r in rows if r["correct"])
        clean = sum(1 for r in rows if not r["invalid_citations"])
        lines += ["", f"  correct:           {passed}/{len(rows)} ({100*passed/len(rows):.0f}%)",
                  f"  clean citations:   {clean}/{len(rows)} ({100*clean/len(rows):.0f}%)"]

    return "\n".join(lines)


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline")
        parser.add_argument("--set", dest="set_name", help="one set to run")
        parser.add_argument("--all", action="store_true", help="run every set")
        parser.add_argument("--full", action="store_true", help="include the LLM (slow)")
        parser.add_argument("--k", type=int, default=4)
        parser.add_argument("--model", default="llama3.2:3b")
        arguments = parser.parse_args()

        if arguments.all:
            names = sorted(available_sets())
        elif arguments.set_name:
            names = [arguments.set_name]
        else:
            parser.error("give --set NAME or --all")

        for name in names:
            question_set = load_set(name)
            if arguments.full:
                report = evaluate_full(question_set, arguments.k, arguments.model)
            else:
                report = evaluate_retrieval(question_set, arguments.k)
            print(format_report(name, report))
            print()

    except SystemExit:
        raise
    except Exception as error:
        if os.environ.get("RAG_DEBUG") == "1":
            raise
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
