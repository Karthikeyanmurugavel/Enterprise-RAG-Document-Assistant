# Enterprise RAG Document Assistant

Ask questions about your PDFs and get answers with page-level citations. Runs
entirely on your own machine — no API keys, no data leaving the box.

Built as a working system rather than a demo: every claim below is measured, and
the measurements include the failures.

---

## What it does

Upload a PDF, ask a question in plain English, get an answer that cites the
filename and page it came from. If the documents do not contain the answer, it
says so instead of guessing.

```
Question: How much economic value could generative AI add?

Answer:
$2.6 trillion to $4.4 trillion in economic benefits annually
[mckinsey-genai-report.pdf, p.11]

Sources:
  [1] mckinsey-genai-report.pdf, p.26
  [2] mckinsey-genai-report.pdf, p.13

Timing: retrieval 7.1s, generation 30.9s
```

## Stack

| layer | choice | why |
|---|---|---|
| PDF parsing | PyMuPDF | fast, keeps page boundaries |
| Embeddings | all-MiniLM-L6-v2 | 384-dim, free, local, good enough at this scale |
| Vector store | ChromaDB | persistent, no server to run |
| Generation | llama3.2:3b via Ollama | fits in 4 GB VRAM, runs offline |
| UI | Streamlit | fastest path to a usable interface |

## Pipeline

```
PDF ─► chunk ─► embed ─► ChromaDB
                            │
question ─► embed ─────► search ─► top-k chunks ─► LLM ─► cited answer
                                                          │
                                                   citation validator
```

Each stage is a separate module with its own entry point, so any of them can be
run and tested alone.

---

## Setup

Requires Python 3.13 and [Ollama](https://ollama.com).

```bash
git clone https://github.com/Karthikeyanmurugavel/Enterprise-RAG-Document-Assistant.git
cd Enterprise-RAG-Document-Assistant

python -m venv .venv
.venv\Scripts\activate                        # Windows
# source .venv/bin/activate                   # macOS / Linux

pip install -r rag_assistant/requirements.txt
ollama pull llama3.2:3b
```

## Usage

All commands run from the `rag_assistant/` directory:

```bash
cd rag_assistant

python cli.py status                      # is the index and Ollama ready?
python cli.py ingest                      # load and embed PDFs from documents/
python cli.py retrieve <question>         # show matching chunks, no LLM
python cli.py ask <question>              # cited answer
```

Or the web UI, which supports drag-and-drop upload:

```bash
streamlit run app.py
```

Ollama must be running — the app talks to it at `localhost:11434`.

Questions do not need quoting — everything after the subcommand is the question.

---

## Evaluation

The part most RAG projects skip. Four question sets with ground truth verified
against the source document, split by failure mode.

```bash
cd rag_assistant
python evaluate.py --all                    # retrieval only, ~26 seconds
python evaluate.py --set citations --full   # includes the LLM, ~4.5 minutes
```

Results on a 68-page McKinsey report (355 chunks):

| set | metric | result |
|---|---|---|
| retrieval | hit@4 | **8/8 (100%)** |
| retrieval | MRR | 0.750 |
| citations | hit@4 | 6/6 (100%) |
| citations | clean citations | **5/6 (83%)** |
| refusal | correct refusals | **4/4 (100%)** |
| hard | hit@4 | **4/5 (80%)** |

Out-of-domain questions score a mean best distance of **1.627**, against
0.32–1.09 for in-domain — a clean separation that makes confidence scoring
possible.

There is deliberately **no LLM-as-judge**. The only judge available locally is
the same 3B model being evaluated, and a judge no more reliable than its subject
produces numbers that look rigorous while meaning little. Every metric is exact
string or set matching, so the evaluation cannot hallucinate.

---

## Design decisions worth explaining

**Chunk overlap of 50 characters.** Chunk boundaries fall mid-sentence. Without
overlap, an answer sitting on a boundary is split across two chunks and neither
matches the question well.

**Minimum chunk length of 100 characters.** Running headers like
`"...productivity frontier"` are short strings made of the document's own
keywords, so they match vague queries structurally rather than semantically. On
one test question three of four results were headers. Filtering them dropped 18
of 373 chunks and replaced all three with real passages.

**Citations are pre-formatted in the prompt.** Each chunk is labelled
`[source.pdf, p.26]` before the model sees it, so it copies a tag rather than
constructing one from metadata. Small models copy far more reliably than they
construct.

**Citations are validated in code, not trusted.** The model still occasionally
invents a page number. `validate_citations()` checks every citation against the
chunks actually retrieved — plain string matching, so it cannot hallucinate in
turn. Three rounds of prompt engineering did not eliminate the problem; the
validator catches it every time.

**Temperature 0.1.** RAG is extraction, not creativity. Every degree of
freedom is a chance to paraphrase a number wrong.

**Per-user isolation at three levels.** Chunk IDs are namespaced
(`alice::policy.pdf_chunk_0`), metadata carries `user_id`, and searches filter
on it. Uploads also land in per-session directories — without that, indexing
after one upload would sweep up every other user's files and re-tag them.

---

## Known limitations

Stated plainly, because a system you cannot trust the limits of is not
trustworthy at all.

- **Invented page numbers.** On roughly half of one test set the model cited a
  page that was never retrieved. The content was correct; the page number was
  not. Flagged with a warning, not blocked — discarding a correct answer over a
  bad tag is the worse trade.
- **Chart pages produce nonsense.** PDF text extraction flattens multi-column
  layouts and exhibits, so axis labels can be assembled into confident-sounding
  claims like `"0.4 productivity gains"`. Citation validation cannot catch this,
  because the cited page is real.
- **CPU-bound.** Around 4.4 tokens/second without GPU acceleration, so answers
  take 20–60 seconds. Time-to-first-token is driven almost entirely by `k`:
  1.0s at k=1 versus 43s at k=4, because prompt processing dominates.
- **Sessions are not accounts.** Session IDs live in browser state, so uploads
  are lost on refresh and their vectors are orphaned.
- **No delete path.** Documents can be added but never removed.
- **Vague queries retrieve loosely.** `"GenAI"` scores 0.950 against the same
  chunk that `"generative AI"` scores 0.311 on. The abbreviation is not strongly
  associated by the embedding model.

## Tests

```bash
cd rag_assistant
python -m pytest test_rag.py -v
```

14 tests covering chunking, citation validation, refusal handling and per-user
isolation. No LLM calls, so the suite runs in about 20 seconds — evaluation and
regression testing are separate jobs.

## Project structure

```
rag_assistant/
├── ingest.py        PDF loading and chunking
├── embed_store.py   embeddings and ChromaDB
├── retrieve.py      semantic search
├── answer.py        prompting, generation, citation validation
├── cli.py           unified entry point
├── app.py           Streamlit UI
├── evaluate.py      evaluation harness
├── test_rag.py      test suite
├── eval_sets/       ground-truth question sets
└── documents/       PDFs to index
```
