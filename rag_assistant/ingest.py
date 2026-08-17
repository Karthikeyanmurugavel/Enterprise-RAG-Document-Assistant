"""
Step 1 of the Enterprise RAG Document Assistant: load PDFs and chunk them.

This file does exactly one job: turn a folder of PDFs into a flat list of small
text pieces ("chunks"), each tagged with the file it came from and the page it
was on. Nothing here embeds, stores, retrieves, or generates anything — those
are later steps that will consume the output of this one.
"""

import os

import fitz  # PyMuPDF is imported under the name "fitz"

# Where this file lives on disk. Folder names below are resolved against this
# rather than against the current working directory, so the script behaves the
# same whether you run it from inside rag_assistant/, from the project root, or
# from your editor's Run button. Relative paths depend on where you happened to
# be standing when you hit enter, which is a surprising thing to debug.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_pdf(filepath):
    """Read one PDF and return a list of {'page_number', 'text'} dicts.

    We extract page by page rather than dumping the whole document into one
    string, because the page number is the thing that makes a citation useful.
    If we flattened the document first, we would have no way to tell the user
    "this answer came from page 7" — and an answer you can't verify is not
    much better than a guess.

    Page numbers are 1-based here (PyMuPDF indexes from 0) so they match what
    a person sees when they open the PDF and scroll to that page.
    """
    pages = []

    with fitz.open(filepath) as doc:
        for page_index, page in enumerate(doc):
            text = page.get_text()

            # Scanned images, spacer pages, and section dividers often produce
            # empty or whitespace-only text. Keeping them would create useless
            # chunks that still get embedded and searched later, so we drop
            # them at the source instead of filtering downstream.
            if not text or not text.strip():
                continue

            pages.append({
                "page_number": page_index + 1,
                "text": text,
            })

    return pages


def chunk_text(text, chunk_size=500, overlap=50, min_length=100):
    """Split text into overlapping chunks of roughly `chunk_size` characters.

    WHY CHUNK AT ALL?
    Two reasons. First, the embedding model turns a piece of text into a single
    vector — one point in meaning-space. If you feed it a whole page covering
    five different topics, that one vector ends up as a blurry average of all
    five and matches nothing well. Small chunks each stay about one thing, so
    the search is sharp. Second, the LLM has a limited context window and we
    want to hand it only the few passages that actually matter, not entire
    documents.

    WHY OVERLAP?
    Chunk boundaries fall in arbitrary places — very often in the middle of a
    sentence. If the answer to a question sits right on that seam, a clean
    split would leave half the answer in chunk A and half in chunk B, and
    neither chunk would look like a good match for the question. Repeating the
    last `overlap` characters at the start of the next chunk means any short
    passage survives intact in at least one chunk. The cost is a little
    duplicated text, which is cheap compared to losing an answer.
    """
    text = text.strip()

    if not text:
        return []

    # Guard against a bad config silently causing an infinite loop: if the
    # overlap were as large as the chunk, the window would never move forward.
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    step = chunk_size - overlap  # how far the window slides each time
    start = 0

    while start < len(text):
        chunk = text[start:start + chunk_size].strip()

        # Discard scraps shorter than min_length. The last slice of a page is
        # often a stub, and PDF page furniture (running headers, page numbers)
        # produces short chunks made of the document title repeated on every
        # page. Those are pure noise: they carry no information, but they still
        # compete for the handful of slots a search returns, and on a short or
        # vague question they can beat the real content — because a header made
        # of the document's own keywords looks superficially like a match.
        # Dropping them costs nothing and measurably improves what comes back.
        if len(chunk) >= min_length:
            chunks.append(chunk)

        start += step

    return chunks


def process_all_pdfs(folder="documents"):
    """Chunk every PDF in `folder` into one flat list ready for embedding.

    Each entry is {'text', 'source', 'page'}. That metadata is the whole point
    of this function. When the vector store later returns a matching chunk, the
    'source' and 'page' travel with it, so the assistant can answer with
    "according to handbook.pdf, page 12" instead of asking the user to take its
    word for it. Metadata added at ingest time is essentially free; metadata
    you forgot to attach cannot be recovered afterwards, because by then the
    chunk is just a vector with no memory of where it came from.

    The list is flat (not grouped by file) because the next step embeds and
    stores every chunk the same way regardless of origin — the file it came
    from lives in the metadata, not in the structure.
    """
    # A bare name like "documents" means "next to this script", not "next to
    # wherever the terminal happens to be pointing". An absolute path passed in
    # by a caller is respected as-is.
    if not os.path.isabs(folder):
        folder = os.path.join(BASE_DIR, folder)

    if not os.path.isdir(folder):
        raise FileNotFoundError(
            f"The folder '{folder}' does not exist. "
            f"Create it and put some PDF files inside."
        )

    pdf_filenames = sorted(
        name for name in os.listdir(folder)
        if name.lower().endswith(".pdf")
    )

    all_chunks = []

    for filename in pdf_filenames:
        filepath = os.path.join(folder, filename)

        for page in load_pdf(filepath):
            for chunk in chunk_text(page["text"]):
                all_chunks.append({
                    "text": chunk,
                    "source": filename,   # filename, not full path — it is what
                                          # gets shown to the user in a citation
                    "page": page["page_number"],
                })

    return all_chunks


if __name__ == "__main__":
    FOLDER = "documents"

    try:
        chunks = process_all_pdfs(FOLDER)
    except FileNotFoundError as error:
        print(f"ERROR: {error}")
        raise SystemExit(1)

    if not chunks:
        print(
            f"No text was extracted from '{FOLDER}'.\n"
            f"Either the folder contains no PDF files, or the PDFs are scanned "
            f"images with no selectable text layer.\n"
            f"Put at least one text-based PDF in '{FOLDER}' and run this again."
        )
        raise SystemExit(1)

    # Count chunks per file so it is obvious at a glance whether every document
    # was actually read — a PDF showing 0 chunks is a red flag worth chasing.
    counts = {}
    for chunk in chunks:
        counts[chunk["source"]] = counts.get(chunk["source"], 0) + 1

    print(f"Found {len(counts)} PDF(s) in '{FOLDER}':")
    for source, count in counts.items():
        print(f"  - {source}: {count} chunks")
    print(f"Total chunks: {len(chunks)}")

    # Print one real chunk. Chunking bugs (garbled text, wrong page numbers,
    # chunks that are far too short) are much easier to spot by eye than by
    # reading the code.
    sample = chunks[0]
    print("\nSample chunk:")
    print(f"  source: {sample['source']}")
    print(f"  page:   {sample['page']}")
    print(f"  text:   {sample['text'][:200]}")
