"""Context Management RAG starter — indexer.

Walks documents/, chunks each file, embeds chunks, persists the index to disk
so the chat backend can load it without re-indexing.

TODO: implement chunk_text(). The embedding and storage code is provided so
you can focus on the structure.
"""
import pickle
from pathlib import Path

from sentence_transformers import SentenceTransformer

# Multilingual (50+ languages), 384-dim — same model as the /embedding project.
# Lets the corpus and the queries be in different languages and still match.
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
INDEX_PATH = Path(__file__).parent / "index.pkl"
DOCS_DIR = Path(__file__).parent / "documents"


# ════════════════════════════════════════════════════════════════
# TODO — implement chunk_text
#
# Split `text` into overlapping chunks. A reasonable default:
#   - ~1000 characters per chunk
#   - ~100 characters of overlap
#   - try to break on paragraph boundaries (\n\n) when possible
#
# Return a list of non-empty strings.
# See the lecture slide on chunking for one working implementation.
# ════════════════════════════════════════════════════════════════

def _looks_like_heading(para: str) -> bool:
    """Heuristic: is this paragraph a section heading rather than body prose?

    The corpus is Wikipedia plain-text extracts, where section titles appear as
    short standalone lines (e.g. "Background", "Prime crew") — NOT markdown
    headers. We treat a single short line with no terminal sentence punctuation
    as a heading. Markdown headers (leading '#') always count.
    """
    if "\n" in para:  # multi-line block is body text, not a heading
        return False
    s = para.strip()
    if not s or len(s) > 80:
        return False
    if s.startswith("#"):
        return True
    return not s.endswith((".", ":", ",", ";", "?", "!", '"')) and len(s.split()) <= 8


def _overlap_tail(text: str, overlap_chars: int) -> str:
    """Return up to `overlap_chars` of trailing text, snapped to a sentence
    boundary so the carried context starts on a clean word/sentence."""
    if overlap_chars <= 0 or len(text) <= overlap_chars:
        return text if len(text) <= overlap_chars else ""
    tail = text[-overlap_chars:]
    # Prefer to start the overlap after a sentence break; fall back to a word break.
    for sep in (". ", "? ", "! ", " "):
        idx = tail.find(sep)
        if idx != -1:
            return tail[idx + len(sep):]
    return tail


def chunk_text(
    text: str,
    target_chars: int = 1000,
    overlap_chars: int = 100,
    doc_title: str | None = None,
) -> list[str]:
    """Split text into overlapping, context-prefixed chunks.

    Strategy:
      - Split on blank lines into paragraphs; track the current section heading
        (short standalone lines like "Background", "Prime crew").
      - Greedily pack body paragraphs into chunks of ~target_chars, never
        crossing a heading boundary (each section starts a fresh chunk).
      - Prefix every chunk with a breadcrumb ("<doc_title> > <heading>") so the
        document and section context ride along into the embedding AND the text
        shown to the model — this is what lets a query naming a specific mission
        match chunks whose body never repeats the mission name.
      - Carry ~overlap_chars of sentence-aligned trailing context across cuts
        within a section so information isn't lost at chunk boundaries.

    Paragraphs longer than target_chars are split by character window.
    Returns a list of non-empty strings.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks: list[str] = []
    current_heading: str | None = None
    current = ""

    def breadcrumb(heading: str | None) -> str:
        parts = [doc_title] if doc_title else []
        # Skip a section heading that just repeats the document title (the H1).
        if heading and heading.strip().lower() != (doc_title or "").strip().lower():
            parts.append(heading)
        return " > ".join(parts)

    def flush():
        nonlocal current
        if current.strip():
            prefix = breadcrumb(current_heading)
            chunks.append(f"{prefix}\n\n{current}" if prefix else current)
        current = ""

    for para in paragraphs:
        if _looks_like_heading(para):
            flush()  # a new section always starts a new chunk
            current_heading = para.lstrip("#").strip()
            continue

        # Split an over-long paragraph into target-sized windows.
        pieces = (
            [para]
            if len(para) <= target_chars
            else [para[i:i + target_chars] for i in range(0, len(para), target_chars)]
        )
        for piece in pieces:
            if current and len(current) + 2 + len(piece) > target_chars:
                tail = _overlap_tail(current, overlap_chars)
                flush()
                current = (tail + "\n\n" + piece) if tail else piece
            else:
                current = (current + "\n\n" + piece) if current else piece

    flush()
    return [c for c in chunks if c.strip()]


# ════════════════════════════════════════════════════════════════
# Provided: embedding (sentence-transformers, no API key required)
# ════════════════════════════════════════════════════════════════

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"Loading embedding model ({MODEL_NAME})... (one-time download ~470MB)")
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed(texts: list[str]) -> list[list[float]]:
    """Embed a list of strings. Returns unit-normalized 384-dim vectors."""
    model = get_model()
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return vectors.tolist()


# ════════════════════════════════════════════════════════════════
# Provided: build / save / load / search
# ════════════════════════════════════════════════════════════════

def _doc_title(text: str, fallback: str) -> str:
    """Use the first markdown H1 ('# Title') as the document title, else the
    filename stem (e.g. 'apollo-11' -> 'apollo 11')."""
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("# "):
            return s[2:].strip()
        if s:
            break
    return fallback.replace("-", " ")


def build_index() -> list[dict]:
    """Walk DOCS_DIR, chunk each file, embed, return list of records."""
    records: list[dict] = []
    chunk_id = 0
    for path in sorted(DOCS_DIR.glob("*")):
        if path.is_dir() or path.suffix.lower() not in (".md", ".txt"):
            continue
        text = path.read_text()
        title = _doc_title(text, path.stem)
        chunks = chunk_text(text, doc_title=title)
        if not chunks:
            continue
        vectors = embed(chunks)
        for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
            records.append({
                "chunk_id": chunk_id,
                "source": path.name,
                "chunk_index": i,
                "text": chunk,
                "embedding": vec,
            })
            chunk_id += 1
        print(f"  {path.name}: {len(chunks)} chunks")
    return records


def save_index(records: list[dict]) -> None:
    with INDEX_PATH.open("wb") as f:
        pickle.dump(records, f)


def load_index() -> list[dict]:
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"No index found at {INDEX_PATH}. Run `python indexer.py` from the project root first."
        )
    with INDEX_PATH.open("rb") as f:
        return pickle.load(f)


def cosine_distance(a: list[float], b: list[float]) -> float:
    # Both vectors are unit-normalized, so cosine distance == 1 - dot product.
    return 1.0 - sum(x * y for x, y in zip(a, b))


def search(query: str, records: list[dict], k: int = 5) -> list[dict]:
    """Embed the query, return top-k records by cosine distance."""
    [query_vec] = embed([query])
    scored = [(cosine_distance(r["embedding"], query_vec), r) for r in records]
    scored.sort(key=lambda x: x[0])
    return [r for _, r in scored[:k]]


def main() -> None:
    print(f"Indexing documents from {DOCS_DIR}/")
    records = build_index()
    save_index(records)
    print(f"\n✓ Indexed {len(records)} chunks → {INDEX_PATH.name}")


if __name__ == "__main__":
    main()
