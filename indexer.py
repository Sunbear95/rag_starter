"""Context Management RAG starter — indexer.

Walks documents/, extracts + chunks each file, embeds chunks, and persists the
index to disk so the chat backend can load it without re-indexing.

The corpus is the Federal Aviation Regulations (14 CFR, Title 14 of the Code of
Federal Regulations). Those ship as two-column PDFs with running page headers
and line-break hyphenation, so the PDF path here does three things that matter
for retrieval quality:

  1. Extraction — split each page into its two columns and read them in order
     (a naive extract interleaves the columns line-by-line), dropping the
     running header/footer band.
  2. Cleanup — rejoin words hyphenated across line breaks, collapse wrapped
     lines, and strip Federal-Register amendment-history noise.
  3. Section-aware chunking — split on `§` section boundaries and prefix every
     chunk with a `14 CFR Part N > Subpart X > § N.M Title` breadcrumb, carrying
     the regulatory structure into both the embedding and the cited text.

Plain .md/.txt files still work via the generic chunker below, so you can swap
in your own corpus.
"""
import bisect
import pickle
import re
from collections import defaultdict
from pathlib import Path

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# Multilingual (50+ languages), 384-dim — same model as the /embedding project.
# Lets the corpus and the queries be in different languages and still match.
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
INDEX_PATH = Path(__file__).parent / "index.pkl"
DOCS_DIR = Path(__file__).parent / "documents"

TARGET_CHARS = 1200      # ~300 tokens; fits the embedding model's window with the breadcrumb
OVERLAP_CHARS = 150


# ════════════════════════════════════════════════════════════════
# Generic text chunking (.md / .txt) — Wikipedia-style prose
# ════════════════════════════════════════════════════════════════

def _looks_like_heading(para: str) -> bool:
    """Heuristic: is this paragraph a section heading rather than body prose?

    For plain-text corpora where section titles appear as short standalone
    lines. Markdown headers (leading '#') always count.
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
    for sep in (". ", "? ", "! ", " "):
        idx = tail.find(sep)
        if idx != -1:
            return tail[idx + len(sep):]
    return tail


def chunk_text(
    text: str,
    target_chars: int = TARGET_CHARS,
    overlap_chars: int = OVERLAP_CHARS,
    doc_title: str | None = None,
) -> list[str]:
    """Split plain text into overlapping, heading-aware chunks.

    Greedily packs body paragraphs into ~target_chars chunks, never crossing a
    heading boundary, prefixing each chunk with a `<doc_title> > <heading>`
    breadcrumb, and carrying ~overlap_chars of sentence-aligned context across
    cuts within a section. Returns a list of non-empty strings.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks: list[str] = []
    current_heading: str | None = None
    current = ""

    def breadcrumb(heading: str | None) -> str:
        parts = [doc_title] if doc_title else []
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
            flush()
            current_heading = para.lstrip("#").strip()
            continue

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
# CFR PDF extraction + cleanup
# ════════════════════════════════════════════════════════════════

# Running header lines repeated on every page — pure noise once we know the §.
_HEADER_LINES = [
    re.compile(r"^Federal Aviation Administration.*$", re.M),
    re.compile(r"^.*\bCFR\b.*Edition.*$", re.M),
    re.compile(r"^\s*VerDate.*$", re.M),
    re.compile(r"^\s*PsN:.*$", re.M),
]
# Bracketed Federal-Register amendment-history blocks, e.g. "[Docket No. ...; 89 FR 80339, Oct. 2, 2024]".
_FR_CITATION = re.compile(r"\[[^\]]*\bFR\b[^\]]*\]", re.S)
# "EFFECTIVE DATE NOTE: ... effective <date>." trailing administrative notes.
_EFFECTIVE_NOTE = re.compile(r"EFFECTIVE DATE NOTE:.*?effective[^.]*\.", re.S | re.I)
# The running page header (the enclosing Subpart's name, e.g. "Fire Protection")
# gets extracted with its internal spaces stripped and lands right after an FR
# citation, right at the tail of a section's body — e.g. "...approved.]
# FIREPROTECTION §27.853 ...". A single glued all-caps run this long never
# occurs in normal CFR prose, so it's a safe tell for this specific artifact.
_TRAILING_HEADER_NOISE = re.compile(r"\s+[A-Z]{8,}\s*$")


def _clean(text: str) -> str:
    """Normalize one page's column-joined text into clean prose."""
    for pat in _HEADER_LINES:
        text = pat.sub("", text)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)   # rejoin words split across line breaks
    text = re.sub(r"[ \t]*\n[ \t]*", " ", text)     # collapse wrapped lines into spaces
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _printed_page(raw: str, fallback: int) -> int:
    """Best-effort CFR printed page number from a page's raw text.

    Even pages carry it at the very start of the header line, odd pages at the
    end. Falls back to the 1-based PDF page index.
    """
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if lines:
        m = re.match(r"^(\d{1,5})\b", lines[0])
        if m:
            return int(m.group(1))
        m = re.search(r"\b(\d{1,5})$", lines[-1])
        if m:
            return int(m.group(1))
    return fallback


def extract_pdf(path: Path) -> tuple[str, list[tuple[int, int]]]:
    """Extract a CFR PDF to a single cleaned text stream.

    Returns (full_text, page_spans) where page_spans is a list of
    (char_offset, printed_page) marking where each page begins in full_text, so
    a chunk's offset can be mapped back to a citable page number.
    """
    import pdfplumber

    parts: list[str] = []
    page_spans: list[tuple[int, int]] = []
    offset = 0
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            w, h = page.width, page.height
            # Crop away the top header / bottom footer band, then read each
            # column top-to-bottom so the two-column layout reads in order.
            left = page.crop((0, 50, w / 2, h - 42)).extract_text() or ""
            right = page.crop((w / 2, 50, w, h - 42)).extract_text() or ""
            cleaned = _clean(left + "\n" + right)
            if not cleaned:
                continue
            page_spans.append((offset, _printed_page(page.extract_text() or "", i + 1)))
            parts.append(cleaned)
            offset += len(cleaned) + 1  # +1 for the joining space below
    return " ".join(parts), page_spans


# ════════════════════════════════════════════════════════════════
# CFR structure parsing — Part / Subpart / § sections
# ════════════════════════════════════════════════════════════════

_PART_RE = re.compile(r"\bPART\s+(\d+[A-Z]?)\s*[—–-]\s*([A-Z][^\n]{0,80})")
_SUBPART_RE = re.compile(r"\bSubpart\s+([A-Z]+)\s*[—–-]\s*([A-Z][A-Za-z,\s]{0,60})")
# Trailing run-on noise after a Subpart title: an ALL-CAPS word (SOURCE, GENERAL)
# or "Sec" that belongs to the following block, not the title.
_SUBPART_TITLE_TAIL = re.compile(r"\s+(?:[A-Z]{2,}|Sec)\b.*$")
# A real § heading: number followed by a Title-case heading ending in a period.
# Cross-references like "§61.3(b)" or "under §61.41 of this part" don't match
# (no Title-case heading word after the number).
# The terminating period must not be immediately followed by another
# "<capital letter>." — that pattern means we're inside a two-letter
# abbreviation like "U.S." or "F.A.A.", not at the title's real end, so the
# (non-greedy, but period-tolerant) title keeps consuming past it instead of
# truncating mid-abbreviation (e.g. "...category U").
_SECTION_RE = re.compile(r"§\s*(\d+\.\d+[a-z]?)\s+([A-Z].{2,140}?\.(?![A-Z]\.))")
# CFR volumes print every part's appendices together in one block after all the
# numbered sections, not immediately after their own part. Appendices don't
# match _SECTION_RE/_PART_RE/_SUBPART_RE, so without this boundary the last
# section before that block would swallow the entire appendix block (and any
# later parts' front matter) into its own body. The heading is normally
# "APPENDIX A TO PART 25—...", but the table-of-contents page listing all of a
# part's appendices gets extracted with its spacing torn apart ("APPENDIXA
# TOPART25 APPENDIXB TOPART25 ..."), so match on the bare, case-sensitive
# "APPENDIX" token (with an optional glued single-letter suffix) rather than
# the full phrase — body prose only ever refers to "appendix" in lowercase.
_APPENDIX_RE = re.compile(r"\bAPPENDIX[A-Z]{0,3}\b")
# Every CFR volume ends with a "Finding Aids" back-matter block (title/chapter
# index, agency lists, amendment-history tables) that isn't part of the
# regulatory text at all. Without this boundary, whatever section happens to
# be the last one matched swallows this entire (large, irrelevant) block.
_FINDING_AIDS_RE = re.compile(r"\bFinding\s+Aids\b")


def _most_recent(positions: list[tuple[int, object]], at: int) -> object | None:
    """Value of the last (offset, value) entry whose offset <= `at`."""
    i = bisect.bisect_right([p[0] for p in positions], at) - 1
    return positions[i][1] if i >= 0 else None


def parse_cfr(full_text: str, page_spans: list[tuple[int, int]], source: str) -> list[dict]:
    """Split a CFR text stream into one record per § section.

    Tracks the enclosing Part and Subpart for each section's breadcrumb, drops
    the front-matter table of contents by keeping only the longest body per
    (part, section), and strips Federal-Register amendment noise from bodies.
    """
    parts_pos = [(m.start(), m.group(1)) for m in _PART_RE.finditer(full_text)]
    subparts_pos = [(m.start(), (m.group(1), _SUBPART_TITLE_TAIL.sub("", m.group(2).strip())))
                    for m in _SUBPART_RE.finditer(full_text)]

    sec_matches = list(_SECTION_RE.finditer(full_text))
    # Body of a section runs until the next structural marker of any kind.
    boundaries = sorted(
        [m.start() for m in sec_matches]
        + [p[0] for p in parts_pos]
        + [p[0] for p in subparts_pos]
        + [m.start() for m in _APPENDIX_RE.finditer(full_text)]
        + [m.start() for m in _FINDING_AIDS_RE.finditer(full_text)]
    )
    offsets = [s[0] for s in page_spans]

    def page_at(off: int) -> int | None:
        i = bisect.bisect_right(offsets, off) - 1
        return page_spans[i][1] if i >= 0 else None

    # Best body per (part, number) — the TOC copy is short, the real one long.
    best: dict[tuple[str, str], dict] = {}
    for m in sec_matches:
        number, title = m.group(1), m.group(2).strip().rstrip(".")
        body_start = m.end()
        bi = bisect.bisect_right(boundaries, body_start - 1)
        body_end = boundaries[bi] if bi < len(boundaries) else len(full_text)
        body = full_text[body_start:body_end]
        body = _FR_CITATION.sub("", body)
        body = _EFFECTIVE_NOTE.sub("", body)
        body = _TRAILING_HEADER_NOISE.sub("", body)
        body = re.sub(r"\s{2,}", " ", body).strip()

        part = _most_recent(parts_pos, m.start())
        sub = _most_recent(subparts_pos, m.start())
        key = (part or "?", number)
        prev = best.get(key)
        if prev is None or len(body) > len(prev["_body"]):
            best[key] = {
                "part": part,
                "subpart": sub,           # (letter, title) or None
                "section": number,
                "section_title": title,
                "page": page_at(m.start()),
                "_body": body,
                "_start": m.start(),
            }

    records = sorted(best.values(), key=lambda r: r["_start"])
    return [r for r in records if r["_body"]]


def _breadcrumb(rec: dict) -> str:
    crumb = f"14 CFR Part {rec['part']}" if rec["part"] else "14 CFR"
    if rec["subpart"]:
        letter, title = rec["subpart"]
        crumb += f" > Subpart {letter}—{title}"
    crumb += f" > § {rec['section']} {rec['section_title']}".rstrip()
    return crumb


def _split_body(body: str, target: int = TARGET_CHARS, overlap: int = OVERLAP_CHARS) -> list[str]:
    """Split an over-long section body at sentence boundaries with overlap."""
    if len(body) <= target:
        return [body]
    pieces: list[str] = []
    start = 0
    while start < len(body):
        end = start + target
        if end >= len(body):
            pieces.append(body[start:].strip())
            break
        # Prefer to cut at a sentence boundary inside the last 200 chars.
        window = body[end - 200:end]
        cut = max(window.rfind(". "), window.rfind("; "))
        if cut != -1:
            end = end - 200 + cut + 1
        pieces.append(body[start:end].strip())
        start = max(end - overlap, start + 1)
    return [p for p in pieces if p]


def cfr_chunks(rec: dict) -> list[dict]:
    """Expand one section record into breadcrumb-prefixed chunk dicts."""
    crumb = _breadcrumb(rec)
    out = []
    for piece in _split_body(rec["_body"]):
        out.append({
            "part": rec["part"],
            "subpart": rec["subpart"][0] if rec["subpart"] else None,
            "section": f"§ {rec['section']}",
            "section_title": rec["section_title"],
            "page": rec["page"],
            "text": f"{crumb}\n\n{piece}",
        })
    return out


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
# Build / save / load / search
# ════════════════════════════════════════════════════════════════

def _doc_title(text: str, fallback: str) -> str:
    """First markdown H1 ('# Title') as the document title, else the filename."""
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("# "):
            return s[2:].strip()
        if s:
            break
    return fallback.replace("-", " ")


def _records_for_file(path: Path) -> list[dict]:
    """Return per-chunk dicts (without embeddings) for one source file."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        full_text, page_spans = extract_pdf(path)
        sections = parse_cfr(full_text, page_spans, path.name)
        chunks: list[dict] = []
        for rec in sections:
            chunks.extend(cfr_chunks(rec))
        return chunks
    if suffix in (".md", ".txt"):
        text = path.read_text()
        title = _doc_title(text, path.stem)
        return [
            {"part": None, "subpart": None, "section": None,
             "section_title": None, "page": None, "text": c}
            for c in chunk_text(text, doc_title=title)
        ]
    return []


def build_index() -> list[dict]:
    """Walk DOCS_DIR, extract + chunk each file, embed, return list of records."""
    records: list[dict] = []
    chunk_id = 0
    for path in sorted(DOCS_DIR.glob("*")):
        if path.is_dir() or path.suffix.lower() not in (".pdf", ".md", ".txt"):
            continue
        chunks = _records_for_file(path)
        if not chunks:
            print(f"  {path.name}: no chunks (skipped)")
            continue
        vectors = embed([c["text"] for c in chunks])
        for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
            records.append({
                "chunk_id": chunk_id,
                "source": path.name,
                "chunk_index": i,
                "part": chunk["part"],
                "subpart": chunk["subpart"],
                "section": chunk["section"],
                "section_title": chunk["section_title"],
                "page": chunk["page"],
                "text": chunk["text"],
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


# CFR section numbers ("91.155", "61.3") first, as a single token — splitting
# on the "." would scatter them into common, low-signal digit tokens ("91",
# "155") that BM25's IDF can't tell apart from any other number in the corpus.
# "§" itself is dropped: every chunk's breadcrumb contains one, so it carries
# no discriminating signal and its BM25 IDF goes sharply negative (present in
# ~100% of docs), which distorted scores more than it helped.
_TOKEN_RE = re.compile(r"\d+\.\d+[a-z]?|[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


# BM25 index, cached per records-list identity (records is loaded once at
# startup and reused for every query, so building this lazily on first use
# and keeping it around is equivalent to building it once upfront).
_bm25_cache: dict[int, BM25Okapi] = {}


def _get_bm25(records: list[dict]) -> BM25Okapi:
    key = id(records)
    bm25 = _bm25_cache.get(key)
    if bm25 is None:
        bm25 = BM25Okapi([_tokenize(r["text"]) for r in records])
        _bm25_cache[key] = bm25
    return bm25


# Cap on candidates considered per retriever before fusion. Keeps the fused
# ranking to documents each retriever actually vouches for, and bounds the
# fusion cost on a large corpus.
_RANKING_LIMIT = 100


def _rrf_fuse(rankings: list[list[int]], k: int = 60) -> dict[int, float]:
    """Reciprocal Rank Fusion over sparse rank-orderings.

    Each ranking may be a different length and need not cover every record —
    in particular, the BM25 ranking below only includes documents with actual
    keyword overlap. An index absent from a ranking contributes 0 from that
    retriever, rather than picking up a "free" rank position (and thus RRF
    score) among documents it has literally zero relevance to. Returns a
    score only for indices that appear in at least one ranking."""
    scores: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            scores[idx] += 1.0 / (k + rank + 1)
    return scores


def search(
    query: str,
    records: list[dict],
    k: int = 5,
    relative_threshold: float = 0.5,
) -> list[dict]:
    """Hybrid search: fuse embedding similarity with BM25 keyword matching.

    Vector search alone blurs the exact identifiers regulatory text leans on
    (e.g. "§ 91.155", "Part 61") since embeddings capture semantic similarity,
    not exact tokens. BM25 catches those verbatim; the two rankings are
    combined with Reciprocal Rank Fusion so neither score scale has to be
    normalized against the other.

    Two trims keep the context block lean (fewer tokens billed, less noise for
    the model to synthesize over) without hurting recall:

      - Section dedup: a long CFR section gets split into overlapping chunks
        ([indexer.py:_split_body]), so two top hits can be near-duplicate text
        from the same (source, section). Keep only the higher-scoring one.
      - Relevance gate: drop hits whose fused score is below
        `relative_threshold` of the best hit's score — they're unlikely to be
        worth their tokens. Always keeps at least the single best hit.
    """
    if not records:
        return []

    [query_vec] = embed([query])
    vector_ranking = sorted(
        range(len(records)), key=lambda i: cosine_distance(records[i]["embedding"], query_vec)
    )[:_RANKING_LIMIT]

    # Only rank documents with actual keyword overlap (score > 0). Sorting
    # the full corpus here would give every zero-relevance document a rank
    # position too, and RRF can't tell "ranked last because barely relevant"
    # apart from "ranked last because completely irrelevant" — letting a
    # vector-similar-but-keyword-unrelated chunk outscore a genuine exact
    # match (e.g. a section number) on the other retriever.
    bm25_scores = _get_bm25(records).get_scores(_tokenize(query))
    bm25_ranking = sorted(
        (i for i in range(len(records)) if bm25_scores[i] > 0),
        key=lambda i: bm25_scores[i], reverse=True,
    )[:_RANKING_LIMIT]

    fused = _rrf_fuse([vector_ranking, bm25_ranking])
    order = sorted(fused, key=lambda i: fused[i], reverse=True)

    deduped: list[tuple[float, dict]] = []
    seen_sections: set[tuple[str, str]] = set()
    for idx in order:
        r = records[idx]
        key = (r["source"], r["section"]) if r["section"] else None
        if key is not None and key in seen_sections:
            continue
        if key is not None:
            seen_sections.add(key)
        deduped.append((fused[idx], r))
        if len(deduped) >= k:
            break

    if not deduped:
        return []
    best = deduped[0][0]
    kept = [r for score, r in deduped if score >= best * relative_threshold]
    return kept or [deduped[0][1]]


# ════════════════════════════════════════════════════════════════
# Structured lookups — exact section/part access, no embedding needed
# ════════════════════════════════════════════════════════════════

def _normalize_section(s: str) -> str:
    """Reduce any section reference to its bare number for matching.

    '§ 91.119', '91.119', 'Sec. 91.119', '§91.119(b)' -> '91.119'.
    """
    m = re.search(r"(\d+[A-Za-z]?\.\d+[A-Za-z0-9\-]*)", s or "")
    return m.group(1) if m else (s or "").strip()


def _section_sort_key(sec: str) -> tuple:
    m = re.search(r"(\d+)\.(\d+)([A-Za-z0-9\-]*)", sec or "")
    if not m:
        return (10**9, 10**9, "")
    return (int(m.group(1)), int(m.group(2)), m.group(3))


def _chunk_body(text: str) -> str:
    """Strip the breadcrumb prefix, returning just the passage body."""
    return text.split("\n\n", 1)[-1].strip()


def _merge_overlapping(bodies: list[str]) -> str:
    """Join a section's chunk bodies back into one passage, removing the
    fixed overlap the chunker carries between consecutive chunks."""
    out = ""
    for b in bodies:
        b = b.strip()
        if not b:
            continue
        if not out:
            out = b
            continue
        # Largest k such that the accumulated text ends with the next body's
        # first k chars — that's the carried-over overlap, drop it once.
        maxk = min(len(out), len(b), OVERLAP_CHARS * 3)
        k = next((c for c in range(maxk, 20, -1) if out.endswith(b[:c])), 0)
        out += b[k:]
    return out


def get_section(records: list[dict], section: str, max_chars: int = 6000) -> dict | None:
    """Reassemble the full text of one CFR section from its chunks.

    Returns None if no section matches. Caps the body at `max_chars` so a
    pathologically long section can't blow up the context window.
    """
    norm = _normalize_section(section)
    if not norm:
        return None
    hits = sorted(
        (r for r in records if _normalize_section(r.get("section") or "") == norm),
        key=lambda r: (r.get("source", ""), r["chunk_index"]),
    )
    if not hits:
        return None
    body = _merge_overlapping([_chunk_body(r["text"]) for r in hits])
    truncated = len(body) > max_chars
    if truncated:
        body = body[:max_chars].rsplit(" ", 1)[0] + " …[truncated — use search_cfr for the remainder]"
    first = hits[0]
    pages = sorted({r.get("page") for r in hits if r.get("page")})
    return {
        "section": first.get("section"),
        "section_title": first.get("section_title"),
        "part": first.get("part"),
        "source": first.get("source"),
        "page": pages[0] if pages else None,
        "chunk_index": first["chunk_index"],
        "chunk_ids": [r["chunk_id"] for r in hits],
        "text": body,
        "n_chunks": len(hits),
        "truncated": truncated,
    }


def list_sections(records: list[dict], part: str, limit: int = 200) -> dict:
    """List the (section, title) pairs within a CFR Part, in section order."""
    p = _normalize_section(part) or str(part).strip().lstrip("Part ").strip()
    # A part reference may itself be a bare number ("67") or embedded in a
    # section number ("67.103" -> part 67); match on the leading part digits.
    p = re.match(r"\d+", p)
    p = p.group(0) if p else str(part).strip()
    seen: dict[str, str] = {}
    for r in records:
        if str(r.get("part") or "") != p:
            continue
        sec = r.get("section")
        if not sec or sec in seen:
            continue
        seen[sec] = r.get("section_title") or ""
    items = sorted(seen.items(), key=lambda kv: _section_sort_key(kv[0]))
    total = len(items)
    return {"part": p, "total": total, "items": items[:limit], "truncated": total > limit}


def main() -> None:
    print(f"Indexing documents from {DOCS_DIR}/")
    records = build_index()
    save_index(records)
    print(f"\n✓ Indexed {len(records)} chunks → {INDEX_PATH.name}")


if __name__ == "__main__":
    main()
