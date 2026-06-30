"""Context Management RAG starter — extended chat backend.

This is the Foundations chat backend with stubs for retrieval-augmented generation.

TODO:
  1. Update SYSTEM_PROMPT with citation rules.
  2. In /api/chat: retrieve top-K chunks for the user's question.
  3. Format chunks as a numbered context block.
  4. Build the user_content with CONTEXT + QUESTION.
  5. Parse citation numbers from the answer; return them to the frontend.
"""
import re
import sys
import time
from pathlib import Path

# Make the parent directory importable so we can use indexer.py
sys.path.insert(0, str(Path(__file__).parent.parent))

from anthropic import Anthropic
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS

from indexer import load_index, search

load_dotenv()  # ANTHROPIC_API_KEY from .env

app = Flask(__name__)
CORS(app)
client = Anthropic()

# Load the index once at startup. Fails fast if no index — run `python indexer.py` first.
INDEX = load_index()
print(f"Loaded {len(INDEX)} chunks from disk")


# ════════════════════════════════════════════════════════════════
# TODO — update SYSTEM_PROMPT with citation rules.
# Suggestions:
#   - Answer ONLY from the provided context.
#   - Cite each factual claim with [n] using the numbers in the context.
#   - If the context doesn't contain the answer, say so explicitly.
# ════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a helpful assistant that answers questions using ONLY the \
sources provided in the CONTEXT block of each message.

Rules:
- Base every statement strictly on the provided context. Do not use outside knowledge, \
and do not guess or infer beyond what the sources state.
- Cite each factual claim with a bracketed source number, e.g. [1] or [2][3], using the \
numbers shown in the context. Place the citation immediately after the claim it supports.
- Only use citation numbers that appear in the context. Never invent a number.
- If the context does not contain enough information to answer, say so explicitly \
(e.g. "The provided sources don't contain an answer to that.") and do not fabricate one.
- If only part of the question is supported, answer that part and clearly state what the \
sources do not cover."""


@app.route("/api/chat", methods=["POST"])
def chat():
    t0 = time.perf_counter()  # measure end-to-end answer latency (retrieval + LLM)
    user_message = request.json["message"]

    # Retrieve the top-K most relevant chunks, then augment the prompt with a
    # numbered context block so the model can ground its answer and cite sources.
    hits = search(user_message, INDEX, k=5)
    context = "\n\n".join(f"[{i + 1}] {h['text']}" for i, h in enumerate(hits))
    user_content = f"CONTEXT:\n{context}\n\nQUESTION:\n{user_message}"

    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
    )
    answer = resp.content[0].text

    # ────────────────────────────────────────────────────────────
    # TODO — citation extraction
    #
    # Parse [n] markers from the answer. Drop invented ones.
    # For each valid citation, return its source filename and chunk index
    # so the frontend can display them.
    # ────────────────────────────────────────────────────────────

    citations = _build_citations(answer, hits)

    usage = {
        "input_tokens": resp.usage.input_tokens,
        "output_tokens": resp.usage.output_tokens,
        "total_tokens": resp.usage.input_tokens + resp.usage.output_tokens,
    }

    latency_ms = round((time.perf_counter() - t0) * 1000)

    return jsonify({
        "reply": answer,
        "citations": citations,
        "usage": usage,
        "latency_ms": latency_ms,
    })


def _build_citations(answer: str, hits: list[dict]) -> list[dict]:
    """Return one citation entry per unique valid [n] used in the answer.

    Extract the numbers, drop out-of-range, and return the matching hit's
    filename, chunk_index, and the chunk text itself so the frontend can show
    the original retrieved context behind each citation.
    """
    used = [int(n) for n in re.findall(r"\[(\d+)\]", answer)]
    seen: set[int] = set()
    citations: list[dict] = []
    for n in used:
        if n in seen or n < 1 or n > len(hits):
            continue
        seen.add(n)
        h = hits[n - 1]
        citations.append({
            "n": n,
            "source": h["source"],
            "chunk_index": h["chunk_index"],
            "text": h["text"],
        })
    return citations


if __name__ == "__main__":
    app.run(port=5000, debug=True)
