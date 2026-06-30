# Context Management Project — RAG over a document corpus

A starter that extends the Foundations chat with retrieval-augmented generation. You'll implement the indexing pipeline and wire retrieval + citations into the chat backend.

## What's in here

```
rag-starter/
├── documents/                  20 real Wikipedia articles (Apollo missions)
├── indexer.py                  walk docs → chunk → embed → store
├── backend/
│   ├── app.py                  extended chat with RAG stubs
│   └── requirements.txt
├── frontend/                   React UI (Foundations chat + Sources display)
└── .env.example
```

**The corpus** is 20 Wikipedia articles (plain-text extracts):

- 15 Apollo missions: 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17
- 5 related: Apollo program (overview), Saturn V, Lunar Module, Command/Service Module, Mission Control Center

Total ~850 KB of text across the 20 files. After chunking (~1000 chars each) you'll have several hundred chunks. Real enough that some questions answer crisply and others surface the seams.

## Setup

```bash
# from this directory
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt

cp .env.example .env
# set ANTHROPIC_API_KEY
```

The first time you run anything that imports `sentence-transformers`, it will download the embedding model (multilingual, ~470 MB). One-time only.

## Your job

### 1. Implement chunking and build the index

Open `indexer.py`. There's a `TODO` for `chunk_text()`. Implement it (see the Context Management lecture slide for one working version). Then:

```bash
python indexer.py
# Indexing documents from documents/
#   01-overview.md: 3 chunks
#   02-streaks.md: 4 chunks
#   ...
# ✓ Indexed N chunks → index.pkl
```

### 2. Wire retrieval into the chat backend

Open `backend/app.py`. There are `TODO`s for:

- Updating `SYSTEM_PROMPT` with citation rules
- Calling `search(user_message, INDEX, k=5)` to get top chunks
- Formatting them as a numbered context block
- Building `user_content` with `CONTEXT:` + `QUESTION:`

The citation parser is already wired — it returns the source filenames the model cited, which the frontend already displays under each answer.

### 3. Run it

```bash
# Terminal 1 — backend
cd backend
python app.py

# Terminal 2 — frontend
cd frontend
npm install
npm run dev
```

Open <http://localhost:5173>. Ask questions about the Apollo program. You should see:
- An answer that draws on the indexed docs
- A `Sources:` line citing which files were used

**Try these (graded easy → hard):**

- *"What was the cause of the Apollo 1 fire?"* — single-doc, factual.
- *"Which Apollo missions landed on the Moon?"* — cross-cutting, enumeration.
- *"Compare the moonwalk durations of Apollo 11 and Apollo 17."* — cross-doc, comparison.
- *"List Apollo missions that used the Saturn V rocket."* — cross-doc, requires reasoning over the corpus.
- *"What is the Artemis program?"* — **out-of-corpus**; the system should say it doesn't know rather than hallucinate.

### 4. Report

Pick **5 test questions** that probe the corpus from different angles. For each:
- The question
- The answer the system gave
- Whether the cited sources are correct (open the file, verify)
- A judgment: did the system answer well, weakly, or hallucinate?

Then write up **2 strengths and 2 weaknesses** of your implementation with the worked examples as evidence.

## What to present

- Your chunking choice (size, overlap, boundary rule) and why
- Your system prompt's citation rules
- One question that works cleanly, with the right citations
- One question that fails — wrong answer, missing citation, or hallucinated source
- What you would change (chunking? retrieval? prompt?) to fix the failure

## Alternative project

Want to RAG over your own corpus (your notes, a docs site you've cloned, a code repo's READMEs)? Replace the contents of `documents/` and re-run `python indexer.py`. The rest of the pipeline works unchanged.
