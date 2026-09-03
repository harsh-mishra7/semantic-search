# RAG semantic search

A retrieval-augmented generation system built from scratch — no LangChain, no
LlamaIndex, no vector database. Hybrid retrieval (dense vectors + BM25 fused by
Reciprocal Rank Fusion), a cross-encoder re-ranker, grounded answers with
checkable citations, and a CLI plus an HTTP API.

It was built to be *understood*, phase by phase, with an evaluation set
deciding every design question. [ARCHITECTURE.md](ARCHITECTURE.md) explains how
the code fits together; [PLAN.md](PLAN.md) is the build log: what was tried,
what the numbers were, and which confident predictions turned out wrong.

## Results

On a 607 KB corpus of continuous prose (18 documents, 1224 chunks), scored
against 34 hand-written questions:

| Retrieval | recall@1 | recall@5 | recall@10 | MRR |
|---|---|---|---|---|
| dense only (`all-MiniLM-L6-v2`) | 0.71 | 0.91 | 0.91 | 0.779 |
| \+ hybrid BM25 / RRF | **0.76** | 0.88 | 0.97 | 0.821 |
| \+ cross-encoder re-rank | 0.71 | **1.00** | **1.00** | 0.810 |
| dense `bge-base-en-v1.5` + hybrid | **0.76** | 0.97 | **1.00** | **0.849** |

Which row is "best" depends on what consumes it. `ask.py` puts *k*=5 chunks in
a prompt, so answer quality is decided by **recall@5** — not by whether the
right chunk ranks first. Re-ranking makes recall@5 perfect while slightly
*lowering* MRR, so it is on by default for answering and off for browsing.
See [PLAN.md §8b](PLAN.md) for the full table and the reasoning.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # first run downloads torch, be patient

# Bring your own corpus: any .md / .txt files under data/
mkdir -p data && cp ~/notes/*.md data/

python scripts/index.py                  # build the index
python scripts/search.py "your question" # retrieval only, no API key needed
```

For generated answers, add one key to `.env` (see `.env.example`):

```bash
echo 'GEMINI_API_KEY=...' > .env         # or ANTHROPIC_API_KEY
python scripts/ask.py "your question"
```

**The corpus in `data/` is gitignored.** This repo was developed against Javier
Marías's *A Heart So White*, which is in copyright and not distributed here.
Use documents you know well — you cannot judge whether a retrieval result is
good on material you have not read.

## How it works

```
INDEXING (offline)
  data/*.{md,txt} → load → chunk → embed → index/{vectors.npy, chunks.json}
                                     └─ sentence-transformers, local, free

QUERY (per question)
  question ─┬─→ embed → cosine over the vector matrix ─┐
            └─→ BM25 over an inverted index ───────────┴→ RRF → top 50
                                                              ↓
                                    cross-encoder re-reads them → top k
                                                              ↓
                              prompt with labelled excerpts → LLM → cited answer
```

Nothing here is deep. The dense search is one `numpy` matrix-vector multiply;
BM25 is ~40 lines over a postings dict; RRF is eight. The engineering is in
knowing which to reach for and being able to prove it helped.

| Module | Role |
|---|---|
| [`rag/loader.py`](rag/loader.py) | files → `Document`, with content hashes |
| [`rag/chunker.py`](rag/chunker.py) | `Document` → `Chunk`, fixed-window or heading-aware |
| [`rag/embedder.py`](rag/embedder.py) | text → L2-normalised vectors |
| [`rag/store.py`](rag/store.py) | persistence + cosine search |
| [`rag/hybrid.py`](rag/hybrid.py) | BM25 and Reciprocal Rank Fusion |
| [`rag/rerank.py`](rag/rerank.py) | cross-encoder second stage |
| [`rag/retriever.py`](rag/retriever.py) | question → ranked chunks, all modes |
| [`rag/generator.py`](rag/generator.py) | prompt + streaming, Claude or Gemini |
| [`rag/pipeline.py`](rag/pipeline.py) | loads everything once, answers many |
| [`rag/api.py`](rag/api.py) | FastAPI: `/search`, `/ask`, `/ask/stream` |

## CLI

```bash
python scripts/search.py "question"              # hybrid, no re-rank, no LLM
python scripts/search.py --mode dense --full "…" # compare retrieval modes
python scripts/search.py --rerank "…"            # add the cross-encoder
python scripts/ask.py "question"                 # full RAG, re-rank on
python scripts/ask.py --no-rerank -k 10 "…"
python scripts/ask.py --dry-run "…"              # print the prompt, spend nothing
python scripts/ask.py --backend claude "…"       # or gemini; auto follows your key
```

## HTTP

```bash
uvicorn rag.api:app --port 8000     # interactive docs at /docs
```

| Endpoint | Purpose |
|---|---|
| `GET /health` | index size, model, chunking and retrieval settings |
| `POST /search` | retrieval only — no LLM, no key, no cost |
| `POST /ask` | full RAG, one JSON answer with citations |
| `POST /ask/stream` | same, as SSE: `sources`, then `delta`×N, then `done` |

```bash
curl -N -X POST localhost:8000/ask/stream \
  -H 'Content-Type: application/json' \
  -d '{"question":"…","k":5}'
```

`sources` is sent before the first token, so a client can show what is being
read from while the model is still thinking. Configure with `RAG_INDEX`,
`RAG_MODE` and `RAG_RERANK`; run single-worker, since the model and index are
per-process.

## Incremental indexing

`scripts/index.py` re-embeds only documents whose **content hash** changed:

```
reused    17/18 documents unchanged, re-embedding 1: heart-so-white/ch05.txt
embedded  (1224, 384) float32  (13 new in 0.14s, 1211 reused)
```

That is 0.14s against 16s for a full rebuild. Hashes are over content, not
mtime, because mtime changes on `git checkout` (when nothing needs re-embedding)
and does *not* change when a file is restored in place (when everything does).

If any input to chunking or embedding differs from the stored index — model,
dimension, chunk size, overlap, strategy, `min_section`, `prepend_context` — it
rebuilds everything and says so. Reusing vectors across settings would produce
a working index with silently wrong rankings, which is the worst failure this
system can have. `--full` forces a rebuild.

## Evaluation

```bash
python eval/verify_questions.py                       # audit the answer key first
python eval/run_eval.py --mode hybrid --rerank
python eval/run_eval.py --sweep 300/75,500/0,1000/75  # compare chunkings
python eval/run_eval.py --mode hybrid --detail        # per-question ranks
```

`verify_questions.py` audits the *labels*, not the system: every
`expected_sources` path resolves, every question's anchor term appears in no
unlisted document, and no document is untargeted. It found 8 labelling defects
the first time it ran. A bad eval label is worse than a bad retriever — it
makes you fix retrieval that was never broken.

## Deliberately not done

Query rewriting (no headroom left at recall@5 = 1.00), metadata filtering (its
only motivating question was solved by re-ranking), approximate nearest
neighbours (brute force is faster than the alternatives below ~100k chunks),
multi-modal and agentic retrieval, conversation memory. Reasons in
[PLAN.md §6](PLAN.md) and §8b.
