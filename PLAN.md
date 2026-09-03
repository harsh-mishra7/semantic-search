# RAG Semantic Search — Build Plan

A learning project. We build a retrieval-augmented generation system from scratch,
in stages, with no framework doing the interesting parts for us.

**Status:** Phases 0-5 complete; Phase 4 re-run on a new corpus. The corpus is
now Javier Marías's *A Heart So White* (18 documents: 16 chapters, Jonathan
Coe's introduction, the Penguin publication note) in place of the quick-clinic
docs. Retrieval on prose: **recall@1 0.71, MRR 0.779** at 500/0 -- down from
0.88/0.901 on technical documentation, and the drop is the point. Generation
running on `gemini-3.6-flash`. Next: Phase 6, where this corpus should pay off.

---

## 1. What we are building

A command-line (later: HTTP) tool that answers questions about *your own documents*.

```
        ┌─────────── INDEXING (offline, run once per corpus change) ───────────┐

  docs/*.md ──▶ load ──▶ chunk ──▶ embed ──▶ store vectors + text + metadata
                                    │
                                    └─ sentence-transformers (local, free)


        ┌─────────── QUERY (online, per question) ───────────┐

  "how do I rotate the API key?"
        │
        ├──▶ embed the question (same model!)
        │
        ├──▶ similarity search against stored vectors ──▶ top-k chunks
        │
        └──▶ stuff chunks into a prompt ──▶ Claude ──▶ answer with citations
```

Two halves. The **retrieval** half is where almost all the engineering lives.
The **generation** half is a single API call with a carefully written prompt.

Most people who say "I built a RAG app" have built a mediocre retrieval system
with a good LLM papering over it. We are going to do the opposite: build the
retrieval well enough that we can *see* it working before we ever call an LLM.

---

## 2. The mental model (read this before writing code)

### The core idea

Keyword search matches **strings**. Semantic search matches **meaning**.

A query for `"how do I reset my password"` should find a document that says
`"account recovery: to choose a new passphrase, visit..."` — zero shared keywords,
same meaning.

We do this by converting text into **vectors** (lists of ~384 floats). The
embedding model is trained so that texts with similar meaning land near each
other in that 384-dimensional space. "Near" is measured with **cosine
similarity** — the cosine of the angle between two vectors, ranging from -1
(opposite) to 1 (identical).

So: semantic search = nearest-neighbour lookup in a vector space.

### Why RAG exists

An LLM knows what was in its training data. It does not know:
- your private documents
- anything after its training cutoff
- specifics it saw once and compressed away

You could paste all your documents into the prompt, but context windows cost
money and attention degrades over very long inputs. So instead: **retrieve the
few most relevant pieces, and put only those in the prompt.** That's it. That is
the entire idea of RAG. Everything else is implementation detail.

### The four places RAG goes wrong

Keep these in mind the whole way through — every design decision maps to one:

1. **Bad chunking.** The answer is split across two chunks, so neither one alone
   is retrievable or sufficient.
2. **Bad retrieval.** The right chunk exists but ranks 14th, and we only pass
   the top 5.
3. **Bad prompting.** The right chunk is in the prompt, but the model ignores it
   and answers from its own memory (or hallucinates a citation).
4. **No evaluation.** All three of the above are happening and you don't know it,
   because you tested with three questions you already knew the answer to.

We will build a small evaluation set in Phase 4 specifically so that 4 does not
happen to us.

---

## 3. Stack, and why

| Choice | What | Why this one |
|---|---|---|
| Language | Python 3.12 | The entire RAG/ML ecosystem is Python-first. When something confuses you, you can read the library source. |
| Embeddings | `sentence-transformers`, model `all-MiniLM-L6-v2` | Runs locally on CPU, ~80 MB, free, no API key, no rate limits. 384 dimensions. Fast enough to re-embed a corpus while experimenting — which we will do a lot. |
| Vector store | numpy array (Phase 3) → optional swap later | A brute-force cosine search over a numpy matrix is ~15 lines and is genuinely fast up to ~100k chunks. Writing it ourselves means we understand what Chroma/Qdrant/pgvector are actually doing before we adopt one. |
| Generation | Claude (`anthropic`, `claude-opus-5`) **or** Gemini (`google-genai`) | The "G" in RAG. Both sit behind one `Generator` protocol, same prompt, so they are directly comparable. `--backend auto` follows whichever key is set. |
| Corpus | A novel, split per chapter into `data/` | Real text you know well, so you can judge whether a retrieval result is actually good. Prose has no headings, which invalidates both Phase 4 winners -- see §8. |
| Interface | CLI first, FastAPI later | A CLI keeps the feedback loop tight. HTTP is a Phase 7 concern. |

**Deliberately not using LangChain or LlamaIndex.** They are fine tools, but they
hide the exact steps we are trying to learn. Once you have built this, those
frameworks become obvious instead of magical — and you'll be able to tell when
they're helping vs. getting in the way.

### A note on embedding models

`all-MiniLM-L6-v2` is small and mediocre by 2026 standards. That is a feature for
learning: its failures are visible, which teaches you more than a model that
quietly works. Better options for later (Phase 6 swap exercise):

- `BAAI/bge-base-en-v1.5` — still local, notably better, ~440 MB, 768 dims
- **Voyage AI** (`voyage-3`) — hosted, retrieval-specialist, costs money. Note:
  Anthropic does not serve an embeddings endpoint; Voyage is the partner they
  point at. Worth knowing so you don't go looking for `client.embeddings.create()`
  in the Anthropic SDK — it doesn't exist.

We build against a small `Embedder` interface so swapping is a one-line change.

---

## 4. Repo layout (target state)

```
semantic-search/
├── PLAN.md                 ← this file
├── README.md               ← written last, in Phase 7
├── requirements.txt
├── .env.example            ← ANTHROPIC_API_KEY=...   (.env itself is gitignored)
├── data/                   ← your source documents (.md, .txt)
├── index/                  ← generated artifacts; gitignored
│   ├── vectors.npy
│   └── chunks.json
├── rag/
│   ├── __init__.py
│   ├── loader.py           ← Phase 1: files → Document objects
│   ├── chunker.py          ← Phase 1: Document → list[Chunk]
│   ├── embedder.py         ← Phase 2: text → vectors
│   ├── store.py            ← Phase 3: save/load vectors, similarity search
│   ├── retriever.py        ← Phase 3/6: query → ranked chunks
│   ├── generator.py        ← Phase 5: chunks + question → cited answer
│   └── pipeline.py         ← Phase 5: wires the above together
├── scripts/
│   ├── index.py            ← build the index
│   ├── search.py           ← retrieval only, prints ranked chunks + scores
│   ├── ask.py              ← full RAG: retrieve + generate
│   └── split_story.py      ← one-off corpus prep: ebook export → per-chapter files
└── eval/
    ├── questions.yaml      ← Phase 4: our test set
    ├── verify_questions.py ← audits the ANSWER KEY (paths, anchors, coverage)
    └── run_eval.py         ← Phase 4: recall@k, MRR
```

---

## 5. The phases

Each phase is independently runnable and ends with a checkpoint you can actually
execute. Do not move to the next phase until the checkpoint passes and you can
explain *why* it passes.

---

### Phase 0 — Setup

**Goal:** a working environment and a real corpus.

Steps:
1. `python3 -m venv .venv && source .venv/bin/activate`
2. `requirements.txt` with `sentence-transformers`, `numpy`, `anthropic`, `python-dotenv`, `pyyaml`
3. `pip install -r requirements.txt` (first run downloads torch — a few hundred MB, be patient)
4. Put **at least 10–20 documents** in `data/`. Your own notes, a project's docs,
   saved articles. They need to be things you know well enough to judge answers on.
5. `.env` with your `ANTHROPIC_API_KEY` (not needed until Phase 5, but set it now)

**Checkpoint:** `python -c "from sentence_transformers import SentenceTransformer; print('ok')"`

**Why the corpus size matters:** with 3 documents everything retrieves perfectly
and you learn nothing. You need enough material that the wrong answer is
plausible.

---

### Phase 1 — Load and chunk

**Goal:** turn a directory of files into a list of text chunks with metadata.

Concepts you will meet:
- **Why chunk at all?** Two reasons. (a) You embed a whole document into one
  vector and it becomes a mush that's near everything and specific to nothing.
  (b) You want to put *relevant excerpts* in the prompt, not entire files.
- **Chunk size trade-off.** Small chunks → precise matching, but each chunk may
  lack the context needed to be useful. Large chunks → more context, but the
  signal gets diluted and you waste prompt tokens on irrelevant text.
- **Overlap.** If you cut at exactly 500 characters you will cut mid-sentence,
  mid-idea. Overlapping consecutive chunks by ~10–20% means an idea that
  straddles a boundary survives in at least one chunk intact.
- **Metadata.** Every chunk carries `source_path`, `chunk_index`, and character
  offsets. This is what makes citation possible later. Do not skip it.

What we write:
- `Document` and `Chunk` dataclasses
- `loader.py` — walk `data/`, read files, return `Document`s
- `chunker.py` — start with a **fixed-size character chunker with overlap**
  (500 chars, 75 overlap). Deliberately naive. We improve it in Phase 4 *after*
  we can measure whether the improvement helps.

**Checkpoint:** a script that prints how many documents were loaded, how many
chunks were produced, and dumps 3 random chunks so you can eyeball whether they
look like coherent units of text.

**Exercise:** find a chunk that got cut in a bad place. Look at it. That defect
is the thing you will be trying to fix in Phase 4 — it is useful to have seen it
early.

---

### Phase 2 — Embeddings

**Goal:** text in, vectors out. Build intuition for what a vector space *is*.

Concepts:
- Loading a model and calling `.encode()`
- **Batching** — encoding 500 chunks in one call is dramatically faster than 500
  calls, because it saturates the CPU/GPU
- **Normalisation** — if you L2-normalise every vector to unit length, then
  cosine similarity becomes a plain dot product. One cheap preprocessing step
  turns your search into a single matrix multiply. Do this.
- **The symmetry rule:** the query must be embedded with the *exact same model*
  as the documents. Different model = different vector space = meaningless
  distances. This is the single most common beginner bug in RAG.

What we write:
- `embedder.py` with a small interface:
  ```python
  class Embedder(Protocol):
      dimension: int
      def encode(self, texts: list[str]) -> np.ndarray: ...
  ```
  and one implementation, `LocalEmbedder`. The interface exists so Phase 6's
  model swap is a one-line change rather than a refactor.

**Checkpoint — the intuition exercise.** Embed these five strings and print the
full 5×5 cosine similarity matrix:

```
"how do I reset my password"
"account recovery instructions"
"the cat sat on the mat"
"password reset procedure"
"feline seating arrangements"
```

Predict the matrix before you run it. Then run it. You should see two tight
clusters and near-zero similarity between them. **If this doesn't make sense to
you, stop and sit with it** — every later phase assumes this is intuitive.

---

### Phase 3 — Store and search

**Goal:** a working semantic search engine. No LLM yet.

Concepts:
- Stack all chunk vectors into one `(n_chunks, 384)` numpy matrix
- Search = `scores = matrix @ query_vector` → one dot product per chunk, done as
  a single BLAS matrix-vector multiply. Then `np.argsort` for the top-k.
- **Why this is fine:** 100,000 chunks × 384 dims is a 150 MB matrix and the
  multiply takes single-digit milliseconds. Approximate nearest-neighbour
  indexes (HNSW, IVF — what Qdrant/FAISS/pgvector use) only start paying off at
  millions of vectors. Knowing where that line is stops you cargo-culting a
  vector database into a project that doesn't need one.
- **Persistence** — `vectors.npy` (numpy binary) plus `chunks.json` (text +
  metadata), index-aligned. Row `i` of the matrix corresponds to chunk `i` in the
  JSON. Keeping them in sync is your responsibility; a mismatch produces
  confidently wrong results with no error.

What we write:
- `store.py` — `save()`, `load()`, `search(query_vector, k) -> [(index, score)]`
- `retriever.py` — ties embedder + store together: `retrieve(question, k) -> list[Chunk]`
- `scripts/index.py` and `scripts/search.py`

**Checkpoint:** `python scripts/search.py "your question here"` prints the top 5
chunks with similarity scores and source file names.

**Exercises — this is the most important phase, spend time here:**
1. Query with words that *do* appear in your docs. Then with synonyms that
   don't. Semantic search should handle both. Does it?
2. Find a query where the top result is wrong. Diagnose *why*: bad chunk, or bad
   embedding? Look at the score. Is the right chunk at rank 3? Rank 30? Absent?
3. Look at the score distribution. What does a good match score? A bad one?
   Is there a threshold below which results are junk? (Answer: sort of, and it's
   corpus-dependent — which is itself worth learning.)

---

### Phase 4 — Measure, then improve

**Goal:** stop guessing. This phase is what separates a toy from a system.

Concepts:
- **Build an eval set.** 15–25 questions in `eval/questions.yaml`, each with the
  document (or chunk) that *should* be retrieved:
  ```yaml
  - question: "How do I rotate the signing key?"
    expected_sources: ["data/security/keys.md"]
  ```
  Write these by reading your docs and asking "what would someone search for to
  find this?" — not by running the system and recording what it does.
- **Metrics:**
  - `recall@k` — of the questions, what fraction have a correct source in the
    top k? This is the number that matters most: if the right chunk isn't
    retrieved, no amount of prompting saves you.
  - `MRR` (mean reciprocal rank) — averages `1/rank_of_first_correct_result`.
    Rewards ranking the right thing *first*, not just somewhere in the list.
- **Now** run experiments, with a number attached to each:
  - chunk size: 300 vs 500 vs 1000 chars
  - overlap: 0 vs 75 vs 150
  - **structure-aware chunking**: split markdown on `##` headings instead of
    blind character counts, so chunks are semantic units
  - **prepending context**: put the document title / heading path at the top of
    each chunk before embedding. Often a large, cheap win — it gives an
    otherwise-ambiguous chunk something to anchor to.
  - `k`: how many chunks do we actually need to retrieve?

What we write: `eval/run_eval.py` printing recall@1, recall@5, recall@10, MRR.

**Checkpoint:** a table in this file (add it below, in §8) recording each
configuration you tried and its scores.

**Exercise:** find one change that *seems* obviously good and makes the metrics
*worse*. There is almost always one. This is the phase's real lesson — intuition
about retrieval is unreliable, which is exactly why the eval set exists.

---

### Phase 5 — Generation

**Goal:** turn retrieved chunks into a grounded, cited answer.

Concepts:
- **Prompt structure.** Retrieved context goes in the user turn, wrapped in
  clear delimiters, each chunk labelled with an ID and source path. Instructions
  about *how to behave* go in the system prompt.
- **Grounding.** The system prompt must tell the model to answer *only* from the
  provided context, cite the chunk IDs it used, and say "I don't know" when the
  context is insufficient. Getting a model to admit ignorance is harder than it
  sounds and is worth testing explicitly.
- **Citations.** Ask for `[1]`-style references to chunk IDs, then map those back
  to source paths for display. This is why Phase 1's metadata mattered.
- **Streaming.** Print tokens as they arrive so the CLI feels responsive.
- **Token budget.** `k` chunks × chunk size = your input cost. Claude Opus 5 is
  $5/M input, $25/M output tokens. 5 chunks of 500 chars ≈ 700 tokens ≈
  $0.0035 per query. Cheap, but worth knowing how the arithmetic works.

What we write:
- `generator.py` — builds the prompt, calls `client.messages.create(...)` with
  `model="claude-opus-5"` and `thinking={"type": "adaptive"}`, streams the reply
- `pipeline.py` — `ask(question) -> Answer` : retrieve → build prompt → generate
- `scripts/ask.py`

**Checkpoint:** `python scripts/ask.py "..."` gives a correct, cited answer, and
`python scripts/ask.py "who won the 2019 cricket world cup"` (something
definitely not in your corpus) says it doesn't know instead of answering from
the model's own memory.

**Exercises:**
1. Delete the "only use the provided context" instruction. Ask an out-of-corpus
   question. Watch it answer confidently from parametric memory. Put it back.
2. Deliberately retrieve *wrong* chunks (hardcode k bad ones). Does the model
   notice the context is irrelevant, or does it try to construct an answer
   anyway? This tells you how much your prompt is protecting you.
3. Compare answers at `k=3`, `k=5`, `k=10`. More context is not monotonically
   better — irrelevant chunks are a distraction, not neutral filler.

---

### Phase 6 — Make retrieval good

**Goal:** the techniques that separate production RAG from tutorial RAG. Pick
these up one at a time, and **re-run the eval after each one** so you know
whether it actually helped.

- **Hybrid search.** Pure semantic search is bad at exact tokens — error codes,
  function names, product SKUs, rare proper nouns. BM25 (classic keyword
  ranking) is excellent at exactly those. Run both, fuse the rankings
  (Reciprocal Rank Fusion is ~10 lines and works well). This is usually the
  single biggest retrieval win available.
- **Re-ranking.** Retrieve 30 candidates cheaply, then score each one against
  the query with a **cross-encoder** (a model that reads query and document
  *together* rather than embedding them separately — much more accurate, far too
  slow to run over the whole corpus). Keep the top 5. Two-stage retrieval.
- **Query rewriting.** User questions are terse and pronoun-laden ("what about
  the other one?"). Have Claude expand the query into a well-formed standalone
  search query before embedding it. Costs one extra small LLM call.
- **Metadata filtering.** Restrict search by source directory, date, or tag
  before the vector search runs.
- **Swap the embedding model.** Change `LocalEmbedder` to `bge-base-en-v1.5`,
  re-index, re-run eval. Quantify what a better model is worth on *your* corpus.

**Checkpoint:** eval table shows measurable improvement over the Phase 4 baseline,
and you can say which technique bought which points.

---

### Phase 7 — Serve it

**Goal:** something you can actually use and show.

- FastAPI with `POST /search` (retrieval only) and `POST /ask` (full RAG,
  streaming via SSE)
- Load the index once at startup, not per request
- Incremental indexing: only re-embed files whose content hash changed
- Write the `README.md`
- Optional: a small web UI, or swap the numpy store for Chroma/pgvector now that
  you know exactly what it's replacing

---

## 6. What we are deliberately not doing

Named so you know they exist and know you skipped them on purpose:

- Multi-modal RAG (images, tables) — different problem
- Agentic / iterative retrieval (model decides to search again mid-answer) —
  natural sequel once this works
- Graph RAG, hierarchical summarisation indexes
- Fine-tuning an embedding model on your corpus
- Distributed / sharded vector stores
- Conversation memory across turns

---

## 7. Glossary

| Term | Meaning |
|---|---|
| **Embedding** | A fixed-length vector of floats representing text meaning. |
| **Dimension** | Length of that vector. MiniLM: 384. bge-base: 768. |
| **Cosine similarity** | Angle-based similarity, -1 to 1. On normalised vectors it's just a dot product. |
| **Chunk** | A slice of a document, embedded and retrieved as one unit. |
| **Top-k** | The k highest-scoring chunks for a query. |
| **recall@k** | Fraction of queries whose correct source appears in the top k. |
| **MRR** | Mean of 1/(rank of first correct result). Rewards ranking correctly. |
| **BM25** | Classic keyword relevance ranking. Strong on exact/rare terms. |
| **Hybrid search** | Combining semantic + keyword rankings. |
| **Cross-encoder** | Model that scores a (query, doc) pair jointly. Accurate, slow. Used for re-ranking. |
| **RRF** | Reciprocal Rank Fusion — simple way to merge two ranked lists. |
| **Grounding** | Constraining the LLM to answer only from provided context. |
| **ANN** | Approximate nearest neighbour — fast, slightly lossy vector search for large corpora. |

---

## 8. Experiment log

Fill this in from Phase 4 onward. Every row is a thing you learned.

All rows: 25 questions, 17 documents, `all-MiniLM-L6-v2`, document-level scoring.
`+title` = heading path prepended to the chunk before embedding (embedding only,
never citation). `min=N` = merge markdown sections shorter than N chars.

**Caveat that governs how to read this table:** 25 questions means one question
is worth 0.04 of recall. A recall@1 difference of 0.04 is *one question* and is
noise. MRR moves in finer increments and is the more trustworthy column, but it
is still 25 samples. Only differences that repeat across several independent
pairings should be believed -- which is precisely why `+title` is convincing
below and no single row is.

### Chunk size and overlap (fixed-window, the Phase 1 chunker)

| Date | Config | chunks | recall@1 | recall@5 | MRR | Notes |
|---|---|---|---|---|---|---|
| 2026-09-03 | 300 / 0 | 279 | 0.68 | 1.00 | 0.813 | |
| 2026-09-03 | 300 / 75 | 363 | 0.76 | 0.96 | 0.855 | best MRR of the fixed sweep |
| 2026-09-03 | 300 / 150 | 530 | 0.72 | 1.00 | 0.840 | 50% overlap, no gain |
| 2026-09-03 | 500 / 0 | 170 | 0.76 | 0.92 | 0.841 | **beats the baseline with NO overlap** |
| 2026-09-03 | **500 / 75 (baseline)** | 196 | **0.76** | **0.92** | **0.823** | the Phase 1 default |
| 2026-09-03 | 500 / 150 | 232 | 0.72 | 0.92 | 0.798 | more overlap, worse |
| 2026-09-03 | 1000 / 0 | 88 | 0.68 | 1.00 | 0.783 | r@5 perfect, r@1 poor: dilution |
| 2026-09-03 | 1000 / 75 | 94 | 0.64 | 1.00 | 0.779 | worst r@1 measured |
| 2026-09-03 | 1000 / 150 | 101 | 0.68 | 0.96 | 0.784 | |
| 2026-09-03 | 1500 / 150 | 65 | 0.72 | 0.96 | 0.807 | |

### Structure-aware chunking and prepended context

| Date | Config | chunks | recall@1 | recall@5 | MRR | Notes |
|---|---|---|---|---|---|---|
| 2026-09-03 | 500/75 `+title` | 196 | 0.84 | 0.88 | 0.874 | +0.051 MRR over baseline |
| 2026-09-03 | 500/75 `heading` | 375 | 0.72 | 0.92 | 0.800 | **worse than baseline** |
| 2026-09-03 | 500/75 `heading +title` | 375 | 0.80 | 0.92 | 0.856 | title rescues it, still < fixed+title |
| 2026-09-03 | 300/75 `+title` | 363 | 0.80 | 0.96 | 0.863 | |
| 2026-09-03 | 300/75 `heading` | 475 | 0.72 | 0.96 | 0.830 | worse than 300/75 fixed |
| 2026-09-03 | 300/75 `heading +title` | 475 | 0.76 | 1.00 | 0.863 | |
| 2026-09-03 | 1000/75 `heading +title` | 341 | 0.72 | 0.88 | 0.795 | |

### Fixing heading mode: merge short sections

Diagnosis: raw `heading` mode put **84 chunks under 100 chars** into a 375-chunk
index (22%). A heading with one line under it becomes an unretrievable fragment
that still competes for a slot in the top k.

| Date | Config | chunks | tiny <100 | recall@1 | recall@5 | MRR | Notes |
|---|---|---|---|---|---|---|---|
| 2026-09-03 | 500/75 `heading +title` min=0 | 375 | 84 | 0.80 | 0.92 | 0.856 | |
| 2026-09-03 | **500/75 `heading +title` min=150** | 275 | 2 | **0.88** | **0.92** | **0.901** | **shipped** (r@10 = 1.00) |
| 2026-09-03 | 500/75 `heading +title` min=300 | 224 | 4 | 0.84 | 0.92 | 0.889 | |
| 2026-09-03 | 500/75 `heading +title` min=500 | 241 | 10 | 0.88 | 0.92 | 0.905 | ties min=150 within noise |
| 2026-09-03 | 300/75 `heading +title` min=150 | 392 | - | 0.80 | 1.00 | 0.877 | only config with perfect r@5 |
| 2026-09-03 | 700/75 `heading +title` min=150 | 250 | - | 0.80 | 0.92 | 0.850 | |

### Phase 5 notes (generation, `gemini-3.6-flash`)

Checkpoints both pass: the in-corpus question answers correctly citing
`DEPLOYMENT.md` chars 0-257; `"who won the 2019 cricket world cup"` returns
"I don't know" despite being handed three timezone chunks at 0.15 similarity.

Exercise 1 (delete the grounding instruction) **did not reproduce as the plan
predicted**, and the reason is the interesting part:

| System prompt | Answer to an out-of-corpus question |
|---|---|
| Full grounding rules | "I don't know. The provided excerpts do not contain..." |
| Rules deleted, corpus framing kept | "Based on the provided documentation, there is no information about..." |
| Generic "helpful assistant" | "**England** won the 2019 Cricket World Cup, defeating New Zealand in the final on a boundary countback..." |

So the numbered rules were doing less work than the single sentence *"You answer
questions about a private documentation corpus"*. Framing carries the grounding;
the rules mostly shape the wording of the refusal. The parametric-memory failure
is entirely real -- it just needs the framing removed to surface, not the rules.
Worth re-testing per model: this is a property of `gemini-3.6-flash`, not a law.

Exercise 3 (k=3 vs 5 vs 10) on *"what do I set as the start command and root
directory when deploying to Render"*:

| k | thinking tokens | outcome |
|---|---|---|
| 3 | 220 | "I don't know" -- the answering chunk was not retrieved |
| 5 | **872** | partial, hedged, split across two cases |
| 10 | 394 | complete and correct: root directory empty, no start command needed |

More context helped here, but not monotonically: the model burned **4x the
thinking tokens at k=5** than at k=3, struggling with partial information, then
needed less again at k=10 once the answer was actually present. k=3 also
demonstrates the plan's failure mode 2 directly -- the right chunk wasn't
retrieved, so no prompt could have saved it.

Exercise 2 (right question, deliberately irrelevant chunks) passed: fed three
timezone chunks and asked about Vercel, the model refused rather than confabulating.

### What was learned

1. **Prepending the heading path is the single biggest win**, and the only change
   that improved every pairing it was applied to (4/4): +0.051, +0.056, +0.008,
   +0.033 MRR. Consistency across independent pairings is the evidence; no one
   delta would be.
2. **Structure-aware chunking made things worse on its own** (3 of 4 pairings) --
   the plan predicted a change that "seems obviously good" would regress, and
   this was it. It only pays off once short sections are merged.
3. **More overlap is not better.** 500/0 beat 500/75 beat 500/150 on MRR. Extra
   overlap adds partially-redundant chunks that compete for top-k slots without
   adding retrievable content.
4. **Chunk size trades r@1 against r@5 exactly as predicted.** 1000-char chunks
   reached r@5 = 1.00 while dropping to r@1 = 0.64: big chunks make a document
   easy to find and hard to rank first.
5. **Net: MRR 0.823 -> 0.901, recall@1 0.76 -> 0.88** (22/25 questions rank 1st).
   The ER-diagram question that ranked 6th in Phase 3 now ranks 1st -- the
   prepended title gives its mermaid chunks something to anchor to.
6. **Remaining misses** (ranks 4, 7, 7): "run frontend and socket backend at the
   same time", "how does a new user get assigned a role", "what authentication
   work is still outstanding". The last two are Phase 6 candidates -- the TODO
   list is bare checkboxes with no prose, which embeds badly.

---

### Corpus change: a novel instead of technical docs (2026-09-03)

Everything above was measured on 17 markdown files of project documentation.
The corpus is now *A Heart So White* -- 18 documents, 607,449 chars, split on
the chapter boundaries the ebook export already contained
(`scripts/split_story.py`, lossless: verified as an identical non-whitespace
character multiset against the source).

**Why split at all.** `recall@k` and `expected_sources` score at *document*
granularity. One 620 KB file makes the metric vacuous -- every retrieved chunk
is trivially from the only correct document, so recall@k and MRR are 1.000 for
every question under every configuration. 18 documents restores a real ranking
problem.

**The eval set was rewritten** (34 questions, all 18 documents targeted) with
two method changes worth keeping: questions are phrased as a reader would ask
them rather than in the book's words (lifting the text's phrasing measures
lexical overlap, which BM25 wins by construction), and thematic questions were
cut rather than labelled, because a question all sixteen chapters partly answer
has no locus that document-level recall can grade. See the header of
`eval/questions.yaml`.

**A new script, because the answer key needs auditing too.**
`eval/verify_questions.py` checks that every `expected_sources` path resolves,
that each question's *anchor* -- the concrete detail making it answerable --
occurs in no unlisted document, and that no document is untargeted. It found 8
labelling defects on its first run, of two kinds: sloppy substring matching
(`doin` matched *doing* in 14 documents; `crack` matched *cracked*) and genuine
recirculation. The second kind changed the labels: `"You must kill her"` is
spoken in ch03 and re-quoted with both speakers named in ch13, so ch13 answers
too and is now listed. **A bad eval label is worse than a bad retriever**,
because it makes you fix retrieval that was never broken.

### Prose: chunk size and overlap

| Date | Config | chunks | recall@1 | recall@5 | recall@10 | MRR | Notes |
|---|---|---|---|---|---|---|---|
| 2026-09-03 | 300 / 0 | 2036 | 0.74 | 0.82 | 0.94 | 0.775 | best r@1 |
| 2026-09-03 | 300 / 75 | 2702 | 0.59 | 0.91 | 0.94 | 0.731 | worst MRR; overlap hurts most at small sizes |
| 2026-09-03 | **500 / 0** | 1224 | 0.71 | **0.91** | 0.91 | **0.779** | **best MRR -- shipped** |
| 2026-09-03 | 500 / 75 | 1437 | 0.68 | 0.88 | 0.91 | 0.759 | the old default |
| 2026-09-03 | 700 / 75 | 980 | 0.71 | 0.85 | 0.91 | 0.770 | |
| 2026-09-03 | 1000 / 75 | 663 | 0.71 | 0.85 | 0.91 | 0.774 | |
| 2026-09-03 | 1500 / 150 | 459 | 0.71 | 0.85 | 0.88 | 0.759 | |

### Prose: the two Phase 4 winners, re-tested

| Date | Config | chunks | recall@1 | recall@5 | MRR | Notes |
|---|---|---|---|---|---|---|
| 2026-09-03 | 500/75 | 1437 | 0.68 | 0.88 | 0.759 | |
| 2026-09-03 | 500/75 `+title` | 1437 | 0.59 | 0.88 | 0.709 | **-0.050 MRR** |
| 2026-09-03 | 500/75 `heading +title` min=150 | 1437 | 0.59 | 0.88 | 0.709 | **bit-identical to the row above** |
| 2026-09-03 | 500/0 | 1224 | 0.71 | 0.91 | 0.779 | |
| 2026-09-03 | 500/0 `+title` | 1224 | 0.65 | 0.85 | 0.747 | -0.032 MRR |

### What was learned from the corpus change

1. **The shipped config was a no-op, and the eval could not have told us.**
   `heading +title min=150` produces spans bit-identical to plain `fixed` here:
   `sections()` finds 18 markdown headings in the whole corpus (one per file,
   i.e. none), so there is nothing to split and nothing for `min_section` to
   merge. A configuration flag that silently does nothing is worse than one
   that does the wrong thing -- it survives every review because the numbers
   look fine.

2. **`+title` inverted, and by almost exactly the same magnitude.** It was the
   most consistent win on the docs corpus (4/4 pairings, +0.051 MRR); here it
   costs -0.050 and -0.032 across the two pairings tried. The mechanism is
   clear once stated: `_title()` finds no h1 in a `.txt` file and falls back to
   the filename stem, so every chunk in ch07 is embedded with the literal
   prefix `ch07`. That carries no semantic signal, but it is not inert -- it
   pulls all of a chapter's chunks toward each other, which is a *clustering*
   effect, not a relevance one. Per-question at 500/0 it rescued one question
   (Venice/Trouville, rank 5 -> 1, where the chapter-mates helped) and damaged
   six, one of them from rank 3 to 7. **Prepending context is only a win when
   the context is discriminative.** The docs corpus made that condition easy to
   miss because real headings satisfy it automatically.

3. **Zero overlap won again, 2/2 pairings** (300/0 > 300/75 by 0.044 MRR;
   500/0 > 500/75 by 0.020). Third corpus-independent confirmation of the
   earlier finding -- extra overlap adds partially-redundant chunks that
   compete for top-k slots without adding retrievable content.

4. **Chunk size barely matters on continuous prose.** Every config from 300 to
   1500 chars lands in MRR 0.759-0.779, a spread smaller than the +title
   effect. On the docs corpus size mattered a lot, because a window either did
   or did not align with a section boundary. Prose has no boundaries to align
   with, so there is nothing for the size knob to get right or wrong.

5. **Retrieval is simply harder here: MRR 0.901 -> 0.779, recall@1 0.88 ->
   0.71, and r@10 no longer reaches 1.00.** Not a regression -- a harder
   problem. Technical docs are written to be *found*: each states its topic in
   a heading and repeats its key terms. A novel is written to be read in order,
   states its subject nowhere, and refers to everything by pronoun.

6. **The three remaining misses are three different failure modes**, which is
   why they are worth naming rather than averaging:
   - *"how long had the narrator known his wife before marrying her"* -- the
     answer is a duration inside a long discursive sentence. Embeddings are
     poor at numeric and quantitative facts; BM25 would find it instantly.
     **Phase 6 hybrid search should fix this one.**
   - *"the woman who asked her husband for poison as a wedding gift"* -- a
     classical digression (Sophonisba and Masinissa) buried in an art-history
     passage. A genuine semantic failure: the paraphrase never gets near the
     passage. **A re-ranker is the candidate here, not BM25** -- the user has
     no rare term to match on.
   - *"is she still answering personal ads by the end"* -- asks about narrative
     *position*, and a chunk vector encodes no position at all. The chapter
     that introduces the personal ads outranks the chapter that resolves them.
     **This needs metadata, not a better embedding** -- which is exactly the
     Phase 6 metadata-filtering item.

7. **This corpus is a better Phase 6 test bed than the docs were.** It is dense
   with rare proper nouns that pure semantic search handles badly and BM25
   handles well, so hybrid search should show a larger, clearer win than it
   would have on deployment guides.

---

## 8b. Phase 6 experiment log

### Hybrid search: BM25 + Reciprocal Rank Fusion (2026-09-03)

BM25 written from scratch in `rag/hybrid.py` (~40 lines of scoring plus an
inverted index), fused with the dense ranking by RRF. No `rank_bm25`
dependency. Cost at query time is one pass over the postings lists of the
query's terms -- no model, no API call, microseconds.

All rows: 34 questions, 18 documents, `all-MiniLM-L6-v2`, 500/0 chunking,
document-level scoring.

| Date | Mode | recall@1 | recall@5 | recall@10 | MRR | Notes |
|---|---|---|---|---|---|---|
| 2026-09-03 | dense (Phase 4 baseline) | 0.71 | 0.91 | 0.91 | 0.779 | |
| 2026-09-03 | bm25 alone | 0.59 | 0.82 | **0.94** | 0.697 | worse overall, better r@10 than dense |
| 2026-09-03 | hybrid, `rrf_k=60` | 0.71 | 0.85 | 0.94 | 0.791 | the literature default: **+0.012, noise** |
| 2026-09-03 | **hybrid, `rrf_k=10`** | **0.76** | 0.88 | **0.97** | **0.821** | **shipped: +0.042 MRR** |

### Tuning RRF's smoothing constant

| Date | `rrf_k` | recall@1 | recall@5 | recall@10 | MRR |
|---|---|---|---|---|---|
| 2026-09-03 | 1 | 0.71 | 0.91 | 0.97 | 0.804 |
| 2026-09-03 | 3 | 0.71 | 0.91 | 0.97 | 0.808 |
| 2026-09-03 | **10** | **0.76** | 0.88 | **0.97** | **0.821** |
| 2026-09-03 | 20 | 0.74 | 0.85 | 0.94 | 0.798 |
| 2026-09-03 | 60 (Cormack et al.) | 0.71 | 0.85 | 0.94 | 0.791 |

### Candidate depth per arm (at `rrf_k=10`)

| Date | candidates | recall@1 | recall@5 | recall@10 | MRR |
|---|---|---|---|---|---|
| 2026-09-03 | 10 | 0.74 | **0.94** | 0.97 | 0.821 |
| 2026-09-03 | 25 | 0.74 | 0.88 | 0.97 | 0.804 |
| 2026-09-03 | 50 | 0.76 | 0.88 | 0.97 | 0.821 |
| 2026-09-03 | 100 | 0.76 | 0.91 | 0.94 | 0.828 |
| 2026-09-03 | 200 | 0.76 | 0.88 | 0.97 | 0.827 |

Spread 0.804-0.828 across a 20x range of depth. That is noise at 34 questions,
so candidate depth is **not** a tuned parameter here -- worth knowing, because
it is the knob people reach for first. Shipped at 50.

### What was learned

1. **The literature default nearly cost us the whole technique.** At the
   canonical `rrf_k=60`, hybrid search is worth +0.012 MRR -- indistinguishable
   from noise, and it *loses* r@5. The plan promised "usually the single
   biggest retrieval win available" and the first measurement said otherwise.
   At `rrf_k=10` the same code is worth +0.042 MRR, +0.05 r@1 and +0.06 r@10.
   Had we run it once at the default and moved on, the conclusion would have
   been "hybrid search doesn't help on prose", which is false.

2. **Why 60 is wrong here, mechanically.** RRF gives rank r the weight
   1/(k+r). At k=60, rank 1 is worth 1/61 = 0.0164, so a chunk found FIRST by
   one ranker and missed by the other scores below a chunk both rankers put
   around 5th (2/65 = 0.0308). That is the intended design -- agreement over
   enthusiasm -- and it is correct when fusing many similar rankers, which is
   what the paper tuned on. It is wrong when fusing exactly two rankers with
   *disjoint competence*, because then "the other ranker didn't find it" is
   not evidence against a result; it is the expected behaviour.

   The single question that proves it: *"the woman who asked her husband for
   poison as a wedding gift"* -- dense MISS, BM25 **rank 1**, hybrid at k=60
   **MISS**, hybrid at k=10 **rank 4**. Fusion at the default threw away a
   result one arm had ranked first.

3. **A Phase 4 prediction was wrong, and the eval caught it.** §8 predicted
   that question needed a *re-ranker*, "not BM25 -- the user has no rare term
   to match on". BM25 put it at rank 1: "poison", "wedding" and "husband" are
   rare enough in a corpus this size. Intuition about which technique fixes
   which failure is no better than intuition about retrieval generally.

4. **BM25 alone is worse but not dominated.** MRR 0.697 vs 0.779, yet its
   r@10 (0.94) beats dense (0.91): it finds documents the embedding cannot
   reach and then ranks them badly. That combination -- worse average, better
   coverage -- is exactly the profile that makes a ranker worth fusing rather
   than adopting or discarding.

5. **The cost is real and lands where predicted.** *"obliging someone to love
   you"* went from dense rank 1 to hybrid rank 8: a purely abstract question
   with no rare terms, which BM25 misses entirely and therefore drags down.
   Note this is the question `eval/questions.yaml` already flags as its weakest
   label -- so the technique's one casualty is the one question whose label was
   least defensible anyway.

6. **Two of the three Phase 4 misses are fixed; the third is untouched, as
   predicted.** *"how long had the narrator known his wife"* (a duration) went
   MISS -> rank 9. The poison question went MISS -> rank 4. But *"is she still
   answering personal ads by the end"* is still the only outright miss, because
   it asks about narrative *position* and neither a vector nor a term count
   encodes where a chunk sits in the book. Confirms the §8 diagnosis: **this
   one needs metadata, and no amount of better ranking will reach it.**

7. **Net Phase 4 -> Phase 6 so far: MRR 0.779 -> 0.821, recall@1 0.71 -> 0.76,
   recall@10 0.91 -> 0.97** (33 of 34 questions now have a correct source in
   the top 10). Still short of the 0.901 the technical-docs corpus reached,
   which remains the honest comparison: a novel is a harder retrieval target.

### Still to do in Phase 6

- **Metadata filtering** -- the remaining miss needs chapter position in the
  index. Cheapest remaining win, and the only one with a named target.
- **Re-ranking** with a cross-encoder over the top ~30 fused candidates.
- **Query rewriting**, which on this corpus mostly means resolving pronouns.
- **Swap the embedding model** to `bge-base-en-v1.5` and re-measure. Now
  interesting for a second reason: a stronger embedder may erode hybrid's
  margin, and knowing whether it does is the difference between "we need BM25"
  and "we needed a better model".
- **Stemming in `tokenize()`** -- currently absent, so "advertised" does not
  match "advertising". Deliberately left as a measurable change rather than an
  assumption.

---

## 9. Progress

- [x] Phase 0 — Setup
- [x] Phase 1 — Load and chunk
- [x] Phase 2 — Embeddings
- [x] Phase 3 — Store and search
- [x] Phase 4 — Measure and improve
- [x] Phase 5 — Generation
- [~] Phase 6 — Better retrieval (hybrid search done: MRR 0.779 → 0.821)
- [ ] Phase 7 — Serve
