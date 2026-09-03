"""Phase 4: measure retrieval. recall@k and MRR over eval/questions.yaml.

    python eval/run_eval.py                                  # baseline only
    python eval/run_eval.py --sweep 300/0,300/75,500/0,500/75,500/150,1000/75
    python eval/run_eval.py --detail                          # per-question failures

Sweeping in one process is deliberate: the model costs ~10s to load and the
question vectors never change between configs, so both are paid once. Only the
chunk embeddings get recomputed per config.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.chunker import chunk_documents
from rag.embedder import DEFAULT_MODEL, LocalEmbedder
from rag.hybrid import RRF_K, BM25Index, reciprocal_rank_fusion
from rag.loader import load_documents
from rag.rerank import DEFAULT_DEPTH, DEFAULT_RERANK_MODEL, CrossEncoderReranker
from rag.store import IndexMeta, VectorStore

KS = (1, 5, 10)
MAX_K = max(KS)


def parse_spec(spec: str) -> tuple[str, tuple[int, int, str, bool]]:
    """"500/75", "300/75/heading", "500/75/heading+title", "500/75+title"."""
    label, body = spec, spec
    prepend = body.endswith("+title")
    if prepend:
        body = body[: -len("+title")]
    parts = body.split("/")
    size, overlap = int(parts[0]), int(parts[1])
    strategy = parts[2] if len(parts) > 2 else "fixed"
    return label, (size, overlap, strategy, prepend)


def ranked_indices(mode: str, store: VectorStore, bm25: BM25Index,
                   query_vector: np.ndarray, question: str,
                   depth: int, rrf_k: int) -> list[int]:
    """Chunk indices for one question under one retrieval mode, best first.

    `depth` is how many candidates each arm contributes before fusion. It does
    not affect the dense or bm25 modes' top-MAX_K at all -- the top 10 of a
    top-50 request is the same top 10 -- so the same depth is used for every
    mode and the comparison stays fair. It matters only to hybrid, where a
    chunk must appear in SOME arm's candidate list to receive any rank points.
    """
    if mode == "dense":
        return [i for i, _ in store.search(query_vector, k=depth)]
    if mode == "bm25":
        return [i for i, _ in bm25.search(question, k=depth)]
    if mode == "hybrid":
        dense = [i for i, _ in store.search(query_vector, k=depth)]
        keyword = [i for i, _ in bm25.search(question, k=depth)]
        return [i for i, _ in reciprocal_rank_fusion([dense, keyword], k=rrf_k)]
    raise ValueError(f"unknown mode {mode!r}")


def first_hit_rank(indices: list[int], chunks: list, expected: set[str]) -> int | None:
    """1-based rank of the highest-ranked chunk from an expected source, or None.

    Note this scores at the *document* level: any chunk of an expected file
    counts. Chunk-level labelling would be more precise but has to be redone
    every time chunk boundaries move -- which is exactly what we are varying.
    """
    for rank, i in enumerate(indices[:MAX_K], start=1):
        if chunks[i].doc_path in expected:
            return rank
    return None


def score(store: VectorStore, bm25: BM25Index, qvecs: np.ndarray,
          questions: list[dict], mode: str, depth: int, rrf_k: int,
          reranker: CrossEncoderReranker | None = None,
          rerank_depth: int = DEFAULT_DEPTH):
    ranks = []
    for i, q in enumerate(questions):
        indices = ranked_indices(mode, store, bm25, qvecs[i], q["question"], depth, rrf_k)
        if reranker is not None:
            # Stage 2 re-reads only the top `rerank_depth` of stage 1. Anything
            # below that is untouched but kept, so the tail still counts toward
            # recall@10 -- a re-ranker should not be able to LOSE a result that
            # stage 1 had at rank 12.
            head, tail = indices[:rerank_depth], indices[rerank_depth:]
            # chunk.text, not embed_text: the cross-encoder actually reads the
            # language, so a synthetic filename prefix would be noise it has to
            # attend to. (Identical under the shipped config, which prepends
            # nothing -- so this choice is reasoned, not yet measured.)
            reordered = reranker.rerank(
                q["question"], [store.chunks[j].text for j in head], head)
            indices = [j for j, _ in reordered] + tail
        ranks.append(first_hit_rank(indices, store.chunks, set(q["expected_sources"])))
    n = len(questions)
    recall = {k: sum(1 for r in ranks if r is not None and r <= k) / n for k in KS}
    # MRR: a question with no correct source in the top MAX_K contributes 0.
    # It rewards ranking the right thing FIRST, not merely somewhere in the list.
    mrr = sum(1.0 / r for r in ranks if r is not None) / n
    return recall, mrr, ranks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--questions", default="eval/questions.yaml")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--sweep", default="500/75", help="comma-separated size/overlap pairs")
    ap.add_argument("--min-section", type=int, default=0,
                    help="heading strategy: merge sections shorter than this")
    ap.add_argument("--detail", action="store_true", help="list per-question ranks")
    ap.add_argument("--mode", default="dense", choices=["dense", "bm25", "hybrid"],
                    help="Phase 6: dense vectors, BM25 keywords, or RRF over both")
    ap.add_argument("--candidates", type=int, default=50,
                    help="candidates per arm before fusion (hybrid only)")
    ap.add_argument("--rrf-k", type=int, default=RRF_K, help="RRF smoothing constant")
    ap.add_argument("--rerank", action="store_true",
                    help="Phase 6: re-read the top candidates with a cross-encoder")
    ap.add_argument("--rerank-depth", type=int, default=DEFAULT_DEPTH,
                    help="how many stage-1 candidates the cross-encoder re-reads")
    ap.add_argument("--rerank-model", default=DEFAULT_RERANK_MODEL)
    ap.add_argument("--stem", action="store_true", help="suffix-strip BM25 tokens")
    args = ap.parse_args()

    questions = yaml.safe_load(Path(args.questions).read_text(encoding="utf-8"))
    docs = load_documents(args.data)
    configs = [parse_spec(spec) for spec in args.sweep.split(",")]

    embedder = LocalEmbedder(args.model)
    # Queries are config-independent: embed them once for the whole sweep.
    qvecs = embedder.encode([q["question"] for q in questions])
    # Loaded once for the whole sweep, like the embedder: the weights do not
    # depend on the chunking config.
    reranker = CrossEncoderReranker(args.rerank_model) if args.rerank else None

    print(f"\n{len(questions)} questions | {len(docs)} documents | {embedder.model_name} "
          f"| mode={args.mode}"
          + (f" candidates={args.candidates} rrf_k={args.rrf_k}" if args.mode == "hybrid" else "")
          + (f" | rerank={args.rerank_model.split('/')[-1]} depth={args.rerank_depth}"
             if reranker else "")
          + "\n")
    print(f"{'config':>12} {'chunks':>7} {'r@1':>7} {'r@5':>7} {'r@10':>7} {'MRR':>7} {'embed':>7}")
    print("-" * 62)

    results = []
    for label, (size, overlap, strategy, prepend) in configs:
        chunks = chunk_documents(docs, size=size, overlap=overlap, strategy=strategy,
                                 prepend_context=prepend, min_section=args.min_section)
        t0 = time.perf_counter()
        # embed_text, not text: this is where a prepended heading path acts.
        vectors = embedder.encode([c.embed_text for c in chunks])
        t_embed = time.perf_counter() - t0
        store = VectorStore(
            chunks, vectors,
            IndexMeta.now(model_name=embedder.model_name, dimension=embedder.dimension,
                          chunk_size=size, overlap=overlap, n_chunks=len(chunks)),
        )
        bm25 = BM25Index([c.embed_text for c in chunks], stem=args.stem)
        recall, mrr, ranks = score(store, bm25, qvecs, questions,
                                   args.mode, args.candidates, args.rrf_k,
                                   reranker, args.rerank_depth)
        results.append((label, len(chunks), recall, mrr, ranks))
        print(f"{label:>12} {len(chunks):>7} "
              f"{recall[1]:>7.2f} {recall[5]:>7.2f} {recall[10]:>7.2f} {mrr:>7.3f} "
              f"{t_embed:>6.1f}s")

    if args.detail:
        for label, _n, _r, _m, ranks in results:
            print(f"\n--- {label}: per-question rank of first correct source")
            for q, r in sorted(zip(questions, ranks), key=lambda p: (p[1] is not None, p[1] or 0)):
                flag = "MISS" if r is None else f"  {r:>2}"
                print(f"  {flag}  {q['question'][:72]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
