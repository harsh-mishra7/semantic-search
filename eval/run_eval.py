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
from rag.loader import load_documents
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


def first_hit_rank(store: VectorStore, query_vector: np.ndarray, expected: set[str]) -> int | None:
    """1-based rank of the highest-ranked chunk from an expected source, or None.

    Note this scores at the *document* level: any chunk of an expected file
    counts. Chunk-level labelling would be more precise but has to be redone
    every time chunk boundaries move -- which is exactly what we are varying.
    """
    for rank, (i, _score) in enumerate(store.search(query_vector, k=MAX_K), start=1):
        if store.chunks[i].doc_path in expected:
            return rank
    return None


def score(store: VectorStore, qvecs: np.ndarray, questions: list[dict]):
    ranks = [
        first_hit_rank(store, qvecs[i], set(q["expected_sources"]))
        for i, q in enumerate(questions)
    ]
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
    args = ap.parse_args()

    questions = yaml.safe_load(Path(args.questions).read_text(encoding="utf-8"))
    docs = load_documents(args.data)
    configs = [parse_spec(spec) for spec in args.sweep.split(",")]

    embedder = LocalEmbedder(args.model)
    # Queries are config-independent: embed them once for the whole sweep.
    qvecs = embedder.encode([q["question"] for q in questions])

    print(f"\n{len(questions)} questions | {len(docs)} documents | {embedder.model_name}\n")
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
        recall, mrr, ranks = score(store, qvecs, questions)
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
