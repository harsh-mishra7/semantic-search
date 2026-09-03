"""Retrieval only -- no LLM. Print the top-k chunks with their scores.

    python scripts/search.py "how do I deploy the socket server"
    python scripts/search.py -k 10 --full "timezone"
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.embedder import LocalEmbedder
from rag.retriever import MODES, Retriever
from rag.store import VectorStore


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="+")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--index", default="index")
    ap.add_argument("--full", action="store_true", help="print whole chunks, not a preview")
    ap.add_argument("--mode", default="hybrid", choices=list(MODES),
                    help="Phase 6: hybrid (default) fuses dense vectors with BM25")
    ap.add_argument("--candidates", type=int, default=50,
                    help="candidates per arm before fusion (hybrid only)")
    args = ap.parse_args()
    question = " ".join(args.question)

    store = VectorStore.load(args.index)
    # Same model the index was built with -- Retriever refuses otherwise.
    retriever = Retriever(LocalEmbedder(store.meta.model_name), store,
                          mode=args.mode, candidates=args.candidates)

    print(f"\nquery   {question!r}")
    print(f"index   {len(store)} chunks, {store.meta.model_name}, "
          f"{store.meta.chunk_size}/{store.meta.overlap}")
    # The score column means something different per mode, so say which.
    kind = {"dense": "cosine", "bm25": "BM25", "hybrid": "RRF"}[args.mode]
    print(f"mode    {args.mode} (scores are {kind})\n")

    for r in retriever.retrieve(question, k=args.k):
        print(f"{r.rank}. {r.score:.3f}  {r.chunk.id}  [{r.chunk.start}:{r.chunk.end}]")
        body = r.chunk.text if args.full else r.chunk.text[:280].rstrip()
        # Collapse blank lines so a preview stays one visual block.
        for line in [ln for ln in body.splitlines() if ln.strip()]:
            print(textwrap.shorten(line, width=96, placeholder=" ..."))
        if not args.full and len(r.chunk.text) > 280:
            print("   [...]")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
