"""Build the index: data/ -> index/vectors.npy + index/chunks.json

    python scripts/index.py
    python scripts/index.py --size 300 --overlap 0     # Phase 4 sweeps
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from rag.chunker import DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP, chunk_document
from rag.embedder import DEFAULT_MODEL, LocalEmbedder
from rag.loader import load_documents
from rag.store import IndexMeta, VectorStore


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--index", default="index")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--size", type=int, default=DEFAULT_CHUNK_SIZE)
    ap.add_argument("--strategy", default="heading", choices=["fixed", "heading"])
    # Defaults below are the measured best for the CURRENT corpus (prose), not
    # universal truths -- see PLAN.md §8. On the previous markdown corpus the
    # winners were overlap=75 and prepend_context=True; on a novel with no
    # headings both lose (+title costs 0.03-0.05 MRR, since _title() falls back
    # to the filename and prepends a non-discriminative token to every chunk).
    # `strategy=heading` is left on because it is provably inert here (no
    # markdown headings exist) and pays off again the moment .md files return.
    ap.add_argument("--overlap", type=int, default=0)
    ap.add_argument("--prepend-context", action="store_true", default=False)
    ap.add_argument("--no-prepend-context", dest="prepend_context", action="store_false")
    ap.add_argument("--min-section", type=int, default=150)
    ap.add_argument("--full", action="store_true",
                    help="rebuild every vector, ignoring the existing index")
    args = ap.parse_args()

    t0 = time.perf_counter()
    docs = load_documents(args.data)
    # An empty corpus otherwise produces a (0, 384) matrix and a perfectly
    # valid, perfectly useless index -- and then search returns nothing with no
    # explanation. Exposed by actually emptying data/.
    if not docs:
        raise SystemExit(f"no .md/.txt documents under {args.data}/ -- nothing to index")

    hashes = {d.path: d.content_hash for d in docs}
    chunk_kw = dict(size=args.size, overlap=args.overlap, strategy=args.strategy,
                    prepend_context=args.prepend_context, min_section=args.min_section)

    # ---- Phase 7: incremental indexing --------------------------------------
    # Reuse a previous run's vectors for documents whose content hash is
    # unchanged. Embedding is the only expensive step (17s for this corpus,
    # 104s with bge-base), and editing one chapter should not re-pay it for
    # the other seventeen.
    #
    # The safety rule is in IndexMeta.reuse_signature: reuse only when EVERY
    # input to chunking and embedding is identical. Mixing vectors across
    # settings yields a working index with silently wrong rankings, so a
    # mismatch downgrades to a full rebuild rather than trying to be clever.
    previous: VectorStore | None = None
    if not args.full:
        try:
            previous = VectorStore.load(args.index)
        except (FileNotFoundError, ValueError, KeyError):
            previous = None   # absent or unreadable: just rebuild

    embedder = LocalEmbedder(args.model)
    wanted = (embedder.model_name, embedder.dimension, args.size, args.overlap,
              args.strategy, args.prepend_context, args.min_section)
    if previous is not None and previous.meta.reuse_signature != wanted:
        print("config    changed since last index -- full rebuild "
              f"(was {previous.meta.reuse_signature}, now {wanted})")
        previous = None

    reusable = previous.meta.doc_hashes if previous is not None else {}

    # Per document: its chunks, and its vectors if they can be lifted from the
    # previous index. Iterating `docs` (already sorted) keeps the corpus-wide
    # chunk order canonical, which is what the row<->chunk invariant needs.
    plan: list[tuple[list, np.ndarray | None]] = []
    reused_docs = 0
    for doc in docs:
        if reusable.get(doc.path) == hashes[doc.path]:
            old_chunks, old_vectors = previous.vectors_for(doc.path)
            if old_chunks:
                plan.append((old_chunks, old_vectors))
                reused_docs += 1
                continue
        plan.append((chunk_document(doc, **chunk_kw), None))

    chunks = [c for cs, _ in plan for c in cs]
    t_chunk = time.perf_counter() - t0
    print(f"loaded    {len(docs)} documents -> {len(chunks)} chunks  ({t_chunk:.2f}s)")
    print(f"chunking  {args.strategy} size={args.size} overlap={args.overlap} "
          f"min_section={args.min_section} prepend_context={args.prepend_context}")
    print(f"model     {embedder.model_name}  {embedder.dimension}-D")

    dropped = sorted(set(reusable) - set(hashes))
    changed = [d.path for d, (_, v) in zip(docs, plan) if v is None]
    if previous is not None:
        print(f"reused    {reused_docs}/{len(docs)} documents unchanged"
              + (f", re-embedding {len(changed)}: {', '.join(changed[:4])}"
                 f"{' ...' if len(changed) > 4 else ''}" if changed else "")
              + (f", dropped {len(dropped)}" if dropped else ""))

    # One batched call for everything that actually needs embedding.
    fresh = [c for cs, v in plan if v is None for c in cs]
    t0 = time.perf_counter()
    if fresh:
        # embed_text: includes the prepended "title > heading path" when enabled.
        new_vectors = embedder.encode([c.embed_text for c in fresh], show_progress=True)
    else:
        new_vectors = np.empty((0, embedder.dimension), dtype=np.float32)
    t_embed = time.perf_counter() - t0

    # Stitch the matrix back together in document order, splicing the freshly
    # embedded rows into the gaps the reused blocks left.
    blocks, cursor = [], 0
    for cs, v in plan:
        if v is None:
            blocks.append(new_vectors[cursor:cursor + len(cs)])
            cursor += len(cs)
        else:
            blocks.append(v)
    vectors = np.vstack(blocks) if blocks else new_vectors

    rate = f", {len(fresh) / t_embed:.0f} chunks/s" if fresh and t_embed else ""
    print(f"embedded  {vectors.shape} {vectors.dtype}  "
          f"({len(fresh)} new in {t_embed:.2f}s{rate}, {len(chunks) - len(fresh)} reused)")

    store = VectorStore(
        chunks=chunks,
        vectors=vectors,
        meta=IndexMeta.now(
            model_name=embedder.model_name,
            dimension=embedder.dimension,
            chunk_size=args.size,
            overlap=args.overlap,
            n_chunks=len(chunks),
            strategy=args.strategy,
            prepend_context=args.prepend_context,
            min_section=args.min_section,
            doc_hashes=hashes,
        ),
    )
    store.save(args.index)

    index_dir = Path(args.index)
    size_kb = sum(f.stat().st_size for f in index_dir.iterdir() if f.is_file()) / 1024
    print(f"saved     {index_dir}/  ({size_kb:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
