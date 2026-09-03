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

from rag.chunker import DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP, chunk_documents
from rag.embedder import DEFAULT_MODEL, LocalEmbedder
from rag.loader import load_documents
from rag.store import IndexMeta, VectorStore


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--index", default="index")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--size", type=int, default=DEFAULT_CHUNK_SIZE)
    ap.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    args = ap.parse_args()

    t0 = time.perf_counter()
    docs = load_documents(args.data)
    chunks = chunk_documents(docs, size=args.size, overlap=args.overlap)
    t_chunk = time.perf_counter() - t0
    print(f"loaded    {len(docs)} documents -> {len(chunks)} chunks  ({t_chunk:.2f}s)")

    t0 = time.perf_counter()
    embedder = LocalEmbedder(args.model)
    t_load = time.perf_counter() - t0
    print(f"model     {embedder.model_name}  {embedder.dimension}-D  (loaded in {t_load:.1f}s)")

    # One call for the whole corpus -- see LocalEmbedder.encode on why batching
    # matters. The chunk order here IS the row order of the matrix.
    t0 = time.perf_counter()
    vectors = embedder.encode([c.text for c in chunks], show_progress=True)
    t_embed = time.perf_counter() - t0
    print(f"embedded  {vectors.shape} {vectors.dtype}  ({t_embed:.2f}s, "
          f"{len(chunks) / t_embed:.0f} chunks/s)")

    store = VectorStore(
        chunks=chunks,
        vectors=vectors,
        meta=IndexMeta.now(
            model_name=embedder.model_name,
            dimension=embedder.dimension,
            chunk_size=args.size,
            overlap=args.overlap,
            n_chunks=len(chunks),
        ),
    )
    store.save(args.index)

    index_dir = Path(args.index)
    size_kb = sum(f.stat().st_size for f in index_dir.iterdir() if f.is_file()) / 1024
    print(f"saved     {index_dir}/  ({size_kb:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
