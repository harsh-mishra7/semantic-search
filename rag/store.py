"""Phase 3: persist chunks + vectors, and search them.

The whole store is a numpy matrix and a dot product. No vector database, on
purpose: 196 chunks x 384 dims is 300 KB, and even 100,000 chunks would be a
150 MB matrix that BLAS multiplies in single-digit milliseconds. Approximate
indexes (HNSW, IVF -- what FAISS/Qdrant/pgvector run) start paying off in the
millions. Knowing where that line sits is what stops you adopting one reflexively.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from rag.chunker import Chunk

VECTORS_FILE = "vectors.npy"
CHUNKS_FILE = "chunks.json"


@dataclass(frozen=True)
class IndexMeta:
    """What built this index. Saved alongside it so it can be checked, not assumed.

    `model_name` is the important one. Embedding a query with a different model
    than the documents gives you two unrelated vector spaces and distances that
    mean nothing -- and it fails silently, returning plausible-looking garbage.
    Recording the name is what lets the retriever refuse instead.
    """

    model_name: str
    dimension: int
    chunk_size: int
    overlap: int
    n_chunks: int
    created_at: str
    # Defaulted so an index written before Phase 7 still loads. All four are
    # inputs to chunking or embedding, and every one of them must match for a
    # cached vector to still be correct -- see `reuse_signature`.
    strategy: str = "fixed"
    prepend_context: bool = False
    min_section: int = 0
    # path -> content hash, for incremental indexing.
    doc_hashes: dict[str, str] = field(default_factory=dict)

    @classmethod
    def now(cls, **kw) -> IndexMeta:
        return cls(created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"), **kw)

    @property
    def reuse_signature(self) -> tuple:
        """Everything that must be identical for a stored vector to be reusable.

        This exists because incremental indexing has one catastrophic failure
        mode: mixing vectors produced under different settings. Change the
        chunk size and reuse a document's old vectors, and that document is
        now indexed with different boundaries than the rest of the corpus --
        which produces a working index, plausible scores, and quietly wrong
        rankings. There is no error and no symptom.

        So the rule is: reuse ONLY when every input to chunking and embedding
        matches, and otherwise rebuild from scratch. Note that `chunk_size` and
        `overlap` alone were not enough -- `strategy`, `prepend_context` and
        `min_section` all change chunk boundaries too, and until Phase 7 the
        index did not record them at all.
        """
        return (self.model_name, self.dimension, self.chunk_size, self.overlap,
                self.strategy, self.prepend_context, self.min_section)


class VectorStore:
    """Index-aligned chunks and unit-length vectors: row i <-> chunks[i]."""

    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, meta: IndexMeta) -> None:
        # This is the invariant the whole system rests on, and the one failure
        # mode with no symptoms: if the rows and the chunk list drift out of
        # alignment, every search returns a real score attached to the wrong
        # text. You get confident, well-cited, wrong answers and no error.
        # So it is checked on construction and again on load.
        if vectors.ndim != 2:
            raise ValueError(f"vectors must be 2-D, got shape {vectors.shape}")
        if vectors.shape[0] != len(chunks):
            raise ValueError(
                f"index misalignment: {vectors.shape[0]} vectors but {len(chunks)} chunks"
            )
        if vectors.shape[1] != meta.dimension:
            raise ValueError(
                f"dimension mismatch: vectors are {vectors.shape[1]}-D, "
                f"meta says {meta.dimension}"
            )

        self.chunks = chunks
        self.vectors = vectors
        self.meta = meta

    def __len__(self) -> int:
        return len(self.chunks)

    def search(self, query_vector: np.ndarray, k: int = 5) -> list[tuple[int, float]]:
        """Return the k highest-scoring (row index, cosine score), best first."""
        if query_vector.shape != (self.meta.dimension,):
            raise ValueError(
                f"query vector must be shape ({self.meta.dimension},), "
                f"got {query_vector.shape}"
            )
        # A non-unit query vector would scale every score by |q|. The ranking
        # would survive, but the numbers stop being cosine similarities -- and
        # in Phase 4 we compare scores across runs, so they have to mean something.
        norm = float(np.linalg.norm(query_vector))
        if not np.isclose(norm, 1.0, atol=1e-3):
            raise ValueError(f"query vector is not L2-normalised (norm={norm:.4f})")

        # The entire search. (n, 384) @ (384,) -> (n,), one BLAS call.
        # Cosine similarity, because both sides are unit length.
        scores = self.vectors @ query_vector

        k = max(1, min(k, len(scores)))
        # argpartition is O(n): it only guarantees the k smallest are on the
        # left, without fully ordering anything. argsort would be O(n log n).
        # Irrelevant at 196 chunks, meaningful at 100k -- and the same answer.
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]  # now sort just those k
        return [(int(i), float(scores[i])) for i in top]

    def vectors_for(self, doc_path: str) -> tuple[list[Chunk], np.ndarray]:
        """This document's chunks and their vectors, in order.

        Used by incremental indexing to lift an unchanged document's work out
        of the previous index. Rows are gathered by position so the returned
        pair stays aligned even though the document's chunks need not be
        contiguous in the matrix.
        """
        rows = [i for i, c in enumerate(self.chunks) if c.doc_path == doc_path]
        return [self.chunks[i] for i in rows], self.vectors[rows]

    def save(self, index_dir: str | Path) -> None:
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        np.save(index_dir / VECTORS_FILE, self.vectors)
        payload = {"meta": asdict(self.meta), "chunks": [asdict(c) for c in self.chunks]}
        (index_dir / CHUNKS_FILE).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, index_dir: str | Path) -> VectorStore:
        index_dir = Path(index_dir)
        vectors_path, chunks_path = index_dir / VECTORS_FILE, index_dir / CHUNKS_FILE
        if not vectors_path.exists() or not chunks_path.exists():
            raise FileNotFoundError(
                f"no index in {index_dir}/ -- run: python scripts/index.py"
            )

        payload = json.loads(chunks_path.read_text(encoding="utf-8"))
        return cls(
            chunks=[Chunk(**c) for c in payload["chunks"]],
            vectors=np.load(vectors_path),
            meta=IndexMeta(**payload["meta"]),
        )
