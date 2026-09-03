"""Phase 3: question -> ranked chunks. Embedder and store, tied together."""

from __future__ import annotations

from dataclasses import dataclass

from rag.chunker import Chunk
from rag.embedder import Embedder
from rag.store import VectorStore


@dataclass(frozen=True)
class Result:
    """One retrieved chunk, with its score and 1-based rank."""

    chunk: Chunk
    score: float
    rank: int


class Retriever:
    def __init__(self, embedder: Embedder, store: VectorStore) -> None:
        # The symmetry rule, enforced rather than remembered.
        #
        # Embedding the query with a different model than the documents is the
        # single most common beginner bug in RAG, and it does not raise: you get
        # two unrelated 384-dimensional spaces, cosine scores that look normal,
        # and rankings that are noise. The only way to notice is to check the
        # names -- so we check the names.
        if embedder.model_name != store.meta.model_name:
            raise ValueError(
                f"model mismatch: index was built with '{store.meta.model_name}', "
                f"querying with '{embedder.model_name}'. Re-index, or use the "
                f"original model -- the two vector spaces are not comparable."
            )
        if embedder.dimension != store.meta.dimension:
            raise ValueError(
                f"dimension mismatch: embedder is {embedder.dimension}-D, "
                f"index is {store.meta.dimension}-D"
            )

        self.embedder = embedder
        self.store = store

    def retrieve(self, question: str, k: int = 5) -> list[Result]:
        # encode() takes a list and returns a matrix; we want the single row.
        query_vector = self.embedder.encode([question])[0]
        return [
            Result(chunk=self.store.chunks[i], score=score, rank=rank)
            for rank, (i, score) in enumerate(self.store.search(query_vector, k=k), start=1)
        ]
