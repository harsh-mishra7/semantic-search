"""Text -> vectors.

The `Embedder` protocol is the seam that makes swapping the model a one-line
change. Everything downstream depends on this interface, never on
sentence-transformers directly.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

DEFAULT_MODEL = "all-MiniLM-L6-v2"


class Embedder(Protocol):
    """Anything that turns text into unit-length vectors.

    `model_name` is part of the interface, not an implementation detail: the
    index records which model built it, so a query embedded with a different
    model can be rejected instead of silently returning nonsense.
    """

    model_name: str
    dimension: int

    def encode(self, texts: list[str]) -> np.ndarray: ...


class LocalEmbedder:
    """all-MiniLM-L6-v2 (or any sentence-transformers model), on CPU."""

    def __init__(self, model_name: str = DEFAULT_MODEL, batch_size: int = 64) -> None:
        # Imported here, not at module scope: pulling in sentence-transformers
        # drags in torch, which costs several seconds. Deferring it means
        # `import rag.embedder` stays instant for anything that only needs the
        # protocol -- and it keeps the cost visible at the point you pay it.
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        # Renamed from get_sentence_embedding_dimension() in sentence-transformers 6.
        # It is typed Optional because a model can omit the metadata; if that
        # ever happens, fail here rather than letting a None reach the store,
        # where it would surface as a confusing dimension-mismatch much later.
        dimension = self._model.get_embedding_dimension()
        if dimension is None:
            raise RuntimeError(f"{model_name} reports no embedding dimension")
        self.dimension: int = dimension
        self.batch_size = batch_size

    def encode(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        """Encode a list of texts into an (len(texts), dimension) float32 array.

        Two things happen here that the rest of the system depends on:

        1. Batching. One call encoding 196 chunks saturates the CPU; 196 calls
           re-pay Python and framework overhead every time. Same arithmetic,
           very different wall clock.

        2. L2 normalisation (`normalize_embeddings=True`). Every vector comes
           back with length 1, which makes cosine similarity identical to a
           plain dot product -- because cosine is (a.b)/(|a||b|) and both norms
           are now 1. That turns the whole of Phase 3's search into a single
           matrix-vector multiply, with no per-query division.
        """
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)

        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        )
        # float32 halves the index size versus float64 and costs nothing:
        # the model computes in float32 anyway.
        return np.asarray(vectors, dtype=np.float32)
