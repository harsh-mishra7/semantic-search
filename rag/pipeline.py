"""Phase 5: the whole thing, wired together. retrieve -> prompt -> generate."""

from __future__ import annotations

from pathlib import Path

from rag.embedder import LocalEmbedder
from rag.generator import Answer, generate
from rag.retriever import Result, Retriever
from rag.store import VectorStore


class Pipeline:
    """Loads the index and model once, then answers many questions.

    Both are expensive to set up (~10s for the model) and free to reuse, which
    is the entire reason Phase 7 keeps a Pipeline alive across HTTP requests
    instead of building one per request.
    """

    def __init__(self, index_dir: str | Path = "index") -> None:
        self.store = VectorStore.load(index_dir)
        # The model recorded in the index, not a default -- Retriever enforces
        # the match, and reading it from the index means the caller cannot
        # accidentally pick a different one.
        self.retriever = Retriever(LocalEmbedder(self.store.meta.model_name), self.store)

    def retrieve(self, question: str, k: int = 5) -> list[Result]:
        return self.retriever.retrieve(question, k=k)

    def ask(self, question: str, k: int = 5, **generate_kwargs) -> Answer:
        return generate(question, self.retrieve(question, k=k), **generate_kwargs)
