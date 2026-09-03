"""Phase 5: the whole thing, wired together. retrieve -> prompt -> generate."""

from __future__ import annotations

from pathlib import Path

from rag.embedder import LocalEmbedder
from rag.generator import Answer, Generator, make_generator
from rag.retriever import Result, Retriever
from rag.store import VectorStore


class Pipeline:
    """Loads the index and model once, then answers many questions.

    Both are expensive to set up (~10s for the model) and free to reuse, which
    is the entire reason Phase 7 keeps a Pipeline alive across HTTP requests
    instead of building one per request.
    """

    def __init__(self, index_dir: str | Path = "index",
                 generator: Generator | None = None,
                 mode: str = "dense", candidates: int = 50) -> None:
        self.store = VectorStore.load(index_dir)
        # Deferred: retrieval works with no API key at all, so a missing key
        # must not stop `scripts/search.py` or the Phase 4 eval from running.
        self._generator = generator
        # The model recorded in the index, not a default -- Retriever enforces
        # the match, and reading it from the index means the caller cannot
        # accidentally pick a different one.
        self.retriever = Retriever(LocalEmbedder(self.store.meta.model_name), self.store,
                                   mode=mode, candidates=candidates)

    def retrieve(self, question: str, k: int = 5) -> list[Result]:
        return self.retriever.retrieve(question, k=k)

    @property
    def generator(self) -> Generator:
        if self._generator is None:
            self._generator = make_generator()
        return self._generator

    def ask(self, question: str, k: int = 5, **generate_kwargs) -> Answer:
        return self.generator.generate(question, self.retrieve(question, k=k),
                                       **generate_kwargs)
