"""Phase 3: question -> ranked chunks. Embedder and store, tied together."""

from __future__ import annotations

from dataclasses import dataclass

from rag.chunker import Chunk
from rag.embedder import Embedder
from rag.hybrid import RRF_K, BM25Index, reciprocal_rank_fusion
from rag.rerank import DEFAULT_DEPTH, CrossEncoderReranker
from rag.store import VectorStore


MODES = ("dense", "bm25", "hybrid")


@dataclass(frozen=True)
class Result:
    """One retrieved chunk, with its score and 1-based rank.

    `score` means different things per mode, and they are not comparable:
    cosine similarity in [-1, 1] for dense, an unbounded BM25 score for bm25,
    and a fused RRF score (roughly 1/k .. 2/(k+1)) for hybrid. Only the RANK is
    meaningful across modes -- which is the same reason RRF fuses ranks rather
    than scores in the first place.
    """

    chunk: Chunk
    score: float
    rank: int


class Retriever:
    def __init__(self, embedder: Embedder, store: VectorStore,
                 mode: str = "dense", candidates: int = 50,
                 rrf_k: int = RRF_K, rerank: bool = False,
                 rerank_depth: int = DEFAULT_DEPTH,
                 reranker: CrossEncoderReranker | None = None) -> None:
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

        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}, expected one of {MODES}")

        self.embedder = embedder
        self.store = store
        self.mode = mode
        self.candidates = candidates
        self.rrf_k = rrf_k

        # Built eagerly for the keyword modes and skipped entirely for dense,
        # because it is pure overhead there. It is only term counting -- no
        # model, no API -- so it costs well under a second on this corpus.
        # embed_text, not text: both arms must see identical input, or a
        # prepend_context experiment would silently change only one of them.
        self.bm25 = (
            BM25Index([c.embed_text for c in store.chunks])
            if mode in ("bm25", "hybrid") else None
        )

        # Stage 2. Costs a ~80 MB model load, so it is built only when asked
        # for; an already-loaded one can be passed in to share it across
        # Retrievers (Phase 7 will want exactly that).
        self.rerank_depth = rerank_depth
        self.reranker = reranker or (CrossEncoderReranker() if rerank else None)

    def retrieve(self, question: str, k: int = 5) -> list[Result]:
        # With a re-ranker, stage 1 must retrieve deeper than k: the whole
        # point is that stage 2 can promote something from rank 30 into the
        # top 5. Retrieving only k would leave it nothing to promote.
        stage1_k = max(k, self.rerank_depth) if self.reranker else k
        if self.mode == "dense":
            # encode() takes a list and returns a matrix; we want the single row.
            scored = self.store.search(self.embedder.encode([question])[0], k=stage1_k)
        elif self.mode == "bm25":
            scored = self.bm25.search(question, k=stage1_k)
        else:
            # Two-arm fusion. Each arm retrieves `candidates` deep so a chunk
            # that only one ranker can find still reaches the fusion step; the
            # top k of the fused list is what comes back.
            depth = max(stage1_k, self.candidates)
            dense = [i for i, _ in self.store.search(
                self.embedder.encode([question])[0], k=depth)]
            keyword = [i for i, _ in self.bm25.search(question, k=depth)]
            scored = reciprocal_rank_fusion([dense, keyword], k=self.rrf_k, top=stage1_k)

        if self.reranker is not None:
            head = [i for i, _ in scored]
            # chunk.text, not embed_text: a cross-encoder actually reads the
            # language, so a synthetic prefix would be noise it has to attend to.
            scored = self.reranker.rerank(
                question, [self.store.chunks[i].text for i in head], head, k=k)

        return [
            Result(chunk=self.store.chunks[i], score=score, rank=rank)
            for rank, (i, score) in enumerate(scored, start=1)
        ]
