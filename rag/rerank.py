"""Cross-encoder re-ranking. The second stage of two-stage retrieval.

BI-ENCODER vs CROSS-ENCODER, which is the whole idea:

Everything so far has been a BI-ENCODER. `all-MiniLM-L6-v2` reads the query and
reads a chunk *separately*, producing one 384-float vector each, and relevance
is the angle between them. That separation is what makes it scale: chunk
vectors are computed once at index time, and a query is then one matrix
multiply against all 1224 of them. But it is also the weakness -- the model
never sees the query and the chunk together, so it cannot notice that the
chunk answers *this particular* question. It compresses each side to a summary
and hopes the summaries line up.

A CROSS-ENCODER concatenates them -- `[CLS] query [SEP] chunk [SEP]` -- and
runs one forward pass over the pair, so attention operates across both at
once. Every token of the query can attend to every token of the chunk. It
outputs a single relevance logit. That is far more accurate, and it is why the
separation between a good and a bad pair is enormous (roughly +9 vs -11 on
`ms-marco-MiniLM-L-6-v2`) where cosine similarities are squeezed into a narrow
band around 0.1-0.6.

The cost is that nothing can be precomputed. There is no chunk vector to store,
because the representation depends on the query. So scoring the corpus means
1224 forward passes per query, which is thousands of times more expensive than
one matrix multiply.

HENCE TWO STAGES. Stage 1 (hybrid) is cheap and tuned for *recall*: get the
right chunk into the top 30 somehow. Stage 2 re-reads those 30 properly and is
tuned for *precision*: put the right one first. This only works if stage 1's
recall@depth is high -- a re-ranker cannot recover a chunk that was never
retrieved, so its ceiling is stage 1's recall@`depth`. That number is the one
to check before blaming the re-ranker.
"""

from __future__ import annotations

import numpy as np

DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# How many stage-1 candidates to re-read. The ceiling on what re-ranking can
# achieve is stage 1's recall@DEPTH, so this trades accuracy against latency:
# every extra candidate is another forward pass on the critical path.
DEFAULT_DEPTH = 30


class CrossEncoderReranker:
    """Re-scores (question, chunk) pairs jointly. Loaded once, reused."""

    def __init__(self, model_name: str = DEFAULT_RERANK_MODEL) -> None:
        # Imported lazily so that `import rag.rerank` costs nothing until a
        # re-ranker is actually built -- the eval imports this module even for
        # configs that never use it.
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def rerank(self, question: str, texts: list[str], indices: list[int],
               k: int | None = None) -> list[tuple[int, float]]:
        """Re-score candidates and return (index, logit) best first.

        `texts` and `indices` are parallel: texts[j] is the text of chunk
        indices[j]. Returning the original chunk indices is what keeps this
        composable with the stores -- the re-ranker never needs to know how
        stage 1 found anything.

        Scores are raw logits: unbounded, roughly -11..+11 for this model, and
        NOT comparable to cosine similarity or to an RRF score. They are also
        not probabilities; apply a sigmoid if you need one, though for ranking
        it changes nothing since sigmoid is monotonic.
        """
        if not indices:
            return []
        # One batched call, for the same reason embedding is batched: 30
        # separate forward passes waste most of the CPU on setup.
        scores = self.model.predict([(question, t) for t in texts])
        order = np.argsort(-np.asarray(scores))
        ranked = [(indices[j], float(scores[j])) for j in order]
        return ranked[:k] if k is not None else ranked
