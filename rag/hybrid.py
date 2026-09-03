"""Phase 6: BM25 keyword ranking, and Reciprocal Rank Fusion to combine it
with the dense vector search from Phase 3.

WHY BOTHER, when semantic search already works?

Because embeddings are lossy in a specific, predictable way. A 384-float vector
is a summary of *meaning*, and a summary necessarily discards the rare, the
exact, and the arbitrary: error codes, function names, SKUs, and -- in this
corpus -- proper nouns like Masinissa and quantities like "twenty-two months".
MiniLM has no room to store that "Masinissa" is a specific Numidian king; it
stores something like "unusual classical-sounding name" and moves on.

BM25 has the opposite bias. It knows nothing about meaning and everything about
which exact tokens are rare. A term appearing in 1 of 1224 chunks gets a large
IDF, so a single match on it dominates the score. That is precisely the signal
the embedding threw away.

So the two rankers fail on disjoint sets of queries, which is the condition
under which combining them helps. Two rankers with correlated errors would just
average into the same mistakes.

THE FORMULA, and why each part is there:

    score(q, d) = SUM over terms t in q of
                      IDF(t) * tf(t,d) * (k1 + 1)
                      -------------------------------------------
                      tf(t,d) + k1 * (1 - b + b * |d| / avgdl)

  IDF(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))

- IDF: rare terms are informative, common ones are not. Note this is why BM25
  needs NO stopword list -- "the" appears in nearly every chunk, so its IDF is
  ~0 and it contributes nothing. Removing stopwords by hand is redundant work
  that also throws away the occasional query where they matter.

- tf, saturating: the k1 term makes term frequency diminishing. The 10th
  occurrence of "Custardoy" in a chunk says much less than the 2nd. Raw tf
  would let one long chunk that repeats a word dominate everything.

- b, length normalisation: divides by chunk length relative to the corpus
  average, so a long chunk does not win merely by containing more words. Our
  chunks are near-uniform (500 chars, no overlap), so this matters far less
  here than it does over whole documents of wildly varying length.

k1=1.5 and b=0.75 are the standard defaults, and are what Lucene ships.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

import numpy as np

# Standard BM25 parameters. k1 controls tf saturation, b length normalisation.
K1 = 1.5
B = 0.75

# RRF's smoothing constant. The paper that introduced RRF (Cormack et al. 2009)
# uses 60, and nearly every implementation copies that number. On this corpus
# 60 is the WORST value tested and 10 is the best: +0.030 MRR between them --
# see PLAN.md §8. The paper tuned k for fusing many similar TREC runs, where
# flattening the top ranks is the right call. Fusing exactly two rankers with
# disjoint competence is a different problem: when one ranker is right and the
# other is blind, agreement-weighting discards the right answer.
RRF_K = 10
PAPER_RRF_K = 60   # kept so the deviation from the literature stays visible

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase, then split on anything that is not a letter or digit.

    Deliberately crude, and NOT stemmed. Two consequences worth knowing:

    - Possessives split: "Ranz's" -> ["ranz", "s"]. That is what we want; a
      query for "Ranz" should match it. The stray "s" appears in almost every
      chunk, so its IDF is ~0 and it costs nothing.
    - No stemming means "advertised" does not match "advertising". A stemmer
      would fix that and would also conflate words that should stay distinct.
      It is a real trade-off, so it belongs in the eval table rather than in a
      decision made here silently.

    The curly apostrophe is normalised first: this corpus is an ebook export
    and uses U+2019 throughout, so a regex written for ASCII "'" would split
    every possessive in a second, invisible way.
    """
    return _TOKEN.findall(text.lower().replace("’", "'"))


class BM25Index:
    """A term -> (chunk, frequency) postings list, plus BM25 scoring over it.

    Index-aligned with the chunk list it was built from, exactly like
    VectorStore: position i here is chunk i there. Same invariant, same
    consequence if it breaks (right scores, wrong text, no error).
    """

    def __init__(self, texts: list[str], k1: float = K1, b: float = B) -> None:
        self.k1, self.b = k1, b
        self.n = len(texts)

        term_freqs = [Counter(tokenize(t)) for t in texts]
        self.lengths = np.array([sum(tf.values()) for tf in term_freqs], dtype=np.float64)
        # An empty corpus, or one of only empty chunks, would divide by zero in
        # the length-normalisation term below.
        self.avgdl = float(self.lengths.mean()) if self.n and self.lengths.sum() else 1.0

        # Inverted index: the whole reason keyword search is fast. Scoring a
        # query touches only the chunks containing its terms, not all 1224.
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, tf in enumerate(term_freqs):
            for term, freq in tf.items():
                self.postings[term].append((i, freq))

        # Robertson-Sparck-Jones IDF, in the form Lucene uses. The +1 inside
        # the log keeps it non-negative: without it, a term in more than half
        # the chunks scores negative and *penalises* the chunks containing it.
        self.idf = {
            term: math.log(1 + (self.n - len(posting) + 0.5) / (len(posting) + 0.5))
            for term, posting in self.postings.items()
        }

    def __len__(self) -> int:
        return self.n

    def scores(self, query: str) -> np.ndarray:
        """BM25 score for every chunk. Chunks matching no query term score 0."""
        out = np.zeros(self.n, dtype=np.float64)
        for term in tokenize(query):
            idf = self.idf.get(term)
            if idf is None:      # term does not occur in the corpus at all
                continue
            for i, freq in self.postings[term]:
                denom = freq + self.k1 * (
                    1.0 - self.b + self.b * self.lengths[i] / self.avgdl
                )
                out[i] += idf * freq * (self.k1 + 1.0) / denom
        return out

    def search(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        """Top k (chunk index, score), best first. Zero-scoring chunks are dropped.

        Dropping them matters for fusion: a chunk that matched no query term at
        all has no keyword evidence, and handing RRF an arbitrary tail of
        0-score chunks would let their positions contribute real rank points.
        """
        scores = self.scores(query)
        k = max(1, min(k, self.n))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(int(i), float(scores[i])) for i in top if scores[i] > 0.0]


def reciprocal_rank_fusion(
    rankings: list[list[int]], k: int = RRF_K, top: int | None = None
) -> list[tuple[int, float]]:
    """Merge ranked lists of chunk indices. Returns (index, fused score), best first.

        RRF(d) = SUM over rankers r of  1 / (k + rank_r(d))

    WHY FUSE RANKS RATHER THAN SCORES. Cosine similarity lives in [-1, 1].
    BM25 is unbounded, and its scale depends on corpus size and term rarity --
    a good BM25 score here is ~8, on another corpus ~30. So `0.6 * cosine +
    0.4 * bm25` is not a weighted average of two comparable things; it is a
    made-up number dominated by whichever ranker happens to have the larger
    units. You can fix that with min-max or z-score normalisation, but both are
    sensitive to outliers and to how many candidates you happened to retrieve.

    RRF sidesteps all of it by throwing the magnitudes away and keeping only
    the ordering -- the one piece of information that genuinely transfers
    between two rankers with incomparable scales. It is ~8 lines and routinely
    competitive with tuned score-weighting schemes.

    WHAT k DOES. It flattens the difference between the top ranks. At k=60,
    rank 1 contributes 1/61 and rank 2 1/62 -- nearly identical. So a chunk
    ranked 2nd by BOTH rankers (2/62 = 0.0323) beats one ranked 1st by a single
    ranker and absent from the other (1/61 = 0.0164). That is the intended
    behaviour: agreement between rankers outweighs enthusiasm from one. A small
    k would make it winner-take-all and undo the point of fusing.
    """
    if k <= 0:
        raise ValueError(f"RRF k must be positive, got {k}")

    fused: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, index in enumerate(ranking, start=1):
            fused[index] += 1.0 / (k + rank)

    # Tie-break on index so the order is deterministic. Ties are common: any
    # two chunks found at the same rank by the same single ranker tie exactly.
    ordered = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
    return ordered[:top] if top is not None else ordered
