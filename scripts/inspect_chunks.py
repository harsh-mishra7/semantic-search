"""Phase 1 checkpoint: load the corpus, chunk it, and look at the result.

    python scripts/inspect_chunks.py
    python scripts/inspect_chunks.py --size 300 --overlap 0 --seed 7

Nothing here is part of the pipeline -- it exists so you can eyeball whether
chunks are coherent units of text before we spend Phase 2 embedding them.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

# Python puts the *script's* directory on sys.path, not the working directory,
# so `import rag` fails without this. The alternative is packaging the project;
# not worth it yet.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.chunker import DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP, Chunk, chunk_documents
from rag.loader import load_documents


def suspicious(chunk: Chunk, doc_text: str) -> str | None:
    """Flag chunks the naive chunker probably mangled.

    Not exhaustive -- just the two defects that are mechanically detectable and
    that Phase 4's structure-aware chunking is meant to fix.
    """
    # An odd number of ``` fences means a code block was cut in half. The chunk
    # ends mid-code with no indication it was ever code.
    if chunk.text.count("```") % 2 == 1:
        return "splits a code fence"
    # Cut mid-word: the character before this chunk's start is alphanumeric and
    # so is the first character of the chunk.
    if chunk.start > 0 and doc_text[chunk.start - 1].isalnum() and chunk.text[:1].isalnum():
        return "starts mid-word"
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data", help="corpus root (default: data)")
    ap.add_argument("--size", type=int, default=DEFAULT_CHUNK_SIZE)
    ap.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    ap.add_argument("-n", "--samples", type=int, default=3, help="random chunks to dump")
    ap.add_argument("--seed", type=int, default=None, help="fix the sample for a reproducible look")
    args = ap.parse_args()

    docs = load_documents(args.data)
    chunks = chunk_documents(docs, size=args.size, overlap=args.overlap)
    by_doc = Counter(c.doc_path for c in chunks)
    text_by_doc = {d.path: d.text for d in docs}

    print(f"config      size={args.size} overlap={args.overlap} stride={args.size - args.overlap}")
    print(f"documents   {len(docs)}")
    print(f"chunks      {len(chunks)}")
    print(f"chars       {sum(len(d.text) for d in docs):,} in docs -> "
          f"{sum(len(c) for c in chunks):,} in chunks "
          f"({sum(len(c) for c in chunks) / max(sum(len(d.text) for d in docs), 1):.2f}x, "
          f"the overlap duplication)")

    lengths = sorted(len(c) for c in chunks)
    print(f"chunk len   min={lengths[0]} median={lengths[len(lengths) // 2]} max={lengths[-1]}")

    print("\nchunks per document")
    for path, n in by_doc.most_common():
        share = n / len(chunks)
        print(f"  {n:>4}  {share:>5.1%}  {path}")

    flagged = [(c, why) for c in chunks if (why := suspicious(c, text_by_doc[c.doc_path]))]
    print(f"\nsuspicious cuts   {len(flagged)} / {len(chunks)} ({len(flagged) / len(chunks):.0%})")
    for reason, n in Counter(why for _, why in flagged).most_common():
        print(f"  {n:>4}  {reason}")

    rng = random.Random(args.seed)
    print(f"\n{'=' * 78}\n{args.samples} random chunks\n{'=' * 78}")
    for c in rng.sample(chunks, min(args.samples, len(chunks))):
        why = suspicious(c, text_by_doc[c.doc_path])
        print(f"\n--- {c.id}  chars [{c.start}:{c.end}]  len={len(c)}"
              f"{'  <<< ' + why if why else ''}")
        print(c.text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
