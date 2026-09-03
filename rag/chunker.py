"""Phase 1: Document -> list[Chunk].

A fixed-size character chunker with overlap. This is deliberately the dumbest
thing that works: it counts characters and cuts, with no idea whether it is
slicing through the middle of a sentence, a code block, or a table.

That naivety is the point. Phase 4 builds an eval set first, and only THEN
tries smarter chunking -- so that "split on markdown headings" is a change we
can attach a number to, rather than a change that merely feels better.
"""

from __future__ import annotations

from dataclasses import dataclass

from rag.loader import Document

# 500 chars is roughly a long paragraph, ~125 tokens. 75 is 15% overlap: enough
# that an idea straddling a cut survives intact in one of the two chunks.
DEFAULT_CHUNK_SIZE = 500
DEFAULT_OVERLAP = 75


@dataclass(frozen=True)
class Chunk:
    """A slice of one document: the unit we embed, retrieve, and cite.

    `start`/`end` are character offsets into the document's raw text, which is
    why the loader does not strip it. They are what makes a citation checkable:
    given a chunk, you can point at the exact span of the real file it came from.
    Without offsets you can cite a filename, which is a much weaker claim.

    `chunk_index` counts within the document, not across the corpus. A chunk's
    position in the corpus-wide list is its row in Phase 3's vector matrix --
    that identity is positional and implicit, and keeping the two in sync is
    the one invariant this project cannot violate.
    """

    doc_path: str
    chunk_index: int
    start: int
    end: int
    text: str

    @property
    def id(self) -> str:
        """Stable, human-readable handle. Phase 5 cites these."""
        return f"{self.doc_path}#{self.chunk_index}"

    def __len__(self) -> int:
        return len(self.text)


def chunk_document(
    doc: Document,
    size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """Cut one document into overlapping fixed-size character windows."""
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")
    # stride = size - overlap. If overlap >= size the stride is <= 0 and the
    # loop below never advances -- an infinite loop that fills memory rather
    # than raising. Cheap to guard, miserable to debug.
    if not 0 <= overlap < size:
        raise ValueError(f"need 0 <= overlap < size, got overlap={overlap} size={size}")

    text = doc.text
    if not text:
        return []

    stride = size - overlap
    chunks: list[Chunk] = []
    start = 0

    while True:
        end = min(start + size, len(text))
        chunks.append(
            Chunk(
                doc_path=doc.path,
                chunk_index=len(chunks),
                start=start,
                end=end,
                text=text[start:end],
            )
        )
        # Stop as soon as a chunk reaches the end of the document. Checking here
        # rather than in the `while` condition is what prevents a redundant
        # trailing sliver: without it, a 430-char document would emit a second
        # chunk covering [425:430] -- five characters, entirely contained in the
        # first chunk, and a garbage vector in the index.
        if end >= len(text):
            return chunks
        start += stride


def chunk_documents(
    docs: list[Document],
    size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """Chunk a whole corpus, preserving document order.

    The returned order is the corpus's canonical order: index i here becomes
    row i of the vector matrix in Phase 3.
    """
    return [c for doc in docs for c in chunk_document(doc, size=size, overlap=overlap)]
