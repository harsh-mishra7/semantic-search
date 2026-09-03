"""Document -> list[Chunk].

A fixed-size character chunker with overlap. This is deliberately the dumbest
thing that works: it counts characters and cuts, with no idea whether it is
slicing through the middle of a sentence, a code block, or a table.

That naivety is the point. The eval set came first, and only THEN the smarter
strategies -- so that "split on markdown headings" is a change we can attach a
number to, rather than a change that merely feels better.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

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
    # Context prepended when EMBEDDING, and never when citing. A chunk that
    # reads "Step 4: Set Environment Variables" is ambiguous on its own; the
    # same chunk embedded as "Deploy Socket Server to Render > Step 4: Set
    # Environment Variables" has something to anchor to. Keeping this separate
    # from `text` means offsets still index the real file and Phase 5 quotes
    # the document rather than our synthetic header.
    prefix: str = ""

    @property
    def embed_text(self) -> str:
        return f"{self.prefix}\n\n{self.text}" if self.prefix else self.text

    @property
    def id(self) -> str:
        """Stable, human-readable handle. Phase 5 cites these."""
        return f"{self.doc_path}#{self.chunk_index}"

    def __len__(self) -> int:
        return len(self.text)


_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def sections(text: str) -> list[tuple[int, int, str]]:
    """Split markdown into (start, end, heading_path) by heading lines.

    Fenced code blocks are tracked and skipped, which is not optional on this
    corpus: the docs are full of shell comments like `# Build and start
    containers` that are indistinguishable from an h1 if you only regex lines.
    Treating those as headings would shatter every code block into fake sections.
    """
    out: list[tuple[int, int, str]] = []
    stack: list[tuple[int, str]] = []
    in_fence = False
    off = cur_start = 0
    cur_path = ""

    for line in text.splitlines(keepends=True):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        elif not in_fence and (m := _HEADING.match(line)):
            if off > cur_start:
                out.append((cur_start, off, cur_path))
            level, title = len(m.group(1)), m.group(2)
            # Drop any deeper-or-equal headings, then push: the stack becomes
            # the breadcrumb trail to this section.
            stack = [(lv, t) for lv, t in stack if lv < level]
            stack.append((level, title))
            cur_path = " > ".join(t for _, t in stack)
            cur_start = off
        off += len(line)

    if off > cur_start:
        out.append((cur_start, off, cur_path))
    return out


def _windows(start: int, end: int, size: int, overlap: int) -> list[tuple[int, int]]:
    """Fixed-size overlapping windows covering [start, end). See chunk_document."""
    stride = size - overlap
    spans, pos = [], start
    while True:
        stop = min(pos + size, end)
        spans.append((pos, stop))
        if stop >= end:
            return spans
        pos += stride


def chunk_document(
    doc: Document,
    size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    strategy: str = "fixed",
    prepend_context: bool = False,
    min_section: int = 0,
) -> list[Chunk]:
    """Cut one document into chunks.

    strategy="fixed"   -- blind character windows over the whole document.
    strategy="heading" -- split on markdown headings first, so each chunk is a
                          semantic unit; only sections longer than `size` are
                          then windowed. Short sections stay whole, which means
                          chunk length varies a lot more than in fixed mode.

    prepend_context=True prefixes each chunk with "title > heading > path" for
    embedding purposes only.
    """
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
    if strategy not in ("fixed", "heading"):
        raise ValueError(f"unknown strategy {strategy!r}, expected 'fixed' or 'heading'")

    secs = sections(text)
    title = _title(doc)

    if strategy == "fixed":
        # Blind windows over the whole document. The end-check sits after the
        # append rather than in a `while` condition, which is what prevents a
        # redundant trailing sliver: otherwise a 430-char document emits a
        # second chunk covering [425:430] -- five characters wholly contained
        # in the first, and a garbage vector in the index.
        spans = _windows(0, len(text), size, overlap)
    else:
        # Sections shorter than `size` survive intact; longer ones get windowed.
        # min_section merges consecutive short sections first: a heading with one
        # line under it becomes a 12-character chunk otherwise, which carries no
        # retrievable signal and still competes for a slot in the top k.
        spans = [w for s, e in _merge_short(secs, min_section)
                 for w in _windows(s, e, size, overlap)]

    chunks: list[Chunk] = []
    for i, (start, end) in enumerate(spans):
        prefix = ""
        if prepend_context:
            # Heading path of the section this chunk STARTS in.
            path = next((p for s, e, p in secs if s <= start < e), "")
            # The heading path already begins with the h1 whenever the document
            # has one, so joining title + path would repeat it.
            if not path:
                prefix = title
            elif path == title or path.startswith(f"{title} > "):
                prefix = path
            else:
                prefix = f"{title} > {path}"
        chunks.append(
            Chunk(doc_path=doc.path, chunk_index=i, start=start, end=end,
                  text=text[start:end], prefix=prefix)
        )
    return chunks


def _merge_short(secs: list[tuple[int, int, str]], floor: int) -> list[tuple[int, int]]:
    """Glue consecutive sections together until each span is >= `floor` chars."""
    if floor <= 0:
        return [(s, e) for s, e, _ in secs]
    out: list[list[int]] = []
    for s, e, _ in secs:
        if out and out[-1][1] - out[-1][0] < floor:
            out[-1][1] = e   # sections are contiguous, so extending is safe
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def _title(doc: Document) -> str:
    """First h1, else the filename -- something for a chunk to anchor to."""
    for line in doc.text.splitlines():
        if (m := _HEADING.match(line)) and len(m.group(1)) == 1:
            return m.group(2)
    return Path(doc.path).stem.replace("_", " ").replace("-", " ")


def chunk_documents(
    docs: list[Document],
    size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    strategy: str = "fixed",
    prepend_context: bool = False,
    min_section: int = 0,
) -> list[Chunk]:
    """Chunk a whole corpus, preserving document order.

    The returned order is the corpus's canonical order: index i here becomes
    row i of the vector matrix in Phase 3.
    """
    return [c for doc in docs
            for c in chunk_document(doc, size=size, overlap=overlap,
                                     strategy=strategy, prepend_context=prepend_context,
                                     min_section=min_section)]
