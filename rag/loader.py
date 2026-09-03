"""Files on disk -> Document objects.

The loader's only job is to find text files and read them. It deliberately knows
nothing about chunking, markdown structure, or embeddings. Keeping it dumb is
what lets the chunking strategy be rewritten without touching this file.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

# .md and .txt only, for now. PDFs would need a parser, and the variable quality
# of PDF text extraction would confound the Phase 4 experiments -- you would not
# be able to tell a chunking regression from a bad column-order extraction.
TEXT_EXTENSIONS = {".md", ".txt"}


@dataclass(frozen=True)
class Document:
    """One source file, read into memory.

    `path` is relative to the corpus root rather than absolute. It is what ends
    up in every chunk's metadata and, in Phase 5, in the citations Claude prints
    back to you -- so it wants to be short and readable. It also keeps the index
    portable: nothing breaks if the project directory moves.
    """

    path: str
    text: str

    def __len__(self) -> int:
        return len(self.text)

    @property
    def content_hash(self) -> str:
        """SHA-256 of the text. Phase 7 uses it to skip re-embedding.

        Of the CONTENT, deliberately -- not the mtime. mtime changes when a
        file is touched, copied or checked out of git, none of which change
        what needs embedding; and it does NOT change when a filesystem
        restores an old copy in place. Hashing the bytes we actually chunk
        means the question "has the work already been done" is answered by the
        work's own input.
        """
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


def load_documents(root: str | Path) -> list[Document]:
    """Read every .md/.txt file under `root`, recursively, in sorted path order.

    The sort matters more than it looks. Chunks are numbered by position, and
    Phase 3 stores their vectors in a numpy matrix where row i corresponds to
    chunk i. `Path.rglob` returns entries in filesystem order, which is not
    guaranteed stable across machines or even across runs -- so without the sort,
    re-indexing could silently renumber every chunk in the corpus.
    """
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(f"corpus root does not exist: {root}")

    paths = sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in TEXT_EXTENSIONS
    )

    documents: list[Document] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")

        # An all-whitespace file would yield an empty chunk, and embedding empty
        # text gives you a vector that is near-equidistant from everything --
        # a permanent low-grade contaminant in the index.
        if not text.strip():
            continue

        # Note: `text` is stored unmodified, NOT stripped. Chunk offsets are
        # offsets into the real file, so `sed -n` on a citation lands on the
        # right line. Stripping here would shift every offset in the document.
        documents.append(Document(path=str(path.relative_to(root)), text=text))

    return documents
