"""One-off corpus prep: split the raw ebook text export into per-chapter files.

WHY SPLIT AT ALL: every metric and technique in Phase 4 scores at *document*
granularity -- `expected_sources` is a list of doc paths, and recall@k asks
whether a correct document appeared in the top k. A corpus of one 620 KB file
makes that measurement vacuous: every chunk retrieved is trivially from the
only correct document, so recall@k == 1.0 and MRR == 1.0 for every question
and every configuration. Splitting on the chapter boundaries the export already
contains restores a real ranking problem: 18 documents, one of which is right.

The raw export marks section breaks with an "OceanofPDF.com" line. Those 20
markers align exactly with the book's structure: Coe's introduction, the title
page, 16 chapters (each verified to open with the small-caps run-in the print
edition uses), the Penguin blurb, and the copyright page.
"""

import re
import sys
from pathlib import Path

MARKER = "OceanofPDF.com"

# Segment index (between markers) -> output filename. Segments 2..17 are the
# 16 chapters. The three boilerplate segments (title page, Penguin blurb,
# copyright/translator page) are merged into one document rather than left as
# three slivers: Phase 4 established that sub-100-char chunks carry no
# retrievable signal while still competing for a slot in the top k.
INTRO = 0
CHAPTERS = range(2, 18)
BOILERPLATE = (1, 18, 19)


def segments(text: str) -> list[str]:
    """Split on marker lines, returning the text between them."""
    out, cur = [], []
    for line in text.split("\n"):
        if line.strip() == MARKER:
            out.append("\n".join(cur))
            cur = []
        else:
            cur.append(line)
    if any(l.strip() for l in cur):
        out.append("\n".join(cur))
    return out


def normalise(text: str) -> str:
    """Collapse blank-line runs to a single blank line; strip edges.

    Purely whitespace. Chunk offsets are offsets into the file we write, so
    this is self-consistent -- and it stops the chunker spending its 500-char
    budget on stacked newlines.
    """
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"


def main(src: Path, dest: Path) -> None:
    segs = segments(src.read_text(encoding="utf-8"))
    dest.mkdir(parents=True, exist_ok=True)

    written = []
    written.append(("introduction.txt", normalise(segs[INTRO])))
    for n, i in enumerate(CHAPTERS, start=1):
        written.append((f"ch{n:02d}.txt", normalise(segs[i])))
    written.append((
        "publication-note.txt",
        normalise("\n\n".join(normalise(segs[i]) for i in BOILERPLATE)),
    ))

    for name, body in written:
        (dest / name).write_text(body, encoding="utf-8")
        print(f"{name:<22} {len(body):>7,} chars")
    total = sum(len(b) for _, b in written)
    print(f"\n{len(written)} documents, {total:,} chars total")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
