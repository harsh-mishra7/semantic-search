"""Audit eval/questions.yaml against the corpus. Run after editing either.

    python eval/verify_questions.py

This is a LABELLING audit, not a metric. It does not run retrieval and does not
need the embedding model -- it only asks whether the answer key is honest.

Three checks:

1. Every `expected_sources` path resolves to a real document. A typo here is
   invisible at runtime: `first_hit_rank` simply never matches, the question
   scores 0, and it reads as a retrieval failure rather than a labelling bug.
   That is the worst kind of eval defect, because it makes you "fix" retrieval
   that was never broken.

2. Each question's `anchor` -- the concrete detail that makes it answerable --
   occurs ONLY inside its expected sources. If an anchor also appears in an
   unlisted document, that document arguably answers the question too, and
   scoring a retrieval of it as wrong would penalise the retriever for being
   right. `anchor: null` exempts a question; the YAML note must say why.

3. Every document is targeted by at least one question. An untargeted document
   can never be retrieved correctly, so it contributes chunks that can only
   ever be noise -- the metric would never notice if indexing it broke.

Matching is case-insensitive and boundary-aware: the guards reject a match that
continues into a longer word, INCLUDING across a typographic apostrophe. Naive
substring matching is what let "doin" match "doing" in fourteen documents on
the first run of this script, and "crack" match "cracked".
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data"
QUESTIONS = ROOT / "eval" / "questions.yaml"
TEXT_EXTENSIONS = {".md", ".txt"}


def occurrences(anchor: str, docs: dict[str, str]) -> set[str]:
    """Documents containing `anchor` as a whole term."""
    pat = re.compile(r"(?<![\w’'])" + re.escape(anchor) + r"(?![\w’'])", re.I)
    return {name for name, text in docs.items() if pat.search(text)}


def main() -> int:
    questions = yaml.safe_load(QUESTIONS.read_text(encoding="utf-8"))
    docs = {
        str(p.relative_to(CORPUS)): p.read_text(encoding="utf-8")
        for p in sorted(CORPUS.rglob("*"))
        if p.is_file() and p.suffix.lower() in TEXT_EXTENSIONS
    }
    if not docs:
        raise SystemExit(f"no documents under {CORPUS}/ -- nothing to verify against")

    problems: list[str] = []
    exempt = 0

    for q in questions:
        question = q["question"]
        expected = set(q["expected_sources"])

        for src in expected:
            if src not in docs:
                problems.append(f"UNKNOWN PATH   {src!r} in {question[:52]!r}")

        if "anchor" not in q:
            problems.append(f"NO ANCHOR KEY  {question[:60]!r}")
            continue
        if q["anchor"] is None:
            exempt += 1
            if not q.get("note"):
                problems.append(f"UNJUSTIFIED EXEMPTION  {question[:52]!r} needs a note")
            continue

        found = occurrences(q["anchor"], docs)
        if not found:
            problems.append(f"ANCHOR ABSENT  {q['anchor']!r} occurs nowhere in the corpus")
        elif leaked := found - expected:
            problems.append(
                f"ANCHOR LEAKS   {q['anchor']!r} also in {sorted(leaked)}; "
                f"either list them in expected_sources or pick a tighter anchor")
        elif missing := expected - found:
            problems.append(
                f"ANCHOR MISSING {q['anchor']!r} absent from its own source(s) {sorted(missing)}")

    targeted = {s for q in questions for s in q["expected_sources"]}
    for name in docs:
        if name not in targeted:
            problems.append(f"UNTARGETED DOC {name!r} -- no question can retrieve it")

    print(f"{len(questions)} questions | {len(docs)} documents | "
          f"{len(questions) - exempt} anchored, {exempt} exempt\n")

    if problems:
        print(f"{len(problems)} problem(s):")
        for p in problems:
            print(f"  {p}")
        return 1

    per_doc = {name: sum(name in q["expected_sources"] for q in questions) for name in docs}
    print("all paths resolve")
    print("all anchors confined to their expected sources")
    print(f"all {len(docs)} documents targeted "
          f"({min(per_doc.values())}-{max(per_doc.values())} questions each)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
