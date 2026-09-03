"""Full RAG: retrieve, then answer with citations.

    python scripts/ask.py "how do I deploy the socket server to Render"
    python scripts/ask.py -k 3 --effort low "..."
    python scripts/ask.py --dry-run "..."      # print the exact prompt, spend nothing
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

from rag.generator import SYSTEM_PROMPT, build_messages
from rag.pipeline import Pipeline


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="+")
    ap.add_argument("-k", type=int, default=5, help="chunks to retrieve (default 5)")
    ap.add_argument("--index", default="index")
    ap.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"],
                    help="thinking depth; API default is high")
    ap.add_argument("--show-thinking", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the prompt and exit without calling the API")
    args = ap.parse_args()
    question = " ".join(args.question)

    # .env is gitignored; ANTHROPIC_API_KEY lives there, never in the code.
    load_dotenv()

    pipe = Pipeline(args.index)
    results = pipe.retrieve(question, k=args.k)

    print(f"\nretrieved {len(results)} chunks for {question!r}")
    for r in results:
        print(f"  [{r.rank}] {r.score:.3f}  {r.chunk.id}")

    if args.dry_run:
        msgs = build_messages(question, results)
        print(f"\n{'=' * 78}\nSYSTEM\n{'=' * 78}\n{SYSTEM_PROMPT}")
        print(f"\n{'=' * 78}\nUSER\n{'=' * 78}\n{msgs[0]['content']}")
        chars = len(SYSTEM_PROMPT) + len(msgs[0]["content"])
        print(f"\n{'=' * 78}\n~{chars:,} chars ~= {chars // 4:,} tokens "
              f"~= ${chars / 4 * 5 / 1e6:.4f} input")
        return 0

    print(f"\n{'-' * 78}")
    answer = pipe.ask(question, k=args.k, effort=args.effort,
                      show_thinking=args.show_thinking)
    print(f"\n{'-' * 78}")

    if answer.citations():
        print("citations")
        for n, r in answer.citations():
            print(f"  [{n}] {r.chunk.doc_path}  chars {r.chunk.start}-{r.chunk.end}")
    else:
        print("citations  none -- the answer cited nothing")

    print(f"\ntokens    {answer.input_tokens} in / {answer.output_tokens} out"
          f"   cost  ${answer.cost:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
