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

from rag.generator import (RATES, SYSTEM_PROMPT, build_user_message,
                           list_gemini_models, make_generator)
from rag.pipeline import Pipeline


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="*")
    ap.add_argument("--backend", default="auto", choices=["auto", "gemini", "claude"],
                    help="auto follows whichever API key is set")
    ap.add_argument("--model", help="override the backend's default model")
    ap.add_argument("--list-models", action="store_true",
                    help="print the Gemini models this key can reach, then exit")
    ap.add_argument("-k", type=int, default=5, help="chunks to retrieve (default 5)")
    ap.add_argument("--index", default="index")
    ap.add_argument("--mode", default="hybrid", choices=["dense", "bm25", "hybrid"],
                    help="Phase 6: hybrid (default) fuses dense vectors with BM25")
    # ON by default here and OFF in search.py, deliberately. ask.py stuffs k
    # chunks into a prompt, so the number that decides answer quality is
    # recall@k -- and re-ranking takes recall@5 from 0.88 to 1.00. search.py
    # exists to inspect the ranking itself, where you want stage 1 unaltered.
    ap.add_argument("--rerank", action="store_true", default=True,
                    help="re-read candidates with a cross-encoder (default on)")
    ap.add_argument("--no-rerank", dest="rerank", action="store_false")
    ap.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"],
                    help="thinking depth; API default is high")
    ap.add_argument("--show-thinking", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the prompt and exit without calling the API")
    args = ap.parse_args()

    # .env is gitignored; the API key lives there, never in the code.
    load_dotenv()

    if args.list_models:
        for name in list_gemini_models():
            print(name)
        return 0
    if not args.question:
        ap.error("a question is required (or use --list-models)")
    question = " ".join(args.question)

    pipe = Pipeline(args.index, mode=args.mode, rerank=args.rerank)
    results = pipe.retrieve(question, k=args.k)

    print(f"\nretrieved {len(results)} chunks for {question!r}")
    for r in results:
        print(f"  [{r.rank}] {r.score:.3f}  {r.chunk.id}")

    if args.dry_run:
        user = build_user_message(question, results)
        print(f"\n{'=' * 78}\nSYSTEM\n{'=' * 78}\n{SYSTEM_PROMPT}")
        print(f"\n{'=' * 78}\nUSER\n{'=' * 78}\n{user}")
        chars = len(SYSTEM_PROMPT) + len(user)
        tokens = chars // 4          # rough: ~4 chars/token for English prose
        model = make_generator(args.backend, args.model).model_name \
            if args.backend != "auto" or args.model else None
        line = f"\n{'=' * 78}\n~{chars:,} chars ~= {tokens:,} input tokens"
        if model and model in RATES:
            line += f" ~= ${tokens * RATES[model][0] / 1e6:.4f} at {model} rates"
        print(line)
        return 0

    generator = make_generator(args.backend, args.model)
    print(f"\nmodel     {generator.model_name}\n{'-' * 78}")
    answer = generator.generate(question, results, effort=args.effort,
                                show_thinking=args.show_thinking)
    print(f"\n{'-' * 78}")

    if answer.citations():
        print("citations")
        for n, r in answer.citations():
            print(f"  [{n}] {r.chunk.doc_path}  chars {r.chunk.start}-{r.chunk.end}")
    else:
        print("citations  none -- the answer cited nothing")

    tokens = f"{answer.input_tokens} in / {answer.output_tokens} out"
    if answer.thinking_tokens:
        tokens += f" / {answer.thinking_tokens} thinking"
    cost = f"${answer.cost:.4f}" if answer.cost is not None else "rates not in RATES"
    print(f"\ntokens    {tokens}\ncost      {cost}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
