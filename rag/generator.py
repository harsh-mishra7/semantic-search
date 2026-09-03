"""Phase 5: retrieved chunks + question -> a grounded, cited answer.

This is the only part of the system that costs money, and the only part where
being wrong is invisible: a confident, well-formatted, fabricated answer looks
exactly like a good one. Almost everything here exists to make that harder.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rag.retriever import Result

MODEL = "claude-opus-5"
# Opus 5 pricing, per million tokens.
COST_IN, COST_OUT = 5.00, 25.00

# The grounding contract. Every clause here was earned by something Phase 3
# measured, not added defensively:
#   - "only from the excerpts" is the whole point of RAG.
#   - the refusal line exists because retrieval ALWAYS returns k chunks with
#     plausible scores. Asking about ERR_CONNECTION_REFUSED, which appears
#     nowhere in this corpus, still produced a 0.535 top hit. Without an
#     explicit escape hatch the model's only options are to invent or to stretch.
#   - "do not cite a label that isn't there" because a fabricated [7] is worse
#     than no citation: it looks checkable and isn't.
SYSTEM_PROMPT = """\
You answer questions about a private documentation corpus using ONLY the \
excerpts provided in the user's message.

Rules:

1. Ground every claim in the excerpts. Do not use your own knowledge of the \
tools involved, even when you are confident and even when the excerpts are \
incomplete.

2. If the excerpts do not contain the answer, say so plainly and stop. Begin \
that reply with "I don't know" and name what is missing. This is the correct \
answer, not a failure -- do not offer a general-knowledge answer as a \
consolation.

3. If the excerpts are only tangentially related to the question, treat that as \
case 2. Do not assemble an answer out of loosely relevant fragments.

4. Cite with the bracketed labels as given, e.g. [2], at the end of each \
sentence that draws on an excerpt. Never cite a label that does not appear in \
the excerpts.

5. If two excerpts disagree, say so and cite both rather than silently picking one.

6. Quote commands, environment variable names, and file paths exactly as they \
appear. Do not tidy or infer them.

Be concise. Prefer a short, exact answer over a thorough, padded one."""


@dataclass
class Answer:
    question: str
    text: str
    results: list[Result]
    cited: list[int] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    refused: bool = False

    @property
    def cost(self) -> float:
        return (self.input_tokens * COST_IN + self.output_tokens * COST_OUT) / 1_000_000

    def citations(self) -> list[tuple[int, Result]]:
        """The (label, result) pairs the answer actually used."""
        return [(n, self.results[n - 1]) for n in self.cited if 1 <= n <= len(self.results)]


def build_context(results: list[Result]) -> str:
    """Format retrieved chunks as labelled excerpts.

    Note this uses `chunk.text`, NOT `chunk.embed_text`. Phase 4 prepends a
    "title > heading path" to each chunk for embedding; putting that in the
    prompt would have Claude quote our synthetic header back as if it were
    document content. The model sees the real file text and the real path.
    """
    blocks = []
    for r in results:
        blocks.append(
            f"[{r.rank}] source: {r.chunk.doc_path} "
            f"(chars {r.chunk.start}-{r.chunk.end}, similarity {r.score:.3f})\n"
            f"{r.chunk.text}"
        )
    return "\n\n".join(blocks)


def build_messages(question: str, results: list[Result]) -> list[dict]:
    # Context first, question last. The question is the volatile part of the
    # prompt, so keeping it at the end is also what a cache breakpoint would
    # want -- see the note in generate() on why we don't cache here yet.
    return [
        {
            "role": "user",
            "content": (
                f"<excerpts>\n{build_context(results)}\n</excerpts>\n\n"
                f"Question: {question}"
            ),
        }
    ]


_CITATION = re.compile(r"\[(\d+)\]")


def parse_citations(text: str, n_results: int) -> tuple[list[int], list[int]]:
    """Return (valid labels cited, in order of first use), (hallucinated labels)."""
    seen: list[int] = []
    bad: list[int] = []
    for m in _CITATION.finditer(text):
        n = int(m.group(1))
        target = seen if 1 <= n <= n_results else bad
        if n not in target:
            target.append(n)
    return seen, bad


def generate(
    question: str,
    results: list[Result],
    client=None,
    effort: str | None = None,
    show_thinking: bool = False,
    stream_to_stdout: bool = True,
    max_tokens: int = 16000,
) -> Answer:
    """Ask Claude the question, grounded in `results`. Streams as it arrives."""
    import anthropic

    if client is None:
        client = anthropic.Anthropic()

    kwargs: dict = {
        "model": MODEL,
        "max_tokens": max_tokens,
        "system": SYSTEM_PROMPT,
        "messages": build_messages(question, results),
        # Adaptive thinking: Claude decides how much to think per request.
        # On Opus 5 this is also the default, so passing it is documentation.
        # `display` defaults to "omitted", which streams empty thinking text --
        # so with a plain text stream you see a pause and then the answer.
        "thinking": {"type": "adaptive", "display": "summarized" if show_thinking else "omitted"},
    }
    # Effort is the cost/quality lever inside a single model. Left at the API
    # default (high) unless asked: grounded extraction over five excerpts is a
    # shallow task, so "medium" or "low" is often just as good and cheaper --
    # but that is a measurement to make, not an assumption to bake in.
    if effort:
        kwargs["output_config"] = {"effort": effort}

    # We do NOT cache the system prompt. Caching is a prefix match and the
    # minimum cacheable prefix is 512-4096 tokens depending on the model; this
    # system prompt is a few hundred, so a cache_control here would silently
    # never hit. Worth knowing before adding one out of habit.

    parts: list[str] = []
    try:
        with client.messages.stream(**kwargs) as stream:
            for event in stream:
                if event.type == "content_block_delta":
                    if event.delta.type == "text_delta":
                        parts.append(event.delta.text)
                        if stream_to_stdout:
                            print(event.delta.text, end="", flush=True)
                    elif event.delta.type == "thinking_delta" and show_thinking:
                        if stream_to_stdout:
                            print(f"\033[2m{event.delta.thinking}\033[0m", end="", flush=True)
            final = stream.get_final_message()
    # Most specific first: collapsing these into one `except APIStatusError`
    # loses the retryable/not-retryable distinction that decides what to do next.
    except anthropic.AuthenticationError:
        raise SystemExit(
            "ANTHROPIC_API_KEY is missing or invalid. Put a real key in .env "
            "(see .env.example) -- retrieval works without one, generation does not."
        )
    except anthropic.RateLimitError as e:
        retry = e.response.headers.get("retry-after", "60")
        raise SystemExit(f"rate limited; retry after {retry}s")
    except anthropic.APIStatusError as e:
        raise SystemExit(f"API error {e.status_code}: {e.message}")
    except anthropic.APIConnectionError:
        raise SystemExit("could not reach the API -- check the network")

    text = "".join(parts)
    cited, bad = parse_citations(text, len(results))
    if bad:
        # Not fatal, but it means the grounding contract leaked. Surface it
        # rather than rendering a citation that points nowhere.
        print(f"\n\n[warning] answer cited labels that were never provided: {bad}")

    return Answer(
        question=question,
        text=text,
        results=results,
        cited=cited,
        input_tokens=final.usage.input_tokens,
        output_tokens=final.usage.output_tokens,
        # stop_details is only populated for a safety refusal, so guard it.
        refused=final.stop_reason == "refusal",
    )
