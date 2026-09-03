"""Phase 5: retrieved chunks + question -> a grounded, cited answer.

This is the only part of the system that costs money, and the only part where
being wrong is invisible: a confident, well-formatted, fabricated answer looks
exactly like a good one. Almost everything here exists to make that harder.

Two backends sit behind one `Generator` protocol, for the same reason Phase 2
put a protocol behind `LocalEmbedder`: the seam is what makes swapping a
one-line change instead of a refactor. The prompt, the citation parsing, and
the grounding contract are shared -- only the transport differs. That also
makes the backends comparable: same prompt, same corpus, different model.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from typing import Protocol

from rag.retriever import Result

CLAUDE_MODEL = "claude-opus-5"
# Verified reachable on this account 2026-09-03. Pinned rather than using the
# `gemini-pro-latest` / `gemini-flash-latest` aliases, for the same reason
# requirements.txt pins versions: a model that changes under you makes Phase 4
# style measurements incomparable across runs.
#
# Two things learned by actually trying, both of which a hardcoded guess would
# have got wrong: `gemini-2.5-pro` (my first guess) is stale, the 3.1-pro
# preview has a free-tier quota of ZERO, and `gemini-2.5-flash` now 404s for
# new keys. `python scripts/ask.py --list-models` is the source of truth.
GEMINI_MODEL = "gemini-3.6-flash"

# USD per million tokens, (input, output). A model absent from this table
# reports token counts and no cost -- better than inventing a rate, since a
# fabricated price is exactly the kind of confident wrong number this project
# is trying to teach you to distrust.
RATES: dict[str, tuple[float, float]] = {
    "claude-opus-5": (5.00, 25.00),
}

# The grounding contract. Every clause was earned by something Phase 3
# measured, not added defensively:
#   - "only from the excerpts" is the whole point of RAG.
#   - the refusal line exists because retrieval ALWAYS returns k chunks with
#     plausible scores. Asking about ERR_CONNECTION_REFUSED, which appears
#     nowhere in this corpus, still produced a 0.535 top hit. Without an
#     explicit escape hatch the model can only invent or stretch.
#   - "never cite a label that isn't there" because a fabricated [7] is worse
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
    model: str = ""
    cited: list[int] = field(default_factory=list)
    hallucinated: list[int] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0

    @property
    def cost(self) -> float | None:
        """None when the model's rates are not in RATES -- see the note there."""
        rates = RATES.get(self.model)
        if rates is None:
            return None
        return (self.input_tokens * rates[0] + self.output_tokens * rates[1]) / 1_000_000

    def citations(self) -> list[tuple[int, Result]]:
        """The (label, result) pairs the answer actually used."""
        return [(n, self.results[n - 1]) for n in self.cited if 1 <= n <= len(self.results)]


def build_context(results: list[Result]) -> str:
    """Format retrieved chunks as labelled excerpts.

    Note this uses `chunk.text`, NOT `chunk.embed_text`. Phase 4 prepends a
    "title > heading path" to each chunk for embedding; putting that in the
    prompt would have the model quote our synthetic header back as if it were
    document content. It sees the real file text and the real path.
    """
    return "\n\n".join(
        f"[{r.rank}] source: {r.chunk.doc_path} "
        f"(chars {r.chunk.start}-{r.chunk.end}, similarity {r.score:.3f})\n"
        f"{r.chunk.text}"
        for r in results
    )


def build_user_message(question: str, results: list[Result]) -> str:
    # Context first, question last: the question is the volatile part, so this
    # is also the ordering a prompt cache would want.
    return f"<excerpts>\n{build_context(results)}\n</excerpts>\n\nQuestion: {question}"


_CITATION = re.compile(r"\[(\d+)\]")


def parse_citations(text: str, n_results: int) -> tuple[list[int], list[int]]:
    """Return (valid labels cited, in first-use order), (labels that don't exist)."""
    seen: list[int] = []
    bad: list[int] = []
    for m in _CITATION.finditer(text):
        n = int(m.group(1))
        target = seen if 1 <= n <= n_results else bad
        if n not in target:
            target.append(n)
    return seen, bad


def api_key(name: str) -> str | None:
    """An env var's value, but only if it looks like a real key.

    .env.example ships placeholders like `sk-ant-...`, and people copy it to
    .env and fill in one line. A bare `os.environ.get` treats the untouched
    placeholder as a present key, so backend auto-detection picks the wrong
    provider and the failure surfaces as a confusing auth error instead of
    "no key set". Cheap to filter here.
    """
    value = (os.environ.get(name) or "").strip()
    return None if not value or "..." in value else value


class Generator(Protocol):
    """Anything that turns (question, retrieved chunks) into a cited Answer."""

    model_name: str

    def generate(self, question: str, results: list[Result], **kwargs) -> Answer: ...
    # Recognised kwargs across backends: effort, show_thinking, stream_to_stdout,
    # max_tokens, on_delta. `on_delta(text)` is called per streamed text chunk.


class _BaseGenerator:
    """Shared prompt handling and citation accounting. Backends add transport."""

    model_name: str

    def _finish(self, question, results, text, in_tok, out_tok, think_tok=0) -> Answer:
        cited, bad = parse_citations(text, len(results))
        if bad:
            # Not fatal, but the grounding contract leaked. Surface it rather
            # than rendering a citation that points nowhere.
            print(f"\n\n[warning] cited labels that were never provided: {bad}")
        return Answer(
            question=question, text=text, results=results, model=self.model_name,
            cited=cited, hallucinated=bad,
            input_tokens=in_tok, output_tokens=out_tok, thinking_tokens=think_tok,
        )


class ClaudeGenerator(_BaseGenerator):
    """Anthropic. Needs ANTHROPIC_API_KEY."""

    def __init__(self, model: str = CLAUDE_MODEL, client=None) -> None:
        self.model_name = model
        self._client = client

    def generate(self, question, results, *, effort=None, show_thinking=False,
                 stream_to_stdout=True, max_tokens=16000, on_delta=None) -> Answer:
        import anthropic

        client = self._client or anthropic.Anthropic()
        kwargs: dict = {
            "model": self.model_name,
            "max_tokens": max_tokens,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": build_user_message(question, results)}],
            # Adaptive thinking: Claude decides how much to think per request.
            # On Opus 5 this is also the default, so passing it is documentation.
            # `display` defaults to "omitted", which streams empty thinking text.
            "thinking": {"type": "adaptive",
                         "display": "summarized" if show_thinking else "omitted"},
        }
        if effort:
            kwargs["output_config"] = {"effort": effort}

        # Deliberately NOT caching the system prompt: caching is a prefix match
        # with a 512-4096 token minimum and this prompt is a few hundred, so a
        # cache_control here would silently never hit.

        parts: list[str] = []
        try:
            with client.messages.stream(**kwargs) as stream:
                for event in stream:
                    if event.type != "content_block_delta":
                        continue
                    if event.delta.type == "text_delta":
                        parts.append(event.delta.text)
                        if stream_to_stdout:
                            print(event.delta.text, end="", flush=True)
                        # Phase 7: the HTTP layer pushes each delta onto a queue
                        # to re-emit as SSE. A callback rather than a generator
                        # so the existing streaming loop stays untouched.
                        if on_delta is not None:
                            on_delta(event.delta.text)
                    elif event.delta.type == "thinking_delta" and show_thinking:
                        if stream_to_stdout:
                            print(f"\033[2m{event.delta.thinking}\033[0m", end="", flush=True)
                final = stream.get_final_message()
        # Most specific first: one broad `except APIStatusError` would lose the
        # retryable/non-retryable distinction that decides what to do next.
        except anthropic.AuthenticationError:
            raise SystemExit("ANTHROPIC_API_KEY missing or invalid (see .env.example)")
        except anthropic.RateLimitError as e:
            raise SystemExit(f"rate limited; retry after "
                             f"{e.response.headers.get('retry-after', '60')}s")
        except anthropic.APIStatusError as e:
            raise SystemExit(f"API error {e.status_code}: {e.message}")
        except anthropic.APIConnectionError:
            raise SystemExit("could not reach the API -- check the network")

        return self._finish(question, results, "".join(parts),
                            final.usage.input_tokens, final.usage.output_tokens)


class GeminiGenerator(_BaseGenerator):
    """Google. Needs GEMINI_API_KEY (or GOOGLE_API_KEY).

    Same prompt and same citation contract as the Claude path -- the point of
    the shared code above. Three things differ mechanically:
      - the system prompt is `config.system_instruction`, not a top-level arg
      - thinking is `thinking_config.thinking_level`, a coarser dial than
        Claude's five effort levels, so the mapping below is lossy
      - thought text arrives as ordinary parts flagged `part.thought`, rather
        than as a distinct delta type, so parts must be inspected not just
        concatenated (`chunk.text` would silently merge reasoning into the answer)
    """

    # Claude effort -> Gemini thinking level. Deliberately lossy: Gemini has
    # four levels to Claude's five, so xhigh and max both land on HIGH.
    _EFFORT = {"low": "LOW", "medium": "MEDIUM", "high": "HIGH",
               "xhigh": "HIGH", "max": "HIGH"}

    def __init__(self, model: str = GEMINI_MODEL, client=None) -> None:
        self.model_name = model
        self._client = client

    @staticmethod
    def _make_client():
        from google import genai

        key = api_key("GEMINI_API_KEY") or api_key("GOOGLE_API_KEY")
        if not key:
            raise SystemExit("set GEMINI_API_KEY in .env (see .env.example)")
        return genai.Client(api_key=key)

    def generate(self, question, results, *, effort=None, show_thinking=False,
                 stream_to_stdout=True, max_tokens=16000, retries=2,
                 on_delta=None) -> Answer:
        from google.genai import errors, types

        client = self._client or self._make_client()
        thinking = types.ThinkingConfig(include_thoughts=bool(show_thinking))
        if effort:
            thinking.thinking_level = self._EFFORT.get(effort, "MEDIUM")

        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=max_tokens,
            thinking_config=thinking,
            # We declare no tools, and without this the SDK prints an
            # automatic-function-calling advisory on every single call.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        user_message = build_user_message(question, results)

        parts: list[str] = []
        usage = None
        for attempt in range(retries + 1):
            try:
                for chunk in client.models.generate_content_stream(
                    model=self.model_name, contents=user_message, config=config,
                ):
                    # usage_metadata is cumulative and only meaningful on the
                    # last chunk that carries it -- keep overwriting.
                    if chunk.usage_metadata is not None:
                        usage = chunk.usage_metadata
                    for cand in chunk.candidates or []:
                        for part in (cand.content.parts if cand.content else None) or []:
                            if not part.text:
                                continue
                            if part.thought:
                                if show_thinking and stream_to_stdout:
                                    print(f"\033[2m{part.text}\033[0m", end="", flush=True)
                            else:
                                parts.append(part.text)
                                if stream_to_stdout:
                                    print(part.text, end="", flush=True)
                                # Phase 7: see the note in ClaudeGenerator.
                                if on_delta is not None:
                                    on_delta(part.text)
                break
            except (errors.ClientError, errors.ServerError) as e:
                # 5xx and 429 are transient -- both were hit while testing this
                # (gemini-3.8-flash returned 503 "high demand" repeatedly). Every
                # other 4xx is our own request being wrong, so it must not retry.
                transient = isinstance(e, errors.ServerError) or e.code == 429
                # Retrying after tokens have already been printed would emit the
                # answer twice. Streaming buys responsiveness at the cost of
                # being unable to cleanly retry mid-response.
                if not transient or parts or attempt == retries:
                    hint = ""
                    if e.code == 404:
                        hint = ("\nrun `python scripts/ask.py --list-models` -- "
                                "Google's model IDs go stale quickly")
                    elif transient and parts:
                        hint = "\n(not retried: output had already started streaming)"
                    raise SystemExit(f"Gemini error {e.code}: {e.message}{hint}")
                delay = 2.0 * (2 ** attempt)
                print(f"\n[{e.code} transient] retrying in {delay:.0f}s "
                      f"({attempt + 1}/{retries})", flush=True)
                time.sleep(delay)

        return self._finish(
            question, results, "".join(parts),
            getattr(usage, "prompt_token_count", 0) or 0,
            getattr(usage, "candidates_token_count", 0) or 0,
            getattr(usage, "thoughts_token_count", 0) or 0,
        )


def list_gemini_models() -> list[str]:
    """What this key can actually reach. Hardcoded model IDs go stale."""
    client = GeminiGenerator._make_client()
    names = []
    for m in client.models.list():
        actions = getattr(m, "supported_actions", None) or []
        if not actions or "generateContent" in actions:
            names.append((m.name or "").removeprefix("models/"))
    return sorted(n for n in names if n)


def make_generator(backend: str = "auto", model: str | None = None) -> Generator:
    """Pick a backend. "auto" follows whichever API key is actually present."""
    if backend == "auto":
        if api_key("GEMINI_API_KEY") or api_key("GOOGLE_API_KEY"):
            backend = "gemini"
        elif api_key("ANTHROPIC_API_KEY"):
            backend = "claude"
        else:
            raise SystemExit("no API key found -- set GEMINI_API_KEY or "
                             "ANTHROPIC_API_KEY in .env (see .env.example)")
    if backend == "gemini":
        return GeminiGenerator(model or GEMINI_MODEL)
    if backend == "claude":
        return ClaudeGenerator(model or CLAUDE_MODEL)
    raise ValueError(f"unknown backend {backend!r}")
