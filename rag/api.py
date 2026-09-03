"""Phase 7: HTTP. Retrieval and grounded answers over the same Pipeline.

    uvicorn rag.api:app --reload          # dev
    uvicorn rag.api:app --port 8000       # then GET http://127.0.0.1:8000/docs

Endpoints:
    GET  /health       index + model stats; no LLM, no key
    POST /search       retrieval only, ranked chunks + scores
    POST /ask          full RAG, one JSON answer with citations
    POST /ask/stream   full RAG, streamed as Server-Sent Events

THE ONE THING THIS FILE EXISTS TO GET RIGHT: load once, serve many.

Building a Pipeline costs ~10s for the embedding model, ~2s for the
cross-encoder, plus reading the index and building the BM25 postings. Doing
that per request would make every query ~12s slower than it needs to be, so it
happens once in the lifespan handler and is then shared. That is the entire
reason Phase 5 put the expensive setup in `Pipeline.__init__` rather than in
`ask()`.

ASYNC vs THREADPOOL, which is easy to get backwards here. Retrieval is
CPU-bound (a forward pass to embed the query, then up to 50 more in the
re-ranker). FastAPI runs a plain `def` endpoint in a threadpool, so CPU work
there does not stall other requests -- but an `async def` endpoint runs ON the
event loop, where the same work blocks every other connection including
in-flight SSE streams. So /search and /ask are deliberately `def`, and
/ask/stream, which must be async to yield, pushes its blocking work to a
thread explicitly.

Deploy single-worker (`--workers 1`) unless you have the RAM to duplicate the
model and index per worker; both are per-process.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
from contextlib import asynccontextmanager
from typing import Any, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from rag.generator import Answer
from rag.pipeline import Pipeline
from rag.retriever import Result

# Retrieval settings are process-level, not per-request: changing the mode
# rebuilds the BM25 index and may load the cross-encoder, which is not
# something an HTTP caller should be able to trigger. Set them at startup.
INDEX_DIR = os.environ.get("RAG_INDEX", "index")
MODE = os.environ.get("RAG_MODE", "hybrid")
RERANK = os.environ.get("RAG_RERANK", "1") not in ("0", "false", "no")

state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()
    # Retrieval only. The generator stays deferred inside Pipeline, so the
    # server starts and /search works with no API key present at all --
    # only /ask and /ask/stream need one.
    state["pipe"] = Pipeline(INDEX_DIR, mode=MODE, rerank=RERANK)
    yield
    state.clear()


app = FastAPI(
    title="RAG semantic search",
    description="Hybrid retrieval (BM25 + dense, RRF) with cross-encoder re-ranking.",
    lifespan=lifespan,
)


class SearchRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    k: int = Field(5, ge=1, le=50)


class AskRequest(SearchRequest):
    effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None
    max_tokens: int = Field(16000, ge=64, le=64000)


class Hit(BaseModel):
    rank: int
    score: float
    chunk_id: str
    source: str
    start: int
    end: int
    text: str


def _hits(results: list[Result]) -> list[Hit]:
    return [
        Hit(rank=r.rank, score=r.score, chunk_id=r.chunk.id, source=r.chunk.doc_path,
            start=r.chunk.start, end=r.chunk.end, text=r.chunk.text)
        for r in results
    ]


def _answer_payload(answer: Answer) -> dict:
    return {
        "question": answer.question,
        "answer": answer.text,
        "model": answer.model,
        # Only the chunks the model actually cited, mapped back to real spans.
        # This is what Phase 1's offsets were for: a citation you can check.
        "citations": [
            {"label": label, "source": r.chunk.doc_path,
             "chunk_id": r.chunk.id, "start": r.chunk.start, "end": r.chunk.end}
            for label, r in answer.citations()
        ],
        "hallucinated_labels": answer.hallucinated,
        "usage": {"input_tokens": answer.input_tokens,
                  "output_tokens": answer.output_tokens,
                  "thinking_tokens": answer.thinking_tokens,
                  "cost_usd": answer.cost},
    }


@app.get("/health")
def health() -> dict:
    pipe: Pipeline = state["pipe"]
    meta = pipe.store.meta
    return {
        "status": "ok",
        "chunks": len(pipe.store),
        "documents": len(meta.doc_hashes) or None,
        "model": meta.model_name,
        "dimension": meta.dimension,
        "chunking": {"size": meta.chunk_size, "overlap": meta.overlap,
                     "strategy": meta.strategy, "min_section": meta.min_section,
                     "prepend_context": meta.prepend_context},
        "retrieval": {"mode": pipe.retriever.mode,
                      "rerank": pipe.retriever.reranker is not None,
                      "rerank_model": getattr(pipe.retriever.reranker, "model_name", None)},
        "built_at": meta.created_at,
    }


@app.post("/search", response_model=list[Hit])
def search(req: SearchRequest) -> list[Hit]:
    """Retrieval only -- no LLM, no API key, no cost."""
    return _hits(state["pipe"].retrieve(req.question, k=req.k))


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    pipe: Pipeline = state["pipe"]
    results = pipe.retrieve(req.question, k=req.k)
    try:
        answer = pipe.generator.generate(
            req.question, results, effort=req.effort,
            max_tokens=req.max_tokens, stream_to_stdout=False)
    # The generators raise SystemExit on API failures, which suits a CLI and is
    # actively dangerous in a server: SystemExit derives from BaseException, so
    # it bypasses `except Exception` and would tear down the worker instead of
    # failing one request. Translate it at the boundary.
    except SystemExit as e:
        raise HTTPException(status_code=502, detail=str(e)) from None
    return {**_answer_payload(answer), "sources": [h.model_dump() for h in _hits(results)]}


def _sse(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode()


@app.post("/ask/stream")
async def ask_stream(req: AskRequest) -> StreamingResponse:
    """Same as /ask, streamed as SSE: `sources`, then `delta`*, then `done`.

    Sending `sources` before the first token is deliberate -- the client can
    render what is being read from while the model is still thinking, which is
    most of the wall clock on a reasoning model.
    """
    pipe: Pipeline = state["pipe"]
    # Retrieval is CPU-bound; keep it off the event loop.
    results = await asyncio.to_thread(pipe.retrieve, req.question, req.k)

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def emit(item) -> None:
        # Called from the generator's thread, so the put must be marshalled
        # back onto the loop rather than touching the queue directly.
        loop.call_soon_threadsafe(queue.put_nowait, item)

    def run() -> None:
        try:
            answer = pipe.generator.generate(
                req.question, results, effort=req.effort,
                max_tokens=req.max_tokens, stream_to_stdout=False,
                on_delta=lambda text: emit(("delta", text)))
            emit(("done", answer))
        except SystemExit as e:
            emit(("error", str(e)))
        except Exception as e:                        # noqa: BLE001
            emit(("error", f"{type(e).__name__}: {e}"))
        finally:
            emit(None)

    async def events():
        yield _sse("sources", {"sources": [h.model_dump() for h in _hits(results)]})
        # Started only now, after `sources` is on the wire, so a client that
        # disconnects during retrieval never causes an API call to be paid for.
        threading.Thread(target=run, daemon=True).start()
        while (item := await queue.get()) is not None:
            kind, payload = item
            if kind == "delta":
                yield _sse("delta", {"text": payload})
            elif kind == "done":
                yield _sse("done", _answer_payload(payload))
            else:
                yield _sse("error", {"detail": payload})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        # no-cache stops a proxy replaying a stale stream; no-transform stops
        # one buffering it, which would defeat streaming entirely.
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
    )
