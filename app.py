import os
import time
import json
import logging
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel # data validation library, used for type-safety, auto-documentation, serialization

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# Lazy singletons (loaded once at startup)
_retriever = None
_generator = None
_judge     = None

def get_retriever():
    global _retriever
    if _retriever is None:
        from src.retrieval.retriever import BasketballRetriever
        log.info("Initialising retriever…")
        _retriever = BasketballRetriever(use_fp16=True, verbose=False)
    return _retriever


def get_generator():
    global _generator
    if _generator is None:
        from src.generator.generator import BasketballGenerator
        log.info("Initialising generator…")
        _generator = BasketballGenerator(verbose=False)
    return _generator

def get_judge():
    global _judge
    if _judge is None:
        from src.evaluation.evaluate import RAGJudge
        log.info("Initialising judge…")
        _judge = RAGJudge(verbose=False)
    return _judge

@asynccontextmanager
# This decorator creates an async context manager — a function that runs code before and after your FastAPI app lifecycle
async def lifespan(app: FastAPI):
    # Code BEFORE app starts (setup phase)
    log.info("Warming up models…")
    try:
        get_retriever()      # Load retriever
        get_generator()      # Load generator

        log.info("Models ready.")
    except Exception as exc:
        log.warning(f"Warm-up failed (will retry on first request): {exc}")
    
    yield  # ← App runs between here
    
    # Code AFTER app shuts down (cleanup phase)
    # (currently empty, but could close DB connections, etc.)


# App instance
# app = FastAPI(title="Basketball Intelligence RAG", lifespan=lifespan) # troubleshooting, because hugging face space taking
# too long to startup
app = FastAPI(title="Basketball Intelligence RAG")

class AskRequest(BaseModel):
    query:    str
    top_k:    int  = 5
    evaluate: bool = True   # set False to skip eval
 
class ClaimItem(BaseModel):
    claim:     str
    supported: bool
    reasoning: str
 
class FaithfulnessPayload(BaseModel):
    score:           float
    supported_count: int
    total:           int
    claims:          list[ClaimItem]
 
class RelevancyPayload(BaseModel):
    score:               float
    generated_questions: list[str]
    similarities:        list[float]
 
class SourceChunk(BaseModel):
    index:               int
    text:                str
    label:               str
    cross_encoder_score: float
    rrf_score:           float
 
class AskResponse(BaseModel):
    answer:        str
    sources:       list[SourceChunk]
    retrieval_ms:  int
    generation_ms: int
    eval_ms:       int
    total_ms:      int
    faithfulness:  FaithfulnessPayload | None = None
    relevancy:     RelevancyPayload    | None = None


# build a readable label from chunk metadata 
def _chunk_label(chunk: dict) -> str:
    meta  = chunk.get("metadata", {})
    layer = meta.get("layer_name", "")
    if layer == "rulebook":
        parts = [p for p in [meta.get("league", ""), "Rulebook"] if p]
        sec, sec_title = meta.get("section", ""), meta.get("section_title", "")
        if sec and sec_title:
            parts.append(f"{sec} ({sec_title})")
        elif sec:
            parts.append(sec)
        return " › ".join(parts) or "Rulebook"
    else:
        parts = ["HoopStudent Encyclopedia"]
        term  = meta.get("term", "")
        ctype = meta.get("chunk_type", "")
        if term:  parts.append(term)
        if ctype: parts.append(ctype.title())
        return " › ".join(parts)


# Routes 
@app.get("/api/health")
def health():
    return {"status": "ok"}

@app.post("/api/ask", response_model=AskResponse)
def ask(body: AskRequest):
    query = body.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="Query cannot be empty.")
    try:
        # Retrieval
        t0 = time.perf_counter()
        chunks = get_retriever().retrieve(query, final_n=body.top_k)
        retrieval_ms = int((time.perf_counter() - t0) * 1000)
 
        # Generation
        t1 = time.perf_counter()
        result = get_generator().generate(query, chunks)
        generation_ms = int((time.perf_counter() - t1) * 1000)
 
        # Evaluation
        eval_ms = 0
        faith_payload = None
        rel_payload   = None
 
        if body.evaluate:
            t2 = time.perf_counter()
            eval_result = get_judge().evaluate(
                query=query,
                answer=result.answer,
                context_chunks=result.context_chunks,
            )
            eval_ms = int((time.perf_counter() - t2) * 1000)
 
            faith_payload = FaithfulnessPayload(
                score=round(eval_result.faithfulness.score, 4),
                supported_count=eval_result.faithfulness.supported_count,
                total=eval_result.faithfulness.total,
                claims=[
                    ClaimItem(
                        claim=cv.claim,
                        supported=cv.supported,
                        reasoning=cv.reasoning,
                    )
                    for cv in eval_result.faithfulness.claims
                ],
            )
            rel_payload = RelevancyPayload(
                score=round(eval_result.relevancy.score, 4),
                generated_questions=eval_result.relevancy.generated_questions,
                similarities=[round(s, 4) for s in eval_result.relevancy.similarities],
            )
        sources = [
            SourceChunk(
                index=i + 1,
                text=c.get("text", ""),
                label=_chunk_label(c),
                cross_encoder_score=round(c.get("cross_encoder_score", 0.0), 4),
                rrf_score=round(c.get("rrf_score", 0.0), 4),
            )
            for i, c in enumerate(result.context_chunks)
        ]
 
        return AskResponse(
            answer=result.answer,
            sources=sources,
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
            eval_ms=eval_ms,
            total_ms=retrieval_ms + generation_ms + eval_ms,
            faithfulness=faith_payload,
            relevancy=rel_payload,
        )
 
    except Exception as exc:
        log.exception("Pipeline error")
        raise HTTPException(status_code=500, detail=str(exc))

# Streaming evaluation endpoint
# Uses Server-Sent Events (SSE) so the frontend gets live progress per query.
# Each event is a JSON line: {"type": "progress"|"result"|"summary"|"error", ...}
 
@app.get("/api/evaluate")
async def run_evaluation():
    """
    Runs the fixed test set and streams results via SSE.

    Why asyncio.to_thread + a queue?
    ---------------------------------
    The rate limiter inside RAGJudge._llm() calls time.sleep() (a blocking
    sleep) which freezes the entire asyncio event loop.  When the loop is
    frozen, the SSE generator cannot yield anything.  After ~60 s of silence
    HuggingFace's nginx proxy declares the connection idle and drops it,
    which is what was causing the "stream error" in the browser.

    The fix has two parts:
      1. asyncio.to_thread  — runs every blocking call (retrieve / generate /
         evaluate) in a worker thread, keeping the event loop free.
      2. SSE keepalive comments — while waiting for the next result the loop
         checks the queue every 15 s and sends ": keepalive" if it's empty.
         SSE comment lines are ignored by the browser but they flush bytes
         through nginx, resetting the proxy's idle timer.
    """
    from src.evaluation.evaluate import TEST_QUERIES

    async def event_stream() -> AsyncGenerator[str, None]:
        def sse(data: dict) -> str:
            return f"data: {json.dumps(data)}\n\n"

        retriever = get_retriever()
        generator = get_generator()
        judge     = get_judge()

        total = len(TEST_QUERIES)
        queue: asyncio.Queue = asyncio.Queue()

        # ------------------------------------------------------------------
        # Worker coroutine — runs the full eval loop and pushes messages
        # into the queue.  Each blocking call is wrapped in asyncio.to_thread
        # so the event loop (and SSE keepalives) stay responsive during
        # rate-limiter sleeps of 40-50 seconds.
        # ------------------------------------------------------------------
        async def run_eval() -> None:
            all_results = []

            for i, query in enumerate(TEST_QUERIES):
                await queue.put({"type": "progress", "index": i,
                                 "total": total, "query": query})
                try:
                    chunks = await asyncio.to_thread(
                        retriever.retrieve, query, final_n=5
                    )
                    gen = await asyncio.to_thread(
                        generator.generate, query, chunks
                    )
                    ev = await asyncio.to_thread(
                        judge.evaluate,
                        query=query,
                        answer=gen.answer,
                        context_chunks=gen.context_chunks,
                    )

                    result_payload = {
                        "type":         "result",
                        "index":        i,
                        "query":        query,
                        "answer":       gen.answer,
                        "faithfulness": round(ev.faithfulness.score, 4),
                        "relevancy":    round(ev.relevancy.score, 4),
                        "claims": [
                            {"claim": c.claim, "supported": c.supported,
                             "reasoning": c.reasoning}
                            for c in ev.faithfulness.claims
                        ],
                        "questions": [
                            {"question": q, "similarity": round(s, 4)}
                            for q, s in zip(
                                ev.relevancy.generated_questions,
                                ev.relevancy.similarities,
                            )
                        ],
                    }
                    all_results.append(result_payload)
                    await queue.put(result_payload)

                except Exception as e:
                    log.exception(f"Eval failed for query {i}: {query}")
                    await queue.put({"type": "error", "index": i,
                                     "query": query, "message": str(e)})

            # Final summary — computed here so the SSE loop just forwards it
            if all_results:
                avg_faith = sum(r["faithfulness"] for r in all_results) / len(all_results)
                avg_rel   = sum(r["relevancy"]    for r in all_results) / len(all_results)
            else:
                avg_faith = avg_rel = 0.0

            await queue.put({
                "type":             "summary",
                "avg_faithfulness": round(avg_faith, 4),
                "avg_relevancy":    round(avg_rel, 4),
                "results":          all_results,
            })

            await queue.put(None)   # sentinel — tells the reader to stop

        # ------------------------------------------------------------------
        # SSE reader loop — drains the queue and sends keepalives when idle
        # ------------------------------------------------------------------
        eval_task = asyncio.create_task(run_eval())
        yield sse({"type": "start", "total": total})

        while True:
            try:
                # Wait up to 15 s for the next message.
                # If nothing arrives the worker is blocked inside a rate-limiter
                # sleep — send a keepalive comment to prevent proxy timeout.
                msg = await asyncio.wait_for(queue.get(), timeout=15.0)
            except asyncio.TimeoutError:
                # ": keepalive" is a valid SSE comment: browsers ignore it,
                # but it flushes bytes through nginx and resets its idle timer.
                yield ": keepalive\n\n"
                continue

            if msg is None:
                break       # sentinel received — eval is complete

            yield sse(msg)

        await eval_task     # propagate any unexpected exceptions from the worker

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering on HF Spaces
        },
    )

# Serve the Single page app
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
def index():
    return FileResponse(static_dir / "index.html")

@app.on_event("startup")
async def startup_log():
    log.info("FastAPI is up — models will load on first request.")