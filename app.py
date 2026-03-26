import os
import time
import json
import logging
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
    Runs the fixed 15-query test set and streams results via SSE.
    """
    from src.evaluation.evaluate import TEST_QUERIES
    async def event_stream() -> AsyncGenerator[str, None]: # returns SEE formatted strings for real-time updates
        def sse(data: dict) -> str: # helper to format data as SSE events
            return f"data: {json.dumps(data)}\n\n"
 
        retriever = get_retriever()
        generator = get_generator()
        judge     = get_judge()
 
        total = len(TEST_QUERIES)
        yield sse({"type": "start", "total": total})
 
        all_results = []
 
        for i, query in enumerate(TEST_QUERIES):
            # Notify frontend which query we're running
            yield sse({"type": "progress", "index": i, "total": total, "query": query})
 
            try:
                # Retrieve + generate
                chunks = retriever.retrieve(query, final_n=5)
                gen    = generator.generate(query, chunks)
 
                # Evaluate
                ev = judge.evaluate(
                    query=query,
                    answer=gen.answer,
                    context_chunks=gen.context_chunks,
                )
 
                result_payload = {
                    "type":        "result",
                    "index":       i,
                    "query":       query,
                    "answer":      gen.answer,
                    "faithfulness": round(ev.faithfulness.score, 4),
                    "relevancy":    round(ev.relevancy.score, 4),
                    "claims": [
                        {"claim": c.claim, "supported": c.supported, "reasoning": c.reasoning}
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
                yield sse(result_payload)
 
            except Exception as e:
                log.exception(f"Eval failed for query {i}: {query}")
                yield sse({"type": "error", "index": i, "query": query, "message": str(e)})
 
        # Final summary
        if all_results:
            avg_faith = sum(r["faithfulness"] for r in all_results) / len(all_results)
            avg_rel   = sum(r["relevancy"]    for r in all_results) / len(all_results)
        else:
            avg_faith = avg_rel = 0.0
 
        yield sse({
            "type":             "summary",
            "avg_faithfulness": round(avg_faith, 4),
            "avg_relevancy":    round(avg_rel, 4),
            "results":          all_results,
        })
 
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
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