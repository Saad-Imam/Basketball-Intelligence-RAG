import os
import time
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel # data validation library, used for type-safety, auto-documentation, serialization

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# Lazy singletons (loaded once at startup)
_retriever = None
_generator = None

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

# ── Request / Response models
class AskRequest(BaseModel):
    query: str
    top_k: int = 5                    # how many final chunks to surface


class SourceChunk(BaseModel):
    index: int
    text: str
    label: str                        # human-readable source label
    cross_encoder_score: float
    rrf_score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    retrieval_ms: int
    generation_ms: int
    total_ms: int


# build a readable label from chunk metadata 
def _chunk_label(chunk: dict) -> str:
    meta = chunk.get("metadata", {})
    layer = meta.get("layer_name", "")
    if layer == "rulebook":
        parts = [p for p in [meta.get("league", ""), "Rulebook"] if p]
        sec = meta.get("section", "")
        sec_title = meta.get("section_title", "")
        if sec and sec_title:
            parts.append(f"{sec} ({sec_title})")
        elif sec:
            parts.append(sec)
        return " › ".join(parts) or "Rulebook"
    else:
        parts = ["HoopStudent Encyclopedia"]
        term = meta.get("term", "")
        ctype = meta.get("chunk_type", "")
        if term:
            parts.append(term)
        if ctype:
            parts.append(ctype.title())
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

        # --- Format sources ---
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
            total_ms=retrieval_ms + generation_ms,
        )

    except Exception as exc:
        log.exception("Pipeline error")
        raise HTTPException(status_code=500, detail=str(exc))


# Serve the Signle page app
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
def index():
    return FileResponse(static_dir / "index.html")

@app.on_event("startup")
async def startup_log():
    log.info("FastAPI is up — models will load on first request.")