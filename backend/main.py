r"""FastAPI app: chat with semantic caching, cache management, analytics,
model status/pulls, and the benchmark runner.

Run from the project root:
    venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
"""

import json
import logging
import threading
import time
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from backend import benchmark, cache as cache_mod, config, embedding, ollama_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("semcache")

_cache = cache_mod.Cache()
_bench_lock = threading.Lock()
_bench_state: dict[str, Any] = {"running": False, "current": None, "error": None, "finished_at": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    log.info("SemCache backend ready. DB at %s", config.DB_PATH)
    yield


app = FastAPI(title="SemCache AI", version="1.0", lifespan=lifespan)

# Restricted CORS: only the local frontend served on port 5500.
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type"],
)


# ---------- consistent error format, no tracebacks to users ----------

@app.exception_handler(ollama_client.OllamaError)
async def _ollama_error_handler(request: Request, exc: ollama_client.OllamaError):
    return JSONResponse(status_code=502,
                        content={"error": {"code": "ollama_unavailable", "message": str(exc)}})


@app.exception_handler(RequestValidationError)
async def _validation_error_handler(request: Request, exc: RequestValidationError):
    # Same error shape everywhere; the pydantic detail is condensed to one line.
    first = exc.errors()[0] if exc.errors() else {}
    field = ".".join(str(x) for x in first.get("loc", [])[1:]) or "body"
    return JSONResponse(status_code=422, content={"error": {
        "code": "validation_error",
        "message": f"Invalid value for {field}: {first.get('msg', 'bad request')}"}})


@app.exception_handler(Exception)
async def _generic_error_handler(request: Request, exc: Exception):
    log.exception("unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500,
                        content={"error": {"code": "internal_error",
                                        "message": "Something broke on the server. Check the backend window for details."}})


# ---------- request models ----------

class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=config.MAX_QUESTION_CHARS)
    system_prompt: str = Field(default=config.DEFAULT_SYSTEM_PROMPT,
                               max_length=config.MAX_SYSTEM_PROMPT_CHARS)
    temperature: float = Field(default=config.DEFAULT_TEMPERATURE, ge=0.0, le=2.0)


class ClearRequest(BaseModel):
    confirm: bool = False


class ConfigUpdate(BaseModel):
    tau: float | None = Field(default=None, ge=0.0, le=1.0)
    delta: float | None = Field(default=None, ge=0.0, le=1.0)
    ttl_hours: float | None = Field(default=None, ge=0.0)


class FeedbackRequest(BaseModel):
    entry_id: int | None = None
    question: str = Field(min_length=1, max_length=config.MAX_QUESTION_CHARS)
    reason: str = Field(default="", max_length=500)


class BenchmarkRequest(BaseModel):
    experiment: str = "all"  # "1".."5" or "all"


# ---------- endpoints ----------

@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "ollama_running": ollama_client.is_running(),
        "tau": _cache.tau,
        "delta": _cache.delta,
        "ttl_hours": _cache.ttl_hours,
    }


@app.get("/models/status")
def models_status() -> dict:
    if not ollama_client.is_running():
        return {"ollama_running": False, "installed": [], "missing":
                [config.CHAT_MODEL, config.EMBED_MODEL]}
    installed = ollama_client.list_models()
    return {
        "ollama_running": True,
        "installed": installed,
        "missing": ollama_client.missing_models(),
        "chat_model": config.CHAT_MODEL,
        "embed_model": config.EMBED_MODEL,
    }


@app.post("/models/pull")
def models_pull(payload: dict[str, Any] = {}) -> Any:
    """Stream real pull progress as newline-delimited JSON."""
    model = payload.get("model") or ""
    if model not in (config.CHAT_MODEL, config.EMBED_MODEL):
        return JSONResponse(status_code=400, content={"error": {
            "code": "bad_model",
            "message": f"Only {config.CHAT_MODEL} and {config.EMBED_MODEL} can be pulled."}})

    def gen():
        try:
            for chunk in ollama_client.pull_stream(model):
                yield json.dumps(chunk) + "\n"
        except ollama_client.OllamaError as e:
            yield json.dumps({"status": "error", "error": str(e)}) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")


@app.post("/chat")
def chat(req: ChatRequest) -> dict:
    """The main flow: normalize -> exact lookup -> embed -> semantic lookup ->
    on miss generate with Qwen and store. Every response carries the cache
    decision and timings so the UI and analytics can show what happened."""
    if not ollama_client.is_running():
        return JSONResponse(status_code=502, content={"error": {
            "code": "ollama_unavailable",
            "message": "Ollama is not running. Start it (the launcher does this automatically) and try again."}})

    missing = ollama_client.missing_models()
    if missing:
        return JSONResponse(status_code=502, content={"error": {
            "code": "models_missing",
            "message": f"Missing Ollama models: {', '.join(missing)}. Pull them from the Settings tab."}})

    t_start = time.perf_counter()
    question = req.question.strip()
    normalized = embedding.normalize_question(question)
    ns = cache_mod.namespace_of(config.CHAT_MODEL, config.EMBED_MODEL,
                                req.system_prompt, req.temperature)

    # exact match first: no embedding needed if the same text was asked before
    t0 = time.perf_counter()
    exact_res = _cache.lookup(ns, normalized, None)
    retrieval_ms = (time.perf_counter() - t0) * 1000
    embedding_ms = 0.0

    result = exact_res
    if not result.hit:
        t0 = time.perf_counter()
        vec, _dim = embedding.embed_query(question)
        embedding_ms = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        result = _cache.lookup(ns, normalized, np.array(vec))
        retrieval_ms = (time.perf_counter() - t0) * 1000

    generation_ms = 0.0
    prompt_tokens = completion_tokens = None
    if result.hit:
        answer = result.answer or ""
        # tokens saved: report what the original generation used
        tokens = _cache._conn.execute(
            "SELECT prompt_tokens, completion_tokens FROM entries WHERE id = ?",
            (result.entry_id,)).fetchone()
        if tokens:
            prompt_tokens, completion_tokens = tokens["prompt_tokens"], tokens["completion_tokens"]
    else:
        t0 = time.perf_counter()
        gen = ollama_client.generate(question, req.system_prompt, req.temperature)
        generation_ms = (time.perf_counter() - t0) * 1000
        answer = gen.get("response", "")
        prompt_tokens = gen.get("prompt_eval_count")
        completion_tokens = gen.get("eval_count")
        if answer:
            vec, _dim = embedding.embed_document(question)
            _cache.insert(ns, question, normalized, vec, answer,
                          config.CHAT_MODEL, config.EMBED_MODEL,
                          prompt_tokens=prompt_tokens or 0,
                          completion_tokens=completion_tokens or 0)

    total_ms = (time.perf_counter() - t_start) * 1000
    _cache.record_request(ns, result.hit, result.exact,
                          result.similarity if result.similarity else None,
                          result.margin if result.margin else None,
                          embedding_ms, retrieval_ms, generation_ms, total_ms,
                          prompt_tokens, completion_tokens, question)

    return {
        "answer": answer,
        "hit": result.hit,
        "exact_hit": result.exact,
        "similarity": round(result.similarity, 4) if result.similarity else None,
        "margin": round(result.margin, 4) if result.margin else None,
        "matched_question": result.question if result.hit else None,
        "entry_id": result.entry_id if result.hit else None,
        "cache_age_seconds": round(result.age_seconds, 1) if result.hit else None,
        "miss_reason": result.reason if not result.hit else None,
        "chat_model": config.CHAT_MODEL,
        "embed_model": config.EMBED_MODEL,
        "timings_ms": {
            "embedding": round(embedding_ms, 1),
            "retrieval": round(retrieval_ms, 1),
            "generation": round(generation_ms, 1),
            "total": round(total_ms, 1),
        },
        "tokens": {"prompt": prompt_tokens, "completion": completion_tokens},
    }


@app.get("/cache")
def cache_list(search: str = "", limit: int = 200) -> dict:
    limit = max(1, min(limit, 1000))
    entries = _cache.list_entries(search=search, limit=limit)
    return {"entries": entries, "count": len(entries)}


@app.delete("/cache/{entry_id}")
def cache_delete(entry_id: int) -> dict:
    deleted = _cache.delete_entry(entry_id)
    if not deleted:
        return JSONResponse(status_code=404, content={"error": {
            "code": "not_found", "message": f"No cache entry with id {entry_id}."}})
    return {"deleted": entry_id}


@app.post("/cache/clear")
def cache_clear(req: ClearRequest) -> dict:
    try:
        removed = _cache.clear_all(req.confirm)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": {
            "code": "confirm_required", "message": str(e)}})
    return {"removed": removed}


@app.get("/config")
def get_config() -> dict:
    return {
        **_cache.get_settings(),
        "chat_model": config.CHAT_MODEL,
        "embed_model": config.EMBED_MODEL,
        "backend_port": config.BACKEND_PORT,
        "frontend_port": config.FRONTEND_PORT,
    }


@app.post("/config")
def update_config(req: ConfigUpdate) -> dict:
    try:
        return _cache.update_settings(req.tau, req.delta, req.ttl_hours)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": {
            "code": "bad_value", "message": str(e)}})


@app.get("/analytics")
def analytics() -> dict:
    return _cache.analytics()


@app.post("/feedback")
def feedback(req: FeedbackRequest) -> dict:
    _cache.flag_wrong(req.entry_id, req.question, req.reason)
    return {"ok": True, "note": "Entry removed from cache and counted as a false-positive report."}


# ---------- benchmark ----------

def _run_benchmarks(which: str) -> None:
    keys = ["1", "2", "3", "4", "5"] if which == "all" else [which]
    try:
        for k in keys:
            _bench_state["current"] = k
            log.info("running benchmark experiment %s", k)
            benchmark.EXPERIMENTS[k]()
    except Exception as e:  # surfaced via /benchmark/results, not a traceback to users
        log.exception("benchmark failed")
        _bench_state["error"] = str(e)
    finally:
        _bench_state["running"] = False
        _bench_state["current"] = None
        _bench_state["finished_at"] = time.time()


@app.post("/benchmark/run")
def benchmark_run(req: BenchmarkRequest) -> dict:
    if req.experiment not in ("1", "2", "3", "4", "5", "all"):
        return JSONResponse(status_code=400, content={"error": {
            "code": "bad_experiment", "message": "experiment must be 1..5 or 'all'."}})
    with _bench_lock:
        if _bench_state["running"]:
            return JSONResponse(status_code=409, content={"error": {
                "code": "already_running", "message": "A benchmark is already running."}})
        _bench_state.update(running=True, error=None, finished_at=None)
    threading.Thread(target=_run_benchmarks, args=(req.experiment,), daemon=True).start()
    return {"started": req.experiment}


@app.get("/benchmark/results")
def benchmark_results() -> dict:
    return {"state": dict(_bench_state), "results": benchmark.load_results()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.BACKEND_HOST, port=config.BACKEND_PORT)
