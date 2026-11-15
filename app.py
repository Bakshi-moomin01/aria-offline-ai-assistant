"""
app.py — FastAPI backend server for ARIA (Offline AI Assistant).

Exposes REST and Streaming endpoints for:
- Real-time conversation with scope guardrails and multi-session memory
- Ollama local model management and status checks
- Multi-model automated benchmark runs and live result fetching
- Explainability and Responsible AI (XAI / RAI) traces
- Web interface serving (Jinja2 templates + static files)
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, ConfigDict, Field

import config
from benchmark import BenchmarkHarness, get_report_html_path, load_latest_report
from explainability import ExplainabilityEngine
from guardrails import check_query
from llm_engine import LLMEngine, ModelNotFoundError, OllamaNotRunningError
from memory import MemoryManager
from models import (
    BenchmarkReport,
    ChatRequest,
    ChatResponse,
    ExplainabilityTrace,
    SystemInfo,
)
from utils import get_system_info, timer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("app")

# Initialize core services
app = FastAPI(
    title="ARIA — Offline AI Assistant",
    description="Offline AI Assistant using local open-source LLMs via Ollama (AI Project)",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static and template directories
TEMPLATES_DIR = config.BASE_DIR / "templates"
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

config.STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(config.STATIC_DIR)), name="static")

# Shared instances
llm_engine = LLMEngine(base_url=config.OLLAMA_BASE_URL)
memory_mgr = MemoryManager(window_size=config.MEMORY_WINDOW_SIZE)
benchmark_harness = BenchmarkHarness(engine=llm_engine)
explainability_engine = ExplainabilityEngine(engine=llm_engine, memory_mgr=memory_mgr)


# ──────────────────────────────────────────────
# Pydantic payloads for extra endpoints
# ──────────────────────────────────────────────

class BenchmarkRunRequest(BaseModel):
    models: list[str] | None = Field(default=None, description="Subset of models to benchmark.")
    question_limit: int | None = Field(default=None, description="Limit number of questions tested.")


class ExplainRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    query: str
    session_id: str = "default"
    model_name: str | None = None


# ──────────────────────────────────────────────
# Web UI routes
# ──────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_index(request: Request):
    """Serve the primary single-page chat and benchmark application."""
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "default_model": config.DEFAULT_CHAT_MODEL,
            "benchmark_models": config.BENCHMARK_MODELS,
        },
    )


# ──────────────────────────────────────────────
# Chat endpoints (Streaming & Sync)
# ──────────────────────────────────────────────

@app.post("/chat")
async def chat_stream(request: ChatRequest):
    """Stream response tokens using server-sent chunks or text stream."""
    session_id = request.session_id
    query = request.message.strip()
    # ------ Few‑shot prompting (optional) ------
    if config.USE_FEW_SHOT:
        # prepend a few example Q/A pairs to guide the model
        examples = "\n".join([f"Q: {q}\nA: {a}" for q, a in config.FEW_SHOT_EXAMPLES])
        query = f"{examples}\n\nQ: {query}\nA:"
    model_name = request.model_name or config.DEFAULT_CHAT_MODEL

    # 1. Scope and Safety Guardrail Check
    guardrail_trace = await check_query(
        query=query,
        llm_engine=llm_engine,
        model_name=model_name,
        use_llm_classifier=True,
    )

    if guardrail_trace.decision == "rejected":
        refusal = guardrail_trace.reason or config.REFUSAL_MESSAGE

        async def refusal_stream():
            yield refusal

        # Record to memory as well
        memory_mgr.add_message(session_id, user_input=query, ai_output=refusal)
        return StreamingResponse(refusal_stream(), media_type="text/plain")

    # 2. Retrieve session history
    history = memory_mgr.get_history(session_id)

    # 3. Stream from Ollama and accumulate for memory persistence
    async def stream_generator():
        accumulated: list[str] = []
        try:
            async for chunk in llm_engine.generate_stream(
                prompt=query,
                model_name=model_name,
                system_prompt=config.SYSTEM_PROMPT,
                history=history,
            ):
                accumulated.append(chunk)
                yield chunk
        except Exception as exc:
            err_msg = f"\n[Inference Error: {exc}]"
            accumulated.append(err_msg)
            yield err_msg
        finally:
            full_reply = "".join(accumulated)
            if full_reply.strip():
                memory_mgr.add_message(session_id, user_input=query, ai_output=full_reply)

    return StreamingResponse(stream_generator(), media_type="text/plain")


@app.post("/chat/sync", response_model=ChatResponse)
async def chat_sync(request: ChatRequest):
    """Synchronous chat endpoint (useful for tests and headless benchmarks)."""
    session_id = request.session_id
    query = request.message.strip()
    # ------ Few‑shot prompting (optional) ------
    if config.USE_FEW_SHOT:
        # prepend a few example Q/A pairs to guide the model
        examples = "\n".join([f"Q: {q}\nA: {a}" for q, a in config.FEW_SHOT_EXAMPLES])
        query = f"{examples}\n\nQ: {query}\nA:"
    model_name = request.model_name or config.DEFAULT_CHAT_MODEL

    start_t = time.perf_counter()

    # Guardrail Check
    guardrail_trace = await check_query(
        query=query,
        llm_engine=llm_engine,
        model_name=model_name,
        use_llm_classifier=True,
    )

    if guardrail_trace.decision == "rejected":
        refusal = guardrail_trace.reason or config.REFUSAL_MESSAGE
        memory_mgr.add_message(session_id, user_input=query, ai_output=refusal)
        latency = (time.perf_counter() - start_t) * 1000
        return ChatResponse(
            session_id=session_id,
            model_name=model_name,
            response=refusal,
            guardrail_triggered=True,
            guardrail_reason=guardrail_trace.reason,
            latency_ms=round(latency, 2),
            timestamp=datetime.utcnow(),
        )

    history = memory_mgr.get_history(session_id)

    try:
        reply = await llm_engine.generate(
            prompt=query,
            model_name=model_name,
            system_prompt=config.SYSTEM_PROMPT,
            history=history,
        )
        memory_mgr.add_message(session_id, user_input=query, ai_output=reply)
        latency = (time.perf_counter() - start_t) * 1000
        return ChatResponse(
            session_id=session_id,
            model_name=model_name,
            response=reply,
            guardrail_triggered=False,
            guardrail_reason=None,
            latency_ms=round(latency, 2),
            timestamp=datetime.utcnow(),
        )
    except Exception as exc:
        logger.error("Chat sync failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ──────────────────────────────────────────────
# Model & Session Management
# ──────────────────────────────────────────────

@app.get("/models")
async def list_models():
    """List all available models in the local Ollama instance."""
    try:
        models = await llm_engine.list_models()
        available_names = [m.get("name") or m.get("model") for m in models]
        return {
            "status": "online",
            "count": len(models),
            "available_models": available_names,
            "raw": models,
        }
    except OllamaNotRunningError as err:
        return {
            "status": "offline",
            "error": str(err),
            "available_models": [],
        }


@app.get("/sessions")
async def get_sessions():
    """List all active conversation sessions and their metadata."""
    return {"sessions": memory_mgr.get_all_sessions()}


@app.delete("/sessions/{session_id}")
async def clear_session(session_id: str):
    """Clear conversation history for a specific session."""
    cleared = memory_mgr.clear_session(session_id)
    return {"session_id": session_id, "cleared": cleared}


# ──────────────────────────────────────────────
# Benchmark endpoints
# ──────────────────────────────────────────────

@app.post("/benchmark/run")
async def run_benchmark(req: BenchmarkRunRequest):
    """Trigger a multi-model benchmark run on the host hardware."""
    models_to_test = req.models or config.BENCHMARK_MODELS
    try:
        report: BenchmarkReport = await benchmark_harness.run_benchmark(
            models=models_to_test,
            limit_questions=req.question_limit,
        )
        return {
            "message": "Benchmark completed successfully",
            "run_id": report.run_id,
            "summary": report.summary,
        }
    except Exception as exc:
        logger.error("Benchmark failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Benchmark execution failed: {exc}")


@app.get("/benchmark/results")
async def get_benchmark_results():
    """Return the most recent benchmark run metrics as JSON."""
    report = load_latest_report()
    if not report:
        return {"status": "no_results_found", "message": "No benchmarks have been run yet."}
    return report.model_dump()


@app.get("/benchmark/report", response_class=HTMLResponse)
async def get_benchmark_html_report():
    """Serve the latest interactive Plotly dashboard report."""
    html_path = get_report_html_path()
    if not html_path or not html_path.exists():
        return HTMLResponse(
            content="<h3>No benchmark visualization report available yet. Run a benchmark first via POST /benchmark/run or the Web UI.</h3>",
            status_code=404,
        )
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


# ──────────────────────────────────────────────
# Explainability & Responsible AI
# ──────────────────────────────────────────────

@app.post("/explain", response_model=ExplainabilityTrace)
async def get_explainability_trace(req: ExplainRequest):
    """Provide complete reasoning trace, prompt composition, and guardrail explanation."""
    try:
        trace = await explainability_engine.generate_trace(
            query=req.query,
            session_id=req.session_id,
            model_name=req.model_name,
        )
        return trace
    except Exception as exc:
        logger.error("Explainability trace failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/explain/audit")
async def get_rai_audit_report(model_name: str | None = None):
    """Run demographic bias and fairness test suite and export audit report."""
    try:
        report = await explainability_engine.build_full_report(model_name=model_name)
        return report.model_dump()
    except Exception as exc:
        logger.error("RAI audit failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ──────────────────────────────────────────────
# Health & System info
# ──────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """System health check and hardware telemetry."""
    is_running = await llm_engine.is_ollama_running()
    sys_info = get_system_info()
    return {
        "status": "healthy" if is_running else "ollama_disconnected",
        "ollama_online": is_running,
        "ollama_url": config.OLLAMA_BASE_URL,
        "timestamp": datetime.utcnow().isoformat(),
        "hardware": sys_info.model_dump(),
    }
