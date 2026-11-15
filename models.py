"""
models.py — Pydantic data models for the Offline AI Assistant (EduBot).

Defines request / response schemas used across the FastAPI endpoints,
benchmark harness, and explainability module.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AppBaseModel(BaseModel):
    model_config = ConfigDict(protected_namespaces=())


# ──────────────────────────────────────────────
# Chat models
# ──────────────────────────────────────────────

class Role(str, Enum):
    """Speaker role in a conversation turn."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ChatMessage(AppBaseModel):
    """A single message in a conversation."""
    role: Role
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ChatRequest(AppBaseModel):
    """Payload sent by the client to the /chat endpoint."""
    session_id: str = Field(
        default="default",
        description="Unique session identifier for conversation continuity.",
    )
    message: str = Field(
        ...,
        min_length=1,
        max_length=4096,
        description="The user's message.",
    )
    model_name: str | None = Field(
        default=None,
        description="Ollama model tag to use (e.g. 'llama3.1:8b'). Falls back to the default model.",
    )


class ChatResponse(AppBaseModel):
    """Response returned by the synchronous /chat/sync endpoint."""
    session_id: str
    model_name: str
    response: str
    guardrail_triggered: bool = False
    guardrail_reason: str | None = None
    latency_ms: float = Field(description="Total response generation time in milliseconds.")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ──────────────────────────────────────────────
# Benchmark models
# ──────────────────────────────────────────────

class QualityScores(BaseModel):
    """Rubric-based quality scores assigned by the LLM-as-judge."""
    relevance: float = Field(ge=1, le=5, description="How relevant the answer is to the query.")
    accuracy: float = Field(ge=1, le=5, description="Factual correctness w.r.t. institute info.")
    completeness: float = Field(ge=1, le=5, description="How thoroughly the query was addressed.")
    tone: float = Field(ge=1, le=5, description="Professionalism and friendliness.")
    average: float = Field(ge=1, le=5, description="Mean of the four rubric scores.")
    rationale: str = Field(default="", description="Brief explanation by the judge model.")


class BenchmarkResult(AppBaseModel):
    """Result of running a single benchmark question against a single model."""
    question_id: int
    question: str
    difficulty: str = Field(description="easy | medium | complex")
    model_name: str
    response: str
    time_to_first_token_ms: float
    total_time_ms: float
    response_length_chars: int
    response_length_tokens: int | None = None
    ram_before_mb: float
    ram_after_mb: float
    ram_delta_mb: float
    vram_before_mb: float | None = None
    vram_after_mb: float | None = None
    vram_delta_mb: float | None = None
    quality_scores: QualityScores | None = None
    error: str | None = None


class BenchmarkReport(BaseModel):
    """Aggregate benchmark report across all models and questions."""
    run_id: str
    started_at: datetime
    completed_at: datetime | None = None
    system_info: SystemInfo | None = None
    models_tested: list[str]
    total_questions: int
    results: list[BenchmarkResult] = []
    summary: dict[str, Any] = Field(
        default_factory=dict,
        description="Per-model aggregate statistics.",
    )


# ──────────────────────────────────────────────
# System information
# ──────────────────────────────────────────────

class GpuInfo(BaseModel):
    """Details about a single GPU."""
    id: int
    name: str
    total_memory_mb: float
    used_memory_mb: float
    free_memory_mb: float
    load_percent: float
    temperature_c: float | None = None


class SystemInfo(BaseModel):
    """Snapshot of the host machine's hardware and OS."""
    os: str
    os_version: str
    cpu: str
    cpu_cores_physical: int
    cpu_cores_logical: int
    ram_total_mb: float
    ram_available_mb: float
    ram_used_percent: float
    gpus: list[GpuInfo] = []
    python_version: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ──────────────────────────────────────────────
# Explainability models
# ──────────────────────────────────────────────

class GuardrailTrace(BaseModel):
    """Detailed trace of the guardrail decision for a query."""
    query: str
    keyword_match_found: bool
    blocked_keyword_found: bool
    llm_classifier_result: str | None = None
    decision: str = Field(description="'accepted' or 'rejected'")
    reason: str = ""


class ExplainabilityTrace(AppBaseModel):
    """Full explainability trace for a single query–response cycle."""
    query: str
    model_name: str
    system_prompt_used: str
    conversation_history: list[dict[str, str]]
    full_prompt_chain: str
    guardrail_trace: GuardrailTrace
    response: str
    decision_points: list[str] = Field(
        default_factory=list,
        description="Key decision factors identified in the response.",
    )
    system_prompt_influence: list[str] = Field(
        default_factory=list,
        description="Sections of the system prompt that influenced the response.",
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class BiasTestCase(BaseModel):
    """A single bias-test pair: same question with demographic variation."""
    original_query: str
    variant_query: str
    variant_label: str
    original_response: str = ""
    variant_response: str = ""
    similarity_score: float | None = None
    flag: str = ""


class ExplainabilityReport(AppBaseModel):
    """Complete explainability & responsible-AI report."""
    model_name: str
    traces: list[ExplainabilityTrace] = []
    bias_tests: list[BiasTestCase] = []
    generated_at: datetime = Field(default_factory=datetime.utcnow)
