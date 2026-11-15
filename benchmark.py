"""
benchmark.py — Multi-model benchmark harness for EduBot.

Runs 25 domain-specific tutoring-centre questions against each configured
Ollama model, measures latency / memory / quality, and produces comparison
charts plus JSON/CSV reports.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import config
from llm_engine import LLMEngine
from models import BenchmarkReport, BenchmarkResult, QualityScores
from utils import get_ram_usage_mb, get_system_info, get_vram_usage_mb, timer, ensure_dir

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Benchmark question bank (25 questions)
# ──────────────────────────────────────────────

BENCHMARK_QUESTIONS: list[dict[str, Any]] = [
    # --- Easy (8 questions: Definitions, Basic Coding, Fundamentals) ---
    {"id": 1, "question": "Explain the difference between a list and a tuple in Python.", "difficulty": "easy"},
    {"id": 2, "question": "What is an API and how does REST architecture work?", "difficulty": "easy"},
    {"id": 3, "question": "Define overfitting in machine learning and how to prevent it.", "difficulty": "easy"},
    {"id": 4, "question": "What is the Big-O time complexity of binary search vs linear search?", "difficulty": "easy"},
    {"id": 5, "question": "Explain what an IP address and DNS do in simple terms.", "difficulty": "easy"},
    {"id": 6, "question": "Write a Python function to check if a string is a palindrome.", "difficulty": "easy"},
    {"id": 7, "question": "What is the difference between synchronous and asynchronous programming?", "difficulty": "easy"},
    {"id": 8, "question": "Hello! What can you help me with as an AI assistant?", "difficulty": "easy"},

    # --- Medium (9 questions: Algorithms, Architecture, Practical Implementation) ---
    {"id": 9, "question": "Explain how Breadth-First Search (BFS) and Depth-First Search (DFS) differ in terms of data structures, time complexity, and space complexity.", "difficulty": "medium"},
    {"id": 10, "question": "Describe the transformer architecture and the self-attention mechanism in modern Large Language Models.", "difficulty": "medium"},
    {"id": 11, "question": "Write a Python implementation of Quicksort with an explanation of its average and worst-case performance.", "difficulty": "medium"},
    {"id": 12, "question": "What are database transactions and ACID properties? Provide a practical example of why they matter.", "difficulty": "medium"},
    {"id": 13, "question": "Compare supervised learning, unsupervised learning, and reinforcement learning with real-world applications for each.", "difficulty": "medium"},
    {"id": 14, "question": "Explain how Retrieval-Augmented Generation (RAG) works and how it mitigates hallucinations in LLMs.", "difficulty": "medium"},
    {"id": 15, "question": "Write a Python script that reads a CSV file, calculates summary statistics (mean, median), and handles missing values.", "difficulty": "medium"},
    {"id": 16, "question": "What is the difference between precision, recall, and F1-score? When is recall preferred over precision?", "difficulty": "medium"},
    {"id": 17, "question": "Explain the purpose of Docker containers and how containerization differs from virtual machines.", "difficulty": "medium"},

    # --- Complex (8 questions: System Design, Advanced AI Theory, Analysis) ---
    {"id": 18, "question": "Design an end-to-end architecture for a low-latency offline AI assistant: detail model quantization (GGUF), memory management, streaming inference, and fallback strategies.", "difficulty": "complex"},
    {"id": 19, "question": "Compare A* search algorithm with Dijkstra's algorithm. Formulate the conditions required for an admissible and consistent heuristic.", "difficulty": "complex"},
    {"id": 20, "question": "Analyze ethical concerns, algorithmic bias, and privacy implications in deploying local conversational AI agents compared to centralized cloud APIs.", "difficulty": "complex"},
    {"id": 21, "question": "Explain gradient descent, stochastic gradient descent (SGD), and Adam optimizer. Contrast their convergence properties and learning rate schedules.", "difficulty": "complex"},
    {"id": 22, "question": "Given a scenario where an offline LLM produces inaccurate technical instructions, design a multi-layer guardrail and verification pipeline to detect and mitigate false outputs.", "difficulty": "complex"},
    {"id": 23, "question": "Explain the mathematical formulation of cross-entropy loss in classification problems and why it is paired with softmax activation.", "difficulty": "complex"},
    {"id": 24, "question": "Write a complete Python class implementing a Priority Queue using a binary min-heap from scratch without external libraries.", "difficulty": "complex"},
    {"id": 25, "question": "Synthesize a comprehensive evaluation strategy for comparing small language models (<8B parameters) on resource-constrained consumer hardware.", "difficulty": "complex"},
]

# ──────────────────────────────────────────────
# LLM-as-judge rubric prompt
# ──────────────────────────────────────────────

_JUDGE_PROMPT_TEMPLATE = """You are an expert AI evaluator assessing responses from an offline AI assistant.

Rate the following response on four criteria using a 1–5 scale:
1. **Relevance** (1-5): Does the response directly address the user's question?
2. **Accuracy** (1-5): Is the information or code technically sound, factually correct, and logically valid?
3. **Completeness** (1-5): Does the response cover all required aspects thoroughly?
4. **Tone** (1-5): Is the response professional, clear, and well-structured?

### User Question
{question}

### Assistant Response
{response}

### Your Rating
Reply in EXACTLY this JSON format (no markdown, no extra text):
{{"relevance": <int>, "accuracy": <int>, "completeness": <int>, "tone": <int>, "rationale": "<one-sentence explanation>"}}
"""


# ──────────────────────────────────────────────
# Benchmark runner
# ──────────────────────────────────────────────

class BenchmarkRunner:
    """Orchestrates benchmark execution across multiple models."""

    def __init__(self, engine: LLMEngine):
        self._engine = engine

    async def run(
        self,
        models: list[str] | None = None,
        question_ids: list[int] | None = None,
    ) -> BenchmarkReport:
        """Execute the benchmark suite.

        Args:
            models:       List of model tags to test. Defaults to ``config.BENCHMARK_MODELS``.
            question_ids: Optional subset of question IDs. ``None`` = all 25.

        Returns:
            A fully-populated ``BenchmarkReport``.
        """
        models = models or config.BENCHMARK_MODELS
        questions = (
            [q for q in BENCHMARK_QUESTIONS if q["id"] in question_ids]
            if question_ids
            else BENCHMARK_QUESTIONS
        )

        run_id = uuid.uuid4().hex[:12]
        report = BenchmarkReport(
            run_id=run_id,
            started_at=datetime.utcnow(),
            models_tested=models,
            total_questions=len(questions),
            system_info=get_system_info(),
        )

        logger.info(
            "Benchmark %s: %d models × %d questions",
            run_id, len(models), len(questions),
        )

        # Filter to models actually available
        available = await self._engine.get_available_model_names()
        skipped = [m for m in models if m not in available]
        if skipped:
            logger.warning("Models not available (skipped): %s", skipped)
        active_models = [m for m in models if m in available]

        for model_name in active_models:
            logger.info("Benchmarking model: %s", model_name)
            for q in questions:
                result = await self._run_single(model_name, q)
                report.results.append(result)

        # LLM-as-judge quality scoring
        judge_model = self._pick_judge(active_models)
        if judge_model:
            await self._score_quality(report, judge_model)

        report.completed_at = datetime.utcnow()
        report.summary = self._compute_summary(report)

        # Persist results
        self._save_results(report)

        return report

    async def run_benchmark(
        self,
        models: list[str] | None = None,
        limit_questions: int | None = None,
    ) -> BenchmarkReport:
        """Alias for run() with optional question count limit."""
        q_ids = None
        if limit_questions:
            q_ids = [q["id"] for q in BENCHMARK_QUESTIONS[:limit_questions]]
        return await self.run(models=models, question_ids=q_ids)

    # ── Single question run ──────────────────

    async def _run_single(self, model_name: str, question: dict) -> BenchmarkResult:
        """Run one question against one model and collect metrics.

        Args:
            model_name: Ollama model tag.
            question:   Question dict with keys ``id``, ``question``, ``difficulty``.

        Returns:
            Populated ``BenchmarkResult``.
        """
        ram_before = get_ram_usage_mb()
        vram_before = get_vram_usage_mb()

        response_text = ""
        ttft: float = 0.0  # time to first token (ms)
        error: str | None = None

        try:
            with timer() as total_t:
                first_token_time: float | None = None
                start = time.perf_counter()

                async for chunk in self._engine.generate_stream(
                    prompt=question["question"],
                    model_name=model_name,
                    system_prompt=config.SYSTEM_PROMPT,
                ):
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    response_text += chunk

            ttft = ((first_token_time - start) * 1000) if first_token_time else total_t.elapsed_ms

        except Exception as exc:
            error = str(exc)
            logger.error("Benchmark error (%s, Q%d): %s", model_name, question["id"], exc)

        ram_after = get_ram_usage_mb()
        vram_after = get_vram_usage_mb()

        return BenchmarkResult(
            question_id=question["id"],
            question=question["question"],
            difficulty=question["difficulty"],
            model_name=model_name,
            response=response_text,
            time_to_first_token_ms=round(ttft, 2),
            total_time_ms=round(total_t.elapsed_ms, 2),
            response_length_chars=len(response_text),
            ram_before_mb=round(ram_before, 2),
            ram_after_mb=round(ram_after, 2),
            ram_delta_mb=round(ram_after - ram_before, 2),
            vram_before_mb=round(vram_before, 2) if vram_before is not None else None,
            vram_after_mb=round(vram_after, 2) if vram_after is not None else None,
            vram_delta_mb=(
                round(vram_after - vram_before, 2)
                if vram_before is not None and vram_after is not None
                else None
            ),
            error=error,
        )

    # ── LLM-as-judge scoring ────────────────

    def _pick_judge(self, active_models: list[str]) -> str | None:
        """Choose the judge model (prefer config.JUDGE_MODEL if available)."""
        if config.JUDGE_MODEL in active_models:
            return config.JUDGE_MODEL
        return active_models[0] if active_models else None

    async def _score_quality(self, report: BenchmarkReport, judge_model: str) -> None:
        """Use *judge_model* to rate every result in the report.

        Args:
            report:      The in-progress benchmark report.
            judge_model: Model tag to use as the judge.
        """
        logger.info("Quality scoring with judge model: %s", judge_model)
        for result in report.results:
            if result.error or not result.response:
                continue
            try:
                prompt = _JUDGE_PROMPT_TEMPLATE.format(
                    question=result.question,
                    response=result.response,
                )
                raw = await self._engine.generate(
                    prompt=prompt,
                    model_name=judge_model,
                    system_prompt="You are a strict JSON evaluator. Respond with ONLY valid JSON.",
                )
                scores = self._parse_judge_response(raw)
                result.quality_scores = scores
            except Exception as exc:
                logger.warning(
                    "Judge scoring failed for Q%d/%s: %s",
                    result.question_id, result.model_name, exc,
                )

    @staticmethod
    def _parse_judge_response(raw: str) -> QualityScores:
        """Parse the judge model's JSON response into a ``QualityScores`` object."""
        # Strip markdown fences if present
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(
                l for l in lines if not l.strip().startswith("```")
            )

        data = json.loads(text)
        relevance = max(1, min(5, int(data.get("relevance", 3))))
        accuracy = max(1, min(5, int(data.get("accuracy", 3))))
        completeness = max(1, min(5, int(data.get("completeness", 3))))
        tone = max(1, min(5, int(data.get("tone", 3))))
        avg = round((relevance + accuracy + completeness + tone) / 4, 2)

        return QualityScores(
            relevance=relevance,
            accuracy=accuracy,
            completeness=completeness,
            tone=tone,
            average=avg,
            rationale=data.get("rationale", ""),
        )

    # ── Summary computation ──────────────────

    @staticmethod
    def _compute_summary(report: BenchmarkReport) -> dict[str, Any]:
        """Compute per-model aggregate statistics.

        Returns:
            Nested dict keyed by model name.
        """
        summary: dict[str, Any] = {}
        for model_name in report.models_tested:
            results = [r for r in report.results if r.model_name == model_name and r.error is None]
            if not results:
                summary[model_name] = {"note": "No successful results."}
                continue

            total_times = [r.total_time_ms for r in results]
            ttft_values = [r.time_to_first_token_ms for r in results]
            lengths = [r.response_length_chars for r in results]
            quality_avgs = [
                r.quality_scores.average for r in results if r.quality_scores
            ]

            summary[model_name] = {
                "questions_answered": len(results),
                "avg_total_time_ms": round(sum(total_times) / len(total_times), 2),
                "avg_ttft_ms": round(sum(ttft_values) / len(ttft_values), 2),
                "avg_response_length": round(sum(lengths) / len(lengths), 0),
                "avg_quality_score": (
                    round(sum(quality_avgs) / len(quality_avgs), 2)
                    if quality_avgs else None
                ),
                "min_time_ms": round(min(total_times), 2),
                "max_time_ms": round(max(total_times), 2),
            }

        return summary

    # ── Persistence ──────────────────────────

    def _save_results(self, report: BenchmarkReport) -> None:
        """Save the report to JSON and CSV files, and generate Plotly charts.

        Args:
            report: The completed benchmark report.
        """
        out_dir = ensure_dir(config.RESULTS_DIR)

        # JSON
        json_path = out_dir / f"benchmark_{report.run_id}.json"
        with open(json_path, "w", encoding="utf-8") as fh:
            fh.write(report.model_dump_json(indent=2))
        logger.info("Saved benchmark JSON: %s", json_path)

        # CSV
        csv_path = out_dir / f"benchmark_{report.run_id}.csv"
        rows: list[dict] = []
        for r in report.results:
            row = r.model_dump(exclude={"quality_scores"})
            if r.quality_scores:
                for k, v in r.quality_scores.model_dump().items():
                    row[f"quality_{k}"] = v
            rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        logger.info("Saved benchmark CSV: %s", csv_path)

        # Charts
        self._generate_charts(report, out_dir)

    def _generate_charts(self, report: BenchmarkReport, out_dir: Path) -> None:
        """Create Plotly comparison charts and save as HTML.

        Args:
            report:  The completed benchmark report.
            out_dir: Directory to write the HTML files.
        """
        results_ok = [r for r in report.results if r.error is None]
        if not results_ok:
            return

        df = pd.DataFrame([r.model_dump() for r in results_ok])

        # --- Chart 1: Average latency by model ---
        avg_latency = df.groupby("model_name")["total_time_ms"].mean().reset_index()
        fig1 = px.bar(
            avg_latency,
            x="model_name", y="total_time_ms",
            title="Average Response Latency by Model",
            labels={"total_time_ms": "Avg Latency (ms)", "model_name": "Model"},
            color="model_name",
        )
        fig1.write_html(str(out_dir / f"chart_latency_{report.run_id}.html"))

        # --- Chart 2: Latency distribution (box plot) ---
        fig2 = px.box(
            df,
            x="model_name", y="total_time_ms",
            title="Response Latency Distribution",
            labels={"total_time_ms": "Latency (ms)", "model_name": "Model"},
            color="model_name",
        )
        fig2.write_html(str(out_dir / f"chart_latency_box_{report.run_id}.html"))

        # --- Chart 3: Quality scores comparison ---
        quality_rows = []
        for r in results_ok:
            if r.quality_scores:
                quality_rows.append({
                    "model_name": r.model_name,
                    "relevance": r.quality_scores.relevance,
                    "accuracy": r.quality_scores.accuracy,
                    "completeness": r.quality_scores.completeness,
                    "tone": r.quality_scores.tone,
                    "average": r.quality_scores.average,
                })
        if quality_rows:
            qdf = pd.DataFrame(quality_rows)
            avg_q = qdf.groupby("model_name")[
                ["relevance", "accuracy", "completeness", "tone"]
            ].mean().reset_index()

            fig3 = go.Figure()
            for metric in ["relevance", "accuracy", "completeness", "tone"]:
                fig3.add_trace(go.Bar(
                    name=metric.capitalize(),
                    x=avg_q["model_name"],
                    y=avg_q[metric],
                ))
            fig3.update_layout(
                barmode="group",
                title="Average Quality Scores by Model",
                yaxis_title="Score (1-5)",
                xaxis_title="Model",
            )
            fig3.write_html(str(out_dir / f"chart_quality_{report.run_id}.html"))

        # --- Chart 4: Response length comparison ---
        avg_len = df.groupby("model_name")["response_length_chars"].mean().reset_index()
        fig4 = px.bar(
            avg_len,
            x="model_name", y="response_length_chars",
            title="Average Response Length by Model",
            labels={"response_length_chars": "Avg Length (chars)", "model_name": "Model"},
            color="model_name",
        )
        fig4.write_html(str(out_dir / f"chart_length_{report.run_id}.html"))

        # --- Chart 5: Latency by difficulty ---
        fig5 = px.bar(
            df.groupby(["model_name", "difficulty"])["total_time_ms"].mean().reset_index(),
            x="model_name", y="total_time_ms", color="difficulty",
            barmode="group",
            title="Average Latency by Model & Question Difficulty",
            labels={"total_time_ms": "Avg Latency (ms)", "model_name": "Model"},
        )
        fig5.write_html(str(out_dir / f"chart_difficulty_{report.run_id}.html"))

        # --- Combined dashboard ---
        dashboard = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Avg Latency", "Latency Distribution",
                "Avg Response Length", "TTFT vs Total Time",
            ),
        )
        for i, model in enumerate(df["model_name"].unique()):
            mdf = df[df["model_name"] == model]
            dashboard.add_trace(
                go.Bar(x=[model], y=[mdf["total_time_ms"].mean()], name=model, showlegend=False),
                row=1, col=1,
            )
            dashboard.add_trace(
                go.Box(y=mdf["total_time_ms"], name=model, showlegend=False),
                row=1, col=2,
            )
            dashboard.add_trace(
                go.Bar(x=[model], y=[mdf["response_length_chars"].mean()], name=model, showlegend=False),
                row=2, col=1,
            )
            dashboard.add_trace(
                go.Scatter(
                    x=mdf["time_to_first_token_ms"], y=mdf["total_time_ms"],
                    mode="markers", name=model,
                ),
                row=2, col=2,
            )
        dashboard.update_layout(
            height=800,
            title_text=f"Benchmark Dashboard — Run {report.run_id}",
        )
        dashboard.write_html(str(out_dir / f"dashboard_{report.run_id}.html"))

        logger.info("Charts saved to %s", out_dir)


# ──────────────────────────────────────────────
# Convenience: load latest report
# ──────────────────────────────────────────────

def load_latest_report() -> BenchmarkReport | None:
    """Load the most recent benchmark report from disk.

    Returns:
        The deserialized ``BenchmarkReport``, or ``None`` if none exist.
    """
    results_dir = config.RESULTS_DIR
    json_files = sorted(results_dir.glob("benchmark_*.json"), reverse=True)
    if not json_files:
        return None
    with open(json_files[0], "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return BenchmarkReport(**data)


def get_report_html_path() -> Path | None:
    """Return the path to the latest benchmark dashboard HTML, or ``None``."""
    results_dir = config.RESULTS_DIR
    html_files = sorted(results_dir.glob("dashboard_*.html"), reverse=True)
    return html_files[0] if html_files else None


# Backward-compatible alias
BenchmarkHarness = BenchmarkRunner

