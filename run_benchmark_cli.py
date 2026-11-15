"""
run_benchmark_cli.py — Comprehensive 3-Model 25-Question Benchmark Runner
Evaluates llama3.2:latest, phi3:mini, and llama3.1:8b across 25 standard AI & Computer Science questions.
"""

import asyncio
import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

# Setup UTF-8 and path
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from benchmark import BENCHMARK_QUESTIONS, BenchmarkRunner, _JUDGE_PROMPT_TEMPLATE
from llm_engine import LLMEngine
from models import BenchmarkReport, BenchmarkResult, QualityScores
from utils import get_ram_usage_mb, get_system_info, get_vram_usage_mb, timer, ensure_dir
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("benchmark_cli")

MODELS_TO_BENCHMARK = [
    "llama3.2:latest",
    "phi3:mini",
    "llama3.1:8b",
]

def score_response_heuristic(question: str, response: str, difficulty: str) -> QualityScores:
    """Deterministic rubric scoring for response quality."""
    resp_len = len(response.strip())
    q_words = set(question.lower().split())
    r_words = set(response.lower().split())
    overlap = len(q_words & r_words) / max(len(q_words), 1)

    # Relevance (1-5)
    if resp_len < 20:
        relevance = 1.0
    elif overlap > 0.4 or resp_len > 150:
        relevance = 5.0
    elif overlap > 0.2:
        relevance = 4.0
    else:
        relevance = 3.0

    # Accuracy (1-5)
    # Check for substantive explanations or code blocks where expected
    has_code = "```" in response or "def " in response or "class " in response
    if "code" in question.lower() or "python" in question.lower() or "function" in question.lower():
        accuracy = 5.0 if has_code else 3.5
    else:
        accuracy = 4.5 if resp_len > 100 else 3.0

    # Completeness (1-5)
    if difficulty == "easy":
        completeness = 5.0 if resp_len > 120 else (4.0 if resp_len > 60 else 2.5)
    elif difficulty == "medium":
        completeness = 5.0 if resp_len > 300 else (4.0 if resp_len > 180 else 3.0)
    else: # complex
        completeness = 5.0 if resp_len > 500 else (4.0 if resp_len > 250 else 3.0)

    # Tone (1-5)
    tone = 5.0 if ("**" in response or "\n-" in response or "\n1." in response or "\n\n" in response) else 4.0

    avg = round((relevance + accuracy + completeness + tone) / 4.0, 2)
    return QualityScores(
        relevance=relevance,
        accuracy=accuracy,
        completeness=completeness,
        tone=tone,
        average=avg,
        rationale=f"Evaluated on {difficulty} technical rubric (Length: {resp_len} chars, Code: {has_code})."
    )

async def run_full_evaluation():
    engine = LLMEngine(base_url=config.OLLAMA_BASE_URL)
    available = await engine.list_models()
    avail_names = [m.get("name") for m in available]
    logger.info("Locally available Ollama models: %s", avail_names)

    active_models = []
    for m in MODELS_TO_BENCHMARK:
        # Match model name or tag
        matched = any(m == a or m.split(":")[0] == a.split(":")[0] for a in avail_names)
        if matched:
            active_models.append(m)
        else:
            logger.warning("Model %s not found in Ollama, skipping.", m)

    logger.info("Benchmarking active models: %s", active_models)
    run_id = f"eval_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    
    report = BenchmarkReport(
        run_id=run_id,
        started_at=datetime.utcnow(),
        models_tested=active_models,
        total_questions=len(BENCHMARK_QUESTIONS),
        system_info=get_system_info(),
    )

    total_runs = len(active_models) * len(BENCHMARK_QUESTIONS)
    current_idx = 0

    print("=" * 70)
    print(f"🚀 STARTING AI PROJECT BENCHMARK: {len(active_models)} Models × {len(BENCHMARK_QUESTIONS)} Questions ({total_runs} Total)")
    print(f"Models: {', '.join(active_models)}")
    print("=" * 70)

    for model_name in active_models:
        print(f"\n▶ Benchmarking Model: {model_name}")
        print("-" * 70)
        
        for q in BENCHMARK_QUESTIONS:
            current_idx += 1
            q_id = q["id"]
            prompt = q["question"]
            diff = q["difficulty"]

            ram_before = get_ram_usage_mb()
            vram_before = get_vram_usage_mb()

            start_t = time.perf_counter()
            first_token_t = None
            response_text = ""
            error = None

            eval_prompt = f"{prompt}\n\nPlease provide a clear, concise, and accurate answer (under 150 words)."

            try:
                async for chunk in engine.generate_stream(
                    prompt=eval_prompt,
                    model_name=model_name,
                    system_prompt=config.SYSTEM_PROMPT,
                ):
                    if first_token_t is None:
                        first_token_t = time.perf_counter()
                    response_text += chunk
            except Exception as exc:
                error = str(exc)
                logger.error("Error generating for %s Q%d: %s", model_name, q_id, exc)

            total_ms = (time.perf_counter() - start_t) * 1000
            ttft_ms = ((first_token_t - start_t) * 1000) if first_token_t else total_ms

            ram_after = get_ram_usage_mb()
            vram_after = get_vram_usage_mb()

            quality = score_response_heuristic(prompt, response_text, diff)

            res = BenchmarkResult(
                question_id=q_id,
                question=prompt,
                difficulty=diff,
                model_name=model_name,
                response=response_text,
                time_to_first_token_ms=round(ttft_ms, 2),
                total_time_ms=round(total_ms, 2),
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
                quality_scores=quality,
                error=error,
            )
            report.results.append(res)

            status_icon = "✅" if not error else "❌"
            print(f"[{current_idx:2d}/{total_runs:2d}] {status_icon} Q{q_id:02d} ({diff.upper():<7}) | Latency: {total_ms:6.1f}ms | TTFT: {ttft_ms:5.1f}ms | Score: {quality.average:.2f}/5.0 | {prompt[:40]}...", flush=True)

    report.completed_at = datetime.utcnow()
    report.summary = BenchmarkRunner._compute_summary(report)

    # Save results
    out_dir = ensure_dir(config.RESULTS_DIR)
    json_path = out_dir / f"benchmark_{report.run_id}.json"
    latest_json_path = out_dir / "benchmark_latest.json"

    with open(json_path, "w", encoding="utf-8") as fh:
        fh.write(report.model_dump_json(indent=2))
    with open(latest_json_path, "w", encoding="utf-8") as fh:
        fh.write(report.model_dump_json(indent=2))

    # Save CSV
    csv_path = out_dir / f"benchmark_{report.run_id}.csv"
    latest_csv_path = out_dir / "benchmark_latest.csv"
    rows = []
    for r in report.results:
        row = r.model_dump(exclude={"quality_scores"})
        if r.quality_scores:
            for k, v in r.quality_scores.model_dump().items():
                row[f"quality_{k}"] = v
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    df.to_csv(latest_csv_path, index=False)

    # Generate charts
    runner = BenchmarkRunner(engine=engine)
    runner._generate_charts(report, out_dir)

    print("\n" + "=" * 70)
    print("📊 BENCHMARK COMPLETE — AGGREGATE SUMMARY:")
    print("=" * 70)
    for model_name, stats in report.summary.items():
        print(f"Model: {model_name}")
        print(f"  • Questions Answered : {stats.get('questions_answered')}")
        print(f"  • Avg Latency        : {stats.get('avg_total_time_ms')} ms")
        print(f"  • Avg TTFT           : {stats.get('avg_ttft_ms')} ms")
        print(f"  • Avg Quality Score  : {stats.get('avg_quality_score')} / 5.0")
        print(f"  • Avg Response Length: {stats.get('avg_response_length')} chars")
        print()

    print(f"✅ Saved raw JSON to: {json_path}")
    print(f"✅ Saved raw CSV  to: {csv_path}")
    print(f"✅ Interactive HTML dashboard generated in {out_dir}")
    print("=" * 70)
    return report

if __name__ == "__main__":
    asyncio.run(run_full_evaluation())
