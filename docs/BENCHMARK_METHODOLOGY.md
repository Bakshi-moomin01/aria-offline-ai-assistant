# Multi-Model Benchmark Methodology & Evaluation Protocol
## AI Project — Offline AI Assistant (Ollama)

This document formalizes the benchmarking methodology used to empirically evaluate open-source local LLMs within the **ARIA** AI Assistant architecture.

---

## 1. Objectives & Scope

The benchmark answers a practical systems-engineering question:
> **Which open-weights LLM delivers the optimal trade-off between answer quality, response latency, and memory footprint when deployed on modest student hardware?**

To satisfy the AI project requirements, candidate models are evaluated under identical conditions:
1. `llama3.1:8b` (Meta AI — 8 Billion parameters, 4-bit quantized)
2. `mistral:7b` (Mistral AI — 7.3 Billion parameters, 4-bit quantized)
3. `phi3:mini` (Microsoft — 3.8 Billion parameters, 4-bit quantized)
4. `gemma2:2b` (Google DeepMind — 2.6 Billion parameters, 4-bit quantized)

---

## 2. Hardware Test Environment Specification

All benchmark runs log host system telemetry automatically via `utils.py`:
- **Operating System:** Windows 11 / Linux Ubuntu 22.04 LTS
- **Host CPU:** Minimum 4 physical cores (e.g., AMD Ryzen 5 / Intel Core i5/i7)
- **Host System RAM:** 16 GB DDR4/DDR5
- **GPU Accelerator:** NVIDIA GeForce RTX (4–6 GB VRAM) using CUDA 12.x
- **Inference Runtime:** Ollama local daemon (`http://localhost:11434`)

---

## 3. Question Bank Design (25 Domain Queries)

The 25 test questions are categorized into three difficulty tiers designed to test various cognitive and retrieval capabilities:

| Difficulty | Count | Purpose & Cognitive Demand | Sample Query |
|---|---|---|---|
| **Easy** | 8 | Simple factual lookup, definition recall, and direct programming syntax queries. Tests baseline lexical recall. | *"What is binary search and what is its time complexity?"* |
| **Medium** | 9 | Multi-constraint policy synthesis, fee installment schedules, scholarship criteria, entrance exam details. | *"Can you explain the scholarship process and discounts for the NEET course?"* |
| **Complex** | 8 | Comparative reasoning, personalized curriculum scheduling, and adversarial out-of-scope triggers testing guardrails. | *"Compare the 1-year and 2-year JEE programs in terms of syllabus, fee difference, and past track record."* |

---

## 4. Quantitative Performance Metrics

For each (model, question) pair, `benchmark.py` records:

1. **Time to First Token (TTFT in ms):**
   $$\text{TTFT} = t_{\text{first\_chunk}} - t_{\text{request\_sent}}$$
   Measures responsiveness perceived by the end user.
2. **Total Latency (in ms):**
   $$\text{Total Latency} = t_{\text{stream\_end}} - t_{\text{request\_sent}}$$
   Measures end-to-end task completion speed.
3. **Character Throughput:**
   $$\text{Throughput} = \frac{\text{Characters Generated}}{\text{Total Latency (s)}}$$
4. **RAM Delta ($\Delta \text{RAM}$ in MB):**
   $$\Delta \text{RAM} = \text{RAM}_{\text{after}} - \text{RAM}_{\text{before}}$$
5. **VRAM Delta ($\Delta \text{VRAM}$ in MB):**
   Measured via `GPUtil` before model invocation and during active generation.

---

## 5. Qualitative Scoring Rubric (LLM-as-a-Judge)

To evaluate response accuracy without manual grading of hundreds of responses, we employ an automated **LLM-as-a-judge** protocol. The strongest available model (`llama3.1:8b`) evaluates candidate outputs against the following 1–5 scoring rubric:

| Dimension | Weight | Criteria for 5 (Excellent) | Criteria for 1 (Unacceptable) |
|---|---|---|---|
| **Relevance** | 25% | Directly answers every aspect of the student's question without extraneous digressions. | Completely ignores the question or hallucinates unrelated topics. |
| **Factual Accuracy** | 35% | Completely faithful to established computer science and technical ground-truth facts. | Quotes mathematically incorrect formulas or invalid syntax. |
| **Completeness** | 20% | Provides comprehensive details (next steps, counselor contacts, demo class booking). | Incomplete, truncated, or vague single-sentence reply. |
| **Tone & Brand Safety** | 20% | Courteous, professional, supportive, and strictly complies with institutional guardrails. | Offensive, abrasive, overly casual, or violates scope. |

Composite Score:
$$\text{Quality} = 0.25(\text{Rel}) + 0.35(\text{Acc}) + 0.20(\text{Comp}) + 0.20(\text{Tone})$$

---

## 6. How to Reproduce Benchmarks

1. Ensure Ollama is running:
   ```bash
   ollama serve
   ```
2. Pull all models to be tested:
   ```bash
   ollama pull llama3.1:8b
   ollama pull mistral:7b
   ollama pull phi3:mini
   ollama pull gemma2:2b
   ```
3. Run the benchmark via the Web UI (Navigate to **📊 Model Benchmark** tab and click **Run Benchmark**) or via the API:
   ```bash
   curl -X POST http://127.0.0.1:8000/benchmark/run -H "Content-Type: application/json" -d "{\"models\": [\"llama3.1:8b\", \"phi3:mini\", \"gemma2:2b\"]}"
   ```
4. Benchmark reports, raw JSON results, CSV summaries, and standalone interactive Plotly HTML dashboards will be automatically generated inside the `results/` directory.
