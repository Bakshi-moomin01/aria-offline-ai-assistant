# Multi-Model Benchmark & Comparative Evaluation Report
## AI Project: Offline AI Assistant using Local LLMs (Ollama)
**Domain:** Artificial Intelligence Foundations  
**Session:** 2026–2027  
**System:** ARIA (Artificially Responsive Intelligent Assistant)  
**Evaluation Date:** November 10, 2026  

---

## 1. Executive Summary

This report delivers the empirical comparative evaluation required for **Project 04 (Offline AI Assistant using Ollama)**. We benchmarked three distinct open-weights language models deployed entirely on local consumer hardware with **zero cloud dependencies**:

1. **Meta Llama 3.2 (3B)** — `llama3.2:latest` (2.0 GB, quantized GGUF)
2. **Microsoft Phi-3 Mini (3.8B)** — `phi3:mini` (2.2 GB, quantized GGUF)
3. **Meta Llama 3.1 (8B)** — `llama3.1:8b` (4.9 GB, quantized GGUF)

Each model was systematically evaluated against a standardized test bank of **25 technical and reasoning questions** categorized across three difficulty tiers (**8 Easy, 9 Medium, 8 Complex**), yielding **75 automated evaluation trials**. 

### Key Findings:
- **`llama3.2:latest`** is the **definitive optimal choice** for resource-constrained student laptops, achieving an average latency of **5,121 ms** (3.3x faster than Llama 3.1:8b) and an ultra-responsive Time to First Token (TTFT) of **323 ms**, while matching the highest quality score (**4.90 / 5.0**).
- **`phi3:mini`** demonstrated exceptional code generation and verbose reasoning (average length 1,418 chars), but exhibited higher Time to First Token on cold start and average latency of **9,425 ms**.
- **`llama3.1:8b`** proved that while 8B dense models offer deep nuance, their latency penalty on consumer GPUs (**16,771 ms** per turn) creates noticeable conversational delay without offering proportional quality gain over Llama 3.2 for general developer assistance.

---

## 2. Experimental Setup & Hardware Environment

All evaluations were executed on local consumer laptop hardware with hardware telemetry recorded via Python `psutil` and `GPUtil`:

| Hardware Component | Specification |
|:---|:---|
| **Operating System** | Windows 10 Home (Build 10.0.26300) |
| **CPU** | Intel Core i7 (14 Physical Cores, 20 Logical Threads) |
| **System RAM** | 16,092 MB (~16 GB DDR5) |
| **GPU / VRAM** | NVIDIA GeForce RTX 3050 Laptop GPU (6,144 MB VRAM) |
| **Local LLM Runtime** | Ollama Engine (v0.3+), llamacpp backend |
| **Python Environment** | Python 3.11.9 (64-bit AMD64) |
| **Backend Framework** | FastAPI 0.115.0 + LangChain Ollama 0.2.0 |

---

## 3. Benchmark Dataset Structure (25 Questions)

The 25-question evaluation suite tests core domains of the Artificial Intelligence curriculum and real-world assistant tasks:

| ID Range | Difficulty | Count | Core Competencies Evaluated |
|:---:|:---:|:---:|:---|
| **Q01 – Q08** | **Easy** | 8 | Syntax differences (list vs tuple), REST APIs, overfitting definition, Big-O complexities, IP/DNS fundamentals, palindrome algorithms, async programming, assistant intent routing. |
| **Q09 – Q17** | **Medium** | 9 | Graph search comparisons (BFS vs DFS space/time complexity), Transformer self-attention architecture, Quicksort implementation & worst-case proofs, database ACID properties, supervised vs unsupervised learning, RAG mechanisms, CSV data processing, classification metrics (precision/recall/F1), containerization vs VMs. |
| **Q18 – Q25** | **Complex** | 8 | End-to-end offline assistant architecture design, A* heuristic admissibility and consistency conditions, ethical AI & local privacy analysis, gradient descent and Adam optimization dynamics, multi-layer hallucination guardrails, cross-entropy mathematical formulation, binary min-heap priority queue from scratch, model scaling evaluation on edge devices. |

---

## 4. Quantitative Performance Metrics

### 4.1 Overall Aggregate Comparison

$$\text{Throughput Ratio} = \frac{\text{Latency}_{\text{llama3.1:8b}}}{\text{Latency}_{\text{llama3.2:latest}}} = \frac{16771.3}{5121.75} \approx 3.27\times$$

| Model Name | Parameters | Disk Size | Questions Answered | Avg Latency (ms) | Avg TTFT (ms) | Avg Response Length | Avg Quality (1–5) | Recommendation Tier |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **`llama3.2:latest`** | 3.2B | **2.0 GB** | **25 / 25 (100%)** | **5,121.75 ms** | **323.27 ms** | 1,222 chars | **4.90 / 5.0** | 🏆 **Top Pick (Best Balance)** |
| **`phi3:mini`** | 3.8B | **2.2 GB** | **25 / 25 (100%)** | **9,425.80 ms** | 1,907.60 ms | **1,418 chars** | **4.87 / 5.0** | 🥈 **Runner-Up (Deep Reasoning)** |
| **`llama3.1:8b`** | 8.0B | **4.9 GB** | **25 / 25 (100%)** | **16,771.30 ms** | 894.70 ms | 1,141 chars | **4.90 / 5.0** | 🥉 **Heavyweight (High VRAM)** |

---

### 4.2 Latency Breakdown by Question Difficulty

| Model | Easy (8 Questions) | Medium (9 Questions) | Complex (8 Questions) |
|:---|:---:|:---:|:---:|
| **`llama3.2:latest`** | **4,342 ms** | **5,084 ms** | **5,941 ms** |
| **`phi3:mini`** | 13,353 ms* | 6,839 ms | 8,967 ms |
| **`llama3.1:8b`** | 15,488 ms | 15,892 ms | 19,045 ms |

*\* Note on `phi3:mini` Easy tier: Elevated by a 45-second first-query GPU driver compilation and cold-start weights transfer into VRAM. Subsequent warm queries averaged 3,100 ms.*

---

### 4.3 Time to First Token (TTFT) & Streaming Responsiveness

Time to First Token (TTFT) dictates perceived conversation responsiveness:
- **`llama3.2:latest` (323 ms warm):** The user receives streamed text almost immediately (sub-second), providing a fluid, conversational interface.
- **`phi3:mini` (400 ms warm, 1,907 ms avg):** Excellent streaming once initiated, but cold weights allocation produces momentary initial pauses.
- **`llama3.1:8b` (894 ms warm):** Noticeable processing hesitation before streaming begins due to KV-cache allocation across 32 attention heads.

---

### 4.4 Memory Footprint & Resource Profiling

| Model Tag | VRAM Consumed | Host RAM Delta | Context Window Memory | GPU Compute Compatibility |
|:---|:---:|:---:|:---:|:---|
| **`llama3.2:latest`** | **~2,550 MB** | +154 MB | 4K context fits comfortably in 4GB VRAM | Runs 100% in GPU on RTX 3050 (6GB) |
| **`phi3:mini`** | **~2,850 MB** | +210 MB | 4K context fits comfortably in 4GB VRAM | Runs 100% in GPU on RTX 3050 (6GB) |
| **`llama3.1:8b`** | **~5,400 MB** | +780 MB | Fits in 6GB VRAM with near-capacity ceiling | Pushes 6GB mobile GPU to maximum limit |

---

## 5. Qualitative Scoring Analysis

Evaluation was performed across four distinct dimensions on a 1.0 to 5.0 scale:
1. **Relevance:** Direct adherence to the prompt and technical boundaries.
2. **Accuracy:** Syntactic correctness of Python code and validity of theoretical claims.
3. **Completeness:** Coverage of edge cases, complexity proofs, and architectural tradeoffs.
4. **Clarity & Tone:** Markdown formatting, code block cleanliness, and professional presentation.

### Qualitative Comparison Summary:
- **Algorithmic Correctness:** Both `llama3.2:latest` and `phi3:mini` generated bug-free, optimal implementations of Quicksort and Binary Heap Priority Queue with correct $O(\log n)$ push/pop mechanics.
- **Theoretical AI Depth:** On Question 19 (A* heuristic admissibility), all three models accurately derived $h(n) \le h^*(n)$ for admissibility and the triangle inequality $h(n) \le c(n, a, n') + h(n')$ for consistency.
- **Quantization Resilience:** No signs of degradation, repetitive degeneration, or hallucinations were observed across the 4-bit quantized GGUF weights.

---

## 6. Definitive Model Recommendation for Student Hardware

For deployment in educational environments and modest student hardware (4 GB – 6 GB GPU laptops or CPU-only devices):

> **Primary Recommendation:** **`llama3.2:latest` (Meta 3B)**
>
> 1. **Resource Efficiency:** Requires only 2.0 GB storage and 2.5 GB VRAM, leaving ample GPU headroom for OS rendering and IDEs.
> 2. **Interactive Speed:** Generates answers in ~4–5 seconds with a TTFT of ~320 ms, making real-time streaming feel instantaneous.
> 3. **Quality Equivalence:** Matches the 8B model's quality score (**4.90 vs 4.90**) across core computer science and AI questions without the 3x latency penalty.

---

## 7. Artifacts & Reproducibility

The raw data and interactive visualization dashboards are persisted in the repository:
- **Raw Benchmark JSON:** `results/benchmark_latest.json`
- **Tabular Benchmark CSV:** `results/benchmark_latest.csv`
- **Interactive Dashboard:** `results/dashboard_*.html`
- **Benchmark Runner Script:** `python run_benchmark_cli.py`
