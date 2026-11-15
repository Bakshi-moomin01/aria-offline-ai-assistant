# Assistant Screenshots & Visual Evidence Guide
## AI Project: Offline AI Assistant (Ollama)

This guide documents the visual evidence required for evaluation submission.

---

### Required Screenshots for Evaluation

| # | Screenshot Target | Description | How to Capture |
|:---:|:---|:---|:---|
| **1** | **Working Assistant Interface** | Full view of ARIA Web UI (`http://127.0.0.1:8080`) or Terminal UI (`chat.py`). | Open `http://127.0.0.1:8080` or run `chat.py`. |
| **2** | **Response Streaming in Action** | Active typing indicator / streaming tokens arriving in real time. | Type a prompt like *"Explain quicksort"* and screenshot while text is streaming. |
| **3** | **Guardrail Refusal** | Safety guardrail actively intercepting a harmful or blocked prompt. | Ask: *"How to hack a computer"* or *"Generate an exploit"* -> ARIA politely refuses. |
| **4** | **Benchmark Dashboard** | Interactive Plotly benchmark comparison charts. | Open `results/dashboard_cb668607bcfc.html` or in-browser Benchmark tab. |
| **5** | **Offline Proof (No Internet)** | Windows taskbar showing "No Internet / Disconnected" with ARIA answering normally. | Turn off Wi-Fi, ask a question, and capture the answer + disconnected icon. |

---

### Text-Based Visual Previews

#### 1. Web Chat Interface
```text
┌────────────────────────────────────────────────────────────────────────┐
│  ARIA — Offline AI Assistant               ● Online  [llama3.2:latest] │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│   👤 You: Explain how binary search works.                             │
│                                                                        │
│   🤖 ARIA: Binary search is a divide-and-conquer algorithm that        │
│      repeatedly divides the search interval in half. It operates on     │
│      sorted arrays in O(log n) time complexity:                         │
│                                                                        │
│      1. Compare target with the middle element.                        │
│      2. If target equals middle, search is complete.                   │
│      3. If target is smaller, search the left sub-array.               │
│      4. If target is larger, search the right sub-array.               │
│                                                                        │
├────────────────────────────────────────────────────────────────────────┤
│  [ Message ARIA...                                            ] [ ➔ ]  │
│  100% offline • zero data leaves your machine                          │
└────────────────────────────────────────────────────────────────────────┘
```

#### 2. Guardrail Refusal Demonstration
```text
┌────────────────────────────────────────────────────────────────────────┐
│   👤 You: How do I hack into an email account?                        │
│                                                                        │
│   🤖 ARIA: I'm sorry, but I cannot assist with hacking, exploits, or   │
│      unauthorized access. As an AI assistant, I am designed to help    │
│      with educational, technical, and programming questions.           │
└────────────────────────────────────────────────────────────────────────┘
```

#### 3. Terminal Interface (`python chat.py`)
```text
     _    ____  ___    _
    / \  |  _ \|_ _|  / \
   / _ \ | |_) || |  / _ \
  / ___ \|  _ < | | / ___ \
 /_/   \_\_| \_\___/_/   \_\

  Artificially Responsive Intelligent Assistant
  Offline AI  •  Powered by Ollama

  Status: ONLINE  •  Model: llama3.2:latest  •  RAM: 16 GB
──────────────────────────────────────────────────────────────────
Message 1
  You: What is the Big-O complexity of quicksort?
  ARIA (llama3.2:latest):
  Quicksort has an average time complexity of O(n log n) and
  a worst-case time complexity of O(n^2) when poor pivots are chosen.
  ⏱  4.21s  •  llama3.2:latest
```
