"""
config.py - ARIA AI Assistant Configuration
AI Project - Offline AI Assistant using Ollama
"""

from pathlib import Path

# Directory paths
BASE_DIR    = Path(__file__).resolve().parent
DATA_DIR    = BASE_DIR / "data"
LOGS_DIR    = BASE_DIR / "logs"
RESULTS_DIR = BASE_DIR / "results"
STATIC_DIR  = BASE_DIR / "static"

for _dir in (DATA_DIR, LOGS_DIR, RESULTS_DIR, STATIC_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# Ollama
OLLAMA_BASE_URL: str = "http://localhost:11434"

# Models
BENCHMARK_MODELS: list[str] = [
    "llama3.2:latest",
    "phi3:mini",
    "llama3.1:8b",
]

JUDGE_MODEL:        str = "llama3.2:latest"
DEFAULT_CHAT_MODEL: str = "llama3.2:latest"

# Few-shot prompting
USE_FEW_SHOT: bool = True
FEW_SHOT_EXAMPLES: list[tuple[str, str]] = [
    (
        "What is machine learning?",
        "Machine learning is a subset of AI where systems learn patterns from data to make predictions or decisions without being explicitly programmed for each task.",
    ),
    (
        "How does a neural network work?",
        "A neural network consists of layers of interconnected nodes (neurons) that process input data, apply weights and activation functions, and propagate signals forward to produce an output.",
    ),
    (
        "What is the difference between supervised and unsupervised learning?",
        "Supervised learning uses labeled training data (input-output pairs) to learn a mapping function. Unsupervised learning finds hidden patterns in unlabeled data without predefined outputs.",
    ),
]

# Memory
MEMORY_WINDOW_SIZE: int = 20
MAX_SESSIONS:       int = 100

# System prompt - General AI Assistant
SYSTEM_PROMPT: str = (
    "You are ARIA (Artificially Responsive Intelligent Assistant), an advanced offline AI assistant "
    "powered by a local large language model. You can help with anything: coding, mathematics, "
    "science, history, writing, reasoning, AI/ML concepts, project planning, and general knowledge. "
    "You are part of an advanced AI project demonstrating offline LLM deployment using Ollama. "
    "Always give clear, structured, helpful answers. Use markdown formatting when appropriate."
)

# Guardrail - very permissive (general assistant)
ON_TOPIC_KEYWORDS: list[str] = [
    # Accept almost everything - only block truly harmful content
    "what", "how", "why", "when", "where", "who", "which", "explain", "tell",
    "help", "code", "python", "java", "ai", "ml", "math", "science", "history",
    "write", "generate", "create", "analyze", "compare", "summarize", "define",
    "example", "project", "ollama", "llm", "model", "benchmark", "data",
    "algorithm", "function", "class", "error", "debug", "fix", "build",
    "hello", "hi", "hey", "thanks", "please", "can", "is", "are", "does",
]

BLOCKED_KEYWORDS: list[str] = [
    "hack", "exploit", "weapon", "narcotic",
    "bomb", "murder", "suicide", "self-harm",
    "porn", "xxx", "terrorism", "extremism",
]

OFF_TOPIC_CATEGORIES: list[str] = [
    "Requests to generate harmful or illegal content",
    "Personal threats or violence",
    "Explicit adult content",
]

GUARDRAIL_CLASSIFIER_PROMPT: str = (
    "You are a safety classifier for a general AI assistant. "
    "Decide if the user query is SAFE or HARMFUL. "
    "Only mark as HARMFUL if it involves: violence, illegal activity, adult content, or self-harm. "
    "Otherwise mark as SAFE. Almost all educational and general questions are SAFE.\n\n"
    'User query: "{user_query}"\n\n'
    "Respond with exactly one word: ON-TOPIC or OFF-TOPIC"
)

REFUSAL_MESSAGE: str = (
    "I'm sorry, I cannot help with that request. "
    "I'm here to assist with educational, technical, and general knowledge questions."
)

BLOCKED_CONTENT_MESSAGE: str = (
    "That request contains content I cannot engage with. "
    "Please ask me something else!"
)

# Logging
LOG_LEVEL:  str = "INFO"
LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"

# Server
APP_HOST:    str = "0.0.0.0"
APP_PORT:    int = 8080
APP_TITLE:   str = "ARIA - Offline AI Assistant"
APP_VERSION: str = "2.0.0"
