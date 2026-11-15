"""
guardrails.py — Scope and safety guardrails for ARIA.

Provides a two-layer filtering pipeline:
1. **Keyword filter** — fast, deterministic check using blocked-word lists
   and technical query verification.
2. **LLM classifier** — asks the model whether a query violates safety boundaries.

Refused queries are logged to ``logs/refused_queries.jsonl`` for later analysis.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import config
from models import GuardrailTrace

if TYPE_CHECKING:
    from llm_engine import LLMEngine

logger = logging.getLogger(__name__)

_REFUSED_LOG_PATH = config.LOGS_DIR / "refused_queries.jsonl"


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

async def check_query(
    query: str,
    llm_engine: "LLMEngine | None" = None,
    model_name: str | None = None,
    use_llm_classifier: bool = True,
) -> GuardrailTrace:
    """Run the full guardrail pipeline on a user query.

    Args:
        query:              Raw user message.
        llm_engine:         Optional LLM engine for the classifier step.
        model_name:         Model to use for classification (defaults to
                            ``config.DEFAULT_CHAT_MODEL``).
        use_llm_classifier: Set ``False`` to skip the (slower) LLM check.

    Returns:
        A ``GuardrailTrace`` recording the decision and reasoning.
    """
    query_lower = query.lower().strip()

    # Layer 1 — blocked-keyword scan
    blocked_hit = _check_blocked_keywords(query_lower)
    if blocked_hit:
        trace = GuardrailTrace(
            query=query,
            keyword_match_found=False,
            blocked_keyword_found=True,
            decision="rejected",
            reason=f"Blocked keyword detected: '{blocked_hit}'",
        )
        _log_refusal(trace)
        return trace

    # Layer 1b — on-topic keyword boost
    keyword_on_topic = _check_on_topic_keywords(query_lower)

    # Very short greetings / thank-you messages are always on-topic
    if len(query_lower.split()) <= 3 and _is_greeting(query_lower):
        return GuardrailTrace(
            query=query,
            keyword_match_found=True,
            blocked_keyword_found=False,
            decision="accepted",
            reason="Short greeting / polite message.",
        )

    # Layer 2 — LLM-based classifier (optional)
    llm_result: str | None = None
    if use_llm_classifier and llm_engine is not None and not keyword_on_topic:
        llm_result = await _llm_classify(query, llm_engine, model_name)

    # Decision logic
    if llm_result and "OFF" in llm_result.upper():
        trace = GuardrailTrace(
            query=query,
            keyword_match_found=keyword_on_topic,
            blocked_keyword_found=False,
            llm_classifier_result=llm_result,
            decision="rejected",
            reason="LLM classifier deemed query off-topic.",
        )
        _log_refusal(trace)
        return trace

    # If keyword matched OR LLM said on-topic OR we skipped the LLM → accept
    decision = "accepted"
    reason = (
        "On-topic keyword matched." if keyword_on_topic
        else "LLM classifier approved." if llm_result
        else "Default accept (classifier skipped)."
    )

    return GuardrailTrace(
        query=query,
        keyword_match_found=keyword_on_topic,
        blocked_keyword_found=False,
        llm_classifier_result=llm_result,
        decision=decision,
        reason=reason,
    )


def get_refusal_message(trace: GuardrailTrace) -> str:
    """Return the appropriate refusal string for a rejected trace.

    Args:
        trace: The guardrail trace that triggered rejection.

    Returns:
        User-facing refusal message.
    """
    if trace.blocked_keyword_found:
        return config.BLOCKED_CONTENT_MESSAGE
    return config.REFUSAL_MESSAGE


# ──────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────

def _check_blocked_keywords(query_lower: str) -> str | None:
    """Return the first blocked keyword found, or ``None``."""
    for kw in config.BLOCKED_KEYWORDS:
        if kw in query_lower:
            logger.warning("Blocked keyword '%s' found in query.", kw)
            return kw
    return None


def _check_on_topic_keywords(query_lower: str) -> bool:
    """Return ``True`` if any on-topic keyword appears in the query."""
    for kw in config.ON_TOPIC_KEYWORDS:
        if kw in query_lower:
            return True
    return False


def _is_greeting(text: str) -> bool:
    """Heuristic check for common greetings / polite phrases."""
    greetings = {
        "hi", "hello", "hey", "good morning", "good afternoon",
        "good evening", "thanks", "thank you", "bye", "goodbye",
        "ok", "okay", "sure", "yes", "no", "please",
    }
    return text.strip().rstrip("!.,?") in greetings


async def _llm_classify(
    query: str,
    llm_engine: "LLMEngine",
    model_name: str | None,
) -> str | None:
    """Ask the LLM to classify a query as ON-TOPIC or OFF-TOPIC.

    Args:
        query:      The user query.
        llm_engine: Initialised LLM engine.
        model_name: Which model to use for classification.

    Returns:
        The raw LLM response string, or ``None`` on failure.
    """
    model = model_name or config.DEFAULT_CHAT_MODEL

    off_topic_list = "\n".join(f"- {cat}" for cat in config.OFF_TOPIC_CATEGORIES)
    prompt = config.GUARDRAIL_CLASSIFIER_PROMPT.format(
        off_topic_list=off_topic_list,
        user_query=query,
    )

    try:
        result = await llm_engine.generate(
            prompt=prompt,
            model_name=model,
            system_prompt="You are a strict scope classifier. Respond with exactly one word.",
            stream=False,
        )
        logger.info("LLM classifier result for query '%s...': %s", query[:40], result.strip())
        return result.strip()
    except Exception as exc:
        logger.error("LLM classifier failed: %s", exc)
        return None


def _log_refusal(trace: GuardrailTrace) -> None:
    """Append a refused-query record to the JSONL log file.

    Args:
        trace: The guardrail trace that was rejected.
    """
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        **trace.model_dump(),
    }
    try:
        with open(_REFUSED_LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    except OSError as exc:
        logger.error("Failed to write refusal log: %s", exc)


def get_refused_queries(limit: int = 100) -> list[dict]:
    """Read the most recent refused-query log entries.

    Args:
        limit: Maximum number of entries to return.

    Returns:
        List of dicts (most recent first).
    """
    if not _REFUSED_LOG_PATH.exists():
        return []
    entries: list[dict] = []
    try:
        with open(_REFUSED_LOG_PATH, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("Failed to read refusal log: %s", exc)
    return entries[-limit:]
