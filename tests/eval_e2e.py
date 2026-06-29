"""End-to-end evaluation with 8 standard metrics.

Metrics (all from published benchmarks):
  1. Task Success Rate  — WebArena / GAIA
  2. Exact Match         — GAIA / HotpotQA official
  3. Answer F1           — HotpotQA official (token-level)
  4. Avg Steps           — AgentBench
  5. Subtask Completion Rate — AgentBench
  6. Faithfulness        — RAGAS (LLM-as-Judge)
  7. Context Precision   — RAGAS
  8. Citation Precision  — Perplexity-style

Experiments:
  --mode full-ab    Experiment 1: Full System A/B — baseline (bare Agent) vs full system
                    (chunk_retriever + obs_retriever + memory all ON). Default mode.
  --mode ltm-ab     Experiment 2: Long-Term Memory A/B — all features ON, only toggle memory.
                    Seeds memory in Round 1, compares Round 2 with/without memory.

Requires:
    DEEPSEEK_API_KEY  — for LLM calls
    SERPAPI_KEY        — for web search

Usage:
    cd d:/Agent
    python -m deepresearch.tests.eval_e2e --config deepresearch/configs/deepseek_eval.yaml
    python -m deepresearch.tests.eval_e2e --config deepresearch/configs/deepseek_eval.yaml --mode ltm-ab
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import math
import os
import re
import string
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is on path
if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from deepresearch.agent.agent import DeepResearchAgent
from deepresearch.agent.synthesizer import SynthesisWriter
from deepresearch.agent.types import ResearchTask, Trajectory
from deepresearch.reward.reward_engine import RewardBreakdown
from deepresearch.utils.config import build_agent_from_config, load_config
from deepresearch.utils.llm_client import AnthropicClient, OpenAICompatibleClient
from deepresearch.utils.memory_store import MemoryStore

_ENV_PLACEHOLDER = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")


# ════════════════��════════════════════════════════════���═════════════════
# HotpotQA Official Metrics: Exact Match & F1
# Reference: https://hotpotqa.github.io/ (Yang et al., EMNLP 2018)
# ═══════════════════════════════════════════════════════════════════════

def _normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace.

    Exactly follows HotpotQA official evaluation script.
    """
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_mrr(relevance_flags: List[bool]) -> float:
    """Mean Reciprocal Rank: 1/rank of the first relevant item.

    Args:
        relevance_flags: ordered list where True = relevant, False = irrelevant

    Reference: standard IR metric (Craswell, 2009)
    """
    for i, relevant in enumerate(relevance_flags, 1):
        if relevant:
            return 1.0 / i
    return 0.0


def compute_ndcg(relevance_flags: List[bool], k: int = 10) -> float:
    """Normalized Discounted Cumulative Gain at k.

    Binary relevance: 1 for relevant, 0 for irrelevant.

    Reference: standard IR metric (Järvelin & Kekäläinen, 2002)
    """
    relevance = [1.0 if r else 0.0 for r in relevance_flags[:k]]
    if not relevance or sum(relevance) == 0:
        return 0.0

    # DCG
    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(relevance))

    # Ideal DCG (all relevant items first)
    ideal = sorted(relevance, reverse=True)
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal))

    return dcg / idcg if idcg > 0 else 0.0


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """HotpotQA official: 1.0 if normalized strings match, else 0.0."""
    return 1.0 if _normalize_answer(prediction) == _normalize_answer(ground_truth) else 0.0


def compute_f1(prediction: str, ground_truth: str) -> float:
    """HotpotQA official: token-level F1 between prediction and ground truth."""
    pred_tokens = _normalize_answer(prediction).split()
    gold_tokens = _normalize_answer(ground_truth).split()

    if not gold_tokens:
        return 1.0 if not pred_tokens else 0.0
    if not pred_tokens:
        return 0.0

    common = collections.Counter(pred_tokens) & collections.Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


# ═══════════════════════════════════════════════════════════════════════
# Answer Extraction (GAIA-style)
# Agent produces long-form answers; extract a concise answer for EM/F1.
# Reference: GAIA (Mialon et al., 2023) uses answer extraction before
# computing Exact Match to handle verbose model outputs.
# ═══════════════════════════════════════════════════════════════════════

async def extract_short_answer(
    llm_client: Any,
    question: str,
    long_answer: str,
) -> str:
    """Extract a concise, direct answer by verbatim extraction from the agent's output.

    Anti-contamination design: the prompt demands verbatim extraction only,
    preventing the LLM from substituting its own knowledge when the agent's
    answer is wrong or incomplete.
    """
    if not long_answer.strip():
        return ""

    prompt = f"""You are an answer extractor. Your job is to extract a short answer from the TEXT below.

STRICT RULES:
- You MUST extract verbatim from the TEXT. Copy the exact words/phrases that answer the question.
- Do NOT use your own knowledge. If the TEXT does not contain a clear answer, output exactly "UNANSWERABLE".
- Output ONLY the extracted phrase, nothing else. No explanation, no prefix, no quotes.

Question: {question}

TEXT:
{long_answer[:3000]}

Extracted answer:"""

    try:
        response = await llm_client.chat(
            messages=[{"role": "user", "content": prompt}],
        )
        extracted = response.strip().strip('"').strip("'").strip()
        # Remove common prefixes the LLM might add
        for prefix in ["The answer is ", "Answer: ", "The short answer is ",
                        "Extracted answer: ", "Extracted answer:"]:
            if extracted.lower().startswith(prefix.lower()):
                extracted = extracted[len(prefix):].strip()
        return extracted
    except Exception:
        return long_answer[:200]


# ═══════════════════════════════════════════════════════════════════════
# LLM-as-Judge Correctness (GAIA / MT-Bench style)
# Binary: does the answer contain the correct factual information?
# ═══════════════════════════════════════════════════════════════════════

async def judge_correctness(
    llm_client: Any,
    question: str,
    reference_answer: str,
    agent_answer: str,
) -> float:
    """LLM-as-Judge correctness: 1.0 if agent's answer is factually correct, else 0.0.

    More robust than EM/F1 for generative answers. The judge checks whether
    the agent's answer contains the same core factual information as the
    reference, regardless of wording or verbosity.
    """
    if not agent_answer.strip():
        return 0.0
    if not reference_answer.strip():
        return 0.0

    prompt = f"""You are a factual correctness judge. Given a question and a reference answer, determine if the agent's answer is factually correct.

QUESTION: {question}

REFERENCE ANSWER: {reference_answer}

AGENT'S ANSWER:
{agent_answer[:3000]}

Rules:
- The agent's answer is CORRECT if it contains the same core factual information as the reference answer, even if phrased differently or with additional details.
- The agent's answer is INCORRECT if it gives a wrong answer, contradicts the reference, or fails to address the question.
- Ignore differences in formatting, verbosity, or extra context — only judge factual correctness.

Output JSON: {{"correct": true}} or {{"correct": false}}
Only output the JSON, nothing else."""

    try:
        data = await _llm_json_call(llm_client, prompt)
        return 1.0 if data.get("correct", False) else 0.0
    except Exception:
        return 0.0


# ═══════════════════════════════════════════════════════════════════════
# RAGAS-style Metrics (LLM-as-Judge)
# Reference: Es et al., EACL 2024 — "RAGAS: Automated Evaluation of RAG"
# ═══════════════════════════════════════════════════════════════════════

async def _llm_json_call(llm_client: Any, prompt: str) -> dict:
    """Call LLM with a prompt expecting JSON output, return parsed dict."""
    response = await llm_client.chat(
        messages=[{"role": "user", "content": prompt}],
        response_format="json",
    )
    text = response.strip()
    # Extract JSON from possible markdown code blocks
    if "```" in text:
        m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        text = m.group(1).strip() if m else text
    return json.loads(text)


async def _faithfulness_step1_decompose(llm_client: Any, answer: str) -> List[str]:
    """RAGAS Step 1: Decompose answer into atomic claims. No context provided.

    The LLM only sees the answer, ensuring claim granularity is not biased
    by what the contexts happen to contain.
    """
    prompt = f"""Given the following answer, create a list of atomic factual statements.
Each statement should be a single, simple, self-contained claim.
Break compound sentences into separate claims.

ANSWER:
{answer[:3000]}

Output JSON: {{"statements": ["statement 1", "statement 2", ...]}}
Only output the JSON, nothing else."""

    data = await _llm_json_call(llm_client, prompt)
    statements = data.get("statements", [])
    return [str(s).strip() for s in statements if str(s).strip()]


async def _faithfulness_step2_verify(
    llm_client: Any,
    statement: str,
    context_text: str,
) -> bool:
    """RAGAS Step 2: NLI verification — is this statement supported by the context?

    Each statement is verified independently against the full context.
    """
    prompt = f"""Consider the following context and statement. Determine whether the
statement is supported by the information in the context.

CONTEXT:
{context_text[:6000]}

STATEMENT: {statement}

Output JSON: {{"verdict": "supported"}} or {{"verdict": "not_supported"}}
Only output the JSON, nothing else."""

    data = await _llm_json_call(llm_client, prompt)
    verdict = str(data.get("verdict", "")).lower().strip()
    return verdict == "supported"


async def compute_faithfulness(
    llm_client: Any,
    answer: str,
    contexts: List[str],
) -> Optional[float]:
    """RAGAS Faithfulness (Es et al., EACL 2024): two-step claim verification.

    Step 1: Decompose answer into atomic claims (answer only, no context).
    Step 2: Verify each claim against retrieved contexts independently.
    Score = supported_claims / total_claims.

    Returns None if contexts is empty (unable to evaluate, not "unfaithful").
    """
    if not answer.strip():
        return 0.0
    if not contexts:
        return None  # Cannot evaluate — not the same as unfaithful

    try:
        # Step 1: Decompose (LLM sees answer only)
        claims = await _faithfulness_step1_decompose(llm_client, answer)
        if not claims:
            return 0.0

        # Step 2: Verify each claim independently
        context_text = "\n\n---\n\n".join(ctx[:2000] for ctx in contexts[:10])
        supported_count = 0
        for claim in claims:
            try:
                if await _faithfulness_step2_verify(llm_client, claim, context_text):
                    supported_count += 1
            except Exception:
                pass  # Treat failed verification as not supported

        return supported_count / len(claims)
    except Exception:
        return 0.0


async def compute_context_precision(
    llm_client: Any,
    query: str,
    contexts: List[str],
) -> Optional[float]:
    """RAGAS Context Precision: fraction of retrieved contexts relevant to the query.

    Score = relevant_contexts / total_contexts.
    Returns None if no contexts available (cannot evaluate).

    Adapted for multi-hop QA: a context is "relevant" if it contains information
    useful for ANY reasoning step toward answering the query, not only if it
    directly contains the final answer.
    """
    if not contexts:
        return None

    context_blocks = []
    for i, ctx in enumerate(contexts[:10], 1):
        context_blocks.append(f"[Context {i}]: {ctx[:1500]}")
    context_text = "\n\n".join(context_blocks)

    prompt = f"""You are evaluating retrieval quality for a multi-hop research question.
The question may require combining information from multiple sources to reach the answer.

QUERY: {query}

{context_text}

For each context, determine if it contains information that is USEFUL for answering the query.
A context is useful if it provides:
- A direct answer or part of the answer
- Background information needed to understand the question
- Bridge information that connects entities in the question (e.g., identifying who/what is being asked about)
- Facts that support or verify claims relevant to the answer

A context is NOT useful if it is completely unrelated to any aspect of the query.

Output JSON: {{"contexts": [{{"id": 1, "relevant": true/false}}, ...]}}
Only output the JSON, nothing else."""

    try:
        data = await _llm_json_call(llm_client, prompt)
        items = data.get("contexts", [])
        if not items:
            return 0.0
        relevant = sum(1 for c in items if c.get("relevant", False))
        return relevant / len(items)
    except Exception:
        return 0.0


async def compute_citation_precision(
    llm_client: Any,
    answer: str,
    citations: List[str],
    contexts: Dict[str, str],
) -> float:
    """Citation Precision (Perplexity-style): fraction of citations that support their claims.

    For each [N] citation in the answer, check if the cited URL's content
    actually supports the statement it's attached to.
    """
    if not citations or not answer.strip():
        return 0.0

    # Build citation context mapping
    citation_blocks = []
    for i, url in enumerate(citations[:10], 1):
        ctx = contexts.get(url, "")[:1500]
        if ctx:
            citation_blocks.append(f"[{i}] URL: {url}\nContent: {ctx}")
        else:
            citation_blocks.append(f"[{i}] URL: {url}\nContent: (not available)")

    citation_text = "\n\n".join(citation_blocks)

    prompt = f"""You are evaluating citation quality. The answer contains numbered citations [1], [2], etc. Check if each citation's source content actually supports the statement it is attached to.

ANSWER:
{answer[:3000]}

CITATION SOURCES:
{citation_text}

For each citation number used in the answer, determine if the cited source supports the claim.
Output JSON: {{"citations": [{{"id": 1, "supported": true/false}}, ...]}}

Only output the JSON, nothing else."""

    try:
        data = await _llm_json_call(llm_client, prompt)
        items = data.get("citations", [])
        if not items:
            return 0.0
        supported = sum(1 for c in items if c.get("supported", False))
        return supported / len(items)
    except Exception:
        return 0.0


# ═══════════════════════════════════════════════════════════════════════
# Over-Refusal Rate (ORR) — dual to Faithfulness
# Detects answers that refuse despite the evidence containing relevant info.
# ═══════════════════════════════════════════════════════════════════════

_REFUSAL_PHRASES = (
    "do not address",
    "does not address",
    "do not specify",
    "does not specify",
    "cannot provide",
    "sources do not",
    "no clear answer",
    "unable to confirm",
    "not explicitly stated",
    "available sources do not",
    "the evidence does not",
    "evidence is insufficient",
    "cannot determine",
    "无法确定",
    "未明确",
    "证据不足",
    "无法提供",
)


def _contains_refusal(answer: str) -> bool:
    if not answer:
        return False
    text = answer.lower()
    return any(p in text for p in _REFUSAL_PHRASES)


async def compute_over_refusal_rate(
    llm_client: Any,
    query: str,
    answer: str,
    contexts: List[str],
) -> Optional[float]:
    """Detect over-refusal: agent says it cannot answer despite the evidence
    containing relevant information.

    Returns:
      0.0 → no refusal phrase OR evidence really insufficient (legitimate refusal)
      1.0 → refusal phrase present AND evidence had relevant info (over-refusal)
      None → no contexts available, cannot evaluate

    Pairs with Faithfulness: Faithfulness=1 + ORR=1 means agent cheated by
    refusing all answers (zero hallucination but useless).
    """
    if not answer or not _contains_refusal(answer):
        return 0.0
    if not contexts:
        return None

    contexts_blob = "\n---\n".join(c[:400] for c in contexts[:6])
    prompt = f"""You are evaluating whether an answer over-refused — i.e., claimed it could not answer despite the evidence containing relevant information.

QUESTION: {query}

ANSWER: {answer[:1500]}

EVIDENCE SNIPPETS:
{contexts_blob}

The answer contains refusal language ("do not address", "cannot provide", etc.).
Evaluate: did the evidence ACTUALLY contain enough information to give a non-refused answer?

- Output {{"over_refused": true}} if the evidence had clear relevant info but the answer refused anyway.
- Output {{"over_refused": false}} if the evidence really lacked the needed info (legitimate refusal).

Output ONLY JSON, nothing else."""

    try:
        data = await _llm_json_call(llm_client, prompt)
        return 1.0 if data.get("over_refused", False) else 0.0
    except Exception:
        return 0.0


# ═══════════════════════════════════════════════════════════════════════
# Task Result & Metrics Aggregation
# ══════════════════════════════════════════���════════════════════════════

@dataclass
class TaskMetrics:
    """Per-task metrics following the benchmark spec."""
    task_id: str
    query: str

    # Task Completion (GAIA / HotpotQA)
    task_success: float         # 1.0 if agent completed without error
    exact_match: float          # HotpotQA official EM
    answer_f1: float            # HotpotQA official F1
    correctness: float          # LLM-as-Judge binary correctness (1.0 or 0.0)

    # Execution Efficiency (AgentBench)
    steps: int                  # total agent steps
    subtask_completion: float   # completed sub-goals / total sub-goals

    # Retrieval Quality (RAGAS)
    faithfulness: Optional[float]   # claims supported / total claims; None = no contexts
    context_precision: Optional[float]  # relevant contexts / total contexts; None = no contexts

    # Answer Quality (Perplexity)
    citation_precision: float   # supported citations / total citations

    # Over-Refusal Rate (dual to Faithfulness — anti-Goodharting)
    over_refusal: Optional[float] = None  # 1.0 = refused despite evidence; None = no contexts

    elapsed_seconds: float = 0.0
    success: bool = True
    error: Optional[str] = None
    extracted_answer: str = ""  # GAIA-style extracted short answer for EM/F1 audit
    key_facts_coverage: Optional[float] = None  # fraction of key_facts found in answer


@dataclass
class AggregateMetrics:
    """Aggregated metrics across all tasks."""
    n_tasks: int
    n_success: int
    task_success_rate: float
    avg_exact_match: float
    avg_answer_f1: float
    avg_correctness: float
    avg_steps: float
    avg_subtask_completion: float
    avg_faithfulness: float
    avg_context_precision: float
    avg_citation_precision: float
    avg_over_refusal: float
    avg_time: float


def _avg_optional(results: List[TaskMetrics], field: str) -> float:
    """Average a metric that may be None, excluding None values."""
    values = [getattr(r, field) for r in results if r.success and getattr(r, field) is not None]
    return sum(values) / len(values) if values else 0.0


def aggregate_metrics(results: List[TaskMetrics]) -> AggregateMetrics:
    """Compute aggregate metrics from per-task results.

    For faithfulness/context_precision, None values (= no contexts available,
    e.g. due to API rate limits) are excluded from the average rather than
    counted as 0.
    """
    n = len(results)
    if n == 0:
        return AggregateMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    successful = [r for r in results if r.success]
    ns = len(successful) or 1  # avoid div by zero

    return AggregateMetrics(
        n_tasks=n,
        n_success=len(successful),
        task_success_rate=len(successful) / n,
        avg_exact_match=sum(r.exact_match for r in successful) / ns,
        avg_answer_f1=sum(r.answer_f1 for r in successful) / ns,
        avg_correctness=sum(r.correctness for r in successful) / ns,
        avg_steps=sum(r.steps for r in successful) / ns,
        avg_subtask_completion=sum(r.subtask_completion for r in successful) / ns,
        avg_faithfulness=_avg_optional(results, "faithfulness"),
        avg_context_precision=_avg_optional(results, "context_precision"),
        avg_citation_precision=sum(r.citation_precision for r in successful) / ns,
        avg_over_refusal=_avg_optional(results, "over_refusal"),
        avg_time=sum(r.elapsed_seconds for r in successful) / ns,
    )


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _resolve_env(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _resolve_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env(item) for item in value]
    if isinstance(value, str):
        m = _ENV_PLACEHOLDER.match(value.strip())
        if m:
            return os.environ.get(m.group(1), "")
    return value


def _load_dataset(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _build_optional_client(cfg_section: dict):
    """Build an LLM client from a YAML section. Returns None if section is empty.

    Supports providers: 'anthropic', 'deepseek', 'openai' (also covers Qwen /
    Moonshot / any OpenAI-compatible API via custom base_url + api_key_env).
    """
    if not cfg_section:
        return None

    provider = str(cfg_section.get("provider", "anthropic")).lower()
    model = str(cfg_section.get("model", ""))
    api_key_env = str(cfg_section.get("api_key_env", "")).strip()

    if provider == "anthropic":
        env_var = api_key_env or "ANTHROPIC_API_KEY"
        api_key = os.environ.get(env_var, "").strip()
        if not api_key:
            raise ValueError(f"{env_var} is not set (required for {model or 'anthropic'}).")
        return AnthropicClient(api_key=api_key, model=model or "claude-haiku-4-5-20251001")

    if provider == "deepseek":
        env_var = api_key_env or "DEEPSEEK_API_KEY"
        default_base = "https://api.deepseek.com"
    elif provider == "openai":
        env_var = api_key_env or "OPENAI_API_KEY"
        default_base = "https://api.openai.com/v1"
    else:
        if not api_key_env:
            raise ValueError(f"provider='{provider}' requires explicit api_key_env in YAML.")
        env_var = api_key_env
        default_base = ""

    api_key = os.environ.get(env_var, "").strip()
    if not api_key:
        raise ValueError(f"{env_var} is not set (required for provider='{provider}').")

    base_url = str(cfg_section.get("base_url", default_base))
    if not base_url:
        raise ValueError(f"base_url required for provider='{provider}'.")

    return OpenAICompatibleClient(api_key=api_key, model=model, base_url=base_url)


def _build_clients(config: dict):
    """Build (llm_client, planner_client, auditor_client) from config."""
    llm_cfg = config.get("llm", {})
    provider = str(llm_cfg.get("provider", "anthropic")).lower()
    model = str(config.get("agent", {}).get("model", "claude-sonnet-4-6"))

    if provider == "deepseek":
        api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY is not set.")
        base_url = str(llm_cfg.get("base_url", "https://api.deepseek.com"))
        llm_client = OpenAICompatibleClient(api_key=api_key, model=model, base_url=base_url)
    else:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set.")
        llm_client = AnthropicClient(api_key=api_key, model=model)

    planner_client = _build_optional_client(config.get("llm_planner", {}))
    auditor_client = _build_optional_client(config.get("llm_auditor", {}))

    return llm_client, planner_client, auditor_client


def _extract_contexts(trajectory: Trajectory) -> Tuple[List[str], Dict[str, str]]:
    """Extract ALL contexts the Agent (and SynthesisWriter) had access to.

    Faithfulness measures: "is the answer faithful to what the Agent saw?"
    So contexts must match what SynthesisWriter used to generate the answer.
    This includes page bodies, search snippets, and cross_check results.

    Deduplicates by URL. Page body takes priority over snippet for the same URL.

    Returns:
        contexts: list of all text contexts (for Faithfulness / Context Precision)
        url_to_content: mapping from URL to best available content (for Citation Precision)
    """
    url_to_content: Dict[str, str] = {}

    for obs in trajectory.observations:
        if not obs.success:
            continue
        result = obs.result

        # PageContent from extract/click — full page body (highest priority)
        if hasattr(result, "body") and hasattr(result, "url"):
            body = str(result.body).strip()
            url = str(result.url).strip()
            if body and url:
                url_to_content[url] = body  # overwrites snippet if same URL

        # SearchResult list from search/cross_check
        elif isinstance(result, list):
            for item in result:
                if hasattr(item, "url") and hasattr(item, "snippet"):
                    url = str(item.url).strip()
                    snippet = str(item.snippet).strip()
                    if url and snippet and url not in url_to_content:
                        url_to_content[url] = snippet

        # String result (e.g. from terminate, cross_check summary)
        elif isinstance(result, str) and result.strip():
            # No URL, but still context the Agent saw
            key = f"_obs_{id(obs)}"
            url_to_content[key] = result.strip()

    # All contexts for RAGAS — matches what SynthesisWriter sees
    contexts = [content[:3000] for content in url_to_content.values()]
    return contexts, url_to_content


def _progress_callback(event: dict) -> None:
    status = "ok" if event["success"] else f"err:{event['error']}"
    print(f"  [{event['step']:02d}] {event['action_type']:<12} {status} | {event['goal'][:50]}")


# ═══════════════════════════════════════════════════════════════════════
# Core: Run one task and compute all 8 metrics
# ═══════════════════════════════════════════════════════════════════════

async def run_one_task(
    agent: DeepResearchAgent,
    synthesizer: SynthesisWriter,
    llm_client: Any,
    record: Dict[str, Any],
    extract_short: bool = True,
    judge_client: Any = None,
) -> TaskMetrics:
    """Run a single task, then compute all 8 benchmark metrics.

    Args:
        extract_short: If True (default), extract a short answer for EM/F1
            (suitable for HotpotQA-style short reference answers).
            If False, use the full long answer for F1 and compute key_facts
            coverage (suitable for long-form reference answers).
        judge_client: LLM used for evaluation-time judge calls (extract_short,
            correctness, faithfulness, context_precision, citation_precision,
            over_refusal). Should be vendor-independent from `llm_client`
            (which generates the answer) to avoid self-evaluation bias.
            Falls back to `llm_client` if None.
    """
    judge = judge_client if judge_client is not None else llm_client
    task = ResearchTask(
        task_id=str(record["task_id"]),
        query=str(record["query"]),
        reference_answer=record.get("reference_answer"),
        key_facts=record.get("key_facts"),
        expected_sub_goals=record.get("expected_sub_goals"),
    )

    t0 = time.monotonic()
    try:
        # Run agent
        env = agent.env
        if callable(getattr(env, "__aenter__", None)):
            async with env:
                trajectory = await agent.run(task)
        else:
            trajectory = await agent.run(task)

        # Synthesize final answer
        answer_body, citations = await synthesizer.synthesize(task.query, trajectory)
        trajectory.final_answer = answer_body
        trajectory.citations = citations

        elapsed = time.monotonic() - t0

        # ── Metric 1: Task Success Rate ──
        task_success = 1.0  # completed without exception

        # ── Metric 2 & 3: Exact Match + Answer F1 ──
        ref = task.reference_answer or ""
        long_pred = trajectory.final_answer or ""

        if extract_short:
            # HotpotQA-style: extract concise answer, compare against short reference
            short_pred = await extract_short_answer(judge, task.query, long_pred)
            em = compute_exact_match(short_pred, ref)
            f1 = compute_f1(short_pred, ref)
        else:
            # Long-form reference: use full answer for F1, skip EM (meaningless)
            short_pred = long_pred
            em = 0.0  # EM not applicable for long references
            f1 = compute_f1(long_pred, ref)

        # ── LLM-as-Judge Correctness ──
        correctness = await judge_correctness(judge, task.query, ref, long_pred)

        # ── Key Facts Coverage (for long-form references) ──
        key_facts = task.key_facts or []
        if key_facts and long_pred:
            answer_lower = _normalize_answer(long_pred)
            matched_facts = sum(
                1 for fact in key_facts
                if _normalize_answer(fact) in answer_lower
                or any(token in answer_lower for token in _normalize_answer(fact).split() if len(token) >= 4)
            )
            kf_coverage = matched_facts / len(key_facts)
        else:
            kf_coverage = None

        # ── Metric 4: Avg Steps (AgentBench) ──
        steps = len(trajectory.observations)

        # ── Metric 5: Subtask Completion Rate (AgentBench) ──
        total_goals = len(trajectory.sub_goals)
        completed_goals = sum(
            1 for g in trajectory.sub_goals if g.status == "completed"
        )
        subtask_completion = completed_goals / total_goals if total_goals > 0 else 0.0

        # Extract contexts for RAGAS metrics
        contexts, url_to_content = _extract_contexts(trajectory)

        # ── Metric 6: Faithfulness (RAGAS) ──
        faithfulness = await compute_faithfulness(judge, long_pred, contexts)

        # ── Metric 7: Context Precision (RAGAS) ──
        context_prec = await compute_context_precision(judge, task.query, contexts)

        # ── Metric 8: Citation Precision (Perplexity-style) ──
        citation_prec = await compute_citation_precision(judge, long_pred, citations, url_to_content)

        # ── Metric 9: Over-Refusal Rate (anti-Goodhart for Faithfulness) ──
        over_refusal = await compute_over_refusal_rate(judge, task.query, long_pred, contexts)

        return TaskMetrics(
            task_id=task.task_id,
            query=task.query[:60],
            task_success=task_success,
            exact_match=em,
            answer_f1=f1,
            correctness=correctness,
            steps=steps,
            subtask_completion=subtask_completion,
            faithfulness=faithfulness,
            context_precision=context_prec,
            citation_precision=citation_prec,
            over_refusal=over_refusal,
            extracted_answer=short_pred,
            key_facts_coverage=kf_coverage,
            elapsed_seconds=round(elapsed, 1),
            success=True,
        )

    except Exception as exc:
        elapsed = time.monotonic() - t0
        return TaskMetrics(
            task_id=str(record["task_id"]),
            query=str(record["query"])[:60],
            task_success=0.0,
            exact_match=0.0,
            answer_f1=0.0,
            correctness=0.0,
            steps=0,
            subtask_completion=0.0,
            faithfulness=None,
            context_precision=None,
            citation_precision=0.0,
            over_refusal=None,
            elapsed_seconds=round(elapsed, 1),
            success=False,
            error=str(exc)[:200],
        )


# ═══════════════════════════════════════════════════════════════════════
# Printing & Reporting
# ═══════════════════════════════════════════════════════════════════════

def print_results(results: List[TaskMetrics], label: str = "") -> AggregateMetrics:
    """Print per-task table and aggregated metrics."""
    agg = aggregate_metrics(results)

    if label:
        print(f"\n{'=' * 70}")
        print(f"  {label}")
        print(f"{'=' * 70}")

    header = (
        f"{'task_id':<14} | {'EM':>4} {'F1':>5} {'COR':>4} | "
        f"{'steps':>5} {'sub%':>5} | "
        f"{'faith':>5} {'ctxP':>5} {'citeP':>5} | "
        f"{'time':>6} | {'ok':>3}"
    )
    print(f"\n{header}")
    print("-" * len(header))

    def _fmt_opt(v: Optional[float]) -> str:
        return f"{v:>5.3f}" if v is not None else "  N/A"

    for r in results:
        ok = "Y" if r.success else "N"
        print(
            f"{r.task_id:<14} | {r.exact_match:>4.0f} {r.answer_f1:>5.3f} {r.correctness:>4.0f} | "
            f"{r.steps:>5} {r.subtask_completion:>5.1%} | "
            f"{_fmt_opt(r.faithfulness)} {_fmt_opt(r.context_precision)} {r.citation_precision:>5.3f} | "
            f"{r.elapsed_seconds:>5.1f}s | {ok:>3}"
        )

    print("-" * len(header))
    print(
        f"{'AVERAGE':<14} | {agg.avg_exact_match:>4.2f} {agg.avg_answer_f1:>5.3f} {agg.avg_correctness:>4.2f} | "
        f"{agg.avg_steps:>5.1f} {agg.avg_subtask_completion:>5.1%} | "
        f"{agg.avg_faithfulness:>5.3f} {agg.avg_context_precision:>5.3f} {agg.avg_citation_precision:>5.3f} | "
        f"{agg.avg_time:>5.1f}s | {agg.task_success_rate:>3.0%}"
    )

    print(f"\n{'─' * 50}")
    print(f"  Task Success Rate:       {agg.task_success_rate:.1%} ({agg.n_success}/{agg.n_tasks})")
    print(f"  Exact Match:             {agg.avg_exact_match:.3f}")
    print(f"  Answer F1:               {agg.avg_answer_f1:.3f}")
    print(f"  Correctness (Judge):     {agg.avg_correctness:.3f}")
    print(f"  Avg Steps:               {agg.avg_steps:.1f}")
    print(f"  Subtask Completion:      {agg.avg_subtask_completion:.1%}")
    print(f"  Faithfulness (RAGAS):    {agg.avg_faithfulness:.3f}")
    print(f"  Context Precision:       {agg.avg_context_precision:.3f}")
    print(f"  Citation Precision:      {agg.avg_citation_precision:.3f}")
    print(f"  Over-Refusal Rate (ORR): {agg.avg_over_refusal:.3f}  (lower is better, target<0.15)")
    print(f"  Avg Time:                {agg.avg_time:.1f}s")
    print(f"{'─' * 50}")

    # Failures
    failures = [r for r in results if not r.success]
    if failures:
        print(f"\nFailed tasks ({len(failures)}):")
        for r in failures:
            print(f"  {r.task_id}: {r.error}")

    return agg


def save_results(results: List[TaskMetrics], path: Path, extra: Optional[Dict] = None) -> None:
    """Save per-task metrics + aggregate to JSON file."""
    agg = aggregate_metrics(results)
    payload: Dict[str, Any] = {
        "aggregate": asdict(agg),
        "per_task": [asdict(r) for r in results],
    }
    if extra:
        payload.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults saved to {path}")


# ═══════════════════════════════════════════════════════════════════════
# Mode: Full A/B (Experiment 1 — Global Comparison)
# Baseline: bare Agent (no chunk_retriever, no obs_retriever, no memory)
# Full:     all enabled (default config)
# ═══════════════════════════════════════════════════════════════════════

async def run_full_ab(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    config = _resolve_env(config)

    llm_client, planner_client, auditor_client = _build_clients(config)
    model = str(config.get("agent", {}).get("model", "claude-sonnet-4-6"))

    synthesizer = SynthesisWriter(llm_client=llm_client, model=model)

    # Vendor-independent evaluation: judge = auditor (Qwen) when configured,
    # otherwise fall back to llm_client (DeepSeek self-eval — known bias).
    judge_client = auditor_client if auditor_client is not None else llm_client
    judge_model = str(config.get("llm_auditor", {}).get("model", model)) if auditor_client else model

    project_root = Path(__file__).resolve().parents[2]
    golden_path = Path(args.golden_set) if args.golden_set else project_root / "deepresearch" / "data" / "golden_set.jsonl"
    records = _load_dataset(golden_path)

    if args.max_tasks:
        records = records[:args.max_tasks]

    print(f"Full A/B Experiment (Baseline vs Full System): {len(records)} tasks")
    print(f"Config: {args.config}")
    print(f"Agent model: {model}")
    print(f"Judge model: {judge_model}  ({'vendor-independent' if auditor_client else 'self-eval (auditor not configured)'})")
    print()

    # ── Run A: Baseline (bare Agent — all RAG components off) ──
    print("=" * 60)
    print("  CONDITION A: Baseline (bare Agent)")
    print("  chunk_retriever=OFF  obs_retriever=OFF  memory=OFF")
    print("=" * 60)

    config_a = dict(config)
    config_a.pop("memory", None)  # no memory
    agent_a = build_agent_from_config(config_a, llm_client, planner_client=planner_client, judge_client=auditor_client)
    agent_a.chunk_retriever = None
    agent_a.obs_retriever = None
    agent_a.memory_store = None
    agent_a.progress_callback = _progress_callback

    results_a: List[TaskMetrics] = []
    for i, record in enumerate(records, 1):
        print(f"[A {i}/{len(records)}] {record['task_id']}: {str(record['query'])[:50]}...")
        result = await run_one_task(agent_a, synthesizer, llm_client, record, judge_client=judge_client)
        results_a.append(result)
        if result.success:
            faith_str = f"{result.faithfulness:.3f}" if result.faithfulness is not None else "N/A"
            print(f"  -> EM={result.exact_match:.0f} F1={result.answer_f1:.3f} "
                  f"faith={faith_str} steps={result.steps}")
        else:
            print(f"  -> FAILED: {result.error}")

    agg_a = print_results(results_a, label="Condition A: Baseline (bare Agent)")

    # ── Run B: Full System (all RAG components on) ──
    print("\n" + "=" * 60)
    print("  CONDITION B: Full System")
    print("  chunk_retriever=ON  obs_retriever=ON  memory=ON")
    print("=" * 60)

    agent_b = build_agent_from_config(config, llm_client, planner_client=planner_client, judge_client=auditor_client)
    agent_b.progress_callback = _progress_callback

    results_b: List[TaskMetrics] = []
    for i, record in enumerate(records, 1):
        print(f"[B {i}/{len(records)}] {record['task_id']}: {str(record['query'])[:50]}...")
        result = await run_one_task(agent_b, synthesizer, llm_client, record, judge_client=judge_client)
        results_b.append(result)
        if result.success:
            faith_str = f"{result.faithfulness:.3f}" if result.faithfulness is not None else "N/A"
            print(f"  -> EM={result.exact_match:.0f} F1={result.answer_f1:.3f} "
                  f"faith={faith_str} steps={result.steps}")
        else:
            print(f"  -> FAILED: {result.error}")

    agg_b = print_results(results_b, label="Condition B: Full System")

    # ── Comparison ──
    _print_ab_comparison(agg_a, agg_b, "Baseline (bare Agent)", "Full System")

    if args.output:
        save_results(results_a, Path(args.output).with_suffix(".baseline.json"))
        save_results(results_b, Path(args.output).with_suffix(".full.json"))

    return 0


# ═════════════════════════════════════════════════════════════════��═════
# Mode: Long-Term Memory A/B (Experiment 2)
# Round 1: seed long-term memory with 7 tasks
# Round 2a: run 7 related tasks WITH long-term memory
# Round 2b: run same 7 tasks WITHOUT long-term memory
# ═══════════════════════════════════════════════════════════════════════

async def run_ltm_ab(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    config = _resolve_env(config)

    llm_client, planner_client, auditor_client = _build_clients(config)
    model = str(config.get("agent", {}).get("model", "claude-sonnet-4-6"))

    synthesizer = SynthesisWriter(llm_client=llm_client, model=model)

    judge_client = auditor_client if auditor_client is not None else llm_client
    judge_model = str(config.get("llm_auditor", {}).get("model", model)) if auditor_client else model

    project_root = Path(__file__).resolve().parents[2]
    pairs_path = project_root / "deepresearch" / "data" / "memory_eval_pairs.jsonl"
    all_records = _load_dataset(pairs_path)

    round1_records = [r for r in all_records if r.get("round") == 1]
    round2_records = [r for r in all_records if r.get("round") == 2]

    print(f"Long-Term Memory A/B Experiment: {len(round1_records)} R1 + {len(round2_records)} R2 tasks")
    print(f"Config: {args.config}")
    print(f"Agent model: {model}")
    print(f"Judge model: {judge_model}  ({'vendor-independent' if auditor_client else 'self-eval (auditor not configured)'})")
    print()

    memory_path = args.memory_path or "deepresearch/data/memory_e2e.jsonl"

    # ── Round 1: Seed memory ──
    print("=" * 60)
    print("  ROUND 1: Seeding Long-Term Memory")
    print("=" * 60)

    config_r1 = dict(config)
    config_r1["memory"] = {"store_path": memory_path, "top_k": 3}
    agent_r1 = build_agent_from_config(config_r1, llm_client, planner_client=planner_client, judge_client=auditor_client)
    agent_r1.progress_callback = _progress_callback

    r1_results: List[TaskMetrics] = []
    for i, record in enumerate(round1_records, 1):
        print(f"[R1 {i}/{len(round1_records)}] {record['task_id']}: {str(record['query'])[:50]}...")
        result = await run_one_task(agent_r1, synthesizer, llm_client, record, extract_short=False, judge_client=judge_client)
        r1_results.append(result)
        if result.success:
            print(f"  -> F1={result.answer_f1:.3f} steps={result.steps}")
        else:
            print(f"  -> FAILED: {result.error}")

    print_results(r1_results, label="Round 1: Memory Seeding")

    # ── Round 2a: WITH memory ──
    print("\n" + "=" * 60)
    print("  ROUND 2a: WITH Long-Term Memory")
    print("=" * 60)

    config_2a = dict(config)
    config_2a["memory"] = {"store_path": memory_path, "top_k": 3}
    agent_2a = build_agent_from_config(config_2a, llm_client, planner_client=planner_client, judge_client=auditor_client)
    agent_2a.progress_callback = _progress_callback
    # Read-only: allow memory search but prevent writes (no pollution of seed memory)
    if agent_2a.memory_store is not None:
        agent_2a.memory_store.add = lambda **kwargs: None  # type: ignore[assignment]

    r2a_results: List[TaskMetrics] = []
    for i, record in enumerate(round2_records, 1):
        print(f"[R2a {i}/{len(round2_records)}] {record['task_id']}: {str(record['query'])[:50]}...")
        result = await run_one_task(agent_2a, synthesizer, llm_client, record, extract_short=False, judge_client=judge_client)
        r2a_results.append(result)
        if result.success:
            print(f"  -> F1={result.answer_f1:.3f} steps={result.steps}")
        else:
            print(f"  -> FAILED: {result.error}")

    agg_2a = print_results(r2a_results, label="Round 2a: WITH memory")

    # ── Round 2b: WITHOUT memory ──
    print("\n" + "=" * 60)
    print("  ROUND 2b: WITHOUT Long-Term Memory (baseline)")
    print("=" * 60)

    config_2b = dict(config)
    config_2b.pop("memory", None)
    agent_2b = build_agent_from_config(config_2b, llm_client, planner_client=planner_client, judge_client=auditor_client)
    agent_2b.progress_callback = _progress_callback

    r2b_results: List[TaskMetrics] = []
    for i, record in enumerate(round2_records, 1):
        print(f"[R2b {i}/{len(round2_records)}] {record['task_id']}: {str(record['query'])[:50]}...")
        result = await run_one_task(agent_2b, synthesizer, llm_client, record, extract_short=False, judge_client=judge_client)
        r2b_results.append(result)
        if result.success:
            print(f"  -> F1={result.answer_f1:.3f} steps={result.steps}")
        else:
            print(f"  -> FAILED: {result.error}")

    agg_2b = print_results(r2b_results, label="Round 2b: WITHOUT memory")

    # ── Comparison ──
    _print_ab_comparison(agg_2a, agg_2b, "With Long-Term Memory", "Without Long-Term Memory")

    # Per-pair comparison
    print(f"\nPer-pair breakdown:")
    for r2a, r2b in zip(r2a_results, r2b_results):
        step_d = r2a.steps - r2b.steps
        f1_d = r2a.answer_f1 - r2b.answer_f1
        print(f"  {r2a.task_id}: steps {r2a.steps} vs {r2b.steps} ({step_d:+d}), "
              f"F1 {r2a.answer_f1:.3f} vs {r2b.answer_f1:.3f} ({f1_d:+.3f})")

    if args.output:
        save_results(r2a_results, Path(args.output).with_suffix(".ltm.json"))
        save_results(r2b_results, Path(args.output).with_suffix(".no_ltm.json"))

    # Clean up
    mem_file = Path(memory_path)
    if mem_file.exists():
        print(f"\nMemory store retained at: {memory_path}")

    return 0


def _print_ab_comparison(agg_a: AggregateMetrics, agg_b: AggregateMetrics, label_a: str, label_b: str) -> None:
    """Print side-by-side comparison of two conditions."""
    print(f"\n{'=' * 60}")
    print(f"  A/B COMPARISON: {label_a} vs {label_b}")
    print(f"{'=' * 60}")

    header = f"{'Metric':<25} | {label_a:>14} | {label_b:>14} | {'Delta':>10}"
    print(f"\n{header}")
    print("-" * len(header))

    metrics = [
        ("Task Success Rate", agg_a.task_success_rate, agg_b.task_success_rate),
        ("Exact Match", agg_a.avg_exact_match, agg_b.avg_exact_match),
        ("Answer F1", agg_a.avg_answer_f1, agg_b.avg_answer_f1),
        ("Correctness (Judge)", agg_a.avg_correctness, agg_b.avg_correctness),
        ("Avg Steps", agg_a.avg_steps, agg_b.avg_steps),
        ("Subtask Completion", agg_a.avg_subtask_completion, agg_b.avg_subtask_completion),
        ("Faithfulness", agg_a.avg_faithfulness, agg_b.avg_faithfulness),
        ("Context Precision", agg_a.avg_context_precision, agg_b.avg_context_precision),
        ("Citation Precision", agg_a.avg_citation_precision, agg_b.avg_citation_precision),
        ("Over-Refusal Rate", agg_a.avg_over_refusal, agg_b.avg_over_refusal),
        ("Avg Time (s)", agg_a.avg_time, agg_b.avg_time),
    ]

    for name, va, vb in metrics:
        delta = va - vb
        sign = "+" if delta > 0 else ""
        # For steps, time, ORR — lower is better
        arrow = ""
        if name in ("Avg Steps", "Avg Time (s)", "Over-Refusal Rate"):
            arrow = " ↓" if delta < 0 else (" ↑" if delta > 0 else "")
        else:
            arrow = " ↑" if delta > 0 else (" ↓" if delta < 0 else "")
        print(f"{name:<25} | {va:>14.3f} | {vb:>14.3f} | {sign}{delta:>9.3f}{arrow}")

    print("-" * len(header))


# ═══════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DeepResearch Benchmark (8 standard metrics)")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--mode", choices=["full-ab", "ltm-ab"], default="full-ab",
                        help="Evaluation mode: full-ab (baseline vs full system), ltm-ab (long-term memory A/B)")
    parser.add_argument("--golden-set", default=None, help="Path to golden_set.jsonl")
    parser.add_argument("--max-tasks", type=int, default=None, help="Limit number of tasks")
    parser.add_argument("--memory-path", default=None, help="Memory store path for memory-ab mode")
    parser.add_argument("--output", default=None, help="Save results JSON to file")
    return parser.parse_args()


def main() -> int:
    try:
        args = _parse_args()
        if args.mode == "ltm-ab":
            return asyncio.run(run_ltm_ab(args))
        else:
            return asyncio.run(run_full_ab(args))
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 1
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
