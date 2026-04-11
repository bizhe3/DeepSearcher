"""End-to-end evaluation with real LLM and real web search.

Runs each task in golden_set.jsonl through the full Agent pipeline
(real LLM + real web environment), computes reward and LLM Judge scores,
and prints a comparison table.

Requires:
    DEEPSEEK_API_KEY  — for LLM calls
    SERPAPI_KEY        — for web search (or BING_API_KEY with --search-engine bing)

Usage:
    cd d:/Agent
    python -m deepresearch.tests.eval_e2e --config configs/deepseek_eval.yaml
    python -m deepresearch.tests.eval_e2e --config configs/deepseek_r1_v3.yaml

Optional flags:
    --max-tasks N      Only run first N tasks (for quick smoke test)
    --no-memory        Disable long-term memory (baseline run)
    --memory-path P    Path to memory_store.jsonl (default: data/memory_e2e.jsonl)
    --output PATH      Save detailed results as JSON
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is on path
if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from deepresearch.agent.agent import DeepResearchAgent
from deepresearch.agent.synthesizer import SynthesisWriter
from deepresearch.agent.types import ResearchTask
from deepresearch.reward.llm_judge import LLMJudge
from deepresearch.reward.reward_engine import RewardBreakdown
from deepresearch.utils.config import build_agent_from_config, load_config
from deepresearch.utils.llm_client import AnthropicClient, OpenAICompatibleClient
from deepresearch.utils.memory_store import MemoryStore

_ENV_PLACEHOLDER = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")


@dataclass
class TaskResult:
    task_id: str
    query: str
    steps: int
    reward: RewardBreakdown
    judge_relevance: int
    judge_completeness: int
    judge_citation: int
    judge_total: float
    judge_reasoning: str
    elapsed_seconds: float
    success: bool
    error: Optional[str] = None


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


def _load_golden_set(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Line {line_num}: expected JSON object")
            records.append(record)
    return records


def _build_clients(config: dict):
    """Build LLM clients from config, same logic as main.py."""
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

    planner_client = None
    planner_llm_cfg = config.get("llm_planner", {})
    if planner_llm_cfg:
        planner_provider = str(planner_llm_cfg.get("provider", "anthropic")).lower()
        planner_model = str(planner_llm_cfg.get("model", "claude-haiku-4-5-20251001"))
        if planner_provider == "deepseek":
            pk = os.environ.get("DEEPSEEK_API_KEY", "").strip()
            if not pk:
                raise ValueError("DEEPSEEK_API_KEY not set for planner.")
            planner_client = OpenAICompatibleClient(
                api_key=pk, model=planner_model,
                base_url=str(planner_llm_cfg.get("base_url", "https://api.deepseek.com")),
            )
        else:
            pk = os.environ.get("ANTHROPIC_API_KEY", "").strip()
            if not pk:
                raise ValueError("ANTHROPIC_API_KEY not set for planner.")
            planner_client = AnthropicClient(api_key=pk, model=planner_model)

    return llm_client, planner_client


async def _run_one_task(
    agent: DeepResearchAgent,
    synthesizer: SynthesisWriter,
    judge: LLMJudge,
    record: Dict[str, Any],
) -> TaskResult:
    """Run a single task end-to-end and return structured result."""
    task = ResearchTask(
        task_id=str(record["task_id"]),
        query=str(record["query"]),
        reference_answer=record.get("reference_answer"),
        key_facts=record.get("key_facts"),
        expected_sub_goals=record.get("expected_sub_goals"),
    )

    t0 = time.monotonic()
    error_msg = None
    try:
        # Run agent with env context manager (for RealWebEnv browser lifecycle)
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

        # Compute reward (with ROUGE-L since we have reference_answer)
        reward = agent.reward_engine.compute(
            trajectory, reference_answer=task.reference_answer
        )

        # LLM Judge scoring
        judge_score = await judge.judge(
            query=task.query,
            answer=trajectory.final_answer or "",
            citations=trajectory.citations,
            key_facts=task.key_facts,
        )
        trajectory.judge_score = judge_score

        elapsed = time.monotonic() - t0
        return TaskResult(
            task_id=task.task_id,
            query=task.query[:60],
            steps=len(trajectory.observations),
            reward=reward,
            judge_relevance=judge_score.relevance,
            judge_completeness=judge_score.completeness,
            judge_citation=judge_score.citation_quality,
            judge_total=judge_score.total,
            judge_reasoning=judge_score.reasoning[:100],
            elapsed_seconds=round(elapsed, 1),
            success=True,
        )

    except Exception as exc:
        elapsed = time.monotonic() - t0
        error_msg = str(exc)[:200]
        return TaskResult(
            task_id=str(record["task_id"]),
            query=str(record["query"])[:60],
            steps=0,
            reward=RewardBreakdown(sub_goal=0, answer=0, citation=0, efficiency_penalty=0, total=0),
            judge_relevance=0,
            judge_completeness=0,
            judge_citation=0,
            judge_total=0,
            judge_reasoning="",
            elapsed_seconds=round(elapsed, 1),
            success=False,
            error=error_msg,
        )


def _print_results(results: List[TaskResult]) -> None:
    """Print formatted result table."""
    header = (
        f"{'task_id':<14} | {'steps':>5} | {'reward':>6} | "
        f"{'rel':>3} {'comp':>4} {'cite':>4} {'judge':>6} | "
        f"{'time':>6} | {'status':<6}"
    )
    print("\n" + header)
    print("-" * len(header))

    for r in results:
        status = "OK" if r.success else f"ERR"
        print(
            f"{r.task_id:<14} | {r.steps:>5} | {r.reward.total:>6.3f} | "
            f"{r.judge_relevance:>3} {r.judge_completeness:>4} {r.judge_citation:>4} {r.judge_total:>6.3f} | "
            f"{r.elapsed_seconds:>5.1f}s | {status:<6}"
        )

    print("-" * len(header))

    n = max(1, len(results))
    successful = [r for r in results if r.success]
    ns = max(1, len(successful))

    avg_steps = sum(r.steps for r in successful) / ns
    avg_reward = sum(r.reward.total for r in successful) / ns
    avg_judge = sum(r.judge_total for r in successful) / ns
    avg_time = sum(r.elapsed_seconds for r in successful) / ns
    success_rate = len(successful) / n

    print(
        f"{'AVERAGE':<14} | {avg_steps:>5.1f} | {avg_reward:>6.3f} | "
        f"{'':>3} {'':>4} {'':>4} {avg_judge:>6.3f} | "
        f"{avg_time:>5.1f}s | {success_rate:.0%}"
    )

    print(f"\nSuccess rate: {len(successful)}/{len(results)} tasks")
    if results:
        print(f"Avg steps: {avg_steps:.1f}")
        print(f"Avg reward total: {avg_reward:.3f}")
        print(f"Avg judge total: {avg_judge:.3f}")
        print(f"Avg time: {avg_time:.1f}s")

    # Print failures
    failures = [r for r in results if not r.success]
    if failures:
        print(f"\nFailed tasks ({len(failures)}):")
        for r in failures:
            print(f"  {r.task_id}: {r.error}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end Agent evaluation")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--golden-set", default=None, help="Path to golden_set.jsonl")
    parser.add_argument("--max-tasks", type=int, default=None, help="Limit number of tasks")
    parser.add_argument("--no-memory", action="store_true", help="Disable long-term memory")
    parser.add_argument("--memory-path", default="data/memory_e2e.jsonl", help="Memory store path")
    parser.add_argument("--output", default=None, help="Save results JSON to file")
    parser.add_argument(
        "--memory-ab", action="store_true",
        help="Run memory A/B test: load memory_eval_pairs.jsonl, run Round 1 to seed memory, "
             "then run Round 2 twice (with/without memory) and compare.",
    )
    return parser.parse_args()


async def async_main() -> int:
    args = _parse_args()

    # Load config
    config = load_config(args.config)
    config = _resolve_env(config)

    # Inject memory config unless disabled
    if not args.no_memory:
        config.setdefault("memory", {})
        config["memory"]["store_path"] = args.memory_path
        config["memory"].setdefault("top_k", 3)

    # Build clients
    llm_client, planner_client = _build_clients(config)

    # Build agent
    agent = build_agent_from_config(config, llm_client, planner_client=planner_client)

    # Progress callback
    def _progress(event: dict) -> None:
        status = "ok" if event["success"] else f"err:{event['error']}"
        print(f"  [{event['step']:02d}] {event['action_type']:<12} {status} | {event['goal'][:50]}")

    agent.progress_callback = _progress

    # Build synthesizer and judge
    model = str(config.get("agent", {}).get("model", "claude-sonnet-4-6"))
    synthesizer = SynthesisWriter(llm_client=llm_client, model=model)

    judge_client = planner_client if planner_client is not None else llm_client
    judge_model = str(config.get("llm_planner", {}).get("model", model))
    judge = LLMJudge(llm_client=judge_client, model=judge_model)

    # Load golden set
    project_root = Path(__file__).resolve().parents[2]
    golden_path = Path(args.golden_set) if args.golden_set else project_root / "data" / "golden_set.jsonl"
    records = _load_golden_set(golden_path)

    if args.max_tasks:
        records = records[:args.max_tasks]

    print(f"Loaded {len(records)} tasks from {golden_path}")
    print(f"Config: {args.config}")
    print(f"Memory: {'disabled' if args.no_memory else args.memory_path}")
    print(f"Model: {model}")
    print()

    # Run tasks sequentially
    results: List[TaskResult] = []
    for i, record in enumerate(records, 1):
        task_id = record["task_id"]
        query_preview = str(record["query"])[:50]
        print(f"[{i}/{len(records)}] {task_id}: {query_preview}...")

        result = await _run_one_task(agent, synthesizer, judge, record)
        results.append(result)

        if result.success:
            print(f"  -> reward={result.reward.total:.3f} judge={result.judge_total:.3f} "
                  f"steps={result.steps} time={result.elapsed_seconds}s")
        else:
            print(f"  -> FAILED: {result.error}")
        print()

    # Print summary
    _print_results(results)

    # Save detailed results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        serializable = []
        for r in results:
            d = {
                "task_id": r.task_id,
                "query": r.query,
                "steps": r.steps,
                "reward_total": r.reward.total,
                "reward_sub_goal": r.reward.sub_goal,
                "reward_answer": r.reward.answer,
                "reward_citation": r.reward.citation,
                "judge_relevance": r.judge_relevance,
                "judge_completeness": r.judge_completeness,
                "judge_citation": r.judge_citation,
                "judge_total": r.judge_total,
                "elapsed_seconds": r.elapsed_seconds,
                "success": r.success,
                "error": r.error,
            }
            serializable.append(d)
        output_path.write_text(json.dumps(serializable, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nDetailed results saved to {output_path}")

    return 0


async def _run_memory_ab(args: argparse.Namespace) -> int:
    """Run two-round memory A/B test using memory_eval_pairs.jsonl.

    Round 1: Run all round=1 tasks → seeds memory_store
    Round 2a: Run round=2 tasks WITH memory (can retrieve Round 1 results)
    Round 2b: Run round=2 tasks WITHOUT memory (fresh, no prior knowledge)
    Compare Round 2a vs 2b.
    """
    config = load_config(args.config)
    config = _resolve_env(config)

    # Load paired dataset
    project_root = Path(__file__).resolve().parents[2]
    pairs_path = project_root / "data" / "memory_eval_pairs.jsonl"
    all_records = _load_golden_set(pairs_path)

    round1_records = [r for r in all_records if r.get("round") == 1]
    round2_records = [r for r in all_records if r.get("round") == 2]

    print(f"Loaded {len(round1_records)} Round 1 tasks + {len(round2_records)} Round 2 tasks")
    print(f"Config: {args.config}")
    print()

    llm_client, planner_client = _build_clients(config)
    model = str(config.get("agent", {}).get("model", "claude-sonnet-4-6"))
    synthesizer = SynthesisWriter(llm_client=llm_client, model=model)
    judge_client = planner_client if planner_client is not None else llm_client
    judge_model = str(config.get("llm_planner", {}).get("model", model))
    judge = LLMJudge(llm_client=judge_client, model=judge_model)

    def _progress(event: dict) -> None:
        status = "ok" if event["success"] else f"err:{event['error']}"
        print(f"  [{event['step']:02d}] {event['action_type']:<12} {status} | {event['goal'][:50]}")

    memory_path = args.memory_path

    # ── Round 1: Seed memory ──
    print("=" * 60)
    print("ROUND 1: Seeding long-term memory")
    print("=" * 60)

    config_r1 = dict(config)
    config_r1["memory"] = {"store_path": memory_path, "top_k": 3}
    agent_r1 = build_agent_from_config(config_r1, llm_client, planner_client=planner_client)
    agent_r1.progress_callback = _progress

    round1_results: List[TaskResult] = []
    for i, record in enumerate(round1_records, 1):
        print(f"\n[R1 {i}/{len(round1_records)}] {record['task_id']}: {str(record['query'])[:50]}...")
        result = await _run_one_task(agent_r1, synthesizer, judge, record)
        round1_results.append(result)
        if result.success:
            print(f"  -> reward={result.reward.total:.3f} steps={result.steps}")
        else:
            print(f"  -> FAILED: {result.error}")

    print(f"\nRound 1 complete. Memory store seeded with {len([r for r in round1_results if r.success])} entries.")
    _print_results(round1_results)

    # ── Round 2a: WITH memory ──
    print("\n" + "=" * 60)
    print("ROUND 2a: Running with long-term memory")
    print("=" * 60)

    config_2a = dict(config)
    config_2a["memory"] = {"store_path": memory_path, "top_k": 3}
    agent_2a = build_agent_from_config(config_2a, llm_client, planner_client=planner_client)
    agent_2a.progress_callback = _progress

    round2a_results: List[TaskResult] = []
    for i, record in enumerate(round2_records, 1):
        print(f"\n[R2a {i}/{len(round2_records)}] {record['task_id']}: {str(record['query'])[:50]}...")
        result = await _run_one_task(agent_2a, synthesizer, judge, record)
        round2a_results.append(result)
        if result.success:
            print(f"  -> reward={result.reward.total:.3f} steps={result.steps}")
        else:
            print(f"  -> FAILED: {result.error}")

    print("\nRound 2a (WITH memory):")
    _print_results(round2a_results)

    # ── Round 2b: WITHOUT memory ──
    print("\n" + "=" * 60)
    print("ROUND 2b: Running WITHOUT long-term memory (baseline)")
    print("=" * 60)

    config_2b = dict(config)
    config_2b.pop("memory", None)
    agent_2b = build_agent_from_config(config_2b, llm_client, planner_client=planner_client)
    agent_2b.progress_callback = _progress

    round2b_results: List[TaskResult] = []
    for i, record in enumerate(round2_records, 1):
        print(f"\n[R2b {i}/{len(round2_records)}] {record['task_id']}: {str(record['query'])[:50]}...")
        result = await _run_one_task(agent_2b, synthesizer, judge, record)
        round2b_results.append(result)
        if result.success:
            print(f"  -> reward={result.reward.total:.3f} steps={result.steps}")
        else:
            print(f"  -> FAILED: {result.error}")

    print("\nRound 2b (WITHOUT memory):")
    _print_results(round2b_results)

    # ── Comparison ──
    print("\n" + "=" * 60)
    print("MEMORY A/B COMPARISON (Round 2: with vs without memory)")
    print("=" * 60)

    def _avg(results: List[TaskResult], field: str) -> float:
        ok = [r for r in results if r.success]
        if not ok:
            return 0.0
        if field == "steps":
            return sum(r.steps for r in ok) / len(ok)
        if field == "reward":
            return sum(r.reward.total for r in ok) / len(ok)
        if field == "judge":
            return sum(r.judge_total for r in ok) / len(ok)
        if field == "time":
            return sum(r.elapsed_seconds for r in ok) / len(ok)
        return 0.0

    header = f"{'':>20} | {'With Memory':>12} | {'No Memory':>12} | {'Delta':>10}"
    print(f"\n{header}")
    print("-" * len(header))

    for field, label in [("steps", "Avg Steps"), ("reward", "Avg Reward"), ("judge", "Avg Judge"), ("time", "Avg Time (s)")]:
        val_a = _avg(round2a_results, field)
        val_b = _avg(round2b_results, field)
        delta = val_a - val_b
        sign = "+" if delta > 0 else ""
        # For steps and time, lower is better → negative delta is good
        print(f"{label:>20} | {val_a:>12.2f} | {val_b:>12.2f} | {sign}{delta:>9.2f}")

    # Per-pair comparison
    print(f"\nPer-pair breakdown:")
    pair_ids = list(dict.fromkeys(r.get("pair_id", "") for r in round2_records))
    for pair_id in pair_ids:
        r2a = next((r for r in round2a_results if r.task_id == f"{pair_id.replace('pair', 'mem')}b"), None)
        r2b = next((r for r in round2b_results if r.task_id == f"{pair_id.replace('pair', 'mem')}b"), None)
        if not r2a or not r2b:
            # Try matching by index
            idx_a = pair_ids.index(pair_id)
            r2a = round2a_results[idx_a] if idx_a < len(round2a_results) else None
            r2b = round2b_results[idx_a] if idx_a < len(round2b_results) else None
        if r2a and r2b:
            step_delta = r2a.steps - r2b.steps
            judge_delta = r2a.judge_total - r2b.judge_total
            step_sign = "+" if step_delta > 0 else ""
            judge_sign = "+" if judge_delta > 0 else ""
            print(f"  {r2a.task_id}: steps {r2a.steps} vs {r2b.steps} ({step_sign}{step_delta}), "
                  f"judge {r2a.judge_total:.3f} vs {r2b.judge_total:.3f} ({judge_sign}{judge_delta:.3f})")

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "round1": [{"task_id": r.task_id, "steps": r.steps, "reward": r.reward.total, "judge": r.judge_total, "success": r.success} for r in round1_results],
            "round2_with_memory": [{"task_id": r.task_id, "steps": r.steps, "reward": r.reward.total, "judge": r.judge_total, "success": r.success} for r in round2a_results],
            "round2_no_memory": [{"task_id": r.task_id, "steps": r.steps, "reward": r.reward.total, "judge": r.judge_total, "success": r.success} for r in round2b_results],
        }
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nResults saved to {output_path}")

    # Clean up memory file used for test
    memory_file = Path(memory_path)
    if memory_file.exists():
        print(f"Memory store at {memory_path} retained for inspection.")

    return 0


def main() -> int:
    try:
        args = _parse_args()
        if args.memory_ab:
            return asyncio.run(_run_memory_ab(args))
        return asyncio.run(async_main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 1
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
