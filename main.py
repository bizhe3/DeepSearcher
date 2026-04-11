"""Command-line interface for running DeepResearch tasks."""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from deepresearch.agent.synthesizer import SynthesisWriter
from deepresearch.agent.types import ResearchTask, Trajectory
from deepresearch.reward.llm_judge import LLMJudge
from deepresearch.utils.config import build_agent_from_config, load_config
from deepresearch.utils.llm_client import AnthropicClient, OpenAICompatibleClient

_ENV_PLACEHOLDER_PATTERN = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for a single DeepResearch run."""
    parser = argparse.ArgumentParser(description="Run DeepResearch from the command line.")
    parser.add_argument("--query", required=True, help="Research query to run.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--output", default=None, help="Optional output markdown file path.")
    return parser.parse_args()


def _resolve_env_placeholders(value: Any) -> Any:
    """Recursively resolve ${ENV_VAR} placeholders in config values."""
    if isinstance(value, dict):
        return {key: _resolve_env_placeholders(val) for key, val in value.items()}

    if isinstance(value, list):
        return [_resolve_env_placeholders(item) for item in value]

    if isinstance(value, str):
        match = _ENV_PLACEHOLDER_PATTERN.match(value.strip())
        if match:
            env_name = match.group(1)
            return os.environ.get(env_name, "")

    return value


def _format_markdown(answer_body: str, citations: list[str]) -> str:
    """Build final markdown output from synthesized answer and citations."""
    body = answer_body.strip()
    if not citations:
        return body

    references = "\n".join(f"- [{index}] {url}" for index, url in enumerate(citations, start=1))
    return f"{body}\n\n## References\n{references}"


async def _run_agent(agent: Any, task: ResearchTask) -> Trajectory:
    """Run agent with optional async context management on environment."""
    env = getattr(agent, "env", None)
    if env is not None and callable(getattr(env, "__aenter__", None)) and callable(getattr(env, "__aexit__", None)):
        async with env:
            return await agent.run(task)
    return await agent.run(task)


def _print_reward_breakdown(reward: Any) -> None:
    """Print reward component breakdown in a readable format."""
    print("\nReward Breakdown")
    print(f"- sub_goal: {reward.sub_goal:.4f}")
    print(f"- answer: {reward.answer:.4f}")
    print(f"- citation: {reward.citation:.4f}")
    print(f"- efficiency_penalty: {reward.efficiency_penalty:.4f}")
    print(f"- total: {reward.total:.4f}")


def _print_judge_score(score) -> None:
    print("\nLLM Judge Scores")
    print(f"- relevance:        {score.relevance}/5")
    print(f"- completeness:     {score.completeness}/5")
    print(f"- citation_quality: {score.citation_quality}/5")
    print(f"- total (weighted): {score.total:.4f}")
    print(f"- reasoning: {score.reasoning}")


async def async_main() -> int:
    """Execute one CLI run from config loading through synthesis and output."""
    args = parse_args()

    config = load_config(args.config)
    config = _resolve_env_placeholders(config)

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
        planner_model_name = str(planner_llm_cfg.get("model", "claude-haiku-4-5-20251001"))
        if planner_provider == "deepseek":
            planner_api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
            if not planner_api_key:
                raise ValueError("DEEPSEEK_API_KEY is not set for planner.")
            planner_base_url = str(planner_llm_cfg.get("base_url", "https://api.deepseek.com"))
            planner_client = OpenAICompatibleClient(
                api_key=planner_api_key, model=planner_model_name, base_url=planner_base_url
            )
        else:
            planner_api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
            if not planner_api_key:
                raise ValueError("ANTHROPIC_API_KEY is not set for planner.")
            planner_client = AnthropicClient(api_key=planner_api_key, model=planner_model_name)

    def _print_progress(event: dict) -> None:
        status = "ok" if event["success"] else f"err:{event['error']}"
        print(
            f"[{event['step']:02d}] {event['action_type']:<12} {status}  "
            f"| {event['goal'][:60]}"
        )

    agent = build_agent_from_config(config, llm_client, planner_client=planner_client)
    # Use planner_client for judging if available — the judge prompt is straightforward
    # and doesn't benefit from R1-style reasoning; using R1 risks truncation mid-JSON.
    judge_client = planner_client if planner_client is not None else llm_client
    planner_model_name = str(config.get("llm_planner", {}).get("model", model))
    judge_model = planner_model_name if planner_client is not None else model
    judge = LLMJudge(llm_client=judge_client, model=judge_model)
    agent.reward_engine.llm_judge = judge
    agent.progress_callback = _print_progress

    task = ResearchTask(task_id="cli_0", query=args.query)

    trajectory = await _run_agent(agent, task)

    synthesizer = SynthesisWriter(llm_client=llm_client, model=model)
    answer_body, citations = await synthesizer.synthesize(task.query, trajectory)

    trajectory.final_answer = answer_body
    trajectory.citations = citations

    output_markdown = _format_markdown(answer_body, citations)
    print(output_markdown)

    if args.output:
        output_path = Path(args.output)
        if output_path.parent and str(output_path.parent) not in {"", "."}:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_markdown, encoding="utf-8")

    reward = await agent.reward_engine.compute_with_judge(trajectory, task)
    agent.last_reward = reward
    _print_reward_breakdown(reward)
    if trajectory.judge_score is not None:
        _print_judge_score(trajectory.judge_score)
    if hasattr(agent.llm_client, "total_input_tokens"):
        print(
            f"\nToken usage: input={agent.llm_client.total_input_tokens} "
            f"output={agent.llm_client.total_output_tokens}"
        )

    return 0


def main() -> int:
    """CLI entrypoint with user-friendly error reporting."""
    try:
        return asyncio.run(async_main())
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

