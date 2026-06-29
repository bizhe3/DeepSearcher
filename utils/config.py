"""Configuration loading and dependency wiring for DeepResearch.

Example YAML shape:

agent:
  planner_model: claude-haiku-4-5-20251001
  decision_model: claude-sonnet-4-6
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from deepresearch.agent.agent import DeepResearchAgent
from deepresearch.agent.auditor import SubGoalAuditor
from deepresearch.agent.planner import SubGoalDecomposer
from deepresearch.envs.real_web_env import RealWebEnv
from deepresearch.envs.sim_env import SimEnv
from deepresearch.reward.reward_engine import RewardEngine
from deepresearch.utils.chunk_retriever import PageChunkRetriever, ObservationRetriever
from deepresearch.utils.memory_store import MemoryStore


def load_config(path: str) -> dict:
    """Load a YAML config file and return its dictionary representation."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file)

    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config at {path} must deserialize to a dictionary.")

    return loaded


def build_agent_from_config(
    config: dict,
    llm_client: Any,
    planner_client: Any = None,
    judge_client: Any = None,
) -> DeepResearchAgent:
    """Build a fully wired DeepResearchAgent from a validated config dictionary."""
    env_cfg: Dict[str, Any] = dict(config.get("env", {}))
    agent_cfg: Dict[str, Any] = dict(config.get("agent", {}))
    reward_cfg: Dict[str, Any] = dict(config.get("reward", {}))

    env_type = str(env_cfg.get("type", "sim")).lower()
    if env_type == "sim":
        env = SimEnv(
            corpus_dir=str(env_cfg.get("corpus_dir", "data/corpus")),
            embedding_model=str(
                env_cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
            ),
            top_k_default=int(env_cfg.get("top_k", 5)),
        )
    elif env_type == "real":
        search_api_key = str(env_cfg.get("search_api_key", "")).strip()
        if not search_api_key:
            raise ValueError("env.search_api_key is required when env.type is 'real'.")

        env = RealWebEnv(
            search_api_key=search_api_key,
            search_engine=str(env_cfg.get("search_engine", "bing")).lower(),
            headless=_as_bool(env_cfg.get("headless", True)),
            request_delay=float(env_cfg.get("request_delay", 1.0)),
        )
    else:
        raise ValueError(f"Unsupported env.type: {env_type}")

    planner_model = str(agent_cfg.get("planner_model", "claude-haiku-4-5-20251001"))
    decision_model = str(
        agent_cfg.get(
            "decision_model",
            agent_cfg.get("model", "claude-sonnet-4-6"),
        )
    )

    effective_planner_client = planner_client if planner_client is not None else llm_client

    planner = SubGoalDecomposer(
        llm_client=effective_planner_client,
        planner_model=planner_model,
        max_sub_goals=int(agent_cfg.get("max_sub_goals", 6)),
    )

    reward_engine = RewardEngine(
        sub_goal_weight=float(reward_cfg.get("sub_goal_weight", 0.2)),
        answer_weight=float(reward_cfg.get("answer_weight", 1.0)),
        citation_weight=float(reward_cfg.get("citation_weight", 0.3)),
        step_penalty=float(reward_cfg.get("step_penalty", 0.01)),
        expected_citations=int(reward_cfg.get("expected_citations", 3)),
    )

    memory_cfg: Dict[str, Any] = dict(config.get("memory", {}))
    memory_store_path = str(memory_cfg.get("store_path", "")).strip()
    memory_store = None
    if memory_store_path:
        # memory_writer LLM: prefer judge_client (vendor-independent) > planner > main llm
        memory_writer_client = (
            judge_client
            if judge_client is not None
            else (planner_client if planner_client is not None else llm_client)
        )
        memory_writer_model = str(
            memory_cfg.get(
                "writer_model",
                config.get("llm_auditor", {}).get(
                    "model", agent_cfg.get("planner_model", "qwen-turbo")
                ),
            )
        )
        memory_store = MemoryStore(
            store_path=memory_store_path,
            top_k=int(memory_cfg.get("top_k", 3)),
            llm_client=memory_writer_client,
            model=memory_writer_model,
        )

    # Auditor: independent judge LLM that tags every step's sub-goal quality.
    # Prefer dedicated judge_client > planner_client > llm_client (fallback).
    auditor_cfg: Dict[str, Any] = dict(config.get("auditor", {}))
    auditor = None
    if _as_bool(auditor_cfg.get("enabled", True)):
        auditor_judge_client = (
            judge_client
            if judge_client is not None
            else (planner_client if planner_client is not None else llm_client)
        )
        auditor_model = str(
            auditor_cfg.get(
                "model",
                agent_cfg.get("planner_model", "claude-haiku-4-5-20251001"),
            )
        )
        auditor = SubGoalAuditor(judge_client=auditor_judge_client, model=auditor_model)

    return DeepResearchAgent(
        env=env,
        planner=planner,
        reward_engine=reward_engine,
        llm_client=llm_client,
        decision_model=decision_model,
        max_steps=int(agent_cfg.get("max_steps", 20)),
        chunk_retriever=PageChunkRetriever(top_k=8),
        obs_retriever=ObservationRetriever(),
        memory_store=memory_store,
        auditor=auditor,
        replan_cooldown_steps=int(auditor_cfg.get("replan_cooldown_steps", 3)),
        max_global_replans=int(auditor_cfg.get("max_global_replans", 4)),
        max_redo_per_subgoal=int(auditor_cfg.get("max_redo_per_subgoal", 1)),
    )


def _as_bool(value: Any) -> bool:
    """Parse common bool-like values without relying on Python truthiness quirks."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)

