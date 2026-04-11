"""Top-level DeepResearch agent orchestration module."""

from __future__ import annotations

import inspect
import json
import logging
import os
import re
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

from deepresearch.agent.planner import SubGoalDecomposer
from deepresearch.agent.types import (
    AgentAction,
    AgentObservation,
    PageContent,
    ResearchTask,
    SearchResult,
    SubGoal,
    Trajectory,
)
from deepresearch.envs.base_env import BaseEnv
from deepresearch.reward import subgoal_reward
from deepresearch.reward.reward_engine import RewardBreakdown, RewardEngine
from deepresearch.utils.checkpoint import load_checkpoint, save_checkpoint
from deepresearch.utils.chunk_retriever import ObservationRetriever, PageChunkRetriever
from deepresearch.utils.memory_store import MemoryStore

logger = logging.getLogger(__name__)

ACTION_SPACE_SCHEMA = """
Choose exactly one action per step. Decision guide:

| action_type  | when to use                                                        | required params                          |
|--------------|--------------------------------------------------------------------|------------------------------------------|
| search       | Need new information; no known URL yet                             | query (str), top_k (int, optional)       |
| extract      | Have a specific URL and need its full content                      | url (str)                                |
| click        | Found a promising link in the current page and want to follow it   | link_url (str)                           |
| scroll       | Current page is paginated and more results are on the next page    | (none, uses current page context)        |
| cross_check  | Need to verify a fact from a different source                      | query (str), top_k (int, optional)       |
| terminate    | Sub-goals are complete and enough information is gathered          | answer (str), citations (list, optional) |

JSON format: {"action_type": "<type>", "params": {<params>}, "step": <N>}

Decision rules:
- Prefer extract over click when you already have the URL from a prior search result.
- Use scroll only when the current page has pagination and partial results.
- Use cross_check when the active sub-goal explicitly requires verifying or confirming a previously found fact (e.g. a market share %, revenue figure, or ranking). Do NOT use search as a substitute for cross_check in these cases.
- Terminate only when the active sub-goal is fully addressed.
"""


class DeepResearchAgent:
    """Coordinate planning, environment interaction, and reward evaluation."""

    def __init__(
        self,
        env: BaseEnv,
        planner: SubGoalDecomposer,
        reward_engine: RewardEngine,
        llm_client: Any,
        decision_model: str = "claude-sonnet-4-6",
        model: Optional[str] = None,
        max_steps: int = 20,
        stall_threshold: int = 3,
        progress_callback: Optional[Callable[[dict], None]] = None,
        checkpoint_dir: Optional[str] = None,
        chunk_retriever: Optional[PageChunkRetriever] = None,
        obs_retriever: Optional[ObservationRetriever] = None,
        memory_store: Optional[MemoryStore] = None,
    ) -> None:
        """Initialize the agent with environment, planning, reward, and LLM dependencies."""
        self.env = env
        self.planner = planner
        self.reward_engine = reward_engine
        self.llm_client = llm_client
        self.decision_model = model if model is not None else decision_model
        self.model = self.decision_model
        self.max_steps = max_steps
        self.stall_threshold = stall_threshold
        self.progress_callback = progress_callback
        self.checkpoint_dir = checkpoint_dir
        self._goal_attempt_counts: dict[str, int] = {}
        self.last_reward: Optional[RewardBreakdown] = None
        self.chunk_retriever = chunk_retriever or PageChunkRetriever()
        self.obs_retriever = obs_retriever or ObservationRetriever()
        self.memory_store = memory_store

    async def run(self, task: ResearchTask) -> Trajectory:
        """Execute a research task and return its full trajectory."""
        self._goal_attempt_counts = {}

        # Long-term memory: retrieve prior research relevant to this task
        self._prior_knowledge: List[str] = []
        if self.memory_store is not None:
            prior_entries = self.memory_store.search(task.query)
            self._prior_knowledge = [entry.format_context() for entry in prior_entries]
            if self._prior_knowledge:
                logger.info("memory_store: retrieved %d prior entries for task=%s",
                            len(self._prior_knowledge), task.task_id)

        checkpoint_path: Optional[str] = None
        trajectory: Optional[Trajectory] = None
        if self.checkpoint_dir:
            checkpoint_path = os.path.join(self.checkpoint_dir, f"{task.task_id}.json")
            restored = load_checkpoint(checkpoint_path)
            if restored is not None:
                if restored.final_answer is not None:
                    self.last_reward = self.reward_engine.compute(
                        restored, reference_answer=getattr(task, "reference_answer", None)
                    )
                    return restored

                for sub_goal in restored.sub_goals:
                    if sub_goal.status == "active":
                        sub_goal.status = "pending"
                trajectory = restored

        if trajectory is None:
            sub_goals = await self.planner.decompose(task.query)
            trajectory = Trajectory(
                task=task.query,
                sub_goals=sub_goals,
                observations=[],
                final_answer=None,
                citations=[],
            )

        current_page_context: Optional[PageContent] = None
        if trajectory.observations and isinstance(trajectory.observations[-1].result, PageContent):
            current_page_context = trajectory.observations[-1].result

        start_step = len(trajectory.observations) + 1
        for step in range(start_step, self.max_steps + 1):
            active_sub_goal = self.planner.get_active_goal(trajectory.sub_goals)
            if active_sub_goal is None:
                break

            self._goal_attempt_counts[active_sub_goal.id] = (
                self._goal_attempt_counts.get(active_sub_goal.id, 0) + 1
            )

            action = await self._decide_action(trajectory, active_sub_goal)
            if action.step != step:
                action = action.model_copy(update={"step": step})

            logger.info(f"step={step} action={action.action_type} goal={active_sub_goal.id}")
            observation = await self.env.execute_action(action, current_page_context)

            # Fallback: if extract/click failed, substitute cached search snippet if available
            if not observation.success and action.action_type in ("extract", "click"):
                failed_url = action.params.get("url") or action.params.get("link_url", "")
                snippet = self._find_cached_snippet(trajectory, failed_url)
                if snippet:
                    observation = observation.model_copy(update={
                        "success": True,
                        "result": snippet,
                        "error": None,
                    })
                    logger.info(f"step={step} fallback to cached snippet for {failed_url}")

            # RAG 改动一：对长页面内容做语义分块过滤，只保留与当前子目标相关的片段
            if (
                observation.success
                and isinstance(observation.result, PageContent)
                and len(observation.result.body) > 1000
            ):
                filtered_body = self.chunk_retriever.filter(
                    body=observation.result.body,
                    query=active_sub_goal.description,
                )
                observation = observation.model_copy(
                    update={
                        "result": observation.result.model_copy(
                            update={"body": filtered_body}
                        )
                    }
                )

            logger.info(f"step={step} success={observation.success} error={observation.error}")
            if self.progress_callback:
                self.progress_callback(
                    {
                        "step": step,
                        "action_type": action.action_type,
                        "goal": active_sub_goal.description,
                        "success": observation.success,
                        "error": observation.error,
                    }
                )
            trajectory.observations.append(observation)
            if checkpoint_path:
                save_checkpoint(trajectory, checkpoint_path)

            if isinstance(observation.result, PageContent):
                current_page_context = observation.result

            completed = subgoal_reward.detect_completion(active_sub_goal, trajectory)
            if completed:
                active_sub_goal.status = "completed"
                active_sub_goal.completed_at_step = step
                if active_sub_goal.summary is None:
                    active_sub_goal.summary = await self._compress_sub_goal(
                        active_sub_goal, trajectory.observations
                    )
                    logger.info(
                        f"sub_goal={active_sub_goal.id} "
                        f"summary={active_sub_goal.summary[:80]!r}"
                    )
            elif active_sub_goal.status == "active":
                active_sub_goal.status = "pending"

            if action.action_type == "terminate":
                final_answer, citations = self._extract_terminate_payload(action, observation)
                trajectory.final_answer = final_answer
                trajectory.citations = citations
                if active_sub_goal.status == "active":
                    active_sub_goal.status = "completed"
                    active_sub_goal.completed_at_step = step
                break

            if (
                not completed
                and self._goal_attempt_counts.get(active_sub_goal.id, 0) >= self.stall_threshold
            ):
                active_sub_goal.status = "failed"
                continue

            if step % 5 == 0:
                completed_goals = [goal for goal in trajectory.sub_goals if goal.status == "completed"]
                new_info = await self._summarize_observations(trajectory.observations)
                replanned = await self.planner.replan(task.query, completed_goals, new_info)
                trajectory.sub_goals = self._merge_replanned_sub_goals(trajectory.sub_goals, replanned)

        self.last_reward = self.reward_engine.compute(
            trajectory, reference_answer=getattr(task, "reference_answer", None)
        )

        # Long-term memory: persist this research for future sessions
        if self.memory_store is not None and trajectory.final_answer:
            summaries = [
                g.summary for g in trajectory.sub_goals
                if g.status == "completed" and g.summary
            ]
            combined_summary = (
                " ".join(summaries) if summaries
                else trajectory.final_answer[:500]
            )
            self.memory_store.add(
                task_id=task.task_id,
                query=task.query,
                summary=combined_summary,
                citations=trajectory.citations,
                key_facts=trajectory.key_facts,
            )

        return trajectory

    async def _decide_action(self, trajectory: Trajectory, active_goal: SubGoal) -> AgentAction:
        """Choose the next action by prompting the LLM with task state context."""
        observations_summary = self._build_hierarchical_context(trajectory, active_goal)

        system_prompt = (
            "You are the action policy for a web research agent. "
            "Output exactly one JSON action object. "
            "Format:\n"
            '{"action_type": "...", "params": {...}, "step": N}'
        )
        user_prompt = (
            f"Original task:\n{trajectory.task}\n\n"
            f"Active sub-goal:\n{active_goal.description}\n\n"
            f"Last observations:\n{observations_summary}\n\n"
            f"Action guide:\n{ACTION_SPACE_SCHEMA}\n\n"
            "Output requirements:\n"
            "- Return ONLY one JSON object\n"
            "- Choose the action_type that best fits the current situation using the decision guide above\n"
            "- Provide all required params for the chosen action\n"
            "- step must be the next integer step"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response_text = await self._chat_completion(messages=messages, response_format="text")

        try:
            thought, action_json = self._extract_thought_and_action(response_text)
            payload = json.loads(action_json)
        except Exception:
            # Retry with a minimal prompt asking only for JSON
            logger.warning("step=%d JSON parse failed, retrying with minimal prompt", len(trajectory.observations) + 1)
            retry_prompt = (
                f"Output ONLY a JSON object for this research task.\n"
                f"Task: {trajectory.task}\n"
                f"Current goal: {active_goal.description}\n"
                f'Format: {{"action_type": "search"|"extract"|"click"|"scroll"|"cross_check"|"terminate", "params": {{}}, "step": {len(trajectory.observations) + 1}}}'
            )
            response_text = await self._chat_completion(
                messages=[{"role": "user", "content": retry_prompt}],
                response_format="text",
            )
            try:
                thought, action_json = self._extract_thought_and_action(response_text)
                payload = json.loads(action_json)
            except Exception as error:
                raise ValueError("Failed to parse action JSON from LLM response.") from error

        if not isinstance(payload, dict):
            raise ValueError("Action payload must be a JSON object.")

        payload = self._normalize_action_payload(payload)
        payload.setdefault("params", {})
        payload.setdefault("step", len(trajectory.observations) + 1)

        action = AgentAction(**payload)
        if thought:
            action = action.model_copy(update={"thought": thought})
        return action

    async def _compress_sub_goal(
        self,
        sub_goal: SubGoal,
        observations: List[AgentObservation],
    ) -> str:
        """Compress a completed sub-goal's observations into a 1-2 sentence summary."""
        successful_obs = [obs for obs in observations if obs.success]
        if not successful_obs:
            return f"完成目标：{sub_goal.description}（无有效观测）"

        obs_lines = []
        for obs in successful_obs:
            result = self._render_observation_result(obs)
            obs_lines.append(f"- [{obs.action.action_type}] {result}")
        formatted_obs = "\n".join(obs_lines)

        messages = [
            {
                "role": "system",
                "content": (
                    "你是一个信息压缩助手。"
                    "将以下研究观测总结为1-2句话，保留关键数据、URL来源和核心发现，去除冗余细节。"
                    "只输出摘要文本，不加任何前缀或标签。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"目标：{sub_goal.description}\n\n"
                    f"观测记录：\n{formatted_obs}"
                ),
            },
        ]
        summary = await self._chat_completion(messages=messages, response_format="text")
        return summary.strip()

    def _build_hierarchical_context(
        self,
        trajectory: Trajectory,
        active_goal: SubGoal,
    ) -> str:
        """Build a hierarchical context string from completed summaries and current raw observations."""
        parts: List[str] = []

        # Layer 0: long-term memory (cross-session prior research)
        if self._prior_knowledge:
            parts.append("已有研究背景：\n" + "\n".join(self._prior_knowledge))

        completed_with_summary = [
            goal for goal in trajectory.sub_goals
            if goal.status == "completed" and goal.summary is not None
        ]
        if completed_with_summary:
            summary_lines = []
            for goal in completed_with_summary:
                label = goal.description[:40]
                summary_lines.append(f"  [{label}] → {goal.summary}")
            parts.append("已完成目标摘要：\n" + "\n".join(summary_lines))

        # RAG 改动二：用当前子目标语义检索最相关的历史观察，替代纯时间窗口 [-5:]
        recent_obs = self.obs_retriever.search(
            query=active_goal.description,
            observations=trajectory.observations,
        )
        if recent_obs:
            obs_lines = []
            for index, obs in enumerate(recent_obs, start=1):
                result_text = self._render_observation_result(obs)
                thought_text = ""
                if obs.action.thought:
                    thought_text = f" thought={obs.action.thought[:80]!r}"
                if not obs.success:
                    failed_url = obs.action.params.get("url") or obs.action.params.get("link_url", "")
                    fail_note = f" [FAILED: {obs.error} — do NOT retry this URL]" if failed_url else f" [FAILED: {obs.error}]"
                    obs_lines.append(
                        f"#{index} action={obs.action.action_type}"
                        f"{thought_text} "
                        f"url={failed_url}{fail_note}"
                    )
                else:
                    obs_lines.append(
                        f"#{index} action={obs.action.action_type}"
                        f"{thought_text} "
                        f"success=True result={result_text}"
                    )
            current_label = active_goal.description[:40]
            parts.append(
                f"当前目标「{current_label}」的最近观测：\n" + " | ".join(obs_lines)
            )
        else:
            parts.append("当前目标尚无观测记录。")

        return "\n\n".join(parts)

    async def _summarize_observations(self, observations: List[AgentObservation]) -> str:
        """Summarize the last three observations into a compact context string."""
        if not observations:
            return "No prior observations."

        lines: List[str] = []
        recent = observations[-3:]
        for index, observation in enumerate(recent, start=1):
            result_text = self._render_observation_result(observation)
            thought_text = ""
            if observation.action.thought:
                thought_text = f" thought={observation.action.thought[:100]!r}"
            lines.append(
                f"#{index} action={observation.action.action_type}"
                f"{thought_text} "
                f"success={observation.success} result={result_text}"
            )

        return " | ".join(lines)

    async def _chat_completion(
        self,
        messages: List[Dict[str, str]],
        response_format: Literal["text", "json"] = "text",
    ) -> str:
        """Call the injected async chat client and return non-empty text."""
        response = self.llm_client.chat(messages=messages, response_format=response_format)
        if inspect.isawaitable(response):
            response = await response

        if not isinstance(response, str):
            response = "" if response is None else str(response)

        text = response.strip()
        if not text:
            raise ValueError("LLM response was empty.")

        return text

    @staticmethod
    def _find_cached_snippet(trajectory: Trajectory, url: str) -> Optional[str]:
        """Return a search snippet for url from prior search observations, or None."""
        if not url:
            return None
        for obs in trajectory.observations:
            if not obs.success:
                continue
            if isinstance(obs.result, list):
                for item in obs.result:
                    if isinstance(item, SearchResult) and item.url == url:
                        return f"[cached snippet] {item.title}\n{item.snippet}"
        return None

    @staticmethod
    def _normalize_action_payload(payload: dict) -> dict:
        """Fix common malformed action payloads from reasoning models."""
        if "action_type" in payload:
            return payload

        # Infer action_type from top-level keys and move them into params
        params = dict(payload.get("params") or {})
        step = payload.get("step")

        if "query" in payload:
            params.setdefault("query", payload["query"])
            action_type = "search"
        elif "url" in payload:
            params.setdefault("url", payload["url"])
            action_type = "click"
        elif "answer" in payload:
            params.setdefault("answer", payload["answer"])
            params.setdefault("citations", payload.get("citations", []))
            action_type = "terminate"
        elif "link_url" in payload:
            params.setdefault("link_url", payload["link_url"])
            action_type = "click"
        else:
            action_type = "search"

        result = {"action_type": action_type, "params": params}
        if step is not None:
            result["step"] = step
        return result

    @staticmethod
    def _extract_thought_and_action(raw_text: str) -> tuple[str, str]:
        """Extract (thought, action_json) from <think>...</think> + JSON output."""
        import re

        thought = ""
        think_match = re.search(
            r"<think>(.*?)</think>",
            raw_text,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if think_match:
            thought = think_match.group(1).strip()
            raw_text = raw_text[think_match.end():]

        text = raw_text.strip()

        # Try fenced code block anywhere in text (re.search, not re.match)
        fenced_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
        if fenced_match:
            return thought, fenced_match.group(1).strip()

        # Find the last complete JSON object by matching braces from the end
        last_brace = text.rfind("}")
        if last_brace == -1:
            raise ValueError("Response does not include a JSON object.")

        depth = 0
        for i in range(last_brace, -1, -1):
            if text[i] == "}":
                depth += 1
            elif text[i] == "{":
                depth -= 1
                if depth == 0:
                    return thought, text[i : last_brace + 1]

        raise ValueError("Response does not include a JSON object.")

    @staticmethod
    def _render_observation_result(observation: AgentObservation) -> str:
        """Render one observation result into a compact plain-text snippet."""
        result = observation.result
        if isinstance(result, PageContent):
            return f"page url={result.url} title={result.title!r} body={result.body[:600]!r}"
        if isinstance(result, list):
            lines = []
            for item in result:
                if isinstance(item, SearchResult):
                    lines.append(f"  [{item.rank}] {item.title} | url={item.url} | {item.snippet[:80]}")
            return "search results:\n" + "\n".join(lines) if lines else "no results"
        if isinstance(result, SearchResult):
            return f"search url={result.url} title={result.title!r} snippet={result.snippet[:120]!r}"
        return str(result)[:140]

    @staticmethod
    def _extract_terminate_payload(
        action: AgentAction,
        observation: AgentObservation,
    ) -> Tuple[str, List[str]]:
        """Extract final answer text and citations from terminate action payloads."""
        answer: Optional[str] = None

        if isinstance(action.params.get("answer"), str):
            answer = action.params["answer"]
        elif isinstance(observation.result, str):
            answer = observation.result

        citations_payload = action.params.get("citations", [])
        citations: List[str] = []
        if isinstance(citations_payload, list):
            citations = [str(item) for item in citations_payload if item is not None]

        return (answer or "", citations)

    @staticmethod
    def _merge_replanned_sub_goals(
        existing_sub_goals: List[SubGoal],
        replanned_sub_goals: List[SubGoal],
    ) -> List[SubGoal]:
        """Preserve terminal goals and replace pending/active goals with replanned ones."""
        preserved = [goal for goal in existing_sub_goals if goal.status in {"completed", "failed"}]
        merged = list(preserved)
        seen_ids = {goal.id for goal in merged}

        for goal in replanned_sub_goals:
            if goal.id in seen_ids:
                continue
            merged.append(goal)
            seen_ids.add(goal.id)

        return merged
