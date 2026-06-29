"""Planning component for DeepResearch agent workflows."""

from __future__ import annotations

import inspect
import json
import re
from typing import Any, Dict, List, Literal, Optional

from deepresearch.agent.types import SubGoal

DECOMPOSE_SYSTEM_PROMPT = """You are a research planning assistant for a web research agent.
Break each research task into 2-6 concrete, independently verifiable sub-goals.
Each sub-goal must be specific enough to be completed by web search.
Do not include vague goals such as \"learn more\" or \"explore topic\".

Planning rules:
- Include at least one sub-goal that explicitly cross-checks a key quantitative claim \
(e.g. market share %, revenue, growth rate) against a second independent source.
- Place data-gathering sub-goals before verification sub-goals.
- The final sub-goal should synthesize findings into a structured conclusion.

Output format requirements:
- Return ONLY a JSON array
- Each item must contain: {\"id\": \"sg_1\", \"description\": \"...\"}
- IDs should follow sg_1, sg_2, ... in execution order
- Descriptions should be concise, actionable, and verifiable
- No markdown, no code fences, no extra keys, no explanatory text

Example output:
[
  {\"id\": \"sg_1\", \"description\": \"Identify the top AI model providers by market share from industry reports\"},
  {\"id\": \"sg_2\", \"description\": \"Cross-verify the market share figures from sg_1 using a second independent source\"},
  {\"id\": \"sg_3\", \"description\": \"Summarize the verified market landscape with citations\"}
]
"""


class SubGoalDecomposer:
    """Generates and updates structured sub-goals for research tasks."""

    def __init__(
        self,
        llm_client: Any,
        planner_model: str = "claude-haiku-4-5-20251001",
        model: Optional[str] = None,
        max_sub_goals: int = 6,
    ) -> None:
        """Initialize the decomposer with a duck-typed async chat client."""
        if max_sub_goals < 2:
            raise ValueError("max_sub_goals must be at least 2.")

        self.llm_client = llm_client
        self.planner_model = model if model is not None else planner_model
        self.model = self.planner_model
        self.max_sub_goals = min(max_sub_goals, 6)

    async def decompose(self, task: str) -> List[SubGoal]:
        """Break a research task into pending, verifiable sub-goals."""
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": DECOMPOSE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Task:\n"
                    f"{task}\n\n"
                    f"Return a JSON array of 2-{self.max_sub_goals} sub-goals. "
                    "Return JSON only."
                ),
            },
        ]

        for attempt in range(2):
            raw_response = await self._chat_completion(messages, response_format="json")
            try:
                return self._parse_sub_goals(raw_response)
            except ValueError as error:
                if attempt == 1:
                    raise ValueError("Failed to parse valid sub-goal JSON from LLM response.") from error

                messages.append({"role": "assistant", "content": raw_response})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous output was invalid. "
                            "Return ONLY a valid JSON array like "
                            "[{\"id\":\"sg_1\",\"description\":\"...\"}] with 2-6 items."
                        ),
                    }
                )

        raise ValueError("Failed to parse valid sub-goal JSON from LLM response.")

    async def replan(
        self,
        task: str,
        completed: List[SubGoal],
        new_info: str,
        quality_signals: Optional[str] = None,
    ) -> List[SubGoal]:
        """Generate updated pending sub-goals based on completed work and new evidence.

        quality_signals (optional) carries auditor verdicts for already-tagged
        sub-goals so the planner can adjust direction (e.g., add cross-check
        sub-goals when prior findings were vague or single-source).
        """
        completed_payload = [
            {
                "id": sub_goal.id,
                "description": sub_goal.description,
                "status": sub_goal.status,
            }
            for sub_goal in completed
        ]

        quality_section = ""
        if quality_signals:
            quality_section = (
                f"\n\nQuality signals from prior sub-goals (auditor tags):\n"
                f"{quality_signals}\n\n"
                "Use these signals to adjust planning: prefer adding cross-check "
                "sub-goals after single-source quantitative findings, and avoid "
                "regenerating sub-goals that already failed."
            )

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": DECOMPOSE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Replan the remaining research steps.\n\n"
                    f"Task:\n{task}\n\n"
                    f"Completed sub-goals:\n{json.dumps(completed_payload, ensure_ascii=False)}\n\n"
                    f"New information:\n{new_info}"
                    f"{quality_section}\n\n"
                    "Return ONLY the remaining new or updated sub-goals as a JSON array."
                ),
            },
        ]

        raw_response = await self._chat_completion(messages, response_format="json")
        try:
            # Replan accepts 0+ new goals; failing to parse should not crash
            # the whole task — caller can keep the existing plan.
            replanned = self._parse_sub_goals(raw_response, min_goals=0)
        except (ValueError, json.JSONDecodeError) as exc:
            import logging
            logging.getLogger(__name__).warning(
                "replan parse failed (%s) — keeping current sub_goals", exc
            )
            return []

        completed_ids = {sub_goal.id for sub_goal in completed}
        completed_descriptions = {
            sub_goal.description.strip().lower() for sub_goal in completed if sub_goal.description.strip()
        }

        pending_updates: List[SubGoal] = []
        for sub_goal in replanned:
            if sub_goal.id in completed_ids:
                continue
            if sub_goal.description.strip().lower() in completed_descriptions:
                continue
            pending_updates.append(sub_goal)

        return pending_updates

    def get_active_goal(self, sub_goals: List[SubGoal]) -> Optional[SubGoal]:
        """Activate and return the first pending sub-goal, if available."""
        for sub_goal in sub_goals:
            if sub_goal.status == "pending":
                sub_goal.status = "active"
                return sub_goal
        return None

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
            raise ValueError("LLM response content is empty.")

        return text

    def _parse_sub_goals(self, raw_text: str, min_goals: int = 2) -> List[SubGoal]:
        """Parse model output into pending SubGoal objects.

        min_goals: minimum count to enforce. decompose() uses 2 (need a real
        plan); replan() uses 0 (replan may legitimately produce 0-1 new goals
        when most work is already complete).
        """
        parsed_payload = json.loads(self._extract_json_array(raw_text))
        if not isinstance(parsed_payload, list):
            raise ValueError("Sub-goal output must be a JSON array.")

        sub_goals: List[SubGoal] = []
        for index, item in enumerate(parsed_payload[: self.max_sub_goals], start=1):
            if not isinstance(item, dict):
                raise ValueError("Each sub-goal item must be a JSON object.")

            description = item.get("description")
            if not isinstance(description, str) or not description.strip():
                raise ValueError("Each sub-goal must include a non-empty description.")

            raw_id = item.get("id", f"sg_{index}")
            sub_goals.append(
                SubGoal(
                    id=str(raw_id),
                    description=description.strip(),
                    status="pending",
                    completed_at_step=None,
                )
            )

        if len(sub_goals) < min_goals:
            raise ValueError(f"LLM returned fewer than {min_goals} sub-goals.")

        return sub_goals

    @staticmethod
    def _extract_json_array(raw_text: str) -> str:
        """Extract a JSON array string from plain text or fenced model output."""
        text = raw_text.strip()

        fenced_match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, flags=re.IGNORECASE | re.DOTALL)
        if fenced_match:
            text = fenced_match.group(1).strip()

        first_bracket = text.find("[")
        last_bracket = text.rfind("]")
        if first_bracket == -1 or last_bracket == -1 or last_bracket < first_bracket:
            raise ValueError("Response does not contain a JSON array.")

        return text[first_bracket : last_bracket + 1]



