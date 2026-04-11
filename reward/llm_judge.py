"""LLM-based answer judge for DeepResearch trajectories."""

from __future__ import annotations

import inspect
import json
import logging
import re
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

from deepresearch.agent.types import JudgeScore, ResearchTask, Trajectory

JUDGE_SYSTEM_PROMPT = (
    "You are an expert research evaluator. Score the agent's answer on\n"
    "three dimensions, each from 1 (very poor) to 5 (excellent):\n"
    "- relevance: Does the answer directly address the research question?\n"
    "- completeness: Does it cover the main aspects? If key_facts are\n"
    "  provided, check how many are present in the answer.\n"
    "- citation_quality: Are citations used and do they support claims?\n"
    "  Score 1 if no citations, 5 if every major claim is cited.\n"
    "Return ONLY a JSON object with keys: relevance (int), completeness\n"
    "(int), citation_quality (int), reasoning (str, one paragraph).\n"
    "No markdown, no extra keys."
)


class LLMJudge:
    """Evaluate agent answers using an LLM as judge."""

    def __init__(self, llm_client: Any, model: str = "claude-sonnet-4-6") -> None:
        """Initialize judge with a chat client and model name."""
        self.llm_client = llm_client
        self.model = model

    async def judge(
        self,
        query: str,
        answer: str,
        citations: List[str],
        key_facts: Optional[List[str]] = None,
    ) -> JudgeScore:
        """Score an answer against query/context and return structured judge scores."""
        truncated_answer = answer[:2000] if len(answer) > 2000 else answer
        numbered_citations = citations[:10]

        citation_lines = [
            f"{index}. {citation}" for index, citation in enumerate(numbered_citations, start=1)
        ]
        citations_block = "\n".join(citation_lines) if citation_lines else "(none)"

        key_facts_block = ""
        if key_facts is not None:
            facts_lines = [f"- {fact}" for fact in key_facts]
            facts_text = "\n".join(facts_lines) if facts_lines else "- (none provided)"
            key_facts_block = (
                "\n\nKey facts:\n"
                "Check that these facts appear in the answer.\n"
                f"{facts_text}"
            )

        user_prompt = (
            f"Research query:\n{query}\n\n"
            f"Agent final answer (possibly truncated):\n{truncated_answer}\n\n"
            f"Citations (numbered, up to 10):\n{citations_block}"
            f"{key_facts_block}"
        )

        raw_response = self.llm_client.chat(
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format="json",
        )
        if inspect.isawaitable(raw_response):
            raw_response = await raw_response

        if not isinstance(raw_response, str):
            raw_response = "" if raw_response is None else str(raw_response)

        extracted = self._extract_json(raw_response)
        try:
            payload = json.loads(extracted)
            if not isinstance(payload, dict):
                raise ValueError("Judge response must be a JSON object.")
            return JudgeScore(**payload)
        except Exception as exc:
            logger.warning("[Judge] parse failed: %s", exc)
            return JudgeScore(
                relevance=1,
                completeness=1,
                citation_quality=1,
                reasoning="Parse error: " + raw_response[:200],
                total=0.0,
            )

    @staticmethod
    def _extract_json(text: str) -> str:
        """Extract a JSON object from text that may contain surrounding prose."""
        # Try fenced code block first
        fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
        if fenced:
            return fenced.group(1).strip()

        # Find last complete JSON object by matching braces from the end
        last_brace = text.rfind("}")
        if last_brace == -1:
            return text
        depth = 0
        for i in range(last_brace, -1, -1):
            if text[i] == "}":
                depth += 1
            elif text[i] == "{":
                depth -= 1
                if depth == 0:
                    return text[i : last_brace + 1]
        return text

    async def judge_trajectory(
        self,
        task: ResearchTask,
        trajectory: Trajectory,
    ) -> JudgeScore:
        """Judge an existing trajectory and persist the score onto the trajectory."""
        score = await self.judge(
            query=task.query,
            answer=trajectory.final_answer or "",
            citations=trajectory.citations,
            key_facts=task.key_facts if hasattr(task, "key_facts") else None,
        )
        trajectory.judge_score = score
        return score
