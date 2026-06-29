"""LLM-as-Auditor for sub-goal quality tagging.

Independent judge LLM that tags every step's sub-goal progress with a
multi-dimensional QualityVerdict. The verdict drives three replan layers:
  - verify hint    (Layer 1: inline cross_check suggestion)
  - redo           (Layer 2: local sub-goal retry)
  - global signals (Layer 3: vague_ratio / total_redo accumulation)
"""

from __future__ import annotations

import inspect
import json
import logging
import re
from typing import Any, List, Set
from urllib.parse import urlparse

from deepresearch.agent.types import (
    AgentObservation,
    PageContent,
    QualityVerdict,
    SearchResult,
    SubGoal,
)

logger = logging.getLogger(__name__)


_AUDIT_PROMPT = """You are a stateless research quality auditor. Tag the current state of a sub-goal.

Sub-goal: {goal}

Findings (compressed summary): {summary}

Source domains used: {sources}
Number of distinct sources: {n_sources}

Tag the following dimensions. Use ONLY the listed enum values.

Output JSON:
{{
  "answer_clarity": "clear" | "vague" | "missing",
  "source_credibility": "multi_source" | "single_source" | "no_source",
  "is_quantitative": true | false,
  "next_action_hint": "proceed" | "verify" | "redo",
  "rationale": "<one short sentence>"
}}

Decision rules:
- answer_clarity = "missing" if findings do NOT actually answer the sub-goal
- answer_clarity = "vague" if findings are hedged, partial, or use modal verbs
- answer_clarity = "clear" only if findings give a direct, specific answer
- source_credibility = "no_source" if 0 source domains
- source_credibility = "single_source" if exactly 1 source domain
- source_credibility = "multi_source" if 2+ distinct source domains
- is_quantitative = true if the sub-goal asks for a number, percentage, date, ranking, or other measurable quantity
- next_action_hint = "redo" if answer_clarity = "missing"
- next_action_hint = "verify" if is_quantitative AND source_credibility != "multi_source" AND answer_clarity != "missing"
- next_action_hint = "proceed" otherwise

Output ONLY the JSON, nothing else."""


class SubGoalAuditor:
    """Tag sub-goal completion quality using an independent judge LLM."""

    def __init__(self, judge_client: Any, model: str) -> None:
        self.judge_client = judge_client
        self.model = model

    async def audit(
        self,
        sub_goal: SubGoal,
        observations: List[AgentObservation],
    ) -> QualityVerdict:
        """Return a QualityVerdict; never raises (returns proceed-default on failure)."""
        try:
            sources = self._extract_source_domains(observations)
            prompt = _AUDIT_PROMPT.format(
                goal=sub_goal.description,
                summary=(sub_goal.summary or "(no summary yet)")[:1200],
                sources=", ".join(sorted(sources)) or "(none)",
                n_sources=len(sources),
            )
            response = self.judge_client.chat(
                messages=[{"role": "user", "content": prompt}],
                response_format="json",
            )
            if inspect.isawaitable(response):
                response = await response
            verdict = self._parse_verdict(str(response))
            logger.info(
                "auditor sg=%s clarity=%s credibility=%s quant=%s hint=%s",
                sub_goal.id,
                verdict.answer_clarity,
                verdict.source_credibility,
                verdict.is_quantitative,
                verdict.next_action_hint,
            )
            return verdict
        except Exception as exc:
            logger.warning("auditor failed for sg=%s: %s", sub_goal.id, exc)
            return QualityVerdict(
                answer_clarity="vague",
                source_credibility="no_source",
                is_quantitative=False,
                next_action_hint="proceed",
                rationale="auditor_failed_default_proceed",
            )

    @staticmethod
    def _extract_source_domains(observations: List[AgentObservation]) -> Set[str]:
        """Collect unique source domains from observation results only.

        Reads only obs.result (URLs from PageContent / SearchResult).
        Ignores obs.action.thought to prevent agent prompt-injection of judge.
        """
        domains: Set[str] = set()
        for obs in observations:
            if not obs.success:
                continue
            result = obs.result
            urls: List[str] = []
            if isinstance(result, PageContent):
                urls.append(result.url)
            elif isinstance(result, list):
                for item in result:
                    if isinstance(item, SearchResult):
                        urls.append(item.url)
            elif isinstance(result, SearchResult):
                urls.append(result.url)
            for url in urls:
                try:
                    netloc = urlparse(url).netloc
                    if netloc:
                        domains.add(netloc)
                except Exception:
                    continue
        return domains

    @staticmethod
    def _parse_verdict(raw: str) -> QualityVerdict:
        text = raw.strip()
        fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if fenced:
            text = fenced.group(1).strip()
        first = text.find("{")
        last = text.rfind("}")
        if first == -1 or last == -1:
            raise ValueError("no JSON object found in auditor response")
        return QualityVerdict(**json.loads(text[first : last + 1]))
