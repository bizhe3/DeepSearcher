"""Reward engine composition for DeepResearch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from deepresearch.agent.types import Trajectory


@dataclass
class RewardBreakdown:
    """Structured reward components computed for a single trajectory."""

    sub_goal: float
    answer: float
    citation: float
    efficiency_penalty: float
    total: float


class RewardEngine:
    """Aggregate sub-goal, answer, citation, and efficiency reward components."""

    def __init__(
        self,
        sub_goal_weight: float = 0.2,
        answer_weight: float = 1.0,
        citation_weight: float = 0.3,
        step_penalty: float = 0.01,
        expected_citations: int = 3,
        llm_judge: Optional[Any] = None,
    ) -> None:
        """Initialize reward component weights and citation expectations."""
        self.sub_goal_weight = sub_goal_weight
        self.answer_weight = answer_weight
        self.citation_weight = citation_weight
        self.step_penalty = step_penalty
        self.expected_citations = expected_citations
        self.llm_judge = llm_judge

    def compute(
        self,
        trajectory: Trajectory,
        reference_answer: Optional[str] = None,
    ) -> RewardBreakdown:
        """Compute and return weighted reward components for a trajectory."""
        completed_count = sum(1 for sub_goal in trajectory.sub_goals if sub_goal.status == "completed")

        sub_goal_reward = self.sub_goal_weight * completed_count
        answer_reference = reference_answer if reference_answer is not None else trajectory.task
        answer_reward = self.answer_weight * self._answer_score(trajectory, reference=answer_reference)
        citation_reward = self.citation_weight * self._citation_score(trajectory)
        efficiency_penalty = self.step_penalty * len(trajectory.observations)

        total = sub_goal_reward + answer_reward + citation_reward - efficiency_penalty
        total = max(0.0, total)

        return RewardBreakdown(
            sub_goal=sub_goal_reward,
            answer=answer_reward,
            citation=citation_reward,
            efficiency_penalty=efficiency_penalty,
            total=total,
        )

    async def compute_with_judge(self, trajectory: Trajectory, task: Any) -> RewardBreakdown:
        """Compute reward and optionally replace answer score using an LLM judge."""
        baseline = self.compute(trajectory, reference_answer=getattr(task, "reference_answer", None))

        if self.llm_judge is not None and trajectory.final_answer is not None:
            score = await self.llm_judge.judge_trajectory(task, trajectory)
            answer_reward = self.answer_weight * score.total
            total = baseline.sub_goal + answer_reward + baseline.citation - baseline.efficiency_penalty
            total = max(0.0, total)
            return RewardBreakdown(
                sub_goal=baseline.sub_goal,
                answer=answer_reward,
                citation=baseline.citation,
                efficiency_penalty=baseline.efficiency_penalty,
                total=total,
            )

        return baseline

    def _answer_score(self, trajectory: Trajectory, reference: Optional[str] = None) -> float:
        """Score answer quality with ROUGE-L F1 against a reference answer."""
        if trajectory.final_answer is None or reference is None:
            return 0.0

        try:
            from rouge_score import rouge_scorer
        except ImportError:
            return 0.0

        try:
            scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
            score = scorer.score(reference, trajectory.final_answer)["rougeL"].fmeasure
        except Exception:
            return 0.0

        return self._clamp(score)

    def _citation_score(self, trajectory: Trajectory) -> float:
        """Score citation coverage against the expected citation count."""
        denominator = max(1, self.expected_citations)
        score = len(trajectory.citations) / denominator
        return self._clamp(score)

    @staticmethod
    def _clamp(value: float) -> float:
        """Clamp floating-point values to the inclusive [0, 1] range."""
        return max(0.0, min(1.0, value))
