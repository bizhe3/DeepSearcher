"""Shared Pydantic data contracts for DeepResearch agent components."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict


class SearchResult(BaseModel):
    """Represents one ranked search result entry."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    url: str
    title: str
    snippet: str
    rank: int


class PageContent(BaseModel):
    """Represents extracted content from a visited page."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    url: str
    title: str
    body: str
    links: List[str]
    is_paginated: bool
    current_page: int
    total_pages: Optional[int]


class SubGoal(BaseModel):
    """Tracks progress for an intermediate research sub-goal."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    description: str
    status: Literal["pending", "active", "completed", "failed"]
    completed_at_step: Optional[int]
    summary: Optional[str] = None


class AgentAction(BaseModel):
    """Defines an action issued by the research agent at a specific step."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    thought: Optional[str] = None
    action_type: Literal["search", "click", "scroll", "extract", "cross_check", "terminate"]
    params: Dict[str, Any]
    step: int


class AgentObservation(BaseModel):
    """Captures the observed result of a single agent action."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    action: AgentAction
    result: Union[List[SearchResult], SearchResult, PageContent, str]
    success: bool
    error: Optional[str]


class Trajectory(BaseModel):
    """Stores the full research trajectory and final outcome for a task."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    task: str
    sub_goals: List[SubGoal]
    observations: List[AgentObservation]
    final_answer: Optional[str]
    citations: List[str]
    key_facts: List[str] = []
    judge_score: Optional[JudgeScore] = None

    def __repr__(self) -> str:
        completed_sub_goals = sum(1 for sub_goal in self.sub_goals if sub_goal.status == "completed")
        total_sub_goals = len(self.sub_goals)
        step_count = len(self.observations)
        return (
            f"Trajectory(steps={step_count}, "
            f"sub_goals_completed={completed_sub_goals}/{total_sub_goals})"
        )


class ResearchTask(BaseModel):
    """Defines a research task specification and optional references."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    task_id: str
    query: str
    expected_sub_goals: Optional[List[str]] = None
    reference_answer: Optional[str] = None
    key_facts: Optional[List[str]] = None


class JudgeScore(BaseModel):
    """Structured scores returned by an LLM judge evaluation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    relevance: int
    completeness: int
    citation_quality: int
    reasoning: str
    total: float = 0.0

    def model_post_init(self, __context: Any) -> None:
        object.__setattr__(
            self,
            "total",
            round((self.relevance * 0.4 + self.completeness * 0.4 + self.citation_quality * 0.2) / 5.0, 4),
        )

