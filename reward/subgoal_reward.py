"""Subgoal reward definitions for DeepResearch."""

from __future__ import annotations

import re
from typing import List, Set

from deepresearch.agent.types import PageContent, SearchResult, SubGoal, Trajectory

_STOPWORDS: Set[str] = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


def detect_completion(sub_goal: SubGoal, trajectory: Trajectory) -> bool:
    """Detect sub-goal completion by matching keywords against observation text."""
    keywords = [token for token in _tokenize(sub_goal.description) if token not in _STOPWORDS]
    if not keywords:
        return False

    observation_text = _collect_observation_text(trajectory).lower()
    if not observation_text:
        return False

    matched_count = sum(1 for keyword in keywords if keyword in observation_text)
    return (matched_count / len(keywords)) >= 0.6


def _tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric tokens."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _collect_observation_text(trajectory: Trajectory) -> str:
    """Concatenate text-bearing fields from all observations into one corpus string."""
    parts: List[str] = []
    for observation in trajectory.observations:
        result = observation.result
        if isinstance(result, PageContent):
            parts.append(result.body)
            parts.append(result.title)
        elif isinstance(result, SearchResult):
            parts.append(result.title)
            parts.append(result.snippet)
        elif isinstance(result, str):
            parts.append(result)

    return " ".join(part for part in parts if part)
