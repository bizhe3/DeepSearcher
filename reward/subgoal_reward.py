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
    """Detect sub-goal completion by matching keywords against observation text.

    Uses a two-tier keyword system: entity keywords (longer tokens likely to be
    proper nouns/technical terms) are weighted higher than common words.
    Completion requires either:
      - 50% of ALL keywords matched, OR
      - 70% of entity keywords (len >= 5) matched
    This prevents false negatives when generic verbs like "identify" or "confirm"
    don't appear verbatim in the observation text.
    """
    keywords = [token for token in _tokenize(sub_goal.description) if token not in _STOPWORDS]
    if not keywords:
        return False

    observation_text = _collect_observation_text(trajectory).lower()
    if not observation_text:
        return False

    matched = [kw for kw in keywords if kw in observation_text]
    all_ratio = len(matched) / len(keywords)

    # Entity keywords: longer tokens are more likely to be meaningful nouns
    entity_keywords = [kw for kw in keywords if len(kw) >= 5]
    if entity_keywords:
        entity_matched = [kw for kw in entity_keywords if kw in observation_text]
        entity_ratio = len(entity_matched) / len(entity_keywords)
    else:
        entity_ratio = all_ratio

    return all_ratio >= 0.5 or entity_ratio >= 0.7


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
        elif isinstance(result, list):
            for item in result:
                if isinstance(item, SearchResult):
                    parts.append(item.title)
                    parts.append(item.snippet)
        elif isinstance(result, str):
            parts.append(result)

    return " ".join(part for part in parts if part)
