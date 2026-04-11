"""Agent integration smoke tests for DeepResearch."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pytest

from deepresearch.agent.agent import DeepResearchAgent
from deepresearch.agent.planner import SubGoalDecomposer
from deepresearch.agent.types import ResearchTask
from deepresearch.envs.sim_env import SimEnv
from deepresearch.reward.reward_engine import RewardEngine


class _FakeEncoder:
    """Deterministic encoder used to avoid external model inference in tests."""

    def encode(
        self,
        texts: List[str],
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        """Encode text list into small deterministic vectors."""
        del convert_to_numpy
        vectors: List[np.ndarray] = []
        for text in texts:
            lower = text.lower()
            if "climate" in lower or "policy" in lower:
                vector = np.array([1.0, 0.2, 0.0, 0.0], dtype=np.float32)
            else:
                vector = np.array([0.1, 1.0, 0.0, 0.0], dtype=np.float32)

            if normalize_embeddings:
                norm = np.linalg.norm(vector)
                if norm > 0:
                    vector = vector / norm
            vectors.append(vector)

        return np.vstack(vectors).astype(np.float32)


class _FakeIndex:
    """FAISS-like index using vector dot-product ranking for deterministic tests."""

    def __init__(self, document_embeddings: np.ndarray) -> None:
        """Store document embeddings used for similarity search."""
        self.document_embeddings = document_embeddings

    def search(self, query_embeddings: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return top-k indices ranked by inner-product similarity."""
        scores = query_embeddings @ self.document_embeddings.T
        top_indices = np.argsort(-scores, axis=1)[:, :top_k]
        top_scores = np.take_along_axis(scores, top_indices, axis=1)
        return top_scores.astype(np.float32), top_indices.astype(np.int64)


@pytest.mark.asyncio
async def test_agent_run_smoke_with_mocked_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    fake_corpus: List[Dict[str, str]],
    mock_llm_client: object,
) -> None:
    """Run an end-to-end agent smoke test without real network or LLM calls."""

    def _patched_build_index(self: SimEnv) -> None:
        self._documents = list(fake_corpus)
        self._documents_by_url = {document["url"]: document for document in self._documents}

        pre_encoded = np.array(
            [[1.0 - (index * 0.01), 0.1, 0.0, 0.0] for index in range(len(self._documents))],
            dtype=np.float32,
        )
        norms = np.linalg.norm(pre_encoded, axis=1, keepdims=True)
        pre_encoded = pre_encoded / np.clip(norms, a_min=1e-8, a_max=None)

        self._encoder = _FakeEncoder()
        self._index = _FakeIndex(pre_encoded)

    monkeypatch.setattr(SimEnv, "_build_index", _patched_build_index)

    env = SimEnv(corpus_dir="unused", top_k_default=5)
    planner = SubGoalDecomposer(llm_client=mock_llm_client, model="claude-sonnet-4-6", max_sub_goals=6)
    reward_engine = RewardEngine()
    agent = DeepResearchAgent(
        env=env,
        planner=planner,
        reward_engine=reward_engine,
        llm_client=mock_llm_client,
        model="claude-sonnet-4-6",
        max_steps=20,
    )

    task = ResearchTask(
        task_id="smoke-1",
        query="Summarize climate policy evidence.",
        expected_sub_goals=None,
        reference_answer=None,
    )

    trajectory = await agent.run(task)

    assert trajectory.observations
    assert any(sub_goal.status == "completed" for sub_goal in trajectory.sub_goals)
    assert trajectory.final_answer is not None
    assert reward_engine.compute(trajectory).total > 0

    thoughts = [
        obs.action.thought
        for obs in trajectory.observations
        if obs.action.thought
    ]
    assert len(thoughts) > 0, "Expected at least one observation with a thought"

