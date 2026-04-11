"""Run DeepResearch agent over a JSONL golden set and report reward metrics."""

from __future__ import annotations

import asyncio
import json
import re
import shutil
from pathlib import Path
from typing import Dict, List
from uuid import uuid4

import numpy as np

from deepresearch.agent.agent import DeepResearchAgent
from deepresearch.agent.planner import SubGoalDecomposer
from deepresearch.agent.types import ResearchTask
from deepresearch.envs.sim_env import SimEnv
from deepresearch.reward.reward_engine import RewardBreakdown, RewardEngine


class _FakeEncoder:
    """Deterministic lightweight encoder to avoid external model downloads."""

    VECTOR_SIZE = 16

    def encode(
        self,
        texts: List[str],
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        """Encode text into normalized hashed vectors."""
        del convert_to_numpy

        vectors: List[np.ndarray] = []
        for text in texts:
            vector = np.zeros(self.VECTOR_SIZE, dtype=np.float32)
            for token in re.findall(r"[a-zA-Z0-9]+", text.lower()):
                index = hash(token) % self.VECTOR_SIZE
                vector[index] += 1.0

            if normalize_embeddings:
                norm = np.linalg.norm(vector)
                if norm > 0:
                    vector = vector / norm
            vectors.append(vector)

        return np.vstack(vectors).astype(np.float32)


class _FakeIndex:
    """FAISS-like index backed by dot-product ranking."""

    def __init__(self, document_embeddings: np.ndarray) -> None:
        """Store document embeddings matrix."""
        self.document_embeddings = document_embeddings

    def search(self, query_embeddings: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return top-k document indices for each query row."""
        scores = query_embeddings @ self.document_embeddings.T
        top_indices = np.argsort(-scores, axis=1)[:, :top_k]
        top_scores = np.take_along_axis(scores, top_indices, axis=1)
        return top_scores.astype(np.float32), top_indices.astype(np.int64)


class GoldenSimEnv(SimEnv):
    """SimEnv variant using deterministic local encoder/index for golden eval."""

    def _create_encoder(self) -> _FakeEncoder:
        """Create deterministic fake encoder."""
        return _FakeEncoder()

    def _create_faiss_index(self, embeddings: np.ndarray) -> _FakeIndex:
        """Create deterministic in-memory index."""
        return _FakeIndex(embeddings)


class ScriptedLLMClient:
    """Minimal async chat client for deterministic planning and actions."""

    def __init__(self, record: Dict[str, object], citation_url: str) -> None:
        """Bind one record's expected goals and reference answer."""
        self._record = record
        self._citation_url = citation_url
        self._action_calls = 0

    async def chat(self, messages: List[Dict[str, str]], response_format: str = "text") -> str:
        """Return deterministic planner/action responses from prompts."""
        del response_format
        system_prompt = ""
        user_prompt = ""
        for message in messages:
            role = message.get("role")
            if role == "system":
                system_prompt = str(message.get("content", ""))
            elif role == "user":
                user_prompt = str(message.get("content", ""))

        if "research planning assistant" in system_prompt:
            expected_sub_goals = self._record.get("expected_sub_goals", [])
            payload: List[Dict[str, str]] = []
            if isinstance(expected_sub_goals, list):
                for index, goal in enumerate(expected_sub_goals, start=1):
                    payload.append({"id": f"sg_{index}", "description": str(goal)})
            return json.dumps(payload, ensure_ascii=False)

        if "action policy for a web research agent" in system_prompt:
            self._action_calls += 1
            if self._action_calls == 1:
                task_match = re.search(r"Original task:\s*(.*?)\n\n", user_prompt, flags=re.DOTALL)
                query_text = task_match.group(1).strip() if task_match else str(self._record.get("query", ""))
                return (
                    "<think>Gather initial evidence from corpus.</think>"
                    + json.dumps(
                        {
                            "action_type": "search",
                            "params": {"query": query_text, "top_k": 1},
                            "step": 1,
                        },
                        ensure_ascii=False,
                    )
                )

            reference_answer = str(self._record.get("reference_answer", ""))
            return (
                "<think>Evidence is sufficient to finalize.</think>"
                + json.dumps(
                    {
                        "action_type": "terminate",
                        "params": {
                            "answer": reference_answer,
                            "citations": [self._citation_url],
                        },
                        "step": 2,
                    },
                    ensure_ascii=False,
                )
            )

        return "{}"


def _load_golden_records(path: Path) -> List[Dict[str, object]]:
    """Load JSONL golden records from disk."""
    records: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Invalid record at line {line_number}: expected object.")
            records.append(payload)
    return records


def _write_corpus(temp_dir: Path, records: List[Dict[str, object]]) -> Dict[str, str]:
    """Write one local JSON corpus doc per record and return citation URLs."""
    citation_urls: Dict[str, str] = {}
    for record in records:
        task_id = str(record["task_id"])
        query = str(record["query"])
        reference_answer = str(record["reference_answer"])

        file_path = temp_dir / f"{task_id}.json"
        payload = {
            "url": f"local://{file_path.name}",
            "title": f"Golden Evidence: {task_id}",
            "body": f"Query: {query}\n\nReference evidence:\n{reference_answer}",
        }
        file_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        citation_urls[task_id] = payload["url"]

    return citation_urls


async def _evaluate_record(
    record: Dict[str, object],
    corpus_dir: Path,
    citation_url: str,
) -> RewardBreakdown:
    """Run one golden task and compute reward breakdown."""
    llm_client = ScriptedLLMClient(record=record, citation_url=citation_url)
    env = GoldenSimEnv(corpus_dir=str(corpus_dir), top_k_default=3)
    planner = SubGoalDecomposer(llm_client=llm_client, max_sub_goals=6)
    reward_engine = RewardEngine()

    task = ResearchTask(
        task_id=str(record["task_id"]),
        query=str(record["query"]),
        expected_sub_goals=[str(goal) for goal in record.get("expected_sub_goals", [])],
        reference_answer=str(record["reference_answer"]),
    )

    agent = DeepResearchAgent(
        env=env,
        planner=planner,
        reward_engine=reward_engine,
        llm_client=llm_client,
        max_steps=5,
    )

    trajectory = await agent.run(task)
    return reward_engine.compute(
        trajectory,
        reference_answer=str(record["reference_answer"]),
    )


def _print_results(rows: List[tuple[str, RewardBreakdown]]) -> None:
    """Print per-task reward table and aggregate averages."""
    header = f"{'task_id':<34} | {'sub_goal':>8} | {'answer':>8} | {'citation':>8} | {'total':>8}"
    print(header)
    print("-" * len(header))

    for task_id, reward in rows:
        print(
            f"{task_id:<34} | "
            f"{reward.sub_goal:8.3f} | "
            f"{reward.answer:8.3f} | "
            f"{reward.citation:8.3f} | "
            f"{reward.total:8.3f}"
        )

    count = max(1, len(rows))
    avg_sub_goal = sum(reward.sub_goal for _, reward in rows) / count
    avg_answer = sum(reward.answer for _, reward in rows) / count
    avg_citation = sum(reward.citation for _, reward in rows) / count
    avg_total = sum(reward.total for _, reward in rows) / count

    print("-" * len(header))
    print(
        f"{'AVERAGE':<34} | "
        f"{avg_sub_goal:8.3f} | "
        f"{avg_answer:8.3f} | "
        f"{avg_citation:8.3f} | "
        f"{avg_total:8.3f}"
    )


async def main() -> None:
    """Load golden set, evaluate all records, and print reward summary."""
    project_root = Path(__file__).resolve().parents[2]
    golden_path = project_root / "data" / "golden_set.jsonl"

    records = _load_golden_records(golden_path)
    if not records:
        raise ValueError(f"Golden set is empty: {golden_path}")

    rows: List[tuple[str, RewardBreakdown]] = []
    workspace_tmp_root = project_root / "data" / ".golden_eval_tmp"
    corpus_dir = workspace_tmp_root / f"run_{uuid4().hex}"
    corpus_dir.mkdir(parents=True, exist_ok=True)

    try:
        citation_urls = _write_corpus(corpus_dir, records)

        for record in records:
            task_id = str(record["task_id"])
            reward = await _evaluate_record(
                record=record,
                corpus_dir=corpus_dir,
                citation_url=citation_urls[task_id],
            )
            rows.append((task_id, reward))
    finally:
        shutil.rmtree(corpus_dir, ignore_errors=True)

    _print_results(rows)


if __name__ == "__main__":
    asyncio.run(main())
