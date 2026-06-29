"""Build golden_set.jsonl from HotpotQA (EMNLP 2018).

Source: HotpotQA distractor split, validation set
  - Multi-hop reasoning with supporting facts
  - Each question requires 2+ retrieval hops

Selection criteria:
  - Only multi-hop (bridge / comparison) questions
  - Only "hard" or "medium" difficulty
  - Has supporting_facts with 2+ unique source documents
  - Skip yes/no answers

Output format:
  {"task_id", "query", "reference_answer", "supporting_facts", "key_facts", "question_type", "difficulty", "source"}

Usage:
    cd d:/Agent
    python -m deepresearch.scripts.build_golden_set --count 50 --output deepresearch/data/golden_set.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List


def _load_hotpotqa(count: int) -> List[Dict[str, Any]]:
    """Load hard/medium multi-hop questions from HotpotQA.

    Uses HF_ENDPOINT env var for mirror support (e.g. https://hf-mirror.com).
    """
    from datasets import load_dataset

    hf_endpoint = os.environ.get("HF_ENDPOINT", "")
    if hf_endpoint:
        print(f"Using HF mirror: {hf_endpoint}")

    print("Loading HotpotQA (distractor split, validation)...")
    ds = load_dataset("hotpot_qa", "distractor", split="validation", trust_remote_code=False)

    candidates = []
    for row in ds:
        # Only multi-hop (bridge or comparison), medium/hard
        if row.get("level") not in ("medium", "hard"):
            continue
        if row.get("type") not in ("bridge", "comparison"):
            continue

        answer = str(row.get("answer", "")).strip()
        question = str(row.get("question", "")).strip()

        # Skip yes/no
        if not answer or not question:
            continue
        if answer.lower() in ("yes", "no"):
            continue

        # Extract supporting fact titles and sentence indices
        sup_facts = row.get("supporting_facts", {})
        sup_titles = sup_facts.get("title", []) if isinstance(sup_facts, dict) else []
        sup_sent_ids = sup_facts.get("sent_id", []) if isinstance(sup_facts, dict) else []
        unique_titles = list(dict.fromkeys(sup_titles))  # dedupe preserving order

        # Require at least 2 source documents (true multi-hop)
        if len(unique_titles) < 2:
            continue

        # Build key_facts from supporting evidence titles
        key_facts = [f"Evidence from: {t}" for t in unique_titles[:4]]
        key_facts.append(f"Answer: {answer}")

        # Build supporting_facts structure for retrieval quality evaluation
        supporting_facts_list = []
        for title, sent_id in zip(sup_titles, sup_sent_ids):
            supporting_facts_list.append({"title": title, "sent_id": int(sent_id)})

        candidates.append({
            "query": question,
            "reference_answer": answer,
            "key_facts": key_facts,
            "supporting_facts": supporting_facts_list,
            "question_type": row.get("type", ""),
            "difficulty": row.get("level", ""),
            "source": "hotpotqa",
        })

    random.shuffle(candidates)
    selected = candidates[:count]
    print(f"  HotpotQA: {len(candidates)} candidates, selected {len(selected)}")
    return selected


def _build_task_id(index: int) -> str:
    return f"hpqa_{index:03d}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build golden_set.jsonl from HotpotQA")
    parser.add_argument("--count", type=int, default=50, help="Number of tasks to sample")
    parser.add_argument("--output", default="deepresearch/data/golden_set.jsonl", help="Output path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)

    records = _load_hotpotqa(args.count)

    # Assign task IDs
    all_records: List[Dict[str, Any]] = []
    for idx, record in enumerate(records, 1):
        task = {
            "task_id": _build_task_id(idx),
            "query": record["query"],
            "reference_answer": record["reference_answer"],
            "key_facts": record["key_facts"],
            "supporting_facts": record["supporting_facts"],
            "question_type": record["question_type"],
            "difficulty": record["difficulty"],
            "source": record["source"],
        }
        all_records.append(task)

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nWritten {len(all_records)} records to {output_path}")

    # Stats
    bridge = sum(1 for r in all_records if r["question_type"] == "bridge")
    comparison = sum(1 for r in all_records if r["question_type"] == "comparison")
    hard = sum(1 for r in all_records if r["difficulty"] == "hard")
    medium = sum(1 for r in all_records if r["difficulty"] == "medium")
    print(f"  bridge: {bridge}, comparison: {comparison}")
    print(f"  hard: {hard}, medium: {medium}")

    # Preview first 3
    print("\nPreview:")
    for record in all_records[:3]:
        print(f"  [{record['task_id']}] ({record['question_type']}/{record['difficulty']})")
        print(f"    Q: {record['query'][:80]}...")
        print(f"    A: {record['reference_answer'][:80]}")
        n_facts = len(record['supporting_facts'])
        print(f"    Supporting facts: {n_facts} sentences")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
