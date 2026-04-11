"""Build golden_set.jsonl from public QA datasets.

Sources:
  - HotpotQA (fullwiki, hard): multi-hop reasoning with supporting facts
  - MuSiQue: multi-hop with decomposed sub-questions

Selection criteria:
  - Only multi-hop / bridge / comparison questions (not single-hop)
  - Only "hard" or "medium" difficulty
  - Answer length >= 2 words (skip yes/no)
  - Has supporting_facts or decomposed sub-questions

Output format matches eval_e2e.py expectations:
  {"task_id", "query", "reference_answer", "key_facts", "expected_sub_goals", "source"}

Usage:
    cd d:/Agent
    python -m deepresearch.scripts.build_golden_set --count 30 --output deepresearch/data/golden_set.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List


def _load_hotpotqa(count: int) -> List[Dict[str, Any]]:
    """Load hard/medium multi-hop questions from HotpotQA."""
    from datasets import load_dataset

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

        # Skip yes/no or very short answers
        if not answer or len(answer.split()) < 2:
            continue
        if answer.lower() in ("yes", "no"):
            continue
        if not question:
            continue

        # Extract supporting fact titles as key_facts
        sup_facts = row.get("supporting_facts", {})
        sup_titles = sup_facts.get("title", []) if isinstance(sup_facts, dict) else []
        unique_titles = list(dict.fromkeys(sup_titles))  # dedupe preserving order

        # Build key_facts from supporting fact sentences
        key_facts = [f"Evidence from: {t}" for t in unique_titles[:4]]
        key_facts.append(f"Answer: {answer}")

        candidates.append({
            "query": question,
            "reference_answer": answer,
            "key_facts": key_facts,
            "type": row.get("type", ""),
            "level": row.get("level", ""),
            "source": "hotpotqa",
        })

    random.shuffle(candidates)
    selected = candidates[:count]
    print(f"  HotpotQA: {len(candidates)} candidates, selected {len(selected)}")
    return selected


def _load_musique(count: int) -> List[Dict[str, Any]]:
    """Load multi-hop questions from MuSiQue with decomposed sub-questions."""
    from datasets import load_dataset

    print("Loading MuSiQue (validation)...")
    try:
        ds = load_dataset("drt/musique", split="validation", trust_remote_code=False)
    except Exception:
        try:
            ds = load_dataset("musique", split="validation", trust_remote_code=False)
        except Exception as exc:
            print(f"  MuSiQue unavailable: {exc}")
            return []

    candidates = []
    for row in ds:
        answer = str(row.get("answer", "")).strip()
        question = str(row.get("question", "")).strip()

        if not answer or not question:
            continue
        if answer.lower() in ("yes", "no"):
            continue
        if len(answer.split()) < 2:
            continue

        # Extract decomposed sub-questions as expected_sub_goals
        decomposition = row.get("question_decomposition", [])
        sub_goals = []
        if isinstance(decomposition, list):
            for item in decomposition:
                if isinstance(item, dict):
                    sq = item.get("question", "")
                    if sq:
                        sub_goals.append(str(sq))
                elif isinstance(item, str):
                    sub_goals.append(item)

        key_facts = [f"Answer: {answer}"]
        if sub_goals:
            for sq in sub_goals[:4]:
                key_facts.append(f"Sub-question: {sq}")

        candidates.append({
            "query": question,
            "reference_answer": answer,
            "key_facts": key_facts,
            "expected_sub_goals": sub_goals[:6] if sub_goals else None,
            "source": "musique",
        })

    random.shuffle(candidates)
    selected = candidates[:count]
    print(f"  MuSiQue: {len(candidates)} candidates, selected {len(selected)}")
    return selected


def _build_task_id(index: int, source: str) -> str:
    prefix = {"hotpotqa": "hpqa", "musique": "musq"}.get(source, "unk")
    return f"{prefix}_{index:03d}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build golden_set.jsonl from public datasets")
    parser.add_argument("--count", type=int, default=30, help="Total number of tasks")
    parser.add_argument("--output", default="deepresearch/data/golden_set.jsonl", help="Output path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)

    # Split count: 70% HotpotQA, 30% MuSiQue
    hotpot_count = int(args.count * 0.7)
    musique_count = args.count - hotpot_count

    hotpot_records = _load_hotpotqa(hotpot_count)
    musique_records = _load_musique(musique_count)

    # Combine and assign task IDs
    all_records: List[Dict[str, Any]] = []
    idx = 1
    for record in hotpot_records:
        task = {
            "task_id": _build_task_id(idx, record["source"]),
            "query": record["query"],
            "reference_answer": record["reference_answer"],
            "key_facts": record["key_facts"],
            "source": record["source"],
        }
        if record.get("expected_sub_goals"):
            task["expected_sub_goals"] = record["expected_sub_goals"]
        all_records.append(task)
        idx += 1

    for record in musique_records:
        task = {
            "task_id": _build_task_id(idx, record["source"]),
            "query": record["query"],
            "reference_answer": record["reference_answer"],
            "key_facts": record["key_facts"],
            "source": record["source"],
        }
        if record.get("expected_sub_goals"):
            task["expected_sub_goals"] = record["expected_sub_goals"]
        all_records.append(task)
        idx += 1

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nWritten {len(all_records)} records to {output_path}")
    print(f"  HotpotQA: {len(hotpot_records)}")
    print(f"  MuSiQue:  {len(musique_records)}")

    # Preview first 3
    print("\nPreview:")
    for record in all_records[:3]:
        print(f"  [{record['task_id']}] ({record['source']}) {record['query'][:60]}...")
        print(f"    Answer: {record['reference_answer'][:80]}...")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
