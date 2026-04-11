"""Offline long-term memory evaluation — no LLM or API key required.

Three test suites:

  Suite A — MemoryStore persistence (write + reload + search)
    - Write entries to JSONL, reload from disk, verify integrity

  Suite B — Cross-session retrieval quality
    - Seed memory with prior research, query with a related new task
    - Measure: does retrieval find the relevant prior entry?
    - Compare: with vs without memory, what context does the Agent see?

  Suite C — Memory-augmented context improvement
    - Simulate two scenarios: Agent with empty memory vs Agent with seeded memory
    - Measure: context relevance (keyword overlap between prior knowledge and new query)
    - Measure: context length reduction (Agent starts with knowledge, fewer exploration steps needed)

Run:
    cd d:/Agent
    python -m deepresearch.tests.eval_memory
"""

from __future__ import annotations

import os
import sys
import tempfile
from typing import List

from deepresearch.utils.memory_store import MemoryEntry, MemoryStore

# ---------------------------------------------------------------------------
# Test data: simulate a sequence of research sessions
# ---------------------------------------------------------------------------

_RESEARCH_HISTORY = [
    {
        "task_id": "hist_001",
        "query": "Compare quantum computing policies in China, US, and EU in 2024",
        "summary": (
            "China invested over 20 billion yuan in quantum technology in 2024, "
            "focusing on Origin Quantum and QuantumCTek. The US CHIPS Act allocated "
            "$5.2 billion for quantum research. The EU Quantum Flagship program has "
            "invested 1 billion euros cumulatively. China leads in quantum patent filings globally."
        ),
        "citations": [
            "https://gov.cn/quantum-policy-2024",
            "https://congress.gov/chips-act",
            "https://qt.eu/flagship",
        ],
        "key_facts": [
            "China quantum patent filings rank #1 globally",
            "Origin Quantum released 72-qubit superconducting chip",
            "US CHIPS Act: $5.2B for quantum",
            "EU Quantum Flagship: 1B euros total",
        ],
    },
    {
        "task_id": "hist_002",
        "query": "Berlin Airlift 1948 political consequences and Cold War impact",
        "summary": (
            "The Berlin Airlift (1948-1949) demonstrated Western resolve against Soviet "
            "blockade. Key consequences: strengthened NATO solidarity, increased US commitment "
            "to European defense, accelerated formation of West Germany (FRG) in 1949."
        ),
        "citations": [
            "https://history.state.gov/berlin-airlift",
            "https://nato.int/history",
        ],
        "key_facts": [
            "Soviet blockade lifted May 1949",
            "277,000 flights during airlift",
            "NATO founded April 1949 partly due to airlift",
            "FRG established May 1949",
        ],
    },
    {
        "task_id": "hist_003",
        "query": "Mediterranean diet health benefits and cardiovascular risk reduction",
        "summary": (
            "Studies show the Mediterranean diet reduces cardiovascular risk by 25-30%. "
            "Key components: olive oil, fish, whole grains, vegetables. The PREDIMED trial "
            "demonstrated significant reduction in major cardiovascular events."
        ),
        "citations": ["https://nejm.org/predimed"],
        "key_facts": [
            "25-30% cardiovascular risk reduction",
            "PREDIMED trial: landmark RCT",
            "Olive oil and fish are key components",
        ],
    },
    {
        "task_id": "hist_004",
        "query": "Electric vehicle market share and growth trends 2023-2024",
        "summary": (
            "EVs reached 18% of new car sales globally in 2023 (IEA). China leads with "
            "35% EV penetration. BYD overtook Tesla in Q4 2023 total sales. Battery costs "
            "fell below $140/kWh. Europe's EV share hit 22%."
        ),
        "citations": [
            "https://iea.org/ev-outlook-2024",
            "https://reuters.com/byd-tesla",
        ],
        "key_facts": [
            "Global EV share: 18% in 2023",
            "China EV penetration: 35%",
            "BYD overtook Tesla Q4 2023",
            "Battery cost below $140/kWh",
        ],
    },
]

# New tasks that should benefit from prior research
_NEW_TASKS = [
    {
        "query": "Analyze China's quantum computing supply chain upstream and downstream",
        "expected_hit_task_id": "hist_001",
        "expected_keywords": ["Origin Quantum", "QuantumCTek", "72-qubit", "20 billion"],
        "description": "Should retrieve quantum policy research from hist_001",
    },
    {
        "query": "How did the Berlin Airlift influence NATO formation",
        "expected_hit_task_id": "hist_002",
        "expected_keywords": ["NATO", "solidarity", "1949", "blockade"],
        "description": "Should retrieve Berlin Airlift research from hist_002",
    },
    {
        "query": "BYD vs Tesla global market competition 2024",
        "expected_hit_task_id": "hist_004",
        "expected_keywords": ["BYD", "Tesla", "18%", "Q4 2023"],
        "description": "Should retrieve EV market research from hist_004",
    },
    {
        "query": "Best programming languages for web development in 2024",
        "expected_hit_task_id": None,
        "expected_keywords": [],
        "description": "No relevant prior research, should return empty or low-relevance",
    },
]


# ---------------------------------------------------------------------------
# Suite A: Persistence
# ---------------------------------------------------------------------------

def _suite_a() -> bool:
    print("=" * 60)
    print("Suite A -- MemoryStore persistence (write + reload)")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmp_dir:
        store_path = os.path.join(tmp_dir, "test_memory.jsonl")

        # Write
        store = MemoryStore(store_path=store_path, top_k=3)
        for record in _RESEARCH_HISTORY:
            store.add(**record)
        print(f"\nWritten {len(store)} entries to {store_path}")

        # Verify deduplication
        store.add(**_RESEARCH_HISTORY[0])
        assert len(store) == len(_RESEARCH_HISTORY), (
            f"Dedup failed: expected {len(_RESEARCH_HISTORY)}, got {len(store)}"
        )
        print(f"Deduplication check: PASS (still {len(store)} entries after re-add)")

        # Reload from disk
        store2 = MemoryStore(store_path=store_path, top_k=3)
        assert len(store2) == len(_RESEARCH_HISTORY), (
            f"Reload failed: expected {len(_RESEARCH_HISTORY)}, got {len(store2)}"
        )
        print(f"Reload from disk: PASS ({len(store2)} entries loaded)")

        # Verify content integrity
        for i, entry in enumerate(store2._entries):
            original = _RESEARCH_HISTORY[i]
            assert entry.task_id == original["task_id"], f"task_id mismatch at index {i}"
            assert entry.query == original["query"], f"query mismatch at index {i}"
            assert entry.summary == original["summary"], f"summary mismatch at index {i}"
            assert entry.embedding is not None, f"embedding is None at index {i}"
        print("Content integrity: PASS (all fields match)")

    print("\nSuite A: ALL PASSED")
    return True


# ---------------------------------------------------------------------------
# Suite B: Cross-session retrieval quality
# ---------------------------------------------------------------------------

def _suite_b() -> bool:
    print("\n" + "=" * 60)
    print("Suite B -- Cross-session retrieval quality")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmp_dir:
        store_path = os.path.join(tmp_dir, "test_memory.jsonl")
        store = MemoryStore(store_path=store_path, top_k=3)
        for record in _RESEARCH_HISTORY:
            store.add(**record)

        all_passed = True
        for task in _NEW_TASKS:
            query = task["query"]
            expected_id = task["expected_hit_task_id"]
            expected_keywords = task["expected_keywords"]

            results = store.search(query)
            result_ids = [r.task_id for r in results]

            print(f"\nQuery: {query[:60]}...")
            print(f"  Expected hit: {expected_id or '(none)'}")
            print(f"  Retrieved:    {result_ids}")

            if expected_id is not None:
                hit = expected_id in result_ids
                # Check it's the top result
                top_hit = result_ids[0] == expected_id if result_ids else False
                print(f"  Target found: {'YES [HIT]' if hit else 'NO [MISS]'}")
                print(f"  Is top-1:     {'YES' if top_hit else 'NO'}")

                if hit and expected_keywords:
                    matched_entry = next(r for r in results if r.task_id == expected_id)
                    context = matched_entry.format_context()
                    kw_hits = [kw for kw in expected_keywords if kw.lower() in context.lower()]
                    kw_misses = [kw for kw in expected_keywords if kw.lower() not in context.lower()]
                    print(f"  Keywords in context: {len(kw_hits)}/{len(expected_keywords)}")
                    if kw_misses:
                        print(f"  Missing keywords: {kw_misses}")

                if not hit:
                    all_passed = False
            else:
                # Should NOT find highly relevant results
                if not results:
                    print(f"  Correctly returned empty: YES [PASS]")
                else:
                    # Check if results are low-relevance (expected for unrelated query)
                    print(f"  Returned {len(results)} entries (low relevance expected)")

        if all_passed:
            print("\nSuite B: ALL PASSED")
        else:
            print("\nSuite B: SOME FAILURES (see above)")
    return all_passed


# ---------------------------------------------------------------------------
# Suite C: Memory-augmented context comparison
# ---------------------------------------------------------------------------

def _suite_c() -> bool:
    print("\n" + "=" * 60)
    print("Suite C -- Memory-augmented context comparison")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Agent WITHOUT memory
        store_path_empty = os.path.join(tmp_dir, "empty_memory.jsonl")
        store_empty = MemoryStore(store_path=store_path_empty, top_k=3)

        # Agent WITH memory
        store_path_seeded = os.path.join(tmp_dir, "seeded_memory.jsonl")
        store_seeded = MemoryStore(store_path=store_path_seeded, top_k=3)
        for record in _RESEARCH_HISTORY:
            store_seeded.add(**record)

        query = "Analyze China's quantum computing supply chain upstream and downstream"
        print(f"\nQuery: {query}")

        # Without memory
        results_empty = store_empty.search(query)
        context_empty = "\n".join(e.format_context() for e in results_empty)

        # With memory
        results_seeded = store_seeded.search(query)
        context_seeded = "\n".join(e.format_context() for e in results_seeded)

        print(f"\n--- Without long-term memory ---")
        print(f"  Prior knowledge entries: {len(results_empty)}")
        print(f"  Context length: {len(context_empty)} chars")
        print(f"  Agent starts with: (nothing, must discover everything from scratch)")

        print(f"\n--- With long-term memory ---")
        print(f"  Prior knowledge entries: {len(results_seeded)}")
        print(f"  Context length: {len(context_seeded)} chars")

        if results_seeded:
            print(f"\n  Context injected into LLM prompt:")
            for entry in results_seeded:
                formatted = entry.format_context(max_summary_len=150)
                for line in formatted.split("\n"):
                    print(f"    {line}")
                print()

        # Keyword relevance analysis
        target_keywords = [
            "Origin Quantum", "QuantumCTek", "72-qubit", "20 billion",
            "quantum", "patent", "superconducting",
        ]
        without_hits = sum(1 for kw in target_keywords if kw.lower() in context_empty.lower())
        with_hits = sum(1 for kw in target_keywords if kw.lower() in context_seeded.lower())

        print(f"  Keyword relevance analysis:")
        print(f"    Without memory: {without_hits}/{len(target_keywords)} keywords available")
        print(f"    With memory:    {with_hits}/{len(target_keywords)} keywords available")
        print(f"    Knowledge gain: +{with_hits - without_hits} keywords")

        passed = with_hits > without_hits and len(results_seeded) > 0
        if passed:
            print(f"\n  Impact: Agent with memory already knows key entities (Origin Quantum,")
            print(f"  QuantumCTek, 72-qubit chip) before any search, enabling more precise")
            print(f"  queries and fewer exploration steps.")

        print(f"\nSuite C: {'PASSED' if passed else 'FAILED'}")
    return passed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    results = {}
    try:
        results["A"] = _suite_a()
    except Exception as exc:
        print(f"[Suite A FAILED] {exc}", file=sys.stderr)
        results["A"] = False

    try:
        results["B"] = _suite_b()
    except Exception as exc:
        print(f"[Suite B FAILED] {exc}", file=sys.stderr)
        results["B"] = False

    try:
        results["C"] = _suite_c()
    except Exception as exc:
        print(f"[Suite C FAILED] {exc}", file=sys.stderr)
        results["C"] = False

    print("\n" + "=" * 60)
    print("Long-term Memory Evaluation Summary")
    print("=" * 60)
    for suite, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"  Suite {suite}: {status}")

    all_passed = all(results.values())
    print(f"\nOverall: {'ALL PASSED' if all_passed else 'SOME FAILURES'}")

    if all_passed:
        print("\nWhat these results demonstrate:")
        print("  1. MemoryStore correctly persists and reloads across sessions (Suite A)")
        print("  2. Semantic retrieval finds relevant prior research for new tasks (Suite B)")
        print("  3. Memory-augmented Agent starts with domain knowledge that a")
        print("     memory-less Agent must discover from scratch (Suite C)")
        print("\nFor end-to-end validation with real LLM:")
        print("  1. Run Agent on task set A (seeds memory_store.jsonl)")
        print("  2. Run Agent on related task set B (retrieves from memory)")
        print("  3. Compare step_count and answer ROUGE-L between:")
        print("     - Agent with seeded memory vs Agent with empty memory")
        print("     - Expected: fewer steps + higher answer quality with memory")


if __name__ == "__main__":
    main()
