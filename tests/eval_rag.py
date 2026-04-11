"""Offline RAG quality benchmark — no LLM or API key required.

Two test suites:
  Suite A — PageChunkRetriever (改动一)
    - Compression ratio: how much body length is reduced
    - Keyword precision: fraction of returned chunks containing query keywords

  Suite B — ObservationRetriever vs recency baseline (改动二)
    - Constructs a synthetic trajectory where the single relevant observation
      is at step 1 (early), surrounded by noise
    - Compares hit-rate of RAG retrieval vs. naive `observations[-top_k:]`

Run:
    cd d:/Agent
    python -m deepresearch.tests.eval_rag
"""

from __future__ import annotations

import re
import sys
import textwrap
from typing import List

from deepresearch.agent.types import AgentAction, AgentObservation, PageContent, SearchResult
from deepresearch.utils.chunk_retriever import ObservationRetriever, PageChunkRetriever

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_obs(step: int, topic: str, snippet: str) -> AgentObservation:
    return AgentObservation(
        action=AgentAction(
            thought=f"Looking into {topic}",
            action_type="search",
            params={"query": topic},
            step=step,
        ),
        result=[
            SearchResult(url=f"http://example.com/{step}", title=topic, snippet=snippet, rank=1)
        ],
        success=True,
        error=None,
    )


def _keyword_precision(chunks: List[str], keywords: List[str]) -> float:
    """Fraction of chunks that contain at least one keyword (case-insensitive)."""
    if not chunks:
        return 0.0
    hits = sum(
        1
        for chunk in chunks
        if any(kw.lower() in chunk.lower() for kw in keywords)
    )
    return hits / len(chunks)


# ---------------------------------------------------------------------------
# Suite A: PageChunkRetriever
# ---------------------------------------------------------------------------

_LONG_PAGE_TEMPLATE = """\
{relevant_block}

{noise_blocks}
"""

_RELEVANT_BLOCK = """\
The Berlin Airlift (1948–1949) was a pivotal Cold War event. Western Allied
aircraft supplied West Berlin after the Soviet Union blocked ground access.
The operation demonstrated the resolve of Western democracies and ultimately
led the Soviets to lift the blockade in May 1949. Political consequences
included strengthened NATO solidarity and increased US commitment to European
defense.
"""

_NOISE_BLOCK = """\
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor
incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis
nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu
fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident.
"""


def _suite_a() -> None:
    print("=" * 60)
    print("Suite A — PageChunkRetriever")
    print("=" * 60)

    query = "Berlin Airlift political consequences"
    keywords = ["Berlin", "Airlift", "NATO", "Soviet", "blockade", "political"]

    # Build a long page: 1 relevant block + 15 noise blocks (~1 relevant + 15 irrelevant paragraphs)
    noise = "\n\n".join([_NOISE_BLOCK.strip()] * 15)
    body = _RELEVANT_BLOCK.strip() + "\n\n" + noise

    retriever = PageChunkRetriever(chunk_size=300, top_k=5, min_body_len=500)

    # Baseline: no filtering
    print(f"\nBody length (before filter) : {len(body):,} chars")

    filtered = retriever.filter(body=body, query=query)
    print(f"Body length (after  filter) : {len(filtered):,} chars")
    print(f"Compression ratio           : {len(filtered)/len(body):.2%}")

    # Precision
    before_chunks = retriever._split(body)
    after_chunks = retriever._split(filtered)
    before_prec = _keyword_precision(before_chunks, keywords)
    after_prec = _keyword_precision(after_chunks, keywords)
    print(f"\nKeyword precision (before)  : {before_prec:.2%}  ({sum(1 for c in before_chunks if any(k.lower() in c.lower() for k in keywords))}/{len(before_chunks)} chunks)")
    print(f"Keyword precision (after)   : {after_prec:.2%}  ({sum(1 for c in after_chunks if any(k.lower() in c.lower() for k in keywords))}/{len(after_chunks)} chunks)")
    print(f"Precision lift              : +{after_prec - before_prec:.2%}")

    # Show what was kept
    print(f"\nRetained content preview:")
    print(textwrap.indent(filtered[:400], "  ") + ("..." if len(filtered) > 400 else ""))


# ---------------------------------------------------------------------------
# Suite B: ObservationRetriever vs recency baseline
# ---------------------------------------------------------------------------

def _suite_b() -> None:
    print("\n" + "=" * 60)
    print("Suite B — ObservationRetriever vs recency baseline [-5:]")
    print("=" * 60)

    query = "Berlin Airlift political consequences of the blockade"

    # Build a 10-step trajectory:
    #   step 1  → highly relevant (Berlin Airlift)       ← target
    #   steps 2–8 → noise on unrelated topics
    #   steps 9–10 → semi-related (Cold War in general)
    observations: List[AgentObservation] = [
        _make_obs(1, "Berlin Airlift 1948",
                  "The Berlin Airlift successfully supplied West Berlin; Soviet blockade lifted May 1949. "
                  "Political consequences: NATO cohesion, US commitment to European defense."),
        _make_obs(2, "French cuisine history",
                  "French cuisine evolved over centuries. Key developments in the 17th and 18th centuries."),
        _make_obs(3, "Amazon rainforest biodiversity",
                  "The Amazon hosts 10% of all species on Earth. Deforestation threatens endemic fauna."),
        _make_obs(4, "Quantum computing basics",
                  "Qubits differ from classical bits by superposition. IBM unveiled a 1000-qubit chip in 2023."),
        _make_obs(5, "Solar panel efficiency 2024",
                  "Perovskite solar cells achieved 33% efficiency in lab conditions this year."),
        _make_obs(6, "World Cup 2022 highlights",
                  "Argentina defeated France on penalties. Messi won the Golden Ball award."),
        _make_obs(7, "Mediterranean diet benefits",
                  "Studies show reduced cardiovascular risk with olive oil and fish consumption."),
        _make_obs(8, "Electric vehicle market share",
                  "EVs reached 18% of new car sales globally in 2023 according to the IEA."),
        _make_obs(9, "Cold War arms race overview",
                  "The Cold War nuclear arms race accelerated after the Soviet atomic bomb test in 1949."),
        _make_obs(10, "Marshall Plan economic impact",
                  "The Marshall Plan provided $13 billion to rebuild Europe and counter Soviet influence."),
    ]

    top_k = 5
    retriever = ObservationRetriever(top_k=top_k)

    # RAG retrieval
    rag_results = retriever.search(query=query, observations=observations)
    rag_steps = [obs.action.step for obs in rag_results]

    # Recency baseline: last top_k
    recency_results = observations[-top_k:]
    recency_steps = [obs.action.step for obs in recency_results]

    target_step = 1  # step 1 is the relevant one

    rag_hit = target_step in rag_steps
    recency_hit = target_step in recency_steps

    print(f"\nQuery: {query[:60]}...")
    print(f"\nTrajectory size  : {len(observations)} observations")
    print(f"top_k            : {top_k}")
    print(f"Target step      : step {target_step} (Berlin Airlift)")
    print()
    print(f"RAG selected steps     : {sorted(rag_steps)}")
    print(f"Recency [-5:] steps    : {sorted(recency_steps)}")
    print()
    print(f"RAG found target       : {'YES [HIT]' if rag_hit else 'NO  [MISS]'}")
    print(f"Recency found target   : {'YES [HIT]' if recency_hit else 'NO  [MISS]'}")

    if rag_hit and not recency_hit:
        print("\n→ RAG recovered the early relevant observation that recency truncation missed.")
    elif rag_hit and recency_hit:
        print("\n→ Both retrieved the target (trajectory not long enough to force a miss).")
    else:
        print("\n→ RAG also missed the target; consider increasing top_k or re-checking embeddings.")

    print("\nRAG-selected observation snippets:")
    for obs in rag_results:
        result = obs.result
        if isinstance(result, list) and result:
            snippet = result[0].snippet[:80]
        else:
            snippet = str(result)[:80]
        marker = " ← TARGET" if obs.action.step == target_step else ""
        print(f"  step {obs.action.step:>2}: {snippet}{marker}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    try:
        _suite_a()
    except Exception as exc:
        print(f"[Suite A FAILED] {exc}", file=sys.stderr)
        raise

    try:
        _suite_b()
    except Exception as exc:
        print(f"[Suite B FAILED] {exc}", file=sys.stderr)
        raise

    print("\n" + "=" * 60)
    print("Eval complete.")
    print("=" * 60)
    print()
    print("Next step for end-to-end comparison:")
    print("  1. Set up your LLM client (real API key)")
    print("  2. Run eval_golden.py on BOTH versions (old vs new agent.py)")
    print("  3. Compare 'total' reward column — expect +answer score")
    print("     from better context passed to the LLM.")


if __name__ == "__main__":
    main()
