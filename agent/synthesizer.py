"""Synthesis component for DeepResearch agent workflows."""

from __future__ import annotations

import inspect
import re
from typing import Any, Dict, List, Tuple

from deepresearch.agent.types import AgentObservation, PageContent, SearchResult, Trajectory

_SYNTHESIS_SYSTEM_PROMPT = """You are a research synthesis assistant that builds answers from provided evidence.

CORE PRINCIPLE: Be DECISIVE based on evidence. Refusing to answer when the evidence
clearly contains the information is just as wrong as making up facts.

RULES:
1. Use the provided evidence as the SOLE source of factual content. Do NOT add
   facts you happen to know but that are not in the evidence.

2. CONFIDENCE LEVELS — match your wording to evidence strength:
   - DIRECT (evidence explicitly states it):
       "X = Y [N]"
   - INFERRED (evidence implies it through related statements):
       "Based on [N], X is Y" or "Evidence indicates X is Y [N]"
   - PARTIAL (only some aspects covered, others not):
       Answer the parts you can; say "the evidence does not cover [specific aspect]"
       only for the parts truly absent.

3. AVOID OVER-REFUSAL. If the evidence mentions an entity, name, number, or fact
   relevant to the question — even indirectly — USE IT. Examples of over-refusal
   to AVOID:
   ❌ Bad: "Sources do not specify the license name."
       Evidence had: "[12] references 'Llama 3 Community License Agreement'"
       ✅ Good: "The license is the 'Llama 3 Community License Agreement' [12]."
   ❌ Bad: "Sources do not address the company's valuation."
       Evidence had: "[7] Hugging Face raised $235M at $4.5B valuation"
       ✅ Good: "Hugging Face was valued at $4.5 billion [7]."
   ❌ Bad: "Sources do not list 3 papers by author X."
       Evidence had: 3 papers attributed to X with titles
       ✅ Good: List the 3 papers, noting any uncertainty about venue/year per item.

4. Only refuse a specific sub-question when the evidence is GENUINELY absent or
   directly contradictory. Do not refuse based on technicalities like "not the
   exact wording" or "not the official source."

5. Every factual claim MUST have an inline citation [N] pointing to the
   specific source supporting it.

6. CRITICAL — citation numbering: Decide which sources you will cite. Number them
   sequentially starting from [1]. Use ONLY these sequential numbers in the text.
   The [N] in the text MUST match the N-th entry in your References list.

7. End with a references section using EXACTLY this heading on its own line: ## References
   Under it, list each source as: {number}. {URL}
   Only list URLs you actually cited. Numbering sequential (1, 2, 3, ...).

8. Do not invent sources. Use only URLs present in the context.
9. Do not use other reference headings (not 参考文献, not 参考来源).
10. Write in English. Use markdown formatting.
"""

_URL_PATTERN = re.compile(r"https?://[^\s)\]>]+", flags=re.IGNORECASE)

# Flexible References-section heading matcher.
# Accepts: # / ## / ### / **References** / ## References: / ## REFERENCES
# Plus EN/CN heading variants and trailing colon (Latin or full-width).
_REFS_HEADING_PATTERN = re.compile(
    r"^(?:#{1,4}\s*|\*\*\s*)"
    r"(References|REFERENCES|Sources|Citations|参考来源|参考文献|参考资料)"
    r"(?:\s*[:：])?"
    r"\s*\*{0,2}\s*$",
    flags=re.IGNORECASE | re.MULTILINE,
)


class SynthesisWriter:
    """Generate final markdown answers and extract citations from references."""

    def __init__(self, llm_client: Any, model: str) -> None:
        """Initialize the synthesis writer with a duck-typed async chat client."""
        self.llm_client = llm_client
        self.model = model
        # Last raw LLM response (before splitting body / citations) — useful
        # for debugging citation extraction or audit-quality issues.
        self.last_raw_response: str = ""

    async def synthesize(self, task: str, trajectory: Trajectory) -> tuple[str, List[str]]:
        """Synthesize a markdown answer and citation list from trajectory observations."""
        successful_observations = [
            observation for observation in trajectory.observations if observation.success
        ]
        context = self._build_context(successful_observations)

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": _SYNTHESIS_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Task:\n{task}\n\n"
                    "Evidence context:\n"
                    f"{context}\n\n"
                    "Produce the final answer now."
                ),
            },
        ]

        response = self.llm_client.chat(messages=messages, response_format="text")
        if inspect.isawaitable(response):
            response = await response

        llm_response = "" if response is None else str(response)
        self.last_raw_response = llm_response
        answer_body, citations = self._extract_citations(llm_response)
        return answer_body, citations

    def _extract_citations(self, llm_response: str) -> tuple[str, List[str]]:
        """Extract answer body and reference URLs from markdown response text.

        Two-tier strategy:
          (1) Find a flexible 'References' / 'Sources' / 'Citations' section
              (supports # / ## / ### / **bold**, optional colon, EN/CN labels).
              If found AND yields URLs, split body | refs and return.
          (2) Fallback: scan the full response for URLs (handles cases where
              the LLM omits a heading, uses a non-recognized heading, or puts
              URLs only inline). The body remains the full text.
        """
        marker_match = _REFS_HEADING_PATTERN.search(llm_response)

        if marker_match is not None:
            split_index = marker_match.start()
            body = llm_response[:split_index].strip()
            references_section = llm_response[marker_match.end():]
            urls = self._extract_urls(references_section)
            if urls:
                return body, urls
            # heading present but section yielded no URLs → fall through

        # Fallback: scan the entire response for URLs.
        body = llm_response.strip()
        urls = self._extract_urls(body)
        return body, urls

    def _build_context(self, observations: List[AgentObservation]) -> str:
        """Build compact evidence context from successful observations."""
        if not observations:
            return "No successful observations available."

        lines: List[str] = []
        index = 1
        for observation in observations:
            evidence_items = self._observation_to_evidence_list(observation)
            for url, snippet in evidence_items:
                if len(snippet) > 800:
                    snippet = f"{snippet[:800]}..."
                lines.append(f"[{index}] URL: {url}\nSnippet: {snippet}")
                index += 1

        return "\n\n".join(lines)

    def _observation_to_evidence_list(self, observation: AgentObservation) -> List[Tuple[str, str]]:
        """Convert an observation result into a list of (url, snippet) pairs."""
        result = observation.result

        if isinstance(result, PageContent):
            snippet = result.body.strip() or result.title.strip()
            return [(result.url, snippet)]

        if isinstance(result, list):
            items = []
            for item in result:
                if isinstance(item, SearchResult):
                    snippet = item.snippet.strip() or item.title.strip()
                    items.append((item.url, snippet))
            return items if items else [("n/a", str(result))]

        if isinstance(result, SearchResult):
            snippet = result.snippet.strip() or result.title.strip()
            return [(result.url, snippet)]

        if isinstance(result, str):
            url = str(observation.action.params.get("url") or observation.action.params.get("link_url") or "n/a")
            return [(url, result.strip())]

        return [("n/a", str(result))]

    @staticmethod
    def _extract_urls(text: str) -> List[str]:
        """Extract unique URLs from text while preserving first-seen order."""
        seen = set()
        urls: List[str] = []
        for match in _URL_PATTERN.findall(text):
            url = match.rstrip(".,;")
            if url in seen:
                continue
            seen.add(url)
            urls.append(url)
        return urls
