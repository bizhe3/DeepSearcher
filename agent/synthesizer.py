"""Synthesis component for DeepResearch agent workflows."""

from __future__ import annotations

import inspect
import re
from typing import Any, Dict, List, Tuple

from deepresearch.agent.types import AgentObservation, PageContent, SearchResult, Trajectory

_SYNTHESIS_SYSTEM_PROMPT = """You are a research synthesis assistant.
Write a comprehensive markdown answer to the user's task using only the provided evidence context.
Requirements:
1. Write a complete, well-structured answer in markdown.
2. Cite evidence inline with bracketed numeric references like [1], [2].
3. End with a references section using EXACTLY this heading on its own line: ## References
4. Under ## References, list each source as: {number}. {URL}
   Example:
   1. https://example.com/article
   2. https://another.com/page
5. Do not invent sources. Use only URLs present in the context.
6. Do not use any other heading for the references section (not 参考文献, not 参考来源).
"""

_URL_PATTERN = re.compile(r"https?://[^\s)\]>]+", flags=re.IGNORECASE)


class SynthesisWriter:
    """Generate final markdown answers and extract citations from references."""

    def __init__(self, llm_client: Any, model: str) -> None:
        """Initialize the synthesis writer with a duck-typed async chat client."""
        self.llm_client = llm_client
        self.model = model

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
        answer_body, citations = self._extract_citations(llm_response)
        return answer_body, citations

    def _extract_citations(self, llm_response: str) -> tuple[str, List[str]]:
        """Extract answer body and reference URLs from markdown response text."""
        marker_match = re.search(r"^##\s*(References|参考来源|参考文献|参考资料)\s*$", llm_response, flags=re.IGNORECASE | re.MULTILINE)

        if marker_match is None:
            body = llm_response.strip()
            return body, []

        split_index = marker_match.start()
        body = llm_response[:split_index].strip()
        references_section = llm_response[marker_match.end() :]

        urls = self._extract_urls(references_section)
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
                if len(snippet) > 300:
                    snippet = f"{snippet[:300]}..."
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
