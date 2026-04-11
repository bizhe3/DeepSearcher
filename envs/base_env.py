"""Base environment abstractions and action execution helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from deepresearch.agent.types import (
    AgentAction,
    AgentObservation,
    PageContent,
    ResearchTask,
    SearchResult,
    SubGoal,
    Trajectory,
)


class BaseEnv(ABC):
    """Abstract base class for research environments."""

    @abstractmethod
    async def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Return top-k ranked search results for a query."""

    @abstractmethod
    async def fetch_page(self, url: str) -> PageContent:
        """Fetch and parse a page by URL into structured content."""

    @abstractmethod
    async def click_link(self, page: PageContent, link_url: str) -> PageContent:
        """Open a link from a page and return the resulting page content."""

    @abstractmethod
    async def get_next_page(self, page: PageContent) -> Optional[PageContent]:
        """Return the next page for paginated content, if one exists."""

    async def execute_action(
        self, action: AgentAction, context: PageContent | None
    ) -> AgentObservation:
        """Execute an agent action and convert it into a standardized observation."""
        try:
            if action.action_type == "search":
                query = str(action.params["query"])
                top_k = int(action.params.get("top_k", 5))
                search_results = await self.search(query=query, top_k=top_k)
                result: list | SearchResult | PageContent | str = (
                    search_results if search_results else "No search results."
                )
            elif action.action_type == "click":
                link_url = str(
                    action.params.get("link_url") or action.params.get("url", "")
                )
                if not link_url:
                    raise ValueError("Click action requires params['link_url'] or params['url'].")
                if context is None:
                    result = await self.fetch_page(url=link_url)
                else:
                    result = await self.click_link(page=context, link_url=link_url)
            elif action.action_type == "scroll":
                if context is None:
                    raise ValueError("Context page is required for scroll action.")
                next_page = await self.get_next_page(page=context)
                result = next_page if next_page is not None else "No next page available."
            elif action.action_type == "extract":
                target_url = action.params.get("url", context.url if context is not None else None)
                if target_url is None:
                    raise ValueError("Extract action requires params['url'] or a context page.")
                result = await self.fetch_page(url=str(target_url))
            elif action.action_type == "cross_check":
                query = str(action.params["query"])
                top_k = int(action.params.get("top_k", 5))
                cross_check_results = await self.search(query=query, top_k=top_k)
                result = cross_check_results if cross_check_results else "No cross-check results."
            elif action.action_type == "terminate":
                result = action.params["answer"]
            else:
                raise ValueError(f"Unsupported action_type: {action.action_type}")

            return AgentObservation(action=action, result=result, success=True, error=None)
        except Exception as e:
            return AgentObservation(action=action, result="", success=False, error=str(e))

