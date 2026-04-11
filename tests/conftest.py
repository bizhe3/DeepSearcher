"""Shared pytest fixtures for DeepResearch tests."""

from __future__ import annotations

import asyncio
import inspect
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import AsyncMock

import pytest

try:
    import pytest_asyncio  # noqa: F401

    _HAS_PYTEST_ASYNCIO = True
except ImportError:
    _HAS_PYTEST_ASYNCIO = False


def pytest_configure(config: pytest.Config) -> None:
    """Register asyncio marker for cleaner output in plugin and fallback modes."""
    config.addinivalue_line("markers", "asyncio: mark test as asynchronous")


def pytest_pyfunc_call(pyfuncitem: pytest.Function) -> bool | None:
    """Run async tests in environments that do not have pytest-asyncio installed."""
    if _HAS_PYTEST_ASYNCIO:
        return None
    if "asyncio" not in pyfuncitem.keywords:
        return None
    if not inspect.iscoroutinefunction(pyfuncitem.obj):
        return None

    funcargs = {name: pyfuncitem.funcargs[name] for name in pyfuncitem._fixtureinfo.argnames}
    asyncio.run(pyfuncitem.obj(**funcargs))
    return True


@pytest.fixture
def fake_corpus() -> List[Dict[str, str]]:
    """Provide ten short synthetic documents for simulation tests."""
    return [
        {
            "url": f"doc://{index}",
            "title": f"Fake Doc {index}",
            "body": (
                f"Fake document {index} includes climate policy evidence and source references "
                "for verification tasks."
            ),
        }
        for index in range(10)
    ]


@pytest.fixture
def mock_llm_client() -> Any:
    """Provide an AsyncMock-based duck-typed chat client."""

    action_counter = {"count": 0}

    async def _chat(*args: Any, **kwargs: Any) -> str:
        del args
        messages = kwargs.get("messages", [])
        system_prompt = ""
        for message in messages:
            if isinstance(message, dict) and message.get("role") == "system":
                system_prompt = str(message.get("content", ""))
                break

        if "research planning assistant" in system_prompt:
            return """[
  {"id": "sg_1", "description": "Find climate policy evidence"},
  {"id": "sg_2", "description": "Produce final answer with citations"}
]"""

        if "action policy for a web research agent" in system_prompt:
            action_counter["count"] += 1
            if action_counter["count"] % 2 == 1:
                return (
                    "<think>\nNeed to find information about climate policy.\n</think>\n"
                    '{"action_type": "search", "params": {"query": "climate policy 2024"}, "step": 1}'
                )

            return (
                "<think>\nEnough information collected to answer.\n</think>\n"
                '{"action_type": "terminate", "params": {"answer": "Test answer.", "citations": []}, "step": 2}'
            )

        return "{}"

    chat_mock = AsyncMock(side_effect=_chat)
    return SimpleNamespace(chat=chat_mock)

