"""Environment tests for DeepResearch."""

from __future__ import annotations

import asyncio
from typing import Dict, List

import pytest

from deepresearch.envs.sim_env import SimEnv


class _FakeEncoder:
    """Minimal encoder double for deterministic test embeddings."""

    def encode(self, texts: List[str], **_: object) -> List[List[float]]:
        """Return one-dimensional embeddings based on text length."""
        return [[float(len(text))] for text in texts]


class _FakeIndex:
    """Minimal FAISS-like index double for deterministic ranking."""

    def __init__(self, document_count: int) -> None:
        """Store corpus size used to cap search outputs."""
        self.document_count = document_count

    def search(self, query_embeddings: object, top_k: int) -> tuple[List[List[float]], List[List[int]]]:
        """Return first k indices as if they were top-ranked matches."""
        del query_embeddings
        result_count = min(top_k, self.document_count)
        scores = [[1.0 for _ in range(result_count)]]
        indices = [[index for index in range(result_count)]]
        return scores, indices


@pytest.fixture
def in_memory_corpus() -> List[Dict[str, str]]:
    """Provide a small corpus for environment tests without filesystem reads."""
    return [
        {"url": "doc://alpha", "title": "Alpha", "body": "A" * 2200},
        {"url": "doc://beta", "title": "Beta", "body": "B" * 120},
        {"url": "doc://gamma", "title": "Gamma", "body": "C" * 80},
    ]


@pytest.fixture
def sim_env(monkeypatch: pytest.MonkeyPatch, in_memory_corpus: List[Dict[str, str]]) -> SimEnv:
    """Create SimEnv with patched index building so tests stay fully in-memory."""

    def _fake_build_index(self: SimEnv) -> None:
        self._documents = list(in_memory_corpus)
        self._documents_by_url = {document["url"]: document for document in self._documents}
        self._encoder = _FakeEncoder()
        self._index = _FakeIndex(len(self._documents))

    monkeypatch.setattr(SimEnv, "_build_index", _fake_build_index)
    return SimEnv(corpus_dir="unused")


def test_search_returns_correct_number_of_results(sim_env: SimEnv) -> None:
    """Search should return exactly top_k results when enough docs are indexed."""
    results = asyncio.run(sim_env.search(query="irrelevant", top_k=2))

    assert len(results) == 2
    assert [result.rank for result in results] == [1, 2]


def test_fetch_page_paginates_correctly(sim_env: SimEnv) -> None:
    """Fetch should paginate long documents into 1000-character pages."""
    first_page = asyncio.run(sim_env.fetch_page(url="doc://alpha"))

    assert first_page.is_paginated is True
    assert first_page.current_page == 1
    assert first_page.total_pages == 3
    assert len(first_page.body) == 1000


def test_get_next_page_returns_none_on_last_page(sim_env: SimEnv) -> None:
    """Next-page retrieval should stop with None at the final page."""
    page = asyncio.run(sim_env.fetch_page(url="doc://alpha"))

    while True:
        next_page = asyncio.run(sim_env.get_next_page(page))
        if next_page is None:
            break
        page = next_page

    assert page.current_page == page.total_pages
    assert asyncio.run(sim_env.get_next_page(page)) is None
