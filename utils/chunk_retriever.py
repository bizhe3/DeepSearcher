"""Chunk-level retrievers for long page content and trajectory observations."""

from __future__ import annotations

import re
from typing import List, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from deepresearch.agent.types import AgentObservation


class PageChunkRetriever:
    """Semantic chunk filter for long page content."""

    def __init__(self, chunk_size: int = 300, top_k: int = 5, min_body_len: int = 1000) -> None:
        self.chunk_size = chunk_size
        self.top_k = top_k
        self.min_body_len = min_body_len
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer("all-MiniLM-L6-v2")
        return self._model

    def _split(self, text: str) -> List[str]:
        chunks: List[str] = []
        paragraphs = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]

        for paragraph in paragraphs:
            if len(paragraph) <= self.chunk_size:
                chunks.append(paragraph)
                continue

            sentences = [
                sentence.strip()
                for sentence in re.split(r"\n+|(?<=[。\.])", paragraph)
                if sentence.strip()
            ]
            if not sentences:
                sentences = [paragraph]

            current_parts: List[str] = []
            current_len = 0

            for sentence in sentences:
                if len(sentence) > self.chunk_size:
                    if current_parts:
                        chunks.append(" ".join(current_parts).strip())
                        current_parts = []
                        current_len = 0

                    start = 0
                    while start < len(sentence):
                        piece = sentence[start : start + self.chunk_size].strip()
                        if piece:
                            chunks.append(piece)
                        start += self.chunk_size
                    continue

                projected_len = current_len + (1 if current_parts else 0) + len(sentence)
                if current_parts and projected_len > self.chunk_size:
                    chunks.append(" ".join(current_parts).strip())
                    current_parts = [sentence]
                    current_len = len(sentence)
                else:
                    current_parts.append(sentence)
                    current_len = projected_len

            if current_parts:
                chunks.append(" ".join(current_parts).strip())

        return [chunk for chunk in chunks if len(chunk) >= 30]

    def filter(self, body: str, query: str) -> str:
        if len(body) <= self.min_body_len:
            return body

        chunks = self._split(body)
        if len(chunks) <= self.top_k or self.top_k <= 0:
            return body

        model = self._get_model()
        chunk_vecs = np.asarray(model.encode(chunks), dtype=np.float32)
        query_vec = np.asarray(model.encode([query]), dtype=np.float32)

        import faiss

        faiss.normalize_L2(chunk_vecs)
        faiss.normalize_L2(query_vec)

        index = faiss.IndexFlatIP(chunk_vecs.shape[1])
        index.add(chunk_vecs)
        _, indices = index.search(query_vec, self.top_k)

        sorted_indices = sorted({int(idx) for idx in indices[0] if idx >= 0})
        if not sorted_indices:
            return body

        return "\n\n".join(chunks[i] for i in sorted_indices)


class ObservationRetriever:
    """Semantic retriever over trajectory observations."""

    def __init__(self, top_k: int = 5) -> None:
        self.top_k = top_k
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer("all-MiniLM-L6-v2")
        return self._model

    def _obs_to_text(self, obs: AgentObservation) -> str:
        from deepresearch.agent.types import PageContent, SearchResult

        thought = obs.action.thought or ""
        if isinstance(obs.result, PageContent):
            result_text = obs.result.body[:200]
        elif isinstance(obs.result, list):
            result_text = " ".join(
                r.snippet[:80]
                for r in obs.result
                if hasattr(r, "snippet")
            )
        else:
            result_text = str(obs.result)[:200]
        return f"{obs.action.action_type} {thought} {result_text}"

    def search(self, query: str, observations: List[AgentObservation]) -> List[AgentObservation]:
        if len(observations) <= self.top_k:
            return observations

        texts = [self._obs_to_text(observation) for observation in observations]
        model = self._get_model()
        obs_vecs = np.asarray(model.encode(texts), dtype=np.float32)
        query_vec = np.asarray(model.encode([query]), dtype=np.float32)

        import faiss

        faiss.normalize_L2(obs_vecs)
        faiss.normalize_L2(query_vec)

        index = faiss.IndexFlatIP(obs_vecs.shape[1])
        index.add(obs_vecs)
        _, indices = index.search(query_vec, self.top_k)

        sorted_indices = sorted({int(idx) for idx in indices[0] if idx >= 0})
        return [observations[i] for i in sorted_indices]



