"""Chunk-level retrievers for long page content and trajectory observations.

Implements a two-stage retrieval pipeline:
  Stage 1 (Recall): FAISS vector search + BM25 keyword search (multi-route)
  Stage 2 (Rerank): Cross-Encoder fine-grained scoring
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from deepresearch.agent.types import AgentObservation


# ═══════════════════════════════════════════════════════════════════════
# BM25 lightweight implementation (no external dependency)
# ═══════════════════════════════════════════════════════════════════════

class _BM25:
    """Minimal BM25 scorer for a small set of documents (in-memory, no index)."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b

    def score(self, query: str, documents: List[str]) -> List[float]:
        """Score each document against the query. Returns list of BM25 scores."""
        query_tokens = self._tokenize(query)
        doc_token_lists = [self._tokenize(d) for d in documents]
        doc_lens = [len(t) for t in doc_token_lists]
        avg_dl = sum(doc_lens) / max(len(doc_lens), 1)
        n_docs = len(documents)

        # Document frequency for each query term
        df: Dict[str, int] = {}
        for qt in query_tokens:
            df[qt] = sum(1 for dt in doc_token_lists if qt in dt)

        scores = []
        for i, doc_tokens in enumerate(doc_token_lists):
            tf_map = Counter(doc_tokens)
            s = 0.0
            for qt in query_tokens:
                if qt not in tf_map:
                    continue
                tf = tf_map[qt]
                idf = math.log((n_docs - df[qt] + 0.5) / (df[qt] + 0.5) + 1.0)
                tf_norm = (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * doc_lens[i] / avg_dl))
                s += idf * tf_norm
            scores.append(s)
        return scores

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[a-z0-9]+", text.lower())


# ═══════════════════════════════════════════════════════════════════════
# PageChunkRetriever: two-stage (recall + rerank)
# ═══════════════════════════════════════════════════════════════════════

class PageChunkRetriever:
    """Two-stage chunk retriever: multi-route recall → Cross-Encoder rerank.

    Stage 1 (Recall): FAISS semantic top-k ∪ BM25 keyword top-k ∪ entity match
    Stage 2 (Rerank): Cross-Encoder scores each (query, chunk) pair, keep top-k
    """

    def __init__(
        self,
        chunk_size: int = 300,
        top_k: int = 8,
        recall_k: int = 20,
        min_body_len: int = 1000,
        use_rerank: bool = True,
    ) -> None:
        self.chunk_size = chunk_size
        self.top_k = top_k          # final output size after rerank
        self.recall_k = recall_k    # candidates from stage 1
        self.min_body_len = min_body_len
        self.use_rerank = use_rerank
        self._bi_encoder = None
        self._cross_encoder = None
        self._bm25 = _BM25()

    def _get_bi_encoder(self):
        if self._bi_encoder is None:
            from sentence_transformers import SentenceTransformer
            self._bi_encoder = SentenceTransformer("all-MiniLM-L6-v2")
        return self._bi_encoder

    def _get_cross_encoder(self):
        if self._cross_encoder is None:
            from sentence_transformers import CrossEncoder
            self._cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        return self._cross_encoder

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
        """Two-stage filtering: multi-route recall → Cross-Encoder rerank."""
        if len(body) <= self.min_body_len:
            return body

        chunks = self._split(body)
        if len(chunks) <= self.top_k or self.top_k <= 0:
            return body

        # ── Stage 1: Multi-route recall ──
        candidates = self._recall(chunks, query)

        # ── Stage 2: Cross-Encoder rerank ──
        if self.use_rerank and len(candidates) > self.top_k:
            candidates = self._rerank(chunks, candidates, query)

        sorted_indices = sorted(candidates[:self.top_k])
        if not sorted_indices:
            return body

        return "\n\n".join(chunks[i] for i in sorted_indices)

    def _recall(self, chunks: List[str], query: str) -> List[int]:
        """Stage 1: Merge candidates from FAISS vector search + BM25 + entity match."""
        candidate_scores: Dict[int, float] = {}

        # Route 1: FAISS semantic search
        model = self._get_bi_encoder()
        chunk_vecs = np.asarray(model.encode(chunks), dtype=np.float32)
        query_vec = np.asarray(model.encode([query]), dtype=np.float32)

        import faiss
        faiss.normalize_L2(chunk_vecs)
        faiss.normalize_L2(query_vec)

        index = faiss.IndexFlatIP(chunk_vecs.shape[1])
        index.add(chunk_vecs)
        scores, indices = index.search(query_vec, min(self.recall_k, len(chunks)))

        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                candidate_scores[int(idx)] = float(score)

        # Route 2: BM25 keyword search
        bm25_scores = self._bm25.score(query, chunks)
        bm25_ranked = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)
        for idx in bm25_ranked[:self.recall_k]:
            if bm25_scores[idx] > 0:
                if idx not in candidate_scores:
                    candidate_scores[idx] = 0.0  # BM25-only candidate
                candidate_scores[idx] += 0.3  # bonus for BM25 match

        # Route 3: Entity keyword match (fallback)
        query_entities = self._extract_entities(query)
        if query_entities:
            for i, chunk in enumerate(chunks):
                if i in candidate_scores:
                    continue
                chunk_lower = chunk.lower()
                if any(ent in chunk_lower for ent in query_entities):
                    candidate_scores[i] = 0.1  # low score, just ensure inclusion

        # Return candidates sorted by recall score
        return sorted(candidate_scores.keys(), key=lambda i: candidate_scores[i], reverse=True)

    def _rerank(self, chunks: List[str], candidate_indices: List[int], query: str) -> List[int]:
        """Stage 2: Cross-Encoder rerank candidates."""
        cross_encoder = self._get_cross_encoder()

        pairs = [(query, chunks[idx]) for idx in candidate_indices]
        ce_scores = cross_encoder.predict(pairs)

        # Sort by Cross-Encoder score descending
        scored = sorted(zip(candidate_indices, ce_scores), key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in scored]

    @staticmethod
    def _extract_entities(query: str) -> List[str]:
        """Extract likely entity tokens from query (capitalized words, numbers)."""
        entities = []
        for token in re.findall(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*", query):
            entities.append(token.lower())
        for token in re.findall(r"\d[\d,.]+", query):
            entities.append(token)
        for token in re.findall(r"[a-zA-Z]{6,}", query):
            entities.append(token.lower())
        return list(set(entities))


# ═══════════════════════════════════════════════════════════════════════
# ObservationRetriever: hybrid temporal + semantic
# ═══════════════════════════════════════════════════════════════════════

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
        """Select observations using hybrid strategy: recent + semantic."""
        if len(observations) <= self.top_k:
            return observations

        recent_count = min(3, self.top_k, len(observations))
        semantic_count = self.top_k - recent_count

        recent_indices = set(range(len(observations) - recent_count, len(observations)))

        if semantic_count > 0:
            texts = [self._obs_to_text(observation) for observation in observations]
            model = self._get_model()
            obs_vecs = np.asarray(model.encode(texts), dtype=np.float32)
            query_vec = np.asarray(model.encode([query]), dtype=np.float32)

            import faiss
            faiss.normalize_L2(obs_vecs)
            faiss.normalize_L2(query_vec)

            index = faiss.IndexFlatIP(obs_vecs.shape[1])
            index.add(obs_vecs)
            _, indices = index.search(query_vec, self.top_k + recent_count)

            semantic_indices = set()
            for idx in indices[0]:
                idx = int(idx)
                if idx < 0:
                    continue
                if idx not in recent_indices:
                    semantic_indices.add(idx)
                    if len(semantic_indices) >= semantic_count:
                        break

            selected = recent_indices | semantic_indices
        else:
            selected = recent_indices

        sorted_indices = sorted(selected)
        return [observations[i] for i in sorted_indices]
