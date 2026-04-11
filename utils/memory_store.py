"""Persistent long-term memory store for cross-session knowledge reuse.

Stores completed research summaries as JSONL entries with pre-computed
embeddings, enabling semantic retrieval of prior research when starting
new tasks.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class MemoryEntry:
    """One record in the long-term memory store."""

    __slots__ = ("task_id", "query", "summary", "citations", "key_facts", "timestamp", "embedding")

    def __init__(
        self,
        task_id: str,
        query: str,
        summary: str,
        citations: Optional[List[str]] = None,
        key_facts: Optional[List[str]] = None,
        timestamp: Optional[str] = None,
        embedding: Optional[List[float]] = None,
    ) -> None:
        self.task_id = task_id
        self.query = query
        self.summary = summary
        self.citations = citations or []
        self.key_facts = key_facts or []
        self.timestamp = timestamp or datetime.now(timezone.utc).isoformat()
        self.embedding = embedding

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "query": self.query,
            "summary": self.summary,
            "citations": self.citations,
            "key_facts": self.key_facts,
            "timestamp": self.timestamp,
            "embedding": self.embedding,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEntry":
        return cls(
            task_id=str(data.get("task_id", "")),
            query=str(data.get("query", "")),
            summary=str(data.get("summary", "")),
            citations=data.get("citations") or [],
            key_facts=data.get("key_facts") or [],
            timestamp=data.get("timestamp"),
            embedding=data.get("embedding"),
        )

    def format_context(self, max_summary_len: int = 300) -> str:
        """Format this entry as a context block for LLM prompts."""
        summary_text = self.summary[:max_summary_len]
        if len(self.summary) > max_summary_len:
            summary_text += "..."

        parts = [f"[Prior Research] {self.query}", f"  Summary: {summary_text}"]
        if self.key_facts:
            facts_text = "; ".join(self.key_facts[:5])
            parts.append(f"  Key facts: {facts_text}")
        if self.citations:
            parts.append(f"  Sources: {', '.join(self.citations[:3])}")
        return "\n".join(parts)


class MemoryStore:
    """JSONL-backed long-term memory with FAISS semantic retrieval."""

    def __init__(self, store_path: str, top_k: int = 3) -> None:
        self.store_path = store_path
        self.top_k = top_k
        self._entries: List[MemoryEntry] = []
        self._model: Any = None
        self._load()

    def _get_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
        return self._model

    def _load(self) -> None:
        """Load existing entries from JSONL file."""
        path = Path(self.store_path)
        if not path.exists():
            self._entries = []
            return

        entries: List[MemoryEntry] = []
        with path.open("r", encoding="utf-8") as f:
            for line_num, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    entries.append(MemoryEntry.from_dict(data))
                except (json.JSONDecodeError, ValueError) as exc:
                    logger.warning("memory_store: skip invalid line %d: %s", line_num, exc)
        self._entries = entries
        logger.info("memory_store: loaded %d entries from %s", len(entries), self.store_path)

    def _save_entry(self, entry: MemoryEntry) -> None:
        """Append one entry to the JSONL file."""
        path = Path(self.store_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")

    def _encode_text(self, text: str) -> List[float]:
        """Encode a single text string into an embedding vector."""
        model = self._get_model()
        vec = model.encode([text], normalize_embeddings=True)
        return np.asarray(vec, dtype=np.float32)[0].tolist()

    def add(
        self,
        task_id: str,
        query: str,
        summary: str,
        citations: Optional[List[str]] = None,
        key_facts: Optional[List[str]] = None,
    ) -> MemoryEntry:
        """Add a completed research result to long-term memory."""
        # Deduplicate: skip if same task_id already exists
        for existing in self._entries:
            if existing.task_id == task_id:
                logger.info("memory_store: task_id=%s already exists, skipping", task_id)
                return existing

        embedding = self._encode_text(f"{query} {summary[:200]}")
        entry = MemoryEntry(
            task_id=task_id,
            query=query,
            summary=summary,
            citations=citations,
            key_facts=key_facts,
            embedding=embedding,
        )
        self._entries.append(entry)
        self._save_entry(entry)
        logger.info("memory_store: added task_id=%s, total=%d", task_id, len(self._entries))
        return entry

    def search(self, query: str, top_k: Optional[int] = None) -> List[MemoryEntry]:
        """Retrieve the most relevant prior research entries for a query."""
        k = top_k if top_k is not None else self.top_k
        if not self._entries or k <= 0:
            return []

        # Collect entries that have embeddings
        valid_entries: List[MemoryEntry] = []
        valid_embeddings: List[List[float]] = []
        for entry in self._entries:
            if entry.embedding is not None:
                valid_entries.append(entry)
                valid_embeddings.append(entry.embedding)

        if not valid_entries:
            return []

        if len(valid_entries) <= k:
            return list(valid_entries)

        import faiss

        model = self._get_model()
        query_vec = np.asarray(model.encode([query], normalize_embeddings=True), dtype=np.float32)
        entry_vecs = np.asarray(valid_embeddings, dtype=np.float32)

        # Vectors are already normalized at encoding time, use inner product
        index = faiss.IndexFlatIP(entry_vecs.shape[1])
        index.add(entry_vecs)
        scores, indices = index.search(query_vec, k)

        results: List[MemoryEntry] = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0 and scores[0][i] > 0.3:  # similarity threshold
                results.append(valid_entries[int(idx)])
        return results

    def __len__(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        """Remove all entries and delete the store file."""
        self._entries = []
        path = Path(self.store_path)
        if path.exists():
            path.unlink()
