"""Simulation environment for DeepResearch experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from deepresearch.agent.types import PageContent, SearchResult
from deepresearch.envs.base_env import BaseEnv


class SimEnv(BaseEnv):
    """Local-corpus simulation environment backed by vector retrieval."""

    PAGE_SIZE = 1000
    DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    DEFAULT_TOP_K = 5

    def __init__(
        self,
        corpus_dir: str,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        top_k_default: int = 5,
    ) -> None:
        """Initialize the simulation environment and build a retrieval index."""
        self.corpus_dir = corpus_dir
        self.embedding_model = embedding_model
        self.top_k_default = top_k_default

        self._documents: List[Dict[str, str]] = []
        self._documents_by_url: Dict[str, Dict[str, str]] = {}
        self._encoder: Any = None
        self._index: Any = None

        self._build_index()

    def _build_index(self) -> None:
        """Load local documents, embed them, and build a FAISS IndexFlatIP."""
        documents = self._load_documents_from_corpus()
        self._initialize_from_documents(documents)

    def _initialize_from_documents(self, documents: List[Dict[str, str]]) -> None:
        """Initialize in-memory corpus structures and FAISS index from documents."""
        self._documents = documents
        self._documents_by_url = {document["url"]: document for document in documents}

        if not documents:
            self._encoder = None
            self._index = None
            return

        self._encoder = self._create_encoder()
        embeddings = self._encode_texts([document["body"] for document in documents])
        self._index = self._create_faiss_index(embeddings)

    def _load_documents_from_corpus(self) -> List[Dict[str, str]]:
        """Read .txt and .json documents from the configured corpus directory."""
        corpus_path = Path(self.corpus_dir)
        if not corpus_path.exists():
            raise FileNotFoundError(f"Corpus directory not found: {self.corpus_dir}")

        documents: List[Dict[str, str]] = []
        for file_path in sorted(corpus_path.rglob("*")):
            if not file_path.is_file():
                continue

            suffix = file_path.suffix.lower()
            if suffix == ".txt":
                relative_path = file_path.relative_to(corpus_path).as_posix()
                body = file_path.read_text(encoding="utf-8")
                documents.append(
                    {
                        "url": f"local://{relative_path}",
                        "title": file_path.stem,
                        "body": body,
                    }
                )
            elif suffix == ".json":
                documents.extend(self._load_documents_from_json(file_path))

        return documents

    @staticmethod
    def _normalize_document_payload(
        payload: Dict[str, Any], fallback_url: str, fallback_title: str
    ) -> Dict[str, str]:
        """Normalize document payloads into the shared url/title/body shape."""
        return {
            "url": str(payload.get("url") or fallback_url),
            "title": str(payload.get("title") or fallback_title),
            "body": str(payload.get("body") or ""),
        }

    def _load_documents_from_json(self, path: Path) -> List[Dict[str, str]]:
        """Load one .json file that contains either a dict or a list of dicts."""
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return [
                self._normalize_document_payload(
                    payload=data,
                    fallback_url=f"local://{path.name}",
                    fallback_title=path.stem,
                )
            ]

        if not isinstance(data, list):
            raise ValueError(f"Unsupported JSON corpus format in {path}")

        documents: List[Dict[str, str]] = []
        for index, item in enumerate(data, start=1):
            if not isinstance(item, dict):
                continue
            documents.append(
                self._normalize_document_payload(
                    payload=item,
                    fallback_url=f"local://{path.stem}/{index}",
                    fallback_title=f"{path.stem}-{index}",
                )
            )
        return documents

    def _create_encoder(self) -> Any:
        """Create the SentenceTransformer encoder used for retrieval embeddings."""
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(self.embedding_model)

    @staticmethod
    def _to_float32_matrix(values: Any) -> Any:
        """Convert embeddings into a 2D float32 matrix when numpy is available."""
        try:
            import numpy as np
        except ImportError:
            return values

        matrix = np.asarray(values, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = np.expand_dims(matrix, axis=0)
        return matrix

    def _encode_texts(self, texts: List[str]) -> Any:
        """Encode text list into embedding vectors using the configured encoder."""
        if self._encoder is None:
            raise RuntimeError("Encoder is not initialized.")

        embeddings = self._encoder.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return self._to_float32_matrix(embeddings)

    @staticmethod
    def _embedding_dimension(embeddings: Any) -> int:
        """Infer embedding dimensionality from a matrix-like embedding container."""
        shape = getattr(embeddings, "shape", None)
        if shape is not None and len(shape) == 2:
            return int(shape[1])

        if isinstance(embeddings, list) and embeddings and isinstance(embeddings[0], list):
            return len(embeddings[0])

        raise ValueError("Unable to infer embedding dimension from provided embeddings.")

    def _create_faiss_index(self, embeddings: Any) -> Any:
        """Build and populate a FAISS IndexFlatIP from document embeddings."""
        import faiss

        dimension = self._embedding_dimension(embeddings)
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        return index

    @staticmethod
    def _first_row(values: Any) -> List[int]:
        """Extract the first row of a 2D index-like container as integers."""
        try:
            row = values[0]
        except Exception:
            return []

        return [int(item) for item in row]

    async def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Search the corpus by query and return ranked retrieval results."""
        if self._index is None or self._encoder is None or not self._documents:
            return []

        requested_top_k = top_k if top_k > 0 else self.top_k_default
        requested_top_k = min(requested_top_k, len(self._documents))
        if requested_top_k <= 0:
            return []

        query_embedding = self._encode_texts([query])
        _, indices = self._index.search(query_embedding, requested_top_k)

        results: List[SearchResult] = []
        rank = 1
        for raw_index in self._first_row(indices):
            if raw_index < 0 or raw_index >= len(self._documents):
                continue

            document = self._documents[raw_index]
            results.append(
                SearchResult(
                    url=document["url"],
                    title=document["title"],
                    snippet=document["body"][:200],
                    rank=rank,
                )
            )
            rank += 1

        return results

    async def fetch_page(self, url: str) -> PageContent:
        """Fetch a document by URL and return the first page-sized slice."""
        document = self._documents_by_url.get(url)
        if document is None:
            raise ValueError(f"Document not found for URL: {url}")

        body = document["body"]
        total_pages = max(1, (len(body) + self.PAGE_SIZE - 1) // self.PAGE_SIZE)
        is_paginated = len(body) > self.PAGE_SIZE

        return PageContent(
            url=document["url"],
            title=document["title"],
            body=body[: self.PAGE_SIZE],
            links=[],
            is_paginated=is_paginated,
            current_page=1,
            total_pages=total_pages,
        )

    async def click_link(self, page: PageContent, link_url: str) -> PageContent:
        """Navigate to another document URL from the current page context."""
        del page
        return await self.fetch_page(link_url)

    async def get_next_page(self, page: PageContent) -> Optional[PageContent]:
        """Return the next page slice for paginated content when available."""
        if not page.is_paginated or page.total_pages is None:
            return None

        if page.current_page >= page.total_pages:
            return None

        document = self._documents_by_url.get(page.url)
        if document is None:
            raise ValueError(f"Document not found for URL: {page.url}")

        next_page_number = page.current_page + 1
        start = (next_page_number - 1) * self.PAGE_SIZE
        end = next_page_number * self.PAGE_SIZE

        return PageContent(
            url=document["url"],
            title=document["title"],
            body=document["body"][start:end],
            links=page.links,
            is_paginated=True,
            current_page=next_page_number,
            total_pages=page.total_pages,
        )

    @classmethod
    def load_from_jsonl(cls, path: str) -> "SimEnv":
        """Load documents from a JSONL file and return an initialized SimEnv."""
        jsonl_path = Path(path)
        if not jsonl_path.exists():
            raise FileNotFoundError(f"JSONL file not found: {path}")

        documents: List[Dict[str, str]] = []
        with jsonl_path.open("r", encoding="utf-8") as file:
            for line_number, raw_line in enumerate(file, start=1):
                line = raw_line.strip()
                if not line:
                    continue

                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError(f"Invalid JSONL payload at line {line_number}: expected object.")

                documents.append(
                    cls._normalize_document_payload(
                        payload=payload,
                        fallback_url=f"jsonl://{jsonl_path.stem}/{line_number}",
                        fallback_title=f"{jsonl_path.stem}-{line_number}",
                    )
                )

        env = cls.__new__(cls)
        env.corpus_dir = str(jsonl_path.parent)
        env.embedding_model = cls.DEFAULT_EMBEDDING_MODEL
        env.top_k_default = cls.DEFAULT_TOP_K
        env._documents = []
        env._documents_by_url = {}
        env._encoder = None
        env._index = None
        env._initialize_from_documents(documents)
        return env
