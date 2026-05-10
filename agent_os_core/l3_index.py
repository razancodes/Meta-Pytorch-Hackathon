"""
AgentOS-Kernel — L3 Persistent Index with Gated Retrieval.

Implements the third tier of the Cognitive Cache Hierarchy:

  L3 (Persistent Index)
    ─ LanceDB embedded vector storage
    ─ BAAI/bge-base-en-v1.5 for semantic embeddings (384-dim)
    ─ BAAI/bge-reranker-v2-m3 for cross-encoder relevance gating
    ─ Results injected at END of context for recency bias

Usage:
    l3 = L3Index(use_mock=False)   # Real models
    l3 = L3Index(use_mock=True)    # Deterministic mocks for testing

    l3.archive_facts("CUST-A", overflowing_facts)
    retrieved = l3.retrieve_gated("PEP Viktor Korev risk")
    messages = L3Index.inject_into_prompt(messages, retrieved)
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import lancedb
import numpy as np
import pyarrow as pa

log = logging.getLogger("agentos.l3")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Constants
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EMBEDDING_DIM = 768  # bge-base-en-v1.5 output dimension
MOCK_EMBEDDING_DIM = 384
DEFAULT_RELEVANCE_THRESHOLD = 0.50
DEFAULT_TOP_K = 5
TABLE_NAME = "memory_facts"

BGE_EMBED_MODEL = "BAAI/bge-base-en-v1.5"
BGE_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Types
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class RetrievedFact:
    """A fact retrieved from L3 with relevance scoring."""

    entity_id: str
    fact_text: str
    vector_distance: float       # Raw L2 distance from LanceDB
    reranker_score: float        # Cross-encoder relevance score (0-1)
    passed_gate: bool            # Whether it passed the relevance threshold
    archived_at: float = 0.0     # Timestamp


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Mock Embedding & Reranking (testing only, no GPU required)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def mock_embed(text: str, dim: int = MOCK_EMBEDDING_DIM) -> List[float]:
    """Deterministic embedding via word-level hashing. Testing only."""
    vec = np.zeros(dim, dtype=np.float32)
    words = text.lower().replace(",", " ").replace(".", " ").replace(":", " ").split()
    words = [w.strip("\"'()[]{}") for w in words if len(w) > 1]
    for word in words:
        h = int(hashlib.md5(word.encode()).hexdigest(), 16)
        for offset in range(3):
            idx = (h + offset * 7919) % dim
            weight = 1.0 + 0.5 * (offset == 0)
            vec[idx] += weight
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()


def mock_cross_encoder_rerank(
    query: str, candidates: List[Tuple[str, str, float]]
) -> List[Tuple[str, str, float, float]]:
    """Mock cross-encoder reranking via word overlap. Testing only."""
    query_lower = query.lower()
    query_words = set(
        w.strip("\"'()[]{}:,.") for w in query_lower.split() if len(w) > 1
    )
    results = []
    for entity_id, fact_text, distance in candidates:
        fact_lower = fact_text.lower()
        fact_words = set(
            w.strip("\"'()[]{}:,.") for w in fact_lower.split() if len(w) > 1
        )
        intersection = query_words & fact_words
        dice = 2.0 * len(intersection) / max(len(query_words) + len(fact_words), 1)
        entity_clean = entity_id.lower().replace("_", " ").replace("-", " ")
        entity_words = set(w for w in entity_clean.split() if len(w) > 1)
        entity_match = len(query_words & entity_words) / max(len(entity_words), 1)
        substring_score = 0.0
        for qw in query_words:
            if len(qw) >= 3 and qw in fact_lower:
                substring_score += 0.15
            if len(qw) >= 3 and qw in entity_clean:
                substring_score += 0.2
        substring_score = min(substring_score, 1.0)
        dist_score = max(0.0, 1.0 - distance / 2.0)
        reranker_score = (
            0.25 * dist_score + 0.25 * dice
            + 0.20 * entity_match + 0.30 * substring_score
        )
        reranker_score = min(1.0, max(0.0, reranker_score))
        results.append((entity_id, fact_text, distance, reranker_score))
    results.sort(key=lambda x: x[3], reverse=True)
    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Real Embedding & Reranking (GPU-accelerated)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class EmbedderEngine:
    """Lazy-loaded sentence-transformers embedder using BAAI/bge-base-en-v1.5."""

    _instance: Optional["EmbedderEngine"] = None
    _model = None

    @classmethod
    def get(cls) -> "EmbedderEngine":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        if EmbedderEngine._model is None:
            log.info("Loading embedding model: %s", BGE_EMBED_MODEL)
            from sentence_transformers import SentenceTransformer
            EmbedderEngine._model = SentenceTransformer(BGE_EMBED_MODEL)
            log.info("Embedding model loaded (dim=%d)", EmbedderEngine._model.get_sentence_embedding_dimension())

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts into dense vectors. Returns (N, 768) float32 array."""
        # BGE instruction prefix for retrieval
        prefixed = [f"Represent this sentence for searching relevant passages: {t}" for t in texts]
        return EmbedderEngine._model.encode(
            prefixed,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

    def encode_single(self, text: str) -> List[float]:
        """Encode a single text. Returns list of floats."""
        return self.encode([text])[0].tolist()


class RerankerEngine:
    """Lazy-loaded cross-encoder reranker using BAAI/bge-reranker-v2-m3."""

    _instance: Optional["RerankerEngine"] = None
    _model = None

    @classmethod
    def get(cls) -> "RerankerEngine":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        if RerankerEngine._model is None:
            log.info("Loading reranker model: %s", BGE_RERANKER_MODEL)
            from sentence_transformers import CrossEncoder
            RerankerEngine._model = CrossEncoder(BGE_RERANKER_MODEL)
            log.info("Reranker model loaded")

    def rerank(
        self, query: str, candidates: List[Tuple[str, str, float]]
    ) -> List[Tuple[str, str, float, float]]:
        """Rerank candidates using the cross-encoder.

        Args:
            query: User query text.
            candidates: List of (entity_id, fact_text, vector_distance).

        Returns:
            List of (entity_id, fact_text, distance, relevance_score), sorted descending.
        """
        if not candidates:
            return []

        # Build pairs for the cross-encoder
        pairs = [(query, fact_text) for _, fact_text, _ in candidates]

        # Get raw logits from the cross-encoder
        raw_scores = RerankerEngine._model.predict(pairs)

        # Sigmoid normalization to [0, 1]
        def sigmoid(x: float) -> float:
            return 1.0 / (1.0 + np.exp(-x))

        results = []
        for (entity_id, fact_text, distance), score in zip(candidates, raw_scores):
            norm_score = float(sigmoid(float(score)))
            results.append((entity_id, fact_text, distance, norm_score))

        results.sort(key=lambda x: x[3], reverse=True)
        return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# L3 Index
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class L3Index:
    """LanceDB-backed persistent memory index with gated retrieval.

    Args:
        db_path: Path to the LanceDB database directory.
        relevance_threshold: Minimum reranker score to pass the gate.
        top_k: Number of candidates to retrieve from vector search.
        use_mock: If True, use deterministic mock embeddings/reranker.
    """

    def __init__(
        self,
        db_path: str = "/tmp/agent_os_l3",
        relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
        top_k: int = DEFAULT_TOP_K,
        use_mock: bool = False,
    ) -> None:
        self._db_path = db_path
        self._relevance_threshold = relevance_threshold
        self._top_k = top_k
        self._use_mock = use_mock
        self._embedding_dim = MOCK_EMBEDDING_DIM if use_mock else EMBEDDING_DIM

        # Connect to embedded LanceDB
        self._db = lancedb.connect(db_path)
        self._table: Optional[lancedb.table.Table] = None
        self._total_archived = 0

        # Try to open existing table
        if TABLE_NAME in self._db.table_names():
            self._table = self._db.open_table(TABLE_NAME)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Internal: model dispatch
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _embed(self, text: str) -> List[float]:
        """Embed a single text using the configured backend."""
        if self._use_mock:
            return mock_embed(text, self._embedding_dim)
        return EmbedderEngine.get().encode_single(text)

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts."""
        if self._use_mock:
            return [mock_embed(t, self._embedding_dim) for t in texts]
        vecs = EmbedderEngine.get().encode(texts)
        return [v.tolist() for v in vecs]

    def _rerank(
        self, query: str, candidates: List[Tuple[str, str, float]]
    ) -> List[Tuple[str, str, float, float]]:
        """Rerank candidates using the configured backend."""
        if self._use_mock:
            return mock_cross_encoder_rerank(query, candidates)
        return RerankerEngine.get().rerank(query, candidates)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Archival
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def archive_facts(self, entity_id: str, facts: List[str]) -> int:
        """Archive facts to the L3 persistent index.

        Generates embeddings for each fact and inserts into LanceDB.
        Uses batch encoding for efficiency with real models.
        """
        if not facts:
            return 0

        now = time.time()

        # Build embed texts with entity context
        embed_texts = [f"{entity_id} {fact}" for fact in facts]
        vectors = self._embed_batch(embed_texts)

        records = []
        for fact, vector in zip(facts, vectors):
            records.append({
                "id": str(uuid.uuid4()),
                "entity_id": entity_id,
                "fact_text": fact,
                "vector": vector,
                "archived_at": now,
            })

        if self._table is None:
            self._table = self._db.create_table(TABLE_NAME, data=records)
        else:
            self._table.add(records)

        self._total_archived += len(records)
        return len(records)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Gated Retrieval
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def retrieve_relevant_context(
        self,
        query: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> List[RetrievedFact]:
        """Retrieve facts relevant to a query via gated vector search.

        Step A: Embed the query via BGE-base-en-v1.5.
        Step B: Vector search in LanceDB (top-K by L2 distance).
        Step C: Cross-encoder reranking via BGE-reranker-v2-m3 + threshold gate.
        """
        if self._table is None or self._total_archived == 0:
            return []

        k = top_k or self._top_k
        gate_threshold = threshold or self._relevance_threshold

        # Step A: Embed the query
        query_vector = self._embed(query)

        # Step B: Vector search in LanceDB
        raw_results = (
            self._table
            .search(query_vector)
            .limit(k)
            .to_list()
        )

        if not raw_results:
            return []

        # Build candidates for reranking
        candidates: List[Tuple[str, str, float]] = []
        archived_times: Dict[str, float] = {}

        for row in raw_results:
            entity_id = row["entity_id"]
            fact_text = row["fact_text"]
            distance = row.get("_distance", 0.0)
            archived_times[f"{entity_id}:{fact_text}"] = row.get("archived_at", 0.0)
            candidates.append((entity_id, fact_text, distance))

        # Step C: Cross-encoder reranking + gate
        reranked = self._rerank(query, candidates)

        results: List[RetrievedFact] = []
        for entity_id, fact_text, distance, score in reranked:
            passed = score >= gate_threshold
            key = f"{entity_id}:{fact_text}"
            results.append(RetrievedFact(
                entity_id=entity_id,
                fact_text=fact_text,
                vector_distance=distance,
                reranker_score=score,
                passed_gate=passed,
                archived_at=archived_times.get(key, 0.0),
            ))

        return results

    def retrieve_gated(
        self,
        query: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> List[RetrievedFact]:
        """Convenience: retrieve ONLY facts that pass the gate."""
        all_results = self.retrieve_relevant_context(query, top_k, threshold)
        return [r for r in all_results if r.passed_gate]

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Prompt Injection
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @staticmethod
    def inject_into_prompt(
        messages: List[Dict[str, str]],
        retrieved_facts: List[RetrievedFact],
    ) -> List[Dict[str, str]]:
        """Inject gated L3 results just BEFORE the last user message.

        End-of-context positioning maximizes attention from the reasoning model.
        """
        if not retrieved_facts:
            return messages

        lines = ["[RETRIEVED HISTORICAL CONTEXT — from long-term memory]"]
        seen = set()

        for fact in retrieved_facts:
            key = f"{fact.entity_id}:{fact.fact_text}"
            if key in seen:
                continue
            seen.add(key)
            lines.append(
                f"  [{fact.entity_id}] {fact.fact_text} "
                f"(relevance: {fact.reranker_score:.2f})"
            )

        lines.append("[END RETRIEVED CONTEXT]")
        context_block = "\n".join(lines)

        result = list(messages)
        last_user_idx = None
        for i in range(len(result) - 1, -1, -1):
            if result[i]["role"] == "user":
                last_user_idx = i
                break

        injection_msg = {"role": "system", "content": context_block}

        if last_user_idx is not None and last_user_idx > 0:
            result.insert(last_user_idx, injection_msg)
        else:
            result.insert(max(len(result) - 1, 1), injection_msg)

        return result

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Diagnostics
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @property
    def total_archived(self) -> int:
        return self._total_archived

    def get_stats(self) -> Dict[str, Any]:
        row_count = self._table.count_rows() if self._table else 0
        return {
            "db_path": self._db_path,
            "total_archived": self._total_archived,
            "rows_in_table": row_count,
            "embedding_dim": self._embedding_dim,
            "relevance_threshold": self._relevance_threshold,
            "top_k": self._top_k,
            "use_mock": self._use_mock,
            "embed_model": "mock" if self._use_mock else BGE_EMBED_MODEL,
            "reranker_model": "mock" if self._use_mock else BGE_RERANKER_MODEL,
        }

    def drop(self) -> None:
        """Drop the table and delete the database directory. For testing."""
        if self._table is not None:
            self._db.drop_table(TABLE_NAME)
            self._table = None
            self._total_archived = 0
        if os.path.exists(self._db_path):
            shutil.rmtree(self._db_path, ignore_errors=True)

    def __repr__(self) -> str:
        stats = self.get_stats()
        mode = "mock" if self._use_mock else "bge"
        return (
            f"L3Index(rows={stats['rows_in_table']}, "
            f"dim={stats['embedding_dim']}, "
            f"threshold={stats['relevance_threshold']}, "
            f"mode={mode})"
        )
