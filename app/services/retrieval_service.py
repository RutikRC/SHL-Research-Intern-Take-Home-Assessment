"""
Service layer for vector-based retrieval with relevance filtering.

Embeds the user query, performs cosine similarity search via pgvector,
and returns only the most relevant Assessment objects from the catalog.
Optimised for Recall@10 with deterministic ordering.
"""

from __future__ import annotations

import re
import time
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging_ import get_logger
from app.database.models import Assessment
from app.repositories.embedding_repository import EmbeddingRepository
from app.services.embedding_client import EmbeddingClient

logger = get_logger(__name__)

# Number of extra results to fetch for re-ranking buffer
SEARCH_OVERFETCH_FACTOR = 3

# Language keywords that need whole-word matching to avoid false positives
LANGUAGE_KEYWORDS = frozenset({
    "java", "python", "ruby", "go", "c#", "c++", "c", "rust",
    "kotlin", "swift", "php", "typescript", "scala", "perl", "r",
})


class RetrievalService:
    """Business logic for semantic / vector search over the catalog."""

    def __init__(
        self,
        repository: EmbeddingRepository | None = None,
        embedding_client: EmbeddingClient | None = None,
    ) -> None:
        self._repository = repository or EmbeddingRepository()
        self._embedding_client = embedding_client or EmbeddingClient()

    async def retrieve(
        self,
        session: AsyncSession,
        query_text: str,
        top_k: int = 10,
    ) -> list[Assessment]:
        """Embed the query and return the most similar catalog items.

        Steps:
            1. Generate an embedding vector for the query text.
            2. Perform pgvector cosine similarity search (overfetched by 3x).
            3. Score each result by keyword overlap with the query.
            4. Deduplicate and return the top-K highest-scored results.

        Args:
            session: Database session.
            query_text: The user's natural-language query.
            top_k: Number of results to return (default 10).

        Returns:
            A list of Assessment objects ordered by relevance (closest first).
            Empty list if the query is empty or no results are found.

        Raises:
            RuntimeError: If the embedding model call fails.
        """
        if not query_text or not query_text.strip():
            logger.warning("retrieval_skipped_empty_query")
            return []

        start = time.monotonic()
        logger.info("retrieval_started", query_chars=len(query_text), top_k=top_k)

        try:
            query_vector = await self._embedding_client.embed(query_text)
            embed_time = time.monotonic() - start
            logger.debug(
                "retrieval_embedding_generated",
                dimension=len(query_vector),
                elapsed_seconds=round(embed_time, 4),
            )

            overfetch_k = top_k * SEARCH_OVERFETCH_FACTOR
            results = await self._repository.search_similar(
                session=session,
                query_embedding=query_vector,
                top_k=overfetch_k,
            )
            search_time = time.monotonic() - start

            # Re-rank by combined keyword + category score
            query_keywords = self._extract_keywords(query_text)
            query_categories = self._extract_categories(query_text)
            scored = [
                (assessment, self._combined_score(assessment, query_keywords, query_categories))
                for assessment in results
            ]
            # Sort by score descending, then by original position as tiebreaker
            scored.sort(key=lambda x: (-x[1], results.index(x[0])))

            # Deduplicate by entity_id (keep highest-scored variant)
            seen_entity_ids: set[str] = set()
            filtered: list[Assessment] = []
            for assessment, _ in scored:
                if assessment.entity_id not in seen_entity_ids:
                    seen_entity_ids.add(assessment.entity_id)
                    filtered.append(assessment)
                    if len(filtered) >= top_k:
                        break

            # Fallback: if no keyword matches, use raw vector results
            if not filtered and results:
                for a in results:
                    if a.entity_id not in seen_entity_ids:
                        seen_entity_ids.add(a.entity_id)
                        filtered.append(a)
                        if len(filtered) >= top_k:
                            break

            logger.info(
                "retrieval_completed",
                query_chars=len(query_text),
                vector_results=len(results),
                filtered_results=len(filtered),
                top_k=top_k,
                elapsed_seconds=round(search_time, 4),
            )

            return filtered

        except (RuntimeError, ValueError) as exc:
            logger.error(
                "retrieval_failed",
                query_chars=len(query_text),
                error=str(exc),
            )
            raise RuntimeError(f"Retrieval failed: {exc}") from exc

    # ── Scoring ─────────────────────────────────────────────────────────────

    def _combined_score(
        self,
        assessment: Assessment,
        query_keywords: set[str],
        query_categories: set[str],
    ) -> int:
        """Compute a combined relevance score for an assessment.

        Combines keyword overlap score with category match bonus.
        """
        score = self._keyword_score(assessment, query_keywords)

        # Category match bonus: +2 per matching category
        if query_categories and assessment.keys:
            for cat in query_categories:
                for key in assessment.keys:
                    if cat in key.lower():
                        score += 2

        return score

    @staticmethod
    def _extract_keywords(text: str) -> set[str]:
        """Extract meaningful keywords from a query string.

        Strips common stop words and returns a set of lowercase tokens.
        """
        stop_words = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "shall", "can",
            "need", "dare", "ought", "used", "to", "of", "in", "for",
            "on", "with", "at", "by", "from", "as", "into", "through",
            "during", "before", "after", "above", "below", "between",
            "out", "off", "over", "under", "again", "further", "then",
            "once", "here", "there", "when", "where", "why", "how",
            "all", "each", "every", "both", "few", "more", "most",
            "other", "some", "such", "no", "nor", "not", "only", "own",
            "same", "so", "than", "too", "very", "just", "because",
            "and", "but", "or", "if", "while", "that", "this", "these",
            "those", "i", "me", "my", "myself", "we", "our", "ours",
            "ourselves", "you", "your", "yours", "yourself", "yourselves",
            "he", "him", "his", "himself", "she", "her", "hers", "herself",
            "it", "its", "itself", "they", "them", "their", "theirs",
            "themselves", "what", "which", "who", "whom", "hiring", "looking",
            "need", "want", "find", "search", "get", "assessments", "assessment",
            "test", "tests", "role", "position", "job", "candidate", "candidates",
            "for", "new", "please", "help", "recommend", "recommendations",
        }
        tokens = re.findall(r"[a-zA-Z0-9#+.]+(?:[/-][a-zA-Z0-9#+]+)*", text.lower())
        return {t for t in tokens if t not in stop_words and len(t) > 1}

    @staticmethod
    def _extract_categories(text: str) -> set[str]:
        """Extract assessment category hints from the query."""
        category_map = {
            "personality": "personality",
            "behavior": "personality",
            "behaviour": "personality",
            "cognitive": "cognitive",
            "ability": "cognitive",
            "aptitude": "cognitive",
            "technical": "technical",
            "programming": "technical",
            "coding": "technical",
            "knowledge": "knowledge",
            "skill": "knowledge",
            "simulation": "simulation",
            "competency": "competency",
            "situational": "situational",
            "development": "development",
        }
        text_lower = text.lower()
        found: set[str] = set()
        for word, category in category_map.items():
            if word in text_lower:
                found.add(category)
        return found

    @staticmethod
    def _keyword_score(assessment: Assessment, query_keywords: set[str]) -> int:
        """Score an assessment by how many query keywords appear in its metadata.

        Checks: name, description, job_levels, keys (categories).
        """
        if not query_keywords:
            return 1

        score = 0

        # Check assessment name (weight: 3 per match)
        name_lower = assessment.name.lower()
        for kw in query_keywords:
            if kw in name_lower:
                if kw in LANGUAGE_KEYWORDS:
                    if re.search(rf'\b{re.escape(kw)}\b', name_lower):
                        score += 3
                else:
                    score += 3

        # Check description (weight: 2 per match)
        desc_lower = assessment.description.lower()
        for kw in query_keywords:
            if kw in desc_lower:
                score += 2

        # Check job levels (weight: 1 per match)
        for level in assessment.job_levels:
            level_lower = level.lower()
            for kw in query_keywords:
                if kw in level_lower:
                    score += 1

        # Check keys/categories (weight: 1 per match)
        for key in assessment.keys:
            key_lower = key.lower()
            for kw in query_keywords:
                if kw in key_lower:
                    score += 1

        return score