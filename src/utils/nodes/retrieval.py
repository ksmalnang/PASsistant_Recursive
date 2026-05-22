"""Retrieval node."""

import asyncio
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage

from src.config import get_settings
from src.services.contracts import DocumentRetriever
from src.utils.nodes.llm import get_llm
from src.utils.nodes.prompts import QUERY_REWRITE_PROMPT
from src.utils.state import AgentState, DocumentType
from src.utils.tools import VectorStoreTools

logger = logging.getLogger(__name__)


class RetrievalNode:
    """
    Retrieves relevant indexed document context from the retrieval stack.

    Delegates to the configured retrieval strategy (similarity, RRF, or reranker)
    and augments the state with hydrated parent/child retrieval context.
    """

    _STOPWORDS = {
        "aku",
        "apa",
        "yang",
        "akan",
        "jika",
        "kalau",
        "kak",
        "kami",
        "gue",
        "gua",
        "gw",
        "nih",
        "dong",
        "sih",
        "kok",
        "loh",
        "yah",
        "ya",
        "deh",
        "saya",
        "dan",
        "atau",
        "di",
        "ke",
        "dari",
        "untuk",
        "dengan",
        "selama",
        "tanpa",
        "mengajukan",
        "mengikuti",
        "ikut",
        "bagaimana",
        "adalah",
        "masih",
        "bisa",
        "nggak",
        "enggak",
        "ga",
        "gak",
        "tdk",
        "the",
        "is",
        "of",
        "to",
        "a",
        "an",
    }
    _POLICY_RULE_MARKERS = (
        "dapat dilakukan setelah",
        "tidak dapat",
        "belum bisa",
        "belum dapat",
        "wajib",
        "harus",
        "syarat",
        "ketentuan",
    )
    _POLICY_SCOPE_TERMS = (
        "ambil sks",
        "maksimal sks",
        "batas sks",
        "beban sks",
        "berapa sks",
        "dpp",
        "spp",
        "cicilan",
        "tagihan",
        "bayar",
        "pembayaran",
        "lunas",
        "perwalian",
        "krs",
        "syarat",
        "ketentuan",
        "aturan",
    )

    def __init__(
        self,
        retriever: DocumentRetriever | None = None,
        llm_provider=get_llm,
    ):
        self.vector_tools = retriever or VectorStoreTools()
        self._llm_provider = llm_provider
        self._llm = None
        self.top_k = get_settings().RETRIEVAL_TOP_K

    async def run(self, state: AgentState) -> dict:
        """
        Retrieve relevant chunks for the current query.

        Args:
            state: Current agent state with retrieval_query

        Returns:
            State updates with retrieved chunks
        """
        if not state.requires_retrieval or not state.retrieval_query:
            return {}

        try:
            doc_type = self._resolve_document_type(
                state.current_intent,
                state.retrieval_query,
            )
            queries = self._build_queries(state)
            if not queries:
                return {
                    "retrieved_chunks": [],
                    "requires_retrieval": False,
                    "response_confidence": 0.0,
                    "retrieval_warning": (
                        "The retriever did not receive a usable query. "
                        "Ask the user to restate the question."
                    ),
                }
            merged_results: list[dict[str, Any]] = []
            query_results_list = await asyncio.gather(
                *[
                    self.vector_tools.search_similar(
                        query=query,
                        document_type=doc_type,
                        top_k=self.top_k,
                        score_threshold=0.2,
                    )
                    for query in queries
                ]
            )
            for query, query_results in zip(queries, query_results_list, strict=True):
                merged_results = self._merge_results(merged_results, query_results, query)

            confidence, warning = self._score_retrieval_confidence(
                query=queries[0],
                results=merged_results,
            )
            logger.info(
                "Retrieved chunks",
                extra={
                    "query": state.retrieval_query,
                    "queries": queries,
                    "results": len(merged_results),
                    "confidence": confidence,
                },
            )
            return {
                "retrieved_chunks": merged_results[: self.top_k],
                "requires_retrieval": False,
                "response_confidence": confidence,
                "retrieval_warning": warning,
            }
        except Exception as exc:
            logger.error("Retrieval failed: %s", exc)
            return {"error": f"Document retrieval failed: {exc}"}

    def _resolve_document_type(
        self,
        intent: str | None,
        query: str | None = None,
    ) -> DocumentType | None:
        """Resolve an optional document type filter from the current intent."""
        if intent == "query_student":
            if self._looks_like_policy_scope(query or ""):
                return None
            return DocumentType.TRANSCRIPT
        return None

    def _looks_like_policy_scope(self, query: str) -> bool:
        """Detect mixed student-record wording that asks for policy/rule context."""
        lowered = query.lower()
        return any(term in lowered for term in self._POLICY_SCOPE_TERMS)

    def _build_queries(self, state: AgentState) -> list[str]:
        """Build original, rewritten, and synonym-expanded retrieval queries."""
        original = (state.retrieval_query or "").strip()
        if not original:
            return []
        if state.current_intent != "query_document":
            return [original]

        queries = [original]
        rewritten = self._rewrite_query(original)
        if rewritten and rewritten.lower() != original.lower():
            queries.append(rewritten)

        expanded = self._expand_query(rewritten or original)
        if expanded and all(expanded.lower() != query.lower() for query in queries):
            queries.append(expanded)
        return queries[:3]

    def _rewrite_query(self, question: str) -> str:
        """Compress a conversational question into a keyword-rich retrieval query."""
        llm = self._get_llm()
        if llm is not None:
            try:
                response = llm.invoke(
                    [HumanMessage(content=QUERY_REWRITE_PROMPT.format(question=question))]
                )
                rewritten = str(response.content).strip()
                if rewritten:
                    return rewritten.splitlines()[0].strip()
            except Exception as exc:
                logger.warning("Query rewrite failed, falling back to heuristics: %s", exc)
        return self._extract_keywords(question)

    def _expand_query(self, query: str) -> str:
        """Add a small synonym layer for Indonesian academic-policy retrieval."""
        expansions = {
            "absen": "ketidakhadiran",
            "tidak aktif": "nonaktif",
            "cuti": "cuti akademik",
            "masa studi": "akhir masa studi",
            "drop out": "dikeluarkan",
        }
        parts = [query.strip()]
        lowered = query.lower()
        for source, target in expansions.items():
            if source in lowered and target not in lowered:
                parts.append(target)
        # Normalize semester numerals (arabic ↔ roman) for lexical matching
        normalized = self._normalize_semester_numerals(" ".join(parts))
        if normalized and normalized.lower() != " ".join(parts).lower():
            return normalized
        return " ".join(part for part in parts if part).strip()

    _ARABIC_TO_ROMAN = {
        "1": "I",
        "2": "II",
        "3": "III",
        "4": "IV",
        "5": "V",
        "6": "VI",
        "7": "VII",
        "8": "VIII",
        "9": "IX",
        "10": "X",
    }
    _ROMAN_TO_ARABIC = {v: k for k, v in _ARABIC_TO_ROMAN.items()}
    _SEMESTER_ARABIC_PATTERN = re.compile(r"\b(semester)\s+(\d{1,2})\b", re.IGNORECASE)
    _SEMESTER_ROMAN_PATTERN = re.compile(r"\b(semester)\s+([IVXL]+)\b", re.IGNORECASE)

    def _normalize_semester_numerals(self, query: str) -> str:
        """Replace 'semester 5' with 'semester V' (and vice versa) for retrieval."""
        # Arabic → Roman (e.g. "semester 5" → "semester V")
        match = self._SEMESTER_ARABIC_PATTERN.search(query)
        if match:
            arabic = match.group(2)
            roman = self._ARABIC_TO_ROMAN.get(arabic)
            if roman:
                return self._SEMESTER_ARABIC_PATTERN.sub(rf"\1 {roman}", query)

        # Roman → Arabic (e.g. "semester V" → "semester 5")
        match = self._SEMESTER_ROMAN_PATTERN.search(query)
        if match:
            roman = match.group(2).upper()
            arabic = self._ROMAN_TO_ARABIC.get(roman)
            if arabic:
                return self._SEMESTER_ROMAN_PATTERN.sub(rf"\1 {arabic}", query)

        return query

    def _extract_keywords(self, text: str) -> str:
        """Heuristic fallback for query rewriting when no LLM is configured."""
        tokens = re.findall(r"\b[\w-]+\b", text.lower())
        kept: list[str] = []
        for token in tokens:
            if token in self._STOPWORDS or len(token) <= 2:
                continue
            if token not in kept:
                kept.append(token)
        return " ".join(kept[:12]) or text.strip()

    def _merge_results(
        self,
        current: list[dict[str, Any]],
        incoming: list[dict[str, Any]],
        query_variant: str,
    ) -> list[dict[str, Any]]:
        """Merge per-query retrieval results by hydrated parent id."""
        merged: dict[str, dict[str, Any]] = {
            self._result_key(result): self._tag_result(result, result.get("matched_query"))
            for result in current
        }
        for result in incoming:
            key = self._result_key(result)
            tagged = self._tag_result(result, query_variant)
            if key not in merged:
                merged[key] = tagged
                continue

            existing = merged[key]
            if float(tagged.get("score") or 0.0) > float(existing.get("score") or 0.0):
                existing.update(
                    {
                        "score": tagged.get("score"),
                        "final_score": tagged.get("final_score", tagged.get("score")),
                        "retrieval_score": tagged.get(
                            "retrieval_score", existing.get("retrieval_score")
                        ),
                        "reranker_score": tagged.get(
                            "reranker_score", existing.get("reranker_score")
                        ),
                        "vector_score": tagged.get("vector_score", existing.get("vector_score")),
                        "bm25_score": tagged.get("bm25_score", existing.get("bm25_score")),
                        "rrf_score": tagged.get("rrf_score", existing.get("rrf_score")),
                        "matched_query": query_variant,
                    }
                )
            for score_field in ("retrieval_score", "vector_score", "bm25_score", "rrf_score"):
                tagged_score = tagged.get(score_field)
                if tagged_score is None:
                    continue
                existing_score = existing.get(score_field)
                existing[score_field] = (
                    tagged_score
                    if existing_score is None
                    else max(float(existing_score), float(tagged_score))
                )
            existing_children = {
                str(child.get("chunk_id") or child.get("text") or "")
                for child in existing.get("matched_children", [])
                if isinstance(child, dict)
            }
            for child in tagged.get("matched_children", []):
                child_key = str(child.get("chunk_id") or child.get("text") or "")
                if child_key and child_key not in existing_children:
                    existing.setdefault("matched_children", []).append(child)
                    existing_children.add(child_key)

        return sorted(
            merged.values(),
            key=lambda item: float(item.get("score") or 0.0),
            reverse=True,
        )

    def _score_retrieval_confidence(
        self,
        query: str,
        results: list[dict[str, Any]],
    ) -> tuple[float, str | None]:
        """Estimate whether the retrieved context is likely on-topic."""
        if not results:
            return 0.0, (
                "The retriever did not find relevant indexed context. "
                "Acknowledge the gap and suggest uploading the specific document or "
                "contacting the academic office."
            )

        query_terms = set(self._extract_keywords(query).split())
        scored_results: list[dict[str, Any]] = []
        overlap_hits = 0
        strong_hits = 0
        strategy = getattr(self.vector_tools, "retrieval_strategy", "similarity")
        for result in results[:5]:
            evidence_text = " ".join(
                [
                    str(result.get("breadcrumb") or ""),
                    str(result.get("text") or ""),
                    " ".join(
                        str(child.get("text") or "")
                        for child in result.get("matched_children", [])
                        if isinstance(child, dict)
                    ),
                ]
            ).lower()
            overlap_terms = sorted(term for term in query_terms if term and term in evidence_text)
            overlap_ratio = len(overlap_terms) / max(1, len(query_terms))
            explicit_policy_support = self._result_has_explicit_policy_rule(result)
            annotated = dict(result)
            annotated["query_overlap_terms"] = overlap_terms
            annotated["query_overlap_ratio"] = overlap_ratio
            annotated["explicit_policy_support"] = explicit_policy_support
            scored_results.append(annotated)
            if overlap_terms:
                overlap_hits += 1
            if self._is_strategy_strong_hit(result, strategy):
                strong_hits += 1

        results[:] = scored_results
        top_score = float(scored_results[0].get("score") or 0.0)
        second_score = (
            float(scored_results[1].get("score") or 0.0)
            if len(scored_results) > 1
            else max(top_score - 0.05, 0.0)
        )
        score_component = self._score_component(top_score, second_score, strategy)
        margin_component = self._margin_component(top_score, second_score, strategy)
        overlap_component = overlap_hits / max(1, min(len(scored_results), 3))
        hit_component = min(strong_hits / 2, 1.0)
        explicit_rule_component = 1.0 if scored_results[0].get("explicit_policy_support") else 0.0
        confidence = round(
            (0.20 * score_component)
            + (0.20 * margin_component)
            + (0.20 * overlap_component)
            + (0.10 * hit_component)
            + (0.30 * explicit_rule_component),
            3,
        )
        top_overlap_ratio = float(scored_results[0].get("query_overlap_ratio") or 0.0)
        if (
            confidence >= 0.45
            or self._has_clear_topical_match(
                confidence=confidence,
                top_score=top_score,
                margin_component=margin_component,
                top_overlap_ratio=top_overlap_ratio,
                strategy=strategy,
            )
            or self._should_trust_explicit_policy_support(
                top_result=scored_results[0],
                top_score=top_score,
                second_score=second_score,
                top_overlap_ratio=top_overlap_ratio,
                strategy=strategy,
            )
        ):
            return confidence, None
        return confidence, (
            "The retrieved excerpts may not directly answer the question. "
            "If the context looks off-topic, say so explicitly and suggest uploading the "
            "specific academic policy document or contacting the academic office."
        )

    def _contains_explicit_policy_rule(self, text: str) -> bool:
        """Detect direct rule/prohibition language in academic-policy evidence."""
        normalized = " ".join(text.split()).strip().lower()
        if not normalized:
            return False
        return any(marker in normalized for marker in self._POLICY_RULE_MARKERS)

    def _result_has_explicit_policy_rule(self, result: dict[str, Any]) -> bool:
        """Check whether a retrieved result contains an explicit answer-bearing rule."""
        texts = [str(result.get("text") or "")]
        texts.extend(
            str(child.get("text") or "")
            for child in result.get("matched_children", [])
            if isinstance(child, dict)
        )
        return any(self._contains_explicit_policy_rule(text) for text in texts)

    def _is_strategy_strong_hit(self, result: dict[str, Any], strategy: str) -> bool:
        """Count strong hits using thresholds appropriate to the active retrieval strategy."""
        score = float(result.get("score") or 0.0)
        if strategy == "rrf":
            return float(result.get("rrf_score") or score) > 0.0
        if strategy == "reranker":
            return float(result.get("reranker_score") or score) >= 0.20
        return score >= 0.55

    def _score_component(self, top_score: float, second_score: float, strategy: str) -> float:
        """Normalize score magnitude for different retrieval strategies."""
        if strategy == "rrf":
            if top_score <= 0.0:
                return 0.0
            return min(max(top_score / max(second_score, 1e-6), 0.0), 2.0) / 2.0
        if strategy == "reranker":
            return min(max((top_score - 0.15) / 0.20, 0.0), 1.0)
        return min(max((top_score - 0.35) / 0.45, 0.0), 1.0)

    def _margin_component(self, top_score: float, second_score: float, strategy: str) -> float:
        """Normalize separation between the top and runner-up candidates."""
        if strategy == "rrf":
            if top_score <= 0.0:
                return 0.0
            return min(max((top_score - second_score) / top_score, 0.0), 1.0)
        if strategy == "reranker":
            return min(max((top_score - second_score + 0.02) / 0.18, 0.0), 1.0)
        return min(max((top_score - second_score + 0.05) / 0.25, 0.0), 1.0)

    def _should_trust_explicit_policy_support(
        self,
        top_result: dict[str, Any],
        top_score: float,
        second_score: float,
        top_overlap_ratio: float,
        strategy: str,
    ) -> bool:
        """Suppress low-confidence warnings when the top chunk directly states the rule."""
        if not top_result.get("explicit_policy_support"):
            return False
        if top_overlap_ratio < 0.45:
            return False
        if strategy == "rrf":
            return top_score > 0.0 and (top_score - second_score) / max(top_score, 1e-6) >= 0.20
        if strategy == "reranker":
            return top_score >= 0.20 and (top_score - second_score) >= 0.05
        return top_score >= 0.35

    def _has_clear_topical_match(
        self,
        confidence: float,
        top_score: float,
        margin_component: float,
        top_overlap_ratio: float,
        strategy: str,
    ) -> bool:
        """Allow near-threshold but clearly topical matches to avoid unnecessary fallback warnings."""
        if confidence < 0.25 or top_overlap_ratio < 0.40:
            return False
        if strategy == "rrf":
            return top_score > 0.0 and margin_component >= 0.20
        if strategy == "reranker":
            return top_score >= 0.20 and margin_component >= 0.30
        return top_score >= 0.35

    def _result_key(self, result: dict[str, Any]) -> str:
        """Build a stable dedupe key for hydrated parent results."""
        return str(
            result.get("parent_id")
            or f"{result.get('document_id', '')}:{result.get('section_id', '')}:{result.get('chunk_id', '')}"
        )

    def _tag_result(self, result: dict[str, Any], query_variant: str | None) -> dict[str, Any]:
        """Attach the query variant that produced a result."""
        tagged = dict(result)
        if "score" in tagged:
            tagged.setdefault("final_score", tagged.get("score"))
            tagged.setdefault("retrieval_score", tagged.get("vector_score", tagged.get("score")))
        if query_variant:
            tagged["matched_query"] = query_variant
        return tagged

    def _get_llm(self):
        """Resolve the current LLM lazily."""
        if self._llm is None:
            self._llm = self._llm_provider()
        return self._llm
