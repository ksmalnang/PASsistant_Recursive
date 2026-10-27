"""Retrieval node."""

import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage

from src.clients.llm import get_rewrite_llm
from src.config import get_settings
from src.services.contracts import DocumentRetriever
from src.services.followup import JevFollowUpClassifier
from src.utils.messages import conversation_transcript
from src.utils.nodes.prompts import QUERY_REWRITE_PROMPT
from src.utils.state import AgentState, DocumentType
from src.utils.tools import VectorStoreTools

logger = logging.getLogger(__name__)


class RetrievalNode:
    """
    Retrieves relevant indexed document context from the retrieval stack.

    Delegates to the configured retrieval strategy (similarity, RRF, or reranker)
    and augments the agent state with retrieved chunk context.
    """

    # --- Policy detection ---
    # Phrases in a retrieved chunk that signal an explicit academic rule or prohibition.
    _POLICY_RULE_MARKERS: tuple[str, ...] = (
        "dapat dilakukan setelah",
        "tidak dapat",
        "belum bisa",
        "belum dapat",
        "wajib",
        "harus",
        "syarat",
        "ketentuan",
    )

    # Domain terms that identify a query as policy/admin rather than personal student records.
    # When present, the document-type filter is widened beyond TRANSCRIPT.
    _POLICY_SCOPE_TERMS: tuple[str, ...] = (
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
        "aturan",
        "cuti",
        "beasiswa",
    )

    # --- Semester numeral normalisation ---
    # Converts between Arabic ("5") and Roman ("V") semester numbers for lexical matching.
    _ARABIC_TO_ROMAN: dict[str, str] = {
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
    _ROMAN_TO_ARABIC: dict[str, str] = {v: k for k, v in _ARABIC_TO_ROMAN.items()}
    _SEMESTER_ARABIC_PATTERN = re.compile(r"\b(semester)\s+(\d{1,2})\b", re.IGNORECASE)
    _SEMESTER_ROMAN_PATTERN = re.compile(r"\b(semester)\s+([IVXL]+)\b", re.IGNORECASE)

    # A rewrite only needs enough of a long pasted message to find its keywords.
    _REWRITE_INPUT_LIMIT = 500
    _HISTORY_LIMIT = 6
    _HISTORY_CHAR_LIMIT = 4000

    # --- Follow-up detection ---
    # Short, referential messages inherit the previous turn's topic and document scope.
    _REFERENCE_WORDS: frozenset[str] = frozenset(
        {"kalau", "kalo", "terus", "itu", "tersebut", "tadi", "sebelumnya"}
    )
    _REFERENCE_PHRASES: tuple[str, ...] = (
        "di atas",
        "yang kedua",
        "gimana dengan",
        "bagaimana dengan",
        "how about",
        "what about",
    )
    _FOLLOW_UP_MAX_WORDS = 8
    _NON_POSSESSIVE_NYA: frozenset[str] = frozenset({"punya", "hanya", "nya"})
    _PERSONAL_RECORD_TERMS: tuple[str, ...] = (
        "nilai saya",
        "ipk saya",
        "ip saya",
        "transkrip",
        "nim",
        "sks saya",
        "status akademik saya",
        "hasil studi saya",
    )
    _MAX_CARRIED_CHUNKS = 2

    def __init__(
        self,
        retriever: DocumentRetriever | None = None,
        llm_provider=get_rewrite_llm,
        follow_up_classifier: JevFollowUpClassifier | None = None,
    ):
        self.vector_tools = retriever or VectorStoreTools()
        self._llm_provider = llm_provider
        self._follow_up_classifier = follow_up_classifier
        self._llm = None
        settings = get_settings()
        self.top_k = settings.RETRIEVAL_TOP_K
        self.max_query_variants = settings.RETRIEVAL_MAX_QUERY_VARIANTS

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

        raw_query = state.retrieval_query.strip()
        history = self._conversation_history(state)
        try:
            follow_up = await self._resolve_follow_up(
                raw_query, history, state.session_id
            )
            doc_type = self._resolve_document_type(
                state.current_intent,
                raw_query,
                state.retrieval_scope,
                follow_up=follow_up,
            )
            queries = await self._build_queries(state, history, follow_up=follow_up)
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
            query_results_list = await self.vector_tools.search_many(
                queries=queries,
                document_type=doc_type,
                top_k=self.top_k,
                score_threshold=0.2,
            )
            for query, query_results in zip(queries, query_results_list, strict=True):
                merged_results = self._merge_results(merged_results, query_results, query)

            canonical_query = (
                queries[0]
                if follow_up
                else (queries[1] if len(queries) > 1 else queries[0])
            )
            confidence, warning = self._score_retrieval_confidence(
                query=canonical_query,
                results=merged_results,
            )
            if (
                follow_up
                and state.conversation_evidence
                and not self._looks_like_policy_scope(raw_query)
                and not self._looks_personal_record(raw_query)
            ):
                merged_results = self._pin_previous_evidence(
                    merged_results, state.conversation_evidence
                )
            scope = doc_type.value if doc_type else "all"
            logger.info(
                "Retrieved chunks",
                extra={
                    "query": state.retrieval_query,
                    "queries": queries,
                    "follow_up": follow_up,
                    "scope": scope,
                    "results": len(merged_results),
                    "confidence": confidence,
                },
            )
            return {
                "retrieved_chunks": merged_results[: self.top_k],
                "resolved_query": canonical_query,
                "retrieval_scope": scope,
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
        previous_scope: str | None = None,
        *,
        follow_up: bool | None = None,
    ) -> DocumentType | None:
        """Resolve an optional document type filter, keeping a follow-up in its prior scope."""
        normalized = query or ""
        is_follow_up = (
            self._looks_referential(normalized) if follow_up is None else follow_up
        )
        if (
            previous_scope
            and is_follow_up
            and not self._looks_personal_record(normalized)
            and not self._looks_like_policy_scope(normalized)
        ):
            if previous_scope == DocumentType.TRANSCRIPT.value:
                return DocumentType.TRANSCRIPT
            return None
        if intent == "query_student":
            if self._looks_like_policy_scope(normalized):
                return None
            return DocumentType.TRANSCRIPT
        return None

    def _looks_like_policy_scope(self, query: str) -> bool:
        """Detect mixed student-record wording that asks for policy/rule context."""
        lowered = query.lower()
        return any(term in lowered for term in self._POLICY_SCOPE_TERMS)

    def _looks_personal_record(self, query: str) -> bool:
        """Detect wording that explicitly asks for the user's own record data."""
        lowered = query.lower()
        return any(
            re.search(rf"\b{re.escape(term)}", lowered)
            for term in self._PERSONAL_RECORD_TERMS
        )

    def _looks_referential(self, query: str) -> bool:
        """Detect a short follow-up whose subject comes from the conversation."""
        lowered = " ".join(query.lower().split())
        words = [word.strip(".,?!;:") for word in lowered.split()]
        if not words or len(words) > self._FOLLOW_UP_MAX_WORDS:
            return False
        has_reference = any(word in self._REFERENCE_WORDS for word in words)
        has_phrase = any(phrase in lowered for phrase in self._REFERENCE_PHRASES)
        has_suffix = any(
            word.endswith("nya") and word not in self._NON_POSSESSIVE_NYA
            for word in words
        )
        return has_reference or has_phrase or has_suffix

    def _pin_previous_evidence(
        self,
        current: list[dict[str, Any]],
        previous: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Prefix a follow-up's results with the previous turn's evidence."""
        seen = {self._result_key(chunk) for chunk in current}
        pinned: list[dict[str, Any]] = []
        for chunk in previous:
            if len(pinned) >= self._MAX_CARRIED_CHUNKS:
                break
            if chunk.get("carried_forward"):
                continue
            key = self._result_key(chunk)
            if key in seen:
                continue
            if float(chunk.get("final_score", chunk.get("score", 0.0)) or 0.0) < 0.0:
                continue
            pinned.append({**chunk, "carried_forward": True})
            seen.add(key)
        return pinned + current

    def _conversation_history(self, state: AgentState) -> str:
        """Render the recent conversation for Jev and the query rewriter."""
        history = conversation_transcript(state.messages[:-1], limit=self._HISTORY_LIMIT)
        return history[-self._HISTORY_CHAR_LIMIT :]

    async def _resolve_follow_up(
        self,
        query: str,
        history: str,
        session_id: str | None,
    ) -> bool:
        """Ask Jev whether the message needs the previous conversation."""
        if self._follow_up_classifier is None:
            return self._looks_referential(query)
        verdict = await self._follow_up_classifier.needs_previous_context(
            query, history, session_id=session_id
        )
        if verdict is None:
            return self._looks_referential(query)
        return verdict

    async def _build_queries(
        self,
        state: AgentState,
        history: str,
        *,
        follow_up: bool,
    ) -> list[str]:
        """Build the query variants used by one retrieval turn."""
        original = (state.retrieval_query or "").strip()
        if not original:
            return []
        if state.current_intent != "query_document":
            return [original]

        if follow_up and state.resolved_query:
            history = f"Previous resolved query: {state.resolved_query}\n\n{history}"
        rewritten = await self._rewrite_query(original, history)
        canonical = rewritten or original

        if follow_up:
            queries = [canonical]
        else:
            queries = [original]
            if rewritten and rewritten.lower() != original.lower():
                queries.append(rewritten)

        expanded = self._expand_query(canonical)
        if expanded and all(expanded.lower() != query.lower() for query in queries):
            queries.append(expanded)
        return queries[: self.max_query_variants]

    async def _rewrite_query(self, question: str, history: str) -> str:
        """Compress a conversational question into a keyword-rich retrieval query."""
        llm = self._get_llm()
        if llm is None:
            return question
        prompt = QUERY_REWRITE_PROMPT.format(
            history=history or "(no prior turns)",
            question=question[: self._REWRITE_INPUT_LIMIT],
        )
        try:
            response = await llm.ainvoke([HumanMessage(content=prompt)])
            rewritten = str(response.content).strip()
            if rewritten:
                return rewritten.splitlines()[0].strip()
            return question
        except Exception as exc:
            logger.warning("Query rewrite failed, using original question: %s", exc)
            return question

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

    def _merge_results(
        self,
        current: list[dict[str, Any]],
        incoming: list[dict[str, Any]],
        query_variant: str,
    ) -> list[dict[str, Any]]:
        """Merge per-query retrieval results by chunk id, keeping the highest score."""
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

        query_terms = set(query.lower().split())
        scored_results: list[dict[str, Any]] = []
        overlap_hits = 0
        strong_hits = 0
        strategy = getattr(self.vector_tools, "retrieval_strategy", "similarity")
        for result in results[:5]:
            evidence_text = str(result.get("text") or "").lower()
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
        return self._contains_explicit_policy_rule(str(result.get("text") or ""))

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
        """Build a stable dedupe key for retrieved chunk results."""
        chunk_id = result.get("chunk_id")
        if chunk_id:
            return str(chunk_id)
        return f"{result.get('document_id', '')}:{str(result.get('text', ''))[:40]}"

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
