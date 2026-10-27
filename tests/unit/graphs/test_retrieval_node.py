"""Unit tests for retrieval query preparation."""

from types import SimpleNamespace
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from src.utils.nodes.retrieval import RetrievalNode
from src.utils.state import AgentState, DocumentType


class _RewriteLLM:
    """Async-only rewrite double; a synchronous call fails the test."""

    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.prompts: list[str] = []

    def invoke(self, messages: Any, /, **kwargs: Any) -> Any:
        raise AssertionError("query rewrite must run through ainvoke")

    async def ainvoke(self, messages: Any, /, **kwargs: Any) -> Any:
        self.prompts.append(str(messages[0].content))
        return SimpleNamespace(content=self.answer)


class _RecordingRetriever:
    """DocumentRetriever double that records the query variants it receives."""

    retrieval_strategy = "similarity"
    reranker_model = None
    reranker_candidate_multiplier = 3

    def __init__(self) -> None:
        self.searched: list[list[str]] = []
        self.document_types: list[Any] = []

    async def search_similar(
        self,
        query: str,
        document_type: Any = None,
        top_k: int = 5,
        score_threshold: float = 0.2,
    ) -> list[dict]:
        return []

    async def search_many(
        self,
        queries: list[str],
        document_type: Any = None,
        top_k: int = 5,
        score_threshold: float = 0.2,
    ) -> list[list[dict]]:
        self.searched.append(list(queries))
        self.document_types.append(document_type)
        return [[] for _ in queries]


class _RankedRetriever:
    """Retriever double returning the same ranked chunks for every query variant."""

    retrieval_strategy = "similarity"
    reranker_model = None
    reranker_candidate_multiplier = 3

    def __init__(self, count: int) -> None:
        self.count = count

    async def search_many(
        self,
        queries: list[str],
        document_type: Any = None,
        top_k: int = 5,
        score_threshold: float = 0.2,
    ) -> list[list[dict]]:
        return [
            [
                {
                    "chunk_id": f"chunk-{index}",
                    "document_id": "doc-1",
                    "filename": "Pedoman.pdf",
                    "text": f"excerpt {index}",
                    "score": 0.9 - index * 0.01,
                    "final_score": 0.9 - index * 0.01,
                }
                for index in range(self.count)
            ]
            for _ in queries
        ]


class _RewriteOverlapRetriever:
    """Retriever double whose only chunk overlaps the rewritten query."""

    retrieval_strategy = "similarity"
    reranker_model = None
    reranker_candidate_multiplier = 3

    async def search_many(
        self,
        queries: list[str],
        document_type: Any = None,
        top_k: int = 5,
        score_threshold: float = 0.2,
    ) -> list[list[dict]]:
        return [
            [
                {
                    "chunk_id": "chunk-1",
                    "document_id": "doc-1",
                    "filename": "Pedoman.pdf",
                    "text": "Syarat cuti akademik adalah ...",
                    "score": 0.4,
                    "final_score": 0.4,
                }
            ]
            for _ in queries
        ]


class _FollowUpDecision:
    """Follow-up classifier double with a fixed verdict."""

    def __init__(self, verdict: bool | None) -> None:
        self.verdict = verdict
        self.calls: list[tuple[str, str, str | None]] = []

    async def needs_previous_context(
        self,
        question: str,
        history: str,
        *,
        session_id: str | None = None,
    ) -> bool | None:
        self.calls.append((question, history, session_id))
        return self.verdict


def _document_state(query: str) -> AgentState:
    return AgentState(
        session_id="retrieval-test",
        current_intent="query_document",
        requires_retrieval=True,
        retrieval_query=query,
    )


def _node(answer: str) -> tuple[RetrievalNode, _RecordingRetriever, _RewriteLLM]:
    retriever = _RecordingRetriever()
    llm = _RewriteLLM(answer)
    return RetrievalNode(retriever=retriever, llm_provider=lambda: llm), retriever, llm


async def test_unit_retrieval_rewrites_async_and_batches_variants() -> None:
    """One batched search receives the original question plus the rewrite."""
    node, retriever, llm = _node("syarat perwalian pembayaran dpp")

    update = await node.run(_document_state("Kak, kalau aku telat bayar DPP, bisa ikut perwalian?"))

    assert retriever.searched == [
        [
            "Kak, kalau aku telat bayar DPP, bisa ikut perwalian?",
            "syarat perwalian pembayaran dpp",
        ]
    ]
    assert len(llm.prompts) == 1
    assert update["retrieved_chunks"] == []


async def test_unit_retrieval_caps_generated_query_variants() -> None:
    """A synonym expansion that would be a third variant is dropped."""
    node, retriever, _ = _node("cuti")

    await node.run(_document_state("Bisakah saya mengambil cuti?"))

    assert retriever.searched == [["Bisakah saya mengambil cuti?", "cuti"]]


async def test_unit_retrieval_rewrite_receives_prior_turn() -> None:
    """The rewrite prompt carries the previous turn so references can be resolved."""
    node, _, llm = _node("syarat mata kuliah pilihan semester lima")
    state = _document_state("kalau yang semester lima?")
    state.messages.append(HumanMessage(content="berapa syarat mata kuliah pilihan?"))
    state.messages.append(AIMessage(content="Minimal 24 SKS."))

    await node.run(state)

    assert "berapa syarat mata kuliah pilihan?" in llm.prompts[0]
    assert "kalau yang semester lima?" in llm.prompts[0]


async def test_unit_retrieval_bounds_rewrite_prompt_input() -> None:
    """A long pasted message is truncated before reaching the rewrite model."""
    node, _, llm = _node("ringkas")
    limit = RetrievalNode._REWRITE_INPUT_LIMIT

    await node.run(_document_state("a" * (limit + 200)))

    assert "a" * limit in llm.prompts[0]
    assert "a" * (limit + 1) not in llm.prompts[0]


async def test_unit_retrieval_exposes_all_ranked_chunks_up_to_top_k() -> None:
    """Response generation receives top_k chunks rather than a fixed five."""
    retriever = _RankedRetriever(count=8)
    node = RetrievalNode(retriever=retriever, llm_provider=lambda: _RewriteLLM("rewrite"))
    node.top_k = 8

    update = await node.run(_document_state("Berapa SKS minimal untuk lulus?"))

    assert len(update["retrieved_chunks"]) == 8


async def test_unit_retrieval_confidence_uses_the_rewritten_query() -> None:
    """A referential fragment does not depress confidence when the rewrite matches the evidence."""
    node = RetrievalNode(
        retriever=_RewriteOverlapRetriever(),
        llm_provider=lambda: _RewriteLLM("syarat cuti akademik"),
    )

    update = await node.run(_document_state("kalau yang tadi gimana?"))

    assert update["response_confidence"] >= 0.45
    assert update["retrieval_warning"] is None


async def test_unit_retrieval_searches_only_rewrite_for_referential_follow_up() -> None:
    """A referential fragment is not sent to the index as its own query."""
    node, retriever, _ = _node("syarat cuti akademik")

    await node.run(_document_state("kalau yang tadi gimana?"))

    assert retriever.searched == [["syarat cuti akademik"]]


async def test_unit_retrieval_rewrite_receives_previous_resolved_query() -> None:
    """The rewrite prompt carries the canonical query from the previous turn."""
    node, _, llm = _node("syarat cuti akademik semester lima")
    state = _document_state("kalau semester lima?")
    state.resolved_query = "syarat cuti akademik"

    await node.run(state)

    assert "Previous resolved query: syarat cuti akademik" in llm.prompts[0]


async def test_unit_retrieval_keeps_previous_all_scope_for_follow_up() -> None:
    """A follow-up stays in the broad document scope used by the previous turn."""
    node, retriever, _ = _node("cuti")
    state = _document_state("kalau semester lima?")
    state.current_intent = "query_student"
    state.retrieval_scope = "all"

    await node.run(state)

    assert retriever.document_types == [None]


async def test_unit_retrieval_keeps_previous_transcript_scope_for_follow_up() -> None:
    """A follow-up stays in the transcript scope even when the intent flips."""
    node, retriever, _ = _node("cuti")
    state = _document_state("kalau semester lalu?")
    state.current_intent = "query_document"
    state.retrieval_scope = "transcript"

    await node.run(state)

    assert retriever.document_types == [DocumentType.TRANSCRIPT]


async def test_unit_retrieval_does_not_match_nim_inside_minimal() -> None:
    """A policy word containing 'nim' is not treated as a personal record term."""
    node, retriever, _ = _node("cuti")
    state = _document_state("kalau minimal SKS?")
    state.current_intent = "query_student"
    state.retrieval_scope = "all"

    await node.run(state)

    assert retriever.document_types == [None]


async def test_unit_retrieval_widens_transcript_scope_for_policy_follow_up() -> None:
    """An explicit policy follow-up leaves a transcript-only scope."""
    node, retriever, _ = _node("cuti")
    state = _document_state("kalau batas SKS?")
    state.current_intent = "query_student"
    state.retrieval_scope = "transcript"

    await node.run(state)

    assert retriever.document_types == [None]


async def test_unit_retrieval_widens_scope_for_new_policy_topic() -> None:
    """A short new policy topic does not inherit a transcript-only scope."""
    node, retriever, _ = _node("cuti")
    state = _document_state("kalau soal cuti?")
    state.current_intent = "query_student"
    state.retrieval_scope = "transcript"

    await node.run(state)

    assert retriever.document_types == [None]


async def test_unit_retrieval_lets_personal_follow_up_override_scope() -> None:
    """An explicit personal-record follow-up does not inherit a broad scope."""
    node, retriever, _ = _node("cuti")
    state = _document_state("kalau nilai saya?")
    state.current_intent = "query_student"
    state.retrieval_scope = "all"

    await node.run(state)

    assert retriever.document_types == [DocumentType.TRANSCRIPT]


async def test_unit_retrieval_pins_previous_evidence_for_follow_up() -> None:
    """The previous turn's top evidence stays in the follow-up's result set."""
    retriever = _RankedRetriever(count=3)
    node = RetrievalNode(
        retriever=retriever,
        llm_provider=lambda: _RewriteLLM("syarat cuti akademik"),
    )
    node.top_k = 4
    state = _document_state("kalau yang tadi gimana?")
    state.conversation_evidence = [
        {"chunk_id": "chunk-0", "filename": "Pedoman.pdf", "text": "same", "final_score": 0.9},
        {"chunk_id": "old-2", "filename": "Old.pdf", "text": "prior rule", "final_score": 0.8},
    ]

    update = await node.run(state)

    assert [chunk["chunk_id"] for chunk in update["retrieved_chunks"]] == [
        "old-2",
        "chunk-0",
        "chunk-1",
        "chunk-2",
    ]
    assert update["retrieved_chunks"][0]["carried_forward"] is True
    assert "carried_forward" not in update["retrieved_chunks"][1]


async def test_unit_retrieval_does_not_pin_evidence_for_new_topic() -> None:
    """A self-contained new question does not carry over the previous evidence."""
    retriever = _RankedRetriever(count=2)
    node = RetrievalNode(
        retriever=retriever,
        llm_provider=lambda: _RewriteLLM("syarat cuti akademik"),
    )
    node.top_k = 4
    state = _document_state("Berapa SKS minimal untuk lulus?")
    state.conversation_evidence = [{"chunk_id": "old-1", "text": "old", "final_score": 0.8}]

    update = await node.run(state)

    assert all("carried_forward" not in chunk for chunk in update["retrieved_chunks"])


async def test_unit_retrieval_does_not_pin_evidence_for_explicit_topic_change() -> None:
    """A follow-up that names a new policy topic does not carry the old evidence."""
    retriever = _RankedRetriever(count=2)
    node = RetrievalNode(
        retriever=retriever,
        llm_provider=lambda: _RewriteLLM("batas SKS"),
    )
    node.top_k = 4
    state = _document_state("kalau batas SKS?")
    state.conversation_evidence = [{"chunk_id": "old-1", "text": "old", "final_score": 0.8}]

    update = await node.run(state)

    assert all("carried_forward" not in chunk for chunk in update["retrieved_chunks"])


async def test_unit_retrieval_does_not_treat_punya_as_a_reference() -> None:
    """A self-contained question containing 'punya' is not a follow-up."""
    retriever = _RankedRetriever(count=2)
    node = RetrievalNode(
        retriever=retriever,
        llm_provider=lambda: _RewriteLLM("syarat KIP"),
    )
    node.top_k = 4
    state = _document_state("Apa syarat punya KIP?")
    state.conversation_evidence = [{"chunk_id": "old-1", "text": "old", "final_score": 0.8}]

    update = await node.run(state)

    assert all("carried_forward" not in chunk for chunk in update["retrieved_chunks"])


async def test_unit_retrieval_uses_jev_follow_up_verdict() -> None:
    """Jev can mark a self-contained question as needing previous context."""
    retriever = _RecordingRetriever()
    classifier = _FollowUpDecision(True)
    node = RetrievalNode(
        retriever=retriever,
        llm_provider=lambda: _RewriteLLM("syarat cuti akademik"),
        follow_up_classifier=classifier,
    )

    await node.run(_document_state("Apa syarat cuti akademik?"))

    assert retriever.searched == [["syarat cuti akademik"]]
    assert classifier.calls[0][0] == "Apa syarat cuti akademik?"


async def test_unit_retrieval_jev_false_keeps_original_and_rewrite() -> None:
    """Jev can override the heuristic and treat a fragment as self-contained."""
    retriever = _RecordingRetriever()
    classifier = _FollowUpDecision(False)
    node = RetrievalNode(
        retriever=retriever,
        llm_provider=lambda: _RewriteLLM("syarat cuti akademik"),
        follow_up_classifier=classifier,
    )

    await node.run(_document_state("kalau yang tadi gimana?"))

    assert retriever.searched == [
        ["kalau yang tadi gimana?", "syarat cuti akademik"]
    ]


async def test_unit_retrieval_falls_back_to_heuristic_when_jev_is_uncertain() -> None:
    """An uncertain Jev verdict leaves the decision to the rule-based fallback."""
    retriever = _RecordingRetriever()
    classifier = _FollowUpDecision(None)
    node = RetrievalNode(
        retriever=retriever,
        llm_provider=lambda: _RewriteLLM("syarat cuti akademik"),
        follow_up_classifier=classifier,
    )

    await node.run(_document_state("kalau yang tadi gimana?"))

    assert retriever.searched == [["syarat cuti akademik"]]


async def test_unit_retrieval_jev_verdict_controls_pinning() -> None:
    """Jev can mark a question as a follow-up and keep the prior evidence."""
    retriever = _RankedRetriever(count=2)
    node = RetrievalNode(
        retriever=retriever,
        llm_provider=lambda: _RewriteLLM("syarat cuti akademik"),
        follow_up_classifier=_FollowUpDecision(True),
    )
    node.top_k = 4
    state = _document_state("Apa syarat semester lima?")
    state.conversation_evidence = [
        {"chunk_id": "old-1", "text": "old", "final_score": 0.8}
    ]

    update = await node.run(state)

    assert update["retrieved_chunks"][0]["chunk_id"] == "old-1"
    assert update["retrieved_chunks"][0]["carried_forward"] is True


async def test_unit_retrieval_jev_verdict_controls_scope() -> None:
    """Jev can mark a question as a follow-up and keep the previous scope."""
    retriever = _RecordingRetriever()
    node = RetrievalNode(
        retriever=retriever,
        llm_provider=lambda: _RewriteLLM("syarat cuti"),
        follow_up_classifier=_FollowUpDecision(True),
    )
    state = _document_state("Apa syarat semester lima?")
    state.current_intent = "query_document"
    state.retrieval_scope = "transcript"

    await node.run(state)

    assert retriever.document_types == [DocumentType.TRANSCRIPT]


async def test_unit_retrieval_sends_history_to_follow_up_classifier() -> None:
    """Jev receives the conversation, not only the latest fragment."""
    classifier = _FollowUpDecision(None)
    node = RetrievalNode(
        retriever=_RecordingRetriever(),
        llm_provider=lambda: _RewriteLLM("syarat cuti akademik"),
        follow_up_classifier=classifier,
    )
    state = _document_state("kalau semester lima?")
    state.messages.append(HumanMessage(content="berapa syarat mata kuliah pilihan?"))
    state.messages.append(AIMessage(content="Minimal 24 SKS."))
    state.messages.append(HumanMessage(content="kalau semester lima?"))

    await node.run(state)

    question, history, session_id = classifier.calls[0]
    assert question == "kalau semester lima?"
    assert "berapa syarat mata kuliah pilihan?" in history
    assert "Minimal 24 SKS." in history
    assert session_id == "retrieval-test"


async def test_unit_retrieval_stores_resolved_query_and_scope() -> None:
    """The turn records the canonical query and scope for the next follow-up."""
    node, _, _ = _node("syarat cuti akademik")

    update = await node.run(_document_state("kalau yang tadi gimana?"))

    assert update["resolved_query"] == "syarat cuti akademik"
    assert update["retrieval_scope"] == "all"
