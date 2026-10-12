"""Unit tests for retrieval query preparation."""

from types import SimpleNamespace
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from src.utils.nodes.retrieval import RetrievalNode
from src.utils.state import AgentState


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
