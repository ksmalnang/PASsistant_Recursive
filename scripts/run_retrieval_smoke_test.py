"""Verify that eval questions retrieve chunks from their expected source documents."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from src.eval.dataset import load_dataset
from src.utils.nodes.retrieval import RetrievalNode
from src.utils.state import AgentState
from src.utils.tools.vector_store import VectorStoreTools


async def run(dataset_path: Path, top_k: int) -> int:
    """Run retrieval checks for dataset rows with `source_document` metadata."""
    retrieval_node = RetrievalNode(retriever=VectorStoreTools())
    samples = load_dataset(dataset_path)
    checked = 0

    for sample in samples:
        expected_document = str(sample.metadata.get("source_document") or "").strip()
        if not expected_document:
            continue
        checked += 1
        updates = await retrieval_node.run(
            AgentState(
                current_intent="query_document",
                requires_retrieval=True,
                retrieval_query=sample.user_input,
            )
        )
        results = list(updates.get("retrieved_chunks", []))[:top_k]
        matched = any(result.get("filename") == expected_document for result in results)
        if not matched:
            filenames = [str(result.get("filename") or "") for result in results]
            raise SystemExit(
                "Retrieval smoke test failed for "
                f"{sample.id}: expected '{expected_document}', got {filenames}"
            )
        print(f"PASS {sample.id}: {expected_document}")

    if checked == 0:
        raise SystemExit("No dataset rows with metadata.source_document were found.")

    print(f"Checked {checked} retrieval case(s).")
    return 0


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="tests/fixtures/ragas_dataset.jsonl",
        help="Path to the evaluation dataset JSONL file.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of retrieval results to inspect per question.",
    )
    args = parser.parse_args()
    raise SystemExit(asyncio.run(run(Path(args.dataset), args.top_k)))


if __name__ == "__main__":
    main()
