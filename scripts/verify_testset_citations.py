"""Verify that every reference_context in a testset exists in the indexed corpus.

Checks the quotes an external generator (NotebookLM, chat LLM) produced against
the Qdrant payloads, fills metadata.source_doc_ids, and reports rows that need
manual repair. Exits non-zero when any quote is not verbatim, so it can gate a
dataset before an expensive evaluation run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qdrant_client import QdrantClient

from src.config import get_settings
from src.eval.citations import (
    DEFAULT_ADJACENT_CHUNKS,
    DEFAULT_FUZZ_THRESHOLD,
    VerificationConfig,
    load_corpus,
    load_rows,
    row_to_dict,
    verify_rows,
)


def main() -> None:
    args = _parse_args()
    settings = get_settings()
    collection = args.collection or settings.QDRANT_COLLECTION_NAME

    client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
    if not client.collection_exists(collection):
        raise SystemExit(f"Qdrant collection '{collection}' does not exist")

    corpus = load_corpus(client, collection)
    rows = load_rows(Path(args.dataset))
    result = verify_rows(
        rows,
        corpus,
        VerificationConfig(
            fuzz_threshold=args.fuzz_threshold,
            adjacent_chunks=args.adjacent_chunks,
        ),
    )

    chunks = sum(len(doc_chunks) for doc_chunks in corpus.by_doc.values())
    matched = result.counts["match"]
    near = result.counts["near"]
    missing = result.counts["missing"]
    print(
        f"Corpus: {chunks} chunks from {len(corpus.by_doc)} document(s) in '{collection}'\n"
        f"Quotes: {matched + near + missing} checked, {matched} verbatim, "
        f"{near} near miss, {missing} not found"
    )
    for row_id, row_problems in result.problems.items():
        for problem in row_problems:
            print(f"FAIL {row_id}: {problem}")

    if args.write_verified:
        target = Path(args.write_verified)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as handle:
            for row in result.rows:
                handle.write(json.dumps(row_to_dict(row), ensure_ascii=False) + "\n")
        print(f"Wrote {target}")

    if args.report:
        target = Path(args.report)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(
                {
                    "dataset": str(args.dataset),
                    "collection": collection,
                    "chunks": chunks,
                    "documents": len(corpus.by_doc),
                    "quotes": {
                        "checked": matched + near + missing,
                        "match": matched,
                        "near": near,
                        "missing": missing,
                    },
                    "rows": [
                        {
                            "id": row.id,
                            "checks": [
                                {
                                    "status": check.status,
                                    "score": round(check.score, 2),
                                    "detail": check.detail,
                                }
                                for check in row_checks
                                if check
                            ],
                            "source_doc_ids": row.metadata.get("source_doc_ids", []),
                            "problems": result.problems.get(row.id, []),
                        }
                        for row, row_checks in zip(result.rows, result.checks, strict=True)
                    ],
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"Wrote {target}")

    raise SystemExit(1 if result.problems else 0)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="testset.jsonl", help="Testset JSONL to verify")
    parser.add_argument("--collection", default=None, help="Qdrant collection override")
    parser.add_argument(
        "--fuzz-threshold",
        type=float,
        default=DEFAULT_FUZZ_THRESHOLD,
        help="Similarity below which a quote counts as missing instead of a near miss",
    )
    parser.add_argument(
        "--adjacent-chunks",
        type=int,
        default=DEFAULT_ADJACENT_CHUNKS,
        help="Chunks to join when checking quotes that span a chunk border",
    )
    parser.add_argument(
        "--write-verified",
        default=None,
        help="Write the dataset with metadata.source_doc_ids filled in",
    )
    parser.add_argument("--report", default=None, help="Write a per-row JSON report")
    return parser.parse_args()


if __name__ == "__main__":
    main()
