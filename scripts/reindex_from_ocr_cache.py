"""Re-index documents from the persisted OCR cache without re-running OCR.

Examples:
    uv run python scripts/reindex_from_ocr_cache.py --all
    uv run python scripts/reindex_from_ocr_cache.py --document-id <document_id>
    uv run python scripts/reindex_from_ocr_cache.py --filename "handbook.pdf"
    uv run python scripts/reindex_from_ocr_cache.py --all --rebuild-text
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


async def run(args: argparse.Namespace) -> int:
    from src.services.reindex import FAILED, REINDEXED, SKIPPED_NO_CACHE, OCRReindexService

    service = OCRReindexService()

    if args.document_id:
        document_ids = list(args.document_id)
    elif args.filename:
        document_ids = service.document_ids_for_filename(args.filename)
        if not document_ids:
            print(f"no OCR cache for {args.filename}, re-upload required")
            return 1
    else:
        document_ids = service.all_cached_document_ids()

    if not document_ids:
        print("No documents with an OCR cache entry were found.")
        return 1

    results = await service.reindex_many(document_ids, rebuild_text=args.rebuild_text)

    counts = Counter(result.status for result in results)
    print("Reindex summary")
    for result in results:
        detail = f" ({result.error})" if result.error else ""
        print(f"  {result.status}: {result.document_id} {result.filename}{detail}")
    print(
        f"reindexed={counts[REINDEXED]} skipped={counts[SKIPPED_NO_CACHE]} failed={counts[FAILED]}"
    )
    return 1 if counts[FAILED] else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--document-id",
        action="append",
        help="Document id to reindex; repeat for multiple documents.",
    )
    group.add_argument(
        "--filename",
        help="Reindex every cached document with this filename.",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Reindex every document that has an OCR cache entry.",
    )
    parser.add_argument(
        "--rebuild-text",
        action="store_true",
        help="Re-derive page text from cached layout blocks before indexing.",
    )
    args = parser.parse_args()
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
