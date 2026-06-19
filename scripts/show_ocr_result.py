"""Run OCR for a local PDF/image and print the extracted result."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DOCUMENT_TYPES = (
    "transcript",
    "id_card",
    "application",
    "recommendation",
    "curriculum",
    "syllabus",
    "certificate",
    "invoice",
    "policy",
    "other",
)


def _document_type(value: str) -> str:
    if value not in DOCUMENT_TYPES:
        valid = ", ".join(DOCUMENT_TYPES)
        raise argparse.ArgumentTypeError(
            f"Invalid document type '{value}'. Valid values: {valid}"
        )
    return value


def _result_to_dict(result: Any) -> dict[str, Any]:
    return {
        "text": result.text,
        "text_quality_score": result.text_quality_score,
        "num_pages": result.num_pages,
        "parsed_pages": result.parsed_pages,
        "failed_pages": result.failed_pages or [],
        "ocr_warnings": result.ocr_warnings or [],
        "page_results": result.page_results or [],
        "layout_details": result.layout_details,
    }


def _print_summary(result: Any, *, show_layout: bool) -> None:
    print("=== OCR Summary ===")
    print(f"Quality score : {result.text_quality_score:.3f}")
    print(f"Pages         : {result.num_pages if result.num_pages is not None else 'unknown'}")
    print(
        "Parsed pages  : "
        f"{result.parsed_pages if result.parsed_pages is not None else 'unknown'}"
    )
    print(f"Failed pages  : {result.failed_pages or []}")

    warnings = result.ocr_warnings or []
    if warnings:
        print("\n=== Warnings ===")
        for warning in warnings:
            print(f"- {warning}")

    page_results = result.page_results or []
    if page_results:
        print("\n=== Page Status ===")
        for page in page_results:
            page_number = int(page.get("page_index", 0)) + 1
            status = page.get("status", "unknown")
            text_length = page.get("text_length", 0)
            layout_blocks = page.get("layout_blocks", 0)
            error = page.get("error")
            line = (
                f"Page {page_number}: {status} "
                f"(text={text_length} chars, blocks={layout_blocks})"
            )
            if error:
                line = f"{line} - {error}"
            print(line)

    print("\n=== Extracted Text ===")
    print(result.text)

    if show_layout:
        print("\n=== Layout Details ===")
        print(json.dumps(result.layout_details or [], ensure_ascii=False, indent=2))


async def run(args: argparse.Namespace) -> int:
    from src.utils.state import DocumentType
    from src.utils.tools.ocr import GLMOCRTool

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise SystemExit(f"Input file does not exist: {input_path}")
    if not input_path.is_file():
        raise SystemExit(f"Input path is not a file: {input_path}")

    tool = GLMOCRTool()
    result = await tool.extract_text(
        str(input_path),
        document_type=DocumentType(args.document_type),
    )

    _print_summary(result, show_layout=args.show_layout)

    if args.json_out:
        output_path = Path(args.json_out).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(_result_to_dict(result), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nSaved JSON result to: {output_path}")

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        help="Path to a PDF, PNG, JPEG, GIF, or WEBP file.",
    )
    parser.add_argument(
        "--document-type",
        type=_document_type,
        default="other",
        help="Document type hint for OCR context.",
    )
    parser.add_argument(
        "--show-layout",
        action="store_true",
        help="Print serialized layout blocks after the extracted text.",
    )
    parser.add_argument(
        "--json-out",
        help="Optional path to save the complete OCR result as JSON.",
    )
    args = parser.parse_args()
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
