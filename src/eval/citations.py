"""Verify that testset reference_contexts exist in the indexed corpus.

External dataset generators (NotebookLM, chat LLMs) can paraphrase a quote while
presenting it as verbatim. A paraphrased reference_context makes context_recall
measure the paraphrase rather than the documents, and ``ragas-eval validate``
cannot detect it. Everything here is pure text matching, so it runs without any
judge or LLM call.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz

SCROLL_PAGE_SIZE = 256
MATCH_SCORE = 100.0
MIN_QUOTE_TOKENS = 4
MIN_QUOTE_CHARS = 20
DEFAULT_FUZZ_THRESHOLD = 80.0
DEFAULT_ADJACENT_CHUNKS = 2

# Near-miss excerpts are printed in a terminal, so keep them short.
CLIP = 140

_WHITESPACE_RE = re.compile(r"\s+")


@dataclass
class Chunk:
    """One indexed chunk, in both raw and comparison-normalized form."""

    doc_id: str
    filename: str
    raw: str
    normalized: str


@dataclass
class Corpus:
    """The indexed text a quote is allowed to come from."""

    by_doc: dict[str, list[Chunk]]
    single: dict[str, str]
    joined: dict[str, str]


@dataclass
class FileRow:
    """A raw dataset row plus its metadata, kept editable for the rewritten file."""

    id: str
    user_input: str
    reference: str
    contexts: list[str]
    metadata: dict[str, Any]
    line_no: int = 0


@dataclass(frozen=True)
class CitationCheck:
    """Verification outcome for one quote."""

    status: str
    score: float
    doc_ids: tuple[str, ...]
    detail: str = ""


@dataclass(frozen=True)
class VerificationConfig:
    """Thresholds for one verification pass."""

    fuzz_threshold: float = DEFAULT_FUZZ_THRESHOLD
    adjacent_chunks: int = DEFAULT_ADJACENT_CHUNKS


@dataclass
class VerificationResult:
    """Everything one pass produces: rewritten rows, counters, and problems."""

    rows: list[FileRow]
    checks: list[list[CitationCheck | None]]
    counts: Counter[str]
    problems: dict[str, list[str]]


def normalize(text: str) -> str:
    """Collapse whitespace and unify punctuation so quotes compare fairly."""
    unified = unicodedata.normalize("NFKC", text).replace("\u00a0", " ")
    return _WHITESPACE_RE.sub(" ", unified).strip()


def is_checkable(quote: str) -> bool:
    """A quote shorter than this is too weak to match on."""
    normalized = normalize(quote)
    return len(normalized) >= MIN_QUOTE_CHARS and len(normalized.split()) >= MIN_QUOTE_TOKENS


def load_corpus(client: Any, collection: str) -> Corpus:
    """Read every indexed payload once, so quote lookups stay in memory."""
    by_doc: dict[str, list[tuple[int | None, str, str, str]]] = {}
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=SCROLL_PAGE_SIZE,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            payload = point.payload or {}
            text = str(payload.get("text") or payload.get("child_text") or "")
            if not text.strip():
                continue
            index = payload.get("chunk_index")
            by_doc.setdefault(str(payload.get("document_id") or "unknown"), []).append(
                (
                    index if isinstance(index, int) else None,
                    str(payload.get("chunk_id") or point.id),
                    str(payload.get("filename") or ""),
                    text,
                )
            )
        if offset is None:
            break
    return _merge_corpus(by_doc)


def _merge_corpus(by_doc: dict[str, list[tuple[int | None, str, str, str]]]) -> Corpus:
    """Index chunks both whole and joined, since quotes can span a chunk border."""
    corpus = Corpus(by_doc={}, single={}, joined={})
    for doc_id, entries in by_doc.items():
        entries.sort(key=lambda entry: (entry[0] is None, entry[0] or 0, entry[1]))
        chunks = [
            Chunk(doc_id=doc_id, filename=filename, raw=text, normalized=normalize(text))
            for _, _, filename, text in entries
        ]
        corpus.by_doc[doc_id] = chunks
        for chunk in chunks:
            corpus.single.setdefault(chunk.normalized, doc_id)
        for position in range(len(chunks) - 1):
            pair = _window(chunks, position, 2)
            corpus.joined.setdefault(pair, doc_id)
    return corpus


def check_quote(
    quote: str,
    corpus: Corpus,
    config: VerificationConfig | None = None,
) -> CitationCheck:
    """Classify one quote as indexed text, a near miss, or missing."""
    config = config or VerificationConfig()
    normalized = normalize(quote)

    doc_id = corpus.single.get(normalized)
    if doc_id is not None:
        return CitationCheck("match", MATCH_SCORE, (doc_id,))

    doc_id = corpus.joined.get(normalized)
    if doc_id is not None:
        return CitationCheck("match", MATCH_SCORE, (doc_id,), "spans two adjacent chunks")

    doc_id = _match_adjacent(normalized, corpus, config.adjacent_chunks)
    if doc_id is not None:
        return CitationCheck(
            "match",
            MATCH_SCORE,
            (doc_id,),
            f"spans chunks within a {config.adjacent_chunks}-chunk window",
        )

    score, detail, matched_doc = _best_partial(normalized, corpus, config.adjacent_chunks)
    if score >= config.fuzz_threshold and matched_doc is not None:
        return CitationCheck("near", score, (matched_doc,), detail)
    return CitationCheck("missing", score, (), detail)


def _window(chunks: list[Chunk], start: int, adjacent_limit: int) -> str:
    """Join chunks the way retrieval reads them: one continuous block of text."""
    return " ".join(chunk.normalized for chunk in chunks[start : start + adjacent_limit])


def _match_adjacent(normalized: str, corpus: Corpus, adjacent_limit: int) -> str | None:
    """Catch quotes whose wording depends on line breaks inside the chunk text."""
    for doc_id, chunks in corpus.by_doc.items():
        for start in range(len(chunks)):
            if normalized in _window(chunks, start, adjacent_limit):
                return doc_id
    return None


def _best_partial(
    normalized: str,
    corpus: Corpus,
    adjacent_limit: int,
) -> tuple[float, str, str | None]:
    """Compare the quote against each window and describe the closest one.

    Classification uses ratio, not partial_ratio: a paraphrase that copies one
    contiguous span verbatim still scores 100 on partial_ratio, which would hide
    exactly the case this check exists to catch.
    """
    best: tuple[float, str, str] | None = None
    best_window = ""
    for doc_id, chunks in corpus.by_doc.items():
        for start in range(len(chunks)):
            window = _window(chunks, start, adjacent_limit)
            score = float(fuzz.ratio(normalized, window))
            if best is None or score > best[0]:
                best = (score, doc_id, window)
                best_window = window

    if best is None:
        return 0.0, "collection has no indexed text", None
    score, doc_id, _ = best
    alignment = fuzz.partial_ratio_alignment(normalized, best_window)
    excerpt = best_window[alignment.dest_start : alignment.dest_end]
    return score, f"{score:.1f}% closest in {doc_id}: {clip(excerpt)}", doc_id


def clip(text: str) -> str:
    """Keep near-miss excerpts short enough to read in a terminal."""
    collapsed = normalize(text)
    return collapsed if len(collapsed) <= CLIP else f"{collapsed[:CLIP]}..."


def load_rows(path: Path) -> list[FileRow]:
    """Read the testset rows that verification needs, preserving unknown metadata."""
    rows: list[FileRow] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"line {line_no}: row must be a JSON object")
        contexts = row.get("reference_contexts") or []
        if not isinstance(contexts, list):
            raise ValueError(f"line {line_no}: reference_contexts must be a list")
        rows.append(
            FileRow(
                id=str(row.get("id") or f"line_{line_no}"),
                user_input=str(row.get("user_input") or ""),
                reference=str(row.get("reference") or ""),
                contexts=[context for context in contexts if isinstance(context, str)],
                metadata=dict(row.get("metadata") or {}),
                line_no=line_no,
            )
        )
    return rows


def verify_rows(
    rows: list[FileRow],
    corpus: Corpus,
    config: VerificationConfig | None = None,
) -> VerificationResult:
    """Check every quote, record matched document ids, and collect row problems."""
    config = config or VerificationConfig()
    checks: list[list[CitationCheck | None]] = []
    counts: Counter[str] = Counter()
    problems: dict[str, list[str]] = {}

    for row_id, times in Counter(row.id for row in rows).items():
        if times > 1:
            problems.setdefault(row_id, []).append(f"id appears {times} times")

    for row in rows:
        row_checks = _check_row(row, corpus, config)
        checks.append(row_checks)
        counts.update(check.status for check in row_checks if check)
        row_problems = _quote_problems(row, row_checks)
        row_problems.extend(_metadata_problems(row, corpus))
        if row_problems:
            problems.setdefault(row.id, []).extend(row_problems)

    return VerificationResult(rows=rows, checks=checks, counts=counts, problems=problems)


def _check_row(
    row: FileRow, corpus: Corpus, config: VerificationConfig
) -> list[CitationCheck | None]:
    """Check each quote; None marks one too short to match on reliably."""
    row_checks: list[CitationCheck | None] = []
    for quote in row.contexts:
        row_checks.append(check_quote(quote, corpus, config) if is_checkable(quote) else None)

    matched = sorted({doc_id for check in row_checks if check for doc_id in check.doc_ids})
    if matched:
        row.metadata["source_doc_ids"] = matched
    if row_checks and all(check and check.status == "match" for check in row_checks):
        row.metadata.pop("reference_context_verified", None)
    else:
        row.metadata["reference_context_verified"] = False
    return row_checks


def _quote_problems(row: FileRow, row_checks: list[CitationCheck | None]) -> list[str]:
    if not row.contexts:
        return ["reference_contexts is empty; nothing to verify"]

    problems: list[str] = []
    for index, (quote, check) in enumerate(zip(row.contexts, row_checks, strict=True)):
        if check is None:
            problems.append(
                f"reference_contexts[{index}] shorter than {MIN_QUOTE_CHARS} chars; not verifiable"
            )
        elif check.status != "match":
            problems.append(
                f"reference_contexts[{index}] {check.status}: {check.detail or clip(quote)}"
            )
    return problems


def _metadata_problems(row: FileRow, corpus: Corpus) -> list[str]:
    problems: list[str] = []
    answerable = row.metadata.get("answerable")
    if answerable is False and row.contexts:
        problems.append("answerable=false but reference_contexts is not empty")
    if answerable is True and not row.contexts:
        problems.append("answerable=true but reference_contexts is empty")
    problems.extend(_source_ref_problems(row, corpus))
    return problems


def _source_ref_problems(row: FileRow, corpus: Corpus) -> list[str]:
    """The generator's own file pointers are a cross-check on its citations."""
    refs = row.metadata.get("source_refs")
    if refs is None:
        return []
    known = {chunk.filename for chunks in corpus.by_doc.values() for chunk in chunks}
    known.discard("")

    problems: list[str] = []
    for index, ref in enumerate(refs):
        if not isinstance(ref, dict):
            problems.append(f"source_refs[{index}] is not an object")
            continue
        filename = str(ref.get("file") or "").strip()
        if not filename:
            problems.append(f"source_refs[{index}] has no file name")
        elif known and filename not in known:
            problems.append(f"source_refs[{index}] names '{filename}', not an indexed file")
    return problems


def row_to_dict(row: FileRow) -> dict[str, Any]:
    return {
        "id": row.id,
        "user_input": row.user_input,
        "reference": row.reference,
        "reference_contexts": list(row.contexts),
        "metadata": row.metadata,
    }
