# Retrieval Pipeline

## Overview

The retrieval pipeline transforms a user question into relevant document context through multiple stages: query processing, vector search, result ranking, and confidence scoring.

## Pipeline Stages

```mermaid
flowchart TD
    Q[User Question] --> QP[1. Query Processing]
    QP -->|"[q1, q2, q3]"| PS[2. Parallel Search]
    PS -->|candidates| RR[3. Reranking]
    RR -->|top-k| CS[4. Confidence Scoring]
    CS --> OUT[Retrieved Context]

    subgraph QP[1. Query Processing]
        direction LR
        LLM_R[LLM Rewrite]
        SYN[Synonym Expand]
        SEM[Semester Normalize]
    end

    subgraph PS[2. Parallel Search]
        direction LR
        E1[Embed + Dense]
        B1[BM25 Sparse]
        RRF[RRF Fusion]
    end
```

## 1. Query Processing

### Multi-Query Generation

For `query_document` intent, up to 3 query variants are generated:

| Variant | Source | Example |
|---------|--------|---------|
| Original | User input | "Kak, kalau aku telat bayar DPP, bisa ikut perwalian?" |
| Rewritten | LLM compression | "syarat perwalian pembayaran DPP" |
| Expanded | Synonym + normalization | "syarat perwalian pembayaran DPP" (with semester normalization) |

### Semester Numeral Normalization

Documents use Roman numerals ("Semester V") while users often type Arabic ("semester 5"). The query expander converts between formats:

- "semester 5" → "semester V" (for matching document headings)
- "semester V" → "semester 5" (for matching informal text)

### Synonym Expansion

Domain-specific Indonesian academic synonyms:
- "absen" → "ketidakhadiran"
- "tidak aktif" → "nonaktif"
- "cuti" → "cuti akademik"
- "drop out" → "dikeluarkan"

## 2. Ingestion Pipeline

### Document Processing Flow

```mermaid
flowchart LR
    PDF[PDF Upload] --> OCR[GLM-OCR]
    OCR -->|per-page| Text[Extracted Text]
    Text --> Split[RecursiveCharacterTextSplitter]
    Split -->|chunks| Embed[Batch Embed]
    Embed --> Qdrant[(Qdrant)]
```

### OCR Cache (Reindex Without OCR)

Raw per-page OCR output is persisted at ingestion time so chunking, table
handling, and indexing can change and documents can be re-indexed without
paying for OCR again. Each document gets one directory:

    data/processed/ocr_cache/{document_id}/pages.json
    data/processed/ocr_cache/{document_id}/meta.json

- `pages.json` stores each page's layout blocks exactly as the OCR provider
  returned them (raw table HTML, labels, and bounding boxes) plus the final
  per-page extracted text, sorted by page index. Failed pages are kept with
  their status and error so page order survives the batch/retry paths.
  Markdown table conversion is lossy (rowspan/colspan are flattened), so
  `layout_details` is the source of truth.
- `meta.json` stores document and OCR metadata (filename, document type,
  number of pages, raw file name, raw file SHA-256, OCR provider/model, page
  status snapshot) and is written **last**: its presence is what marks a cache
  entry as complete. Both files are written atomically (temp file + rename).

The cache is written only by the knowledge-base ingestion path (`POST /upload`).
A cache write failure is logged and never fails ingestion. Set
`OCR_CACHE_ENABLED=false` to disable it. See
[Reindexing from Cache](#7-reindexing-from-cache) below.

### Chunked Indexing with RecursiveCharacterTextSplitter

Each ingested document is split into overlapping chunks using LangChain's
`RecursiveCharacterTextSplitter`. The splitter tries separators in order
(`\n\n`, `\n`, `. `, ` `, `""`) to find natural break points.

| Parameter | Default | Env Var |
|-----------|---------|---------|
| Chunk size | 1000 chars | `CHUNK_SIZE` |
| Chunk overlap | 200 chars | `CHUNK_OVERLAP` |
| Separators | `["\n\n", "\n", ". ", " ", ""]` | `CHUNK_SEPARATORS` |

Each chunk carries:
- `chunk_id` — `{document_id}:chunk:{index}`
- `text` — the chunk content
- `document_id`, `filename`, `doc_title`, `document_type`
- `chunk_type` — `"chunk"`
- `chunk_index` — zero-based position in the document
- `source_locations` — list of `{"page": N}` objects (1-based page numbers
  derived from OCR page boundaries)

### Page Number Tracking

Page numbers are reconstructed from per-page OCR text lengths stored in
`DocumentUpload.ocr_page_status`. Each chunk's character span is mapped
against page boundaries so that chunks spanning multiple pages carry all
relevant page numbers.

### Embedding and Indexing

Chunks are embedded in batches (32 at a time by default) using the
configured embedding model. When hybrid retrieval (`rrf` or `reranker`)
is active, a BM25 sparse vector is also generated for each chunk.
All vectors and metadata are upserted into Qdrant as individual points.

### Cache Invalidation

On each document indexing:
1. **Blanket invalidation** — all cached search results are cleared
   (`delete_prefix("search:")`).
2. **Per-document invalidation** — any cache entries scoped to the
   specific document ID are also removed (`delete_prefix("doc:{id}:")`).

The same invalidation runs when a document is deleted
(`DELETE /documents/by-filename/{filename}`), which also removes the OCR cache
directory and the raw file. Pass `?keep_files=true` to keep the cache and raw
file while still deleting vectors and cached search results.

## 3. Search Strategies

### Pure Similarity (`RETRIEVAL_STRATEGY=similarity`)

- Dense cosine similarity only
- `score_threshold=0.2` minimum
- Candidate multiplier: 3x top_k

### RRF Hybrid (`RETRIEVAL_STRATEGY=rrf`)

- Dense + BM25 sparse vectors searched independently
- Fused with Reciprocal Rank Fusion (k=60)
- No absolute score threshold (relative filtering at 50% of top score)
- BM25 uses `Qdrant/bm25` model with stemmer enabled

### Reranker (`RETRIEVAL_STRATEGY=reranker`)

- First stage: same as RRF or similarity (wider candidate pool, 6x multiplier)
- Second stage: cross-encoder reranker scores each candidate (OpenRouter provider primary, with automatic Jina AI fallback)
- Final threshold: score > 0.0

### Score Fields

- `retrieval_score` is the first-stage vector similarity or RRF score.
- `reranker_score` is present only when reranker mode scores the chunk.
- `final_score` is the final confidence score computed by the Confidence Scoring stage.

## 4. Confidence Scoring

A custom 5-component model evaluates the final retrieved context to assign a confidence level (High, Medium, Low) and determine if an honest fallback is required:
1. **Score Magnitude**: Checks if the top chunk's final score passes a baseline threshold.
2. **Margin**: Compares the top chunk to the rest to ensure it strongly stands out.
3. **Overlap**: Checks if semantic overlap across top results confirms the answer.
4. **Strong Hits**: Counts how many chunks cross a "strong" relevance threshold.
5. **Explicit Policy Rules**: Detects if the query relates to rigid academic rules and ensures policy documents are retrieved.

## 5. Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `RETRIEVAL_TOP_K` | 10 | Final number of retrieval records returned |
| `RETRIEVAL_STRATEGY` | reranker | Ranking strategy |
| `RERANKER_CANDIDATE_MULTIPLIER` | 6 | First-stage overfetch ratio |
| `CHUNK_SIZE` | 1000 | Target character length per chunk |
| `CHUNK_OVERLAP` | 200 | Character overlap between consecutive chunks |
| `OCR_CACHE_ENABLED` | true | Persist raw per-page OCR output for reindexing |
| `OCR_CACHE_DIR` | `DATA_DIR/processed/ocr_cache` | OCR cache location |
| `REINDEX_JOB_TTL_SECONDS` | 86400 | TTL for async bulk reindex job records |
| `ADMIN_API_KEY` | unset | `X-Admin-Key` value required by reindex endpoints |

## 6. Caching

Search results are cached in Redis with key based on:
- Query text
- Document type filter
- Top-k and score threshold
- Retrieval strategy and reranker config
- BM25 settings

Cache is automatically invalidated on document ingestion or deletion (`delete_prefix("search:")`).
Per-document cache entries are also cleared on re-ingestion (`delete_prefix("doc:{id}:")`).


## 7. Reindexing from Cache

Reindexing rebuilds chunks and vectors from `pages.json` and never calls OCR.
It reuses the ingestion helper `_build_ocr_result_from_pages`, so the page
order, the `"\n\n"` page join, and page-number offsets are identical to a fresh
ingest. Old points for the document are deleted right before the new upsert, so
reindexing does not create duplicates. Documents ingested before OCR caching
existed have no cache and must be re-uploaded.

By default the cached per-page text is used. With `rebuild_text` (API) or
`--rebuild-text` (CLI), each page's text is re-derived from its cached layout
blocks with the current table conversion; pages that yield no block text fall
back to their cached text.

The CLI and the API share one service (`src/services/reindex.py`):

    uv run python scripts/reindex_from_ocr_cache.py --document-id <id>
    uv run python scripts/reindex_from_ocr_cache.py --filename "handbook.pdf"
    uv run python scripts/reindex_from_ocr_cache.py --all --rebuild-text

API endpoints (all require `X-Admin-Key` and the chat rate limit):

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/documents/{document_id}/reindex` | Reindex one document (404 without cache, 409 when locked) |
| POST | `/documents/reindex/by-filename/{filename}` | Reindex every cached document with that filename |
| POST | `/documents/reindex-jobs` | Queue a bulk job (202 + `Location`; 409 when one is active) |
| GET | `/documents/reindex-jobs/{job_id}` | Job progress and per-document results |
| DELETE | `/documents/reindex-jobs/{job_id}` | Request cancellation |

Bulk jobs run sequentially in a single process. The job record is stored in
Redis under `reindex_job:{job_id}` with a TTL; a queued/running job whose
heartbeat is stale is reported as `failed` with error `interrupted`.
