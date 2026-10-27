# API Reference

PASsistant exposes a FastAPI HTTP API at the root path (there is no `/api`
version prefix). When the server is running locally, the interactive Swagger UI
is at `http://localhost:8000/docs`, ReDoc is at `http://localhost:8000/redoc`,
and the OpenAPI schema is at `http://localhost:8000/openapi.json`.

Start the server with:

```bash
uv run uvicorn src.api:app --reload --port 8000
```

## Conventions and access

- Chat endpoints accept `multipart/form-data`, including when no files are
  attached. Send `message`, and optionally `thread_id` and one or more `files`.
- Other request and response bodies are JSON unless stated otherwise.
- A `thread_id` may be omitted to create a chat session; reuse the returned ID
  to continue that conversation. Sessions and streaming replay state are held
  in memory.
- Reindex endpoints require the `X-Admin-Key` header to match
  `ADMIN_API_KEY`. They return `403` when the key is missing, invalid, or not
  configured.
- Telegram webhook requests require Telegram support to be enabled and the
  `X-Telegram-Bot-Api-Secret-Token` header to match
  `TELEGRAM_WEBHOOK_SECRET_TOKEN`.
- Chat requests and reindex endpoints are rate-limited by client IP. The
  default is 20 requests per minute (`RATE_LIMIT_PER_MINUTE`). WebSocket
  messages are checked against the same limiter.
- Most errors use FastAPI's standard JSON shape: `{"detail":"..."}`. Chat
  guardrail rejections return `400`; rate limits return `429`.
- The current CORS middleware allows `GET`, `POST`, and `OPTIONS`, with
  `Content-Type` and `Authorization` headers. Cross-origin browser clients
  using `DELETE` or `X-Admin-Key` need the deployment's CORS policy updated.

## Endpoint summary

| Method | Path | Description | Access |
| --- | --- | --- | --- |
| `GET` | `/health` | API and dependency health | Public |
| `POST` | `/chat` | Chat request; optional files are used for this turn only | Rate-limited |
| `POST` | `/chat/stream` | Chat response as Server-Sent Events (SSE) | Rate-limited |
| `POST` | `/upload` | OCR and ingest PDF/image files into the shared index | Public |
| `GET` | `/documents` | List indexed documents | Public |
| `DELETE` | `/documents/by-filename/{filename}` | Delete indexed documents by filename | Public |
| `POST` | `/documents/{document_id}/reindex` | Reindex one cached document | Admin key |
| `POST` | `/documents/reindex/by-filename/{filename}` | Reindex cached documents by filename | Admin key |
| `POST` | `/documents/reindex-jobs` | Start a bulk reindex job | Admin key |
| `GET` | `/documents/reindex-jobs/{job_id}` | Get bulk job status | Admin key |
| `DELETE` | `/documents/reindex-jobs/{job_id}` | Request bulk job cancellation | Admin key |
| `POST` | `/telegram/webhook` | Receive a Telegram update (when enabled) | Telegram secret |
| `WS` | `/ws/{thread_id}` | Stream chat events over WebSocket | Rate-limited |

## Chat

### `POST /chat`

Send a message and receive the complete answer. Files attached to this request
are temporary context for this turn; they are **not** added to the shared
retrieval index. Use `POST /upload` to ingest documents into that index.

Form fields:

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `message` | string | Yes | User's message |
| `thread_id` | string | No | Existing conversation ID; omit to start a session |
| `files` | file, repeatable | No | Context files for this turn |

Example:

```bash
curl -X POST http://localhost:8000/chat \
  -F 'message=What are the requirements for graduation?' \
  -F 'thread_id=my-thread-id'
```

To send files with the same endpoint:

```bash
curl -X POST http://localhost:8000/chat \
  -F 'message=Summarize this document' \
  -F 'files=@handbook.pdf'
```

A successful response is a `ChatResponse`:

```json
{
  "id": "message-uuid",
  "response": "The answer text...",
  "thread_id": "my-thread-id",
  "intent": "query_document",
  "documents_processed": 0,
  "citations": []
}
```

`citations` contains source details when documents support the answer. Citation
items may include `id`, `document_id`, `filename`, `title`, `section`, `page`,
`source_locations`, `score`, `chunk_id`, `snippet`, `formatted`, and `is_cited`.
`formatted` is the ready-to-display IEEE-style reference string for that citation.

Common responses include `400` when the input guard rejects a message, `422`
for invalid form data, and `429` when the rate limit is exceeded.

### `POST /chat/stream`

Uses the same multipart fields as `/chat`, but returns `text/event-stream`.
Each SSE frame has an `id`, an `event` name, and a JSON `data` field. The JSON
payload is a `ChatStreamEvent`:

```json
{
  "event_id": "run-uuid:1",
  "event_type": "run.started",
  "thread_id": "my-thread-id",
  "run_id": "run-uuid",
  "timestamp": "2026-01-01T12:00:00Z",
  "sequence": 1,
  "data": {
    "message": "user request accepted",
    "resume_supported": true
  }
}
```

Event names and their `data` fields:

| Event | Data fields |
| --- | --- |
| `run.started` | `message`, `resume_supported` |
| `run.status` | `stage`, `label`, `meta` |
| `message.delta` | `text`, `index` |
| `run.completed` | `response`, `intent`, `documents_processed`, `citations` |
| `run.failed` | `code`, `message`, `retryable` |

Example request:

```bash
curl -N -X POST http://localhost:8000/chat/stream \
  -F 'message=Summarize the academic calendar' \
  -F 'thread_id=my-thread-id'
```

To resume an in-memory run, send the last received event ID using either the
`Last-Event-ID` header or the `last_event_id` query parameter. Event IDs have
the form `<run_id>:<sequence>`. Replay state is process-local and is lost when
the API process restarts.

## Documents

### `POST /upload`

Upload one or more files as repeated `files` form fields. Supported formats are
PDF, PNG, JPEG, GIF, and WEBP. The per-file limit is 50 MiB for PDFs and 10 MiB
for images. The endpoint returns `201 Created` with one result per file; a
processing failure is reported in that file's result and does not stop the
other uploads.

```bash
curl -X POST http://localhost:8000/upload \
  -F 'files=@handbook.pdf' \
  -F 'files=@calendar.png'
```

Each result includes `success`, `document_id`, `filename`, `document_type`, and
`status`. It may also include `document_title`, `parsed_pages`, `failed_pages`,
`ocr_warnings`, `quality_warning`, or `error`. Unsupported media types return
`415`; files that exceed their limits return `413`.

### `GET /documents`

Return a list of indexed documents:

```json
[
  {
    "document_id": "document-uuid",
    "filename": "handbook.pdf",
    "chunks": 42,
    "doc_title": "Academic Handbook"
  }
]
```

### `DELETE /documents/by-filename/{filename}`

Delete indexed content matching the filename. URL-encode special characters
in the filename. By default, deletion also removes the OCR cache and original
file. Set `keep_files=true` to remove only the vectors and search cache.

```bash
curl -X DELETE \
  'http://localhost:8000/documents/by-filename/handbook.pdf?keep_files=true'
```

Returns `success`, `document_id`, and `filename`. Returns `404` when no matching
document exists and `409` if the document is currently locked.

## Reindexing (admin)

All reindex endpoints require `X-Admin-Key` and share the configured rate
limit. Reindexing rebuilds the index from OCR cache without running OCR again.

### `POST /documents/{document_id}/reindex`

Optional query parameter: `rebuild_text` (default `false`), which re-derives
page text from cached layout blocks before indexing.

```bash
curl -X POST \
  'http://localhost:8000/documents/document-uuid/reindex?rebuild_text=false' \
  -H 'X-Admin-Key: YOUR_ADMIN_API_KEY'
```

Returns `document_id`, `filename`, `status`, and `chunks`. Returns `404` when
OCR cache is missing, `409` when the document is locked, and `500` if
reindexing fails.

### `POST /documents/reindex/by-filename/{filename}`

Reindex every cached document matching the URL-encoded filename. Accepts the
same optional `rebuild_text` query parameter and returns a list of per-document
results (`document_id`, `filename`, `status`, `chunks`, and optional `error`).

### Bulk reindex jobs

`POST /documents/reindex-jobs` accepts a JSON body and returns `202 Accepted`.
`document_ids: null` selects every document with an OCR cache.

```json
{
  "rebuild_text": false,
  "document_ids": ["document-uuid"]
}
```

Response:

```json
{
  "job_id": "job-uuid",
  "status": "queued",
  "total": 1
}
```

The response includes a `Location` header pointing to the job status endpoint.
Use `GET /documents/reindex-jobs/{job_id}` to read progress, including
`processed`, `succeeded`, `skipped_no_cache`, `failed`, and per-document
`results`. `DELETE` on the same path requests cancellation; the active document
is allowed to finish before the job stops. Unknown jobs return `404`; a job
that cannot be cancelled returns `409`.

## WebSocket chat

Connect to `ws://localhost:8000/ws/{thread_id}`. Send either a plain text
message or a JSON object with `message` and optional `last_event_id`:

```json
{
  "message": "What does the handbook say about exams?",
  "last_event_id": null
}
```

The server sends JSON-encoded `ChatStreamEvent` objects using the same event
names as `/chat/stream`. WebSocket chat currently accepts text only (no file
attachments). Rate-limit or input-guard rejections are returned as a
`run.failed` event.

## Health check

`GET /health` returns `status`, `version`, `environment`, and `redis` and
`qdrant` dependency statuses. The overall status is `degraded` when either
configured dependency is unhealthy; an unconfigured Redis dependency is
reported as `disabled`.

## Telegram webhook

When Telegram support is enabled, Telegram sends its update JSON to
`POST /telegram/webhook` with the configured secret header. The endpoint
acknowledges a valid update with `{"ok": true}` and processes it in a
background task. A disabled integration returns `404`; an unset webhook secret
returns `503`; an invalid secret returns `403`.
