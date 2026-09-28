# Project Structure

This document explains how the PASsistant repository is organized, what each major folder is responsible for, and where to make common changes.

## Repository Layout

```text
PASsistant/
├── .streamlit/              # Streamlit configuration
├── data/                    # Local data used by ingestion and development
│   └── raw/                 # Source documents before processing
├── docs/                    # Technical and operational documentation
├── scripts/                 # Setup, infrastructure, dataset, and smoke-test scripts
├── src/                     # Application source code
├── tests/                   # Automated tests and evaluation fixtures
│   └── fixtures/            # Reusable JSONL datasets and test data
├── .gitignore               # Files excluded from version control
├── LICENSE                  # MIT license
├── README.md                # Project overview and quick-start commands
├── langgraph.json           # LangGraph deployment configuration
├── pyproject.toml           # Dependencies, CLI commands, and tool configuration
└── uv.lock                  # Locked Python dependency versions
```

The `.venv/`, cache directories, Python bytecode, local secrets, and other generated files are development artifacts. They should not be treated as application source or committed to the repository.

## Application Source: `src/`

The application uses a layered structure. API and user-interface adapters sit at the outside, the LangGraph workflow coordinates a request, and services and utilities provide the underlying business and infrastructure operations.

```text
src/
├── agent.py
├── api/
├── config/
├── eval/
├── frontend/
├── graphs/
├── guardrails/
├── services/
├── telegram_bot/
└── utils/
```

### Entry points

- `src/agent.py` — Main LangGraph application entry point and interactive CLI client. It also exposes the compiled graph used by LangGraph deployment.
- `src/api/__init__.py` — Creates the FastAPI application and registers API routes.
- `src/frontend/app.py` — Streamlit chat interface and administration dashboard.
- `src/frontend/run_frontend.py` — Starts the Streamlit application on port `8501`.
- `src/telegram_bot/polling.py` — Runs the Telegram bot in polling mode for local development.

The corresponding command-line entry points are declared in `pyproject.toml`:

```text
uv run chatbot       # Interactive CLI
uv run frontend      # Streamlit frontend
uv run ragas-eval    # RAGAS evaluation
```

### `src/api/` — HTTP and streaming interfaces

The API layer exposes the application to external clients through FastAPI.

- `routes/chat.py` — Chat, file-upload chat, and streaming chat endpoints.
- `routes/documents.py` — Document upload, listing, and deletion endpoints.
- `routes/health.py` — Health and dependency-status endpoints.
- `routes/telegram.py` — Telegram webhook endpoints.
- `routes/websocket.py` — WebSocket streaming endpoint.
- `routes/router.py` — Route registration and composition.
- `models.py` — Pydantic request and response schemas.
- `services.py` — API-level orchestration around graph execution and streaming.
- `sessions.py` — In-memory conversation/session management.
- `helpers.py` — Shared API helpers.

### `src/config/` — Configuration and logging

- `settings.py` — Environment-backed application settings and defaults.
- `logging.py` — Structured logging configuration.

This is the preferred location for configuration behavior. Avoid reading environment variables directly in individual feature modules when a setting belongs in `settings.py`.

### `src/graphs/` — LangGraph workflow

- `workflow.py` — Defines the graph, nodes, conditional routing, and workflow compilation.

The graph coordinates intent routing, document processing, student-record lookup, retrieval, response generation, error handling, and output safety checks. Changes to the request flow generally belong here and in the node modules under `src/utils/nodes/`.

### `src/services/` — Application business logic

Services contain reusable operations that are independent of a particular transport such as HTTP, Streamlit, or Telegram.

- `contracts.py` — Shared service contracts and interfaces.
- `document_processing.py` — Document processing and ingestion orchestration.
- `indexing.py` — Chunk indexing and vector-store updates.
- `ingestion_health.py` — Ingestion quality and health reporting.
- `intent.py` — User-intent classification support.
- `response_generation.py` — LLM response-generation service.
- `session_registry.py` — Session registry behavior.
- `student_records.py` — Student-record lookup operations.

### `src/guardrails/` — Safety and traffic controls

- `input_guard.py` — Detects unsafe or prompt-injection-style input.
- `output_guard.py` — Filters sensitive information and prevents unsafe output.
- `rate_limit.py` — In-memory request rate limiting.

Guardrails should be applied at the boundaries of the application so all supported interfaces receive consistent protection.

### `src/utils/` — Shared workflow and infrastructure helpers

This package contains the shared state, graph node implementations, tools, cache integration, and vector-search implementation.

- `state.py` — Shared LangGraph state and core data models.
- `cache.py` — Redis cache integration.
- `nodes/` — LangGraph node implementations:
  - `router.py` — Intent-based workflow routing.
  - `retrieval.py` — Query expansion, retrieval, merging, and confidence handling.
  - `response.py` — Response construction.
  - `document_processing.py` — Document-processing node behavior.
  - `student_record.py` — Student-record node behavior.
  - `error_handler.py` — Workflow error handling.
  - `llm.py` — Shared LLM node helpers.
  - `prompts/` — Prompt templates grouped by workflow purpose.
- `tools/` — Tool functions exposed to the workflow:
  - `document.py` — Document-related operations.
  - `ocr.py` — OCR operations.
  - `student.py` — Student-record operations.
  - `vector_store.py` — Vector-store operations.
- `vector_store/` — Qdrant and retrieval implementations:
  - `collection.py` — Collection setup and management.
  - `search.py` — Vector search behavior.
  - `bm25.py` — Sparse/BM25 retrieval.
  - `reranker.py` — Cross-encoder reranking.
  - `tools.py` — Vector-store helper functions.

### `src/frontend/` — Streamlit application

The frontend is intentionally separate from the API. It calls the backend rather than duplicating graph and service logic.

- `app.py` — Chat UI, document-management controls, health information, and dashboard views.
- `run_frontend.py` — Local/server entry point that launches Streamlit.

Streamlit-specific settings are stored in `.streamlit/config.toml`.

### `src/telegram_bot/` — Telegram integration

- `adapter.py` — Telegram-to-application adapter.
- `files.py` — Telegram file and image handling.
- `formatting.py` — Telegram response formatting.
- `polling.py` — Development polling runner.

Production webhook handling is exposed through the FastAPI routes in `src/api/routes/telegram.py`.

### `src/eval/` — RAG evaluation

The `eval/ragas/` package contains the RAGAS evaluation workflow:

- `cli.py` — Command-line argument handling.
- `data.py` — Evaluation dataset loading and preparation.
- `evaluator.py` — Evaluation execution.
- `models.py` — Evaluation-specific models.
- `reporting.py` — Result formatting and reporting.
- `__main__.py` — Module entry point.

Use this package to measure retrieval and answer quality; it is excluded from the default type-checking scope in `pyproject.toml` because it has optional dependencies.

## Documentation: `docs/`

- `api-keys.md` — How to obtain and configure external API credentials.
- `architecture.md` — System architecture, request lifecycle, and design decisions.
- `deployment.md` — Infrastructure, environment configuration, and deployment guidance.
- `evaluation.md` — RAGAS evaluation instructions.
- `retrieval-pipeline.md` — Document indexing and retrieval behavior.
- `project-structure.md` — This repository and source-code structure guide.

Update the relevant document when changing an external interface, deployment process, retrieval strategy, or major architectural boundary.

## Scripts: `scripts/`

Scripts are operational helpers and are not imported as part of the application package.

- `setup.sh` — Full local setup helper.
- `run_qdrant.sh` — Starts Qdrant through Docker.
- `build_ragas_dataset.py` — Builds or validates RAGAS datasets.
- `genereate_eval_datasets.py` — Generates evaluation datasets.
- `run_ragas_eval.py` — Runs evaluation from a script.
- `run_retrieval_smoke_test.py` — Exercises retrieval pipelines.
- `show_ocr_result.py` — Displays OCR output for inspection.
- `bump_version.sh` — Updates the application version.

## Data and fixtures

- `data/raw/` contains source academic documents used for ingestion. Treat these files as input data, not generated application output.
- `tests/fixtures/` contains deterministic JSONL documents and RAGAS datasets used by tests and evaluation runs.
- Qdrant storage and Redis data are external runtime infrastructure. They are not stored in the Python package.

Do not store API keys, tokens, or other secrets in `data/`, fixtures, source files, or documentation. Use the local environment configuration described in `README.md` and `docs/api-keys.md`.

## Tests: `tests/`

Tests are organized around application behavior rather than implementation folders. Examples include:

- API, WebSocket, and Telegram adapter behavior.
- Guardrail and citation sanitization behavior.
- LLM, retrieval, tool, and workflow behavior.
- Logging and production-readiness checks.
- RAGAS evaluation behavior.

Run the full test suite with:

```bash
uv run pytest
```

## Where to make common changes

| Change | Primary location |
|---|---|
| Add or modify an HTTP endpoint | `src/api/routes/` and `src/api/models.py` |
| Change request routing or graph flow | `src/graphs/workflow.py` and `src/utils/nodes/` |
| Change retrieval or ranking | `src/utils/vector_store/`, `src/utils/nodes/retrieval.py`, and `src/services/indexing.py` |
| Change OCR or document ingestion | `src/utils/tools/ocr.py`, `src/utils/tools/document.py`, and `src/services/document_processing.py` |
| Change prompts | `src/utils/nodes/prompts/` |
| Change Streamlit UI | `src/frontend/app.py` and `.streamlit/config.toml` |
| Change Telegram behavior | `src/telegram_bot/` and `src/api/routes/telegram.py` |
| Add or modify configuration | `src/config/settings.py` |
| Add tests or fixtures | `tests/` and `tests/fixtures/` |
| Change dependencies or developer commands | `pyproject.toml`, then refresh `uv.lock` |

## Dependency direction

A useful rule when adding code is to keep dependencies flowing inward:

```text
External clients
    ↓
API / Streamlit / Telegram adapters
    ↓
LangGraph workflow and nodes
    ↓
Services and shared state
    ↓
Tools, vector-store helpers, and external infrastructure
```

For example, a new chat behavior should not be implemented only in the Streamlit page. Put the behavior in the workflow or service layer, then keep each interface responsible for translating its own input and output format.
