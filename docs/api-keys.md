# Obtaining External API Credentials

This document provides step-by-step instructions on how to acquire and configure the external API keys required to run the PASsistant chatbot.

---

## 1. OpenRouter API Key (Primary LLM Provider)

- **Purpose**: Power the core chatbot workflow, response generation, query expansion/reformulation, and final validation checks.
- **How to obtain**:
  1. Go to [OpenRouter](https://openrouter.ai/).
  2. Click **Login** / **Sign up** in the top right.
  3. Go to **Console / Settings** -> **API Keys**.
  4. Click **Create Key**, give it a name (e.g. `passistant-key`), and click **Create**.
  5. Copy the generated key immediately.
  6. In your `.env` file, paste the key and base URL:
     ```env
     OPENAI_API_KEY=your_copied_key_here
     OPENAI_BASE_URL=https://openrouter.ai/api/v1
     ```

---

## 2. Zhipu AI API Key (GLM-4 OCR PDF Ingestion)

- **Purpose**: Powers layout-aware, page-by-page OCR extraction (retaining tabular format structures) when documents are uploaded.
- **How to obtain**:
  1. Go to [Zhipu AI Open Platform (Zhipu AI)](https://z.ai/).
  2. Register or log in to your account.
  3. Go to the dashboard console and click on **API Keys** / **API Key Management**.
  4. Copy your unique API Key.
  5. In your `.env` file, set:
     ```env
     ZHIPU_API_KEY=your_copied_key_here
     ZHIPU_BASE_URL=https://open.bigmodel.cn/api/paas/v4
     ```

---

## 3. Reranker API Keys (OpenRouter Primary and Jina AI Fallback - Optional)

- **Purpose**: Re-ranks the retrieved raw document chunks to surface the most relevant contexts for the final answer.
- **Primary Provider (OpenRouter)**:
  - The reranker defaults to OpenRouter (`cohere/rerank-v3.5`) via your existing `OPENAI_API_KEY` and `OPENAI_BASE_URL`.
  - No additional key is needed if you already configured OpenRouter in section 1.
  - You can optionally set `RERANKER_MODEL`, `RERANKER_BASE_URL`, or `OPENROUTER_API_KEY`.
- **Fallback Provider (Jina AI)**:
  - If OpenRouter fails or is unavailable, the system automatically falls back to Jina AI.
  - How to obtain Jina API key:
    1. Visit [Jina AI Reranker](https://jina.ai/reranker/).
    2. Create an account or log in.
    3. Navigate to **API Console** and create a new access token.
    4. In your `.env` file, set:
       ```env
       JINA_API_KEY=your_copied_key_here
       ```
- **Local Fallback**: If neither remote provider is configured or available, the system can fall back to local **FastEmbed** cross-encoders on CPU.

---

## 4. Telegram Bot Token (Telegram client - Optional)

- **Purpose**: Connects the chatbot workflow to a Telegram bot client for direct messaging and photo uploads.
- **How to obtain**:
  1. Open Telegram and search for [@BotFather](https://t.me/BotFather) (ensure it has the official verification badge).
  2. Send `/newbot` to start the creation process.
  3. Choose a name and a username for your bot.
  4. BotFather will output the **HTTP API token**. Copy it.
  5. In your `.env` file, set:
     ```env
     TELEGRAM_BOT_TOKEN=your_copied_token_here
     TELEGRAM_ENABLED=true
     ```

---

## 5. LangSmith API Key (Observability & Tracing - Optional)

- **Purpose**: Traces and visualizes the internal states of the LangGraph chatbot, showing detailed run execution times, search queries, and prompt costs.
- **How to obtain**:
  1. Go to [LangSmith](https://smith.langchain.com/).
  2. Sign up or log in.
  3. Click the **Settings** (gear icon) in the bottom-left corner of the sidebar.
  4. Under **API Keys**, click **Create API Key**.
  5. Copy the generated key.
  6. In your `.env` file, set:
     ```env
     LANGSMITH_API_KEY=your_copied_key_here
     LANGSMITH_TRACING=true
     ```
