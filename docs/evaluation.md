# Evaluation Guide

Panduan lengkap untuk menjalankan evaluasi RAG pipeline PASsistant menggunakan framework [RAGAS](https://docs.ragas.io/).

---

## Daftar Isi

- [Overview](#overview)
- [Prasyarat](#prasyarat)
- [Instalasi](#instalasi)
- [Menjalankan Evaluasi](#menjalankan-evaluasi)
  - [Cara 1 — Script (Recommended)](#cara-1--script-recommended)
  - [Cara 2 — Module Langsung](#cara-2--module-langsung)
- [Mode Evaluasi](#mode-evaluasi)
  - [Fixture Mode](#fixture-mode)
  - [Live Mode](#live-mode)
- [Metrics](#metrics)
- [Membandingkan Retrieval Strategy](#membandingkan-retrieval-strategy)
- [Dataset](#dataset)
- [Laporan Hasil](#laporan-hasil)
- [Troubleshooting](#troubleshooting)

---

## Overview

Evaluasi mengukur kualitas seluruh pipeline RAG — dari retrieval hingga response generation — menggunakan RAGAS sebagai LLM-as-judge. Ada dua entry point:

| Entry Point | Output Nama File | Cocok Untuk |
|---|---|---|
| `scripts/run_ragas_eval.py` | `YYYYMMDD_HHMMSS_<dataset>_ragas_report.json` | Eksperimen, perbandingan antar run |
| `uv run ragas-eval` | Manual via `--output` | Skrip otomatis / CI |

---

## Prasyarat

- Python 3.11+
- [UV](https://docs.astral.sh/uv/getting-started/installation/) terinstal
- File `.env` sudah dikonfigurasi (minimal `OPENAI_API_KEY`, `OPENAI_BASE_URL`)
- Untuk **live mode**: Qdrant berjalan dan dokumen sudah diindeks

---

## Instalasi

Install dependensi eval (terpisah dari dependensi utama):

```bash
uv sync --extra eval
```

Verifikasi instalasi:

```bash
uv run python -c "import ragas; print(ragas.__version__)"
```

> **Jika `uv sync` gagal** karena error `Access is denied` pada file `.exe`:
> ada proses lain yang mengunci file tersebut (biasanya Streamlit frontend masih berjalan).
> Hentikan proses tersebut dulu, lalu ulangi `uv sync --extra eval`.

---

## Menjalankan Evaluasi

### Cara 1 — Script (Recommended)

Script `scripts/run_ragas_eval.py` secara otomatis memberi nama output file dengan format
`YYYYMMDD_HHMMSS_<dataset-stem>_ragas_report.json` di folder `src/eval/reports/`.

**Sintaks dasar:**

```bash
python scripts/run_ragas_eval.py <path-ke-dataset>
```

**Contoh:**

```bash
# Fixture mode (default) — cepat, tidak perlu pipeline berjalan
python scripts/run_ragas_eval.py tests/fixtures/140726_kurikulum_01.jsonl

# Live mode — menjalankan pipeline RAG secara penuh
python scripts/run_ragas_eval.py tests/fixtures/140726_kurikulum_01.jsonl --mode live

# Extended metrics (8 metrik)
python scripts/run_ragas_eval.py tests/fixtures/140726_kurikulum_01.jsonl --metrics-tier extended

# Ganti direktori output
python scripts/run_ragas_eval.py tests/fixtures/140726_kurikulum_01.jsonl --output-dir my_reports
```

**Output yang dihasilkan:**

```
src/eval/reports/20260715_103042_140726_kurikulum_01_ragas_report.json
```

**Semua flag yang tersedia:**

| Flag | Default | Keterangan |
|---|---|---|
| `dataset` | *(wajib)* | Path ke file JSONL |
| `--mode` | `fixture` | `fixture` atau `live` |
| `--metrics-tier` | `core` | `core` (4 metrik) atau `extended` (8 metrik) |
| `--k-eval` | `5` | Jumlah chunk yang diambil saat live mode |
| `--output-dir` | `src/eval/reports` | Direktori output laporan |
| `--fixture` | *(tidak ada)* | Path ke pre-computed responses JSON (fixture mode) |
| `--openrouter-max-retries` | `6` | Maks retry untuk error transient OpenRouter |
| `--openrouter-min-interval-seconds` | `0.3` | Jeda minimum antar request ke OpenRouter |
| `--openrouter-backoff-base-seconds` | `2.0` | Base backoff eksponensial untuk retry |

> **Ctrl+C:** Script menggunakan `os._exit(1)` — menekan Ctrl+C akan langsung menghentikan
> proses termasuk semua thread yang sedang blocking. Tidak perlu force-kill dari Task Manager.

---

### Cara 2 — Module Langsung

Gunakan ini jika butuh kontrol penuh atas nama file output:

```bash
uv run python -m src.eval.ragas \
    --dataset tests/fixtures/140726_kurikulum_01.jsonl \
    --mode fixture \
    --metrics-tier core \
    --output src/eval/reports/my_custom_report.json
```

---

## Mode Evaluasi

### Fixture Mode

Menggunakan jawaban yang sudah ada di kolom `answer` dan konteks di kolom `contexts` dalam dataset JSONL.
**Tidak** memanggil pipeline RAG secara langsung.

- ✅ Cepat
- ✅ Tidak butuh Qdrant / pipeline berjalan
- ✅ Reproducible — hasil konsisten di setiap run
- ❌ Tidak mengukur kualitas retrieval terkini

```bash
python scripts/run_ragas_eval.py tests/fixtures/140726_kurikulum_01.jsonl --mode fixture
```

Jika punya file responses terpisah (pre-computed dari run sebelumnya):

```bash
python scripts/run_ragas_eval.py tests/fixtures/140726_kurikulum_01.jsonl \
    --mode fixture \
    --fixture path/to/responses.json
```

Format file fixture:

```json
{
  "ragas_q1": {
    "response": "Jawaban yang sudah digenerate...",
    "retrieved_contexts": ["Teks chunk 1...", "Teks chunk 2..."]
  }
}
```

---

### Live Mode

Menjalankan seluruh pipeline RAG untuk setiap sampel: retrieval dari Qdrant → response generation via LLM.
Hasil kemudian di-scoring oleh RAGAS.

- ✅ Mengukur performa pipeline yang sebenarnya
- ✅ Berguna untuk membandingkan konfigurasi retrieval
- ❌ Butuh Qdrant berjalan dan dokumen sudah diindeks
- ❌ Lebih lambat dan ada biaya API

**Prasyarat live mode:**

```bash
# Pastikan Qdrant berjalan
bash scripts/run_qdrant.sh

# Pastikan dokumen sudah diindeks
curl http://localhost:6333/collections
```

```bash
python scripts/run_ragas_eval.py tests/fixtures/140726_kurikulum_01.jsonl \
    --mode live \
    --k-eval 5
```

---

## Metrics

### Tier Core (default)

Selalu dijalankan. Mengukur kualitas inti pipeline RAG.

| Metrik | Yang Diukur | Rentang |
|---|---|---|
| **Faithfulness** | Apakah jawaban berdasarkan konteks yang diambil? (deteksi halusinasi) | 0–1 |
| **Answer Relevancy** | Apakah jawaban relevan dan fokus ke pertanyaan? | 0–1 |
| **Context Precision** | Signal-to-noise ratio — apakah dokumen relevan di-ranking tertinggi? | 0–1 |
| **Context Recall** | Apakah retrieval menemukan semua informasi yang dibutuhkan? | 0–1 |

### Tier Extended (opsional)

Tambahan 4 metrik di atas metrik core.

| Metrik | Yang Diukur |
|---|---|
| **Factual Correctness** | Akurasi faktual dibanding ground truth |
| **Semantic Similarity** | Kemiripan embedding antara jawaban dan referensi |
| **Context Entity Recall** | Apakah entitas dari ground truth ada di konteks? |
| **Noise Sensitivity** | Ketahanan terhadap chunk yang tidak relevan |

```bash
python scripts/run_ragas_eval.py tests/fixtures/140726_kurikulum_01.jsonl \
    --metrics-tier extended
```

---

## Membandingkan Retrieval Strategy

Untuk membandingkan `similarity` vs `rrf` vs `reranker`, set environment variable
`RETRIEVAL_STRATEGY` sebelum menjalankan evaluasi. Laporan secara otomatis merekam strategi
yang digunakan di field `config.retrieval_strategy`.

**CMD:**

```cmd
set RETRIEVAL_STRATEGY=similarity
python scripts/run_ragas_eval.py tests/fixtures/140726_kurikulum_01.jsonl --mode live

set RETRIEVAL_STRATEGY=rrf
python scripts/run_ragas_eval.py tests/fixtures/140726_kurikulum_01.jsonl --mode live
```

**PowerShell:**

```powershell
$env:RETRIEVAL_STRATEGY = "similarity"
python scripts/run_ragas_eval.py tests/fixtures/140726_kurikulum_01.jsonl --mode live

$env:RETRIEVAL_STRATEGY = "rrf"
python scripts/run_ragas_eval.py tests/fixtures/140726_kurikulum_01.jsonl --mode live
```

Hasilnya dua file dengan timestamp berbeda di `src/eval/reports/`. Bandingkan
`aggregate_scores` dari keduanya untuk melihat pengaruh strategi retrieval.

---

## Dataset

### Dataset yang Tersedia

| File | Deskripsi |
|---|---|
| `tests/fixtures/140726_kurikulum_01.jsonl` | Dataset kurikulum TI Unpas |
| `tests/fixtures/ragas_dataset_kurikulum_if_unpas.jsonl` | Dataset kurikulum IF Unpas |
| `tests/fixtures/ragas_dataset_pedoman_kemahasiswaan.jsonl` | Dataset pedoman kemahasiswaan |
| `tests/fixtures/ragas_dataset_15.jsonl` | Dataset campuran 15 sampel |
| `tests/fixtures/ragas_dataset.jsonl` | Dataset utama |

### Schema JSONL

Setiap baris adalah satu JSON object:

```json
{
  "id": "ragas_q1",
  "question": "Min, total SKS yang harus ditempuh buat lulus S1 Teknik Informatika Unpas berapa sih?",
  "contexts": [
    { "text": "Total 144", "is_relevant": true },
    { "text": "Jumlah Beban Studi Semester I 19 0 0 19", "is_relevant": true }
  ],
  "answer": "Total SKS yang harus ditempuh adalah 144 SKS.",
  "ground_truth": "Total beban studi untuk kurikulum Teknik Informatika Universitas Pasundan adalah 144 SKS.",
  "metadata": {
    "source_file": "Kurikulum Teknik Informatika Unpas 2021.pdf",
    "difficulty": "easy",
    "reasoning_type": "single-hop",
    "noise_level": "low"
  }
}
```

| Field | Tipe | Wajib | Keterangan |
|---|---|---|---|
| `id` | `str` | ✅ | Identifier unik per sampel |
| `question` | `str` | ✅ | Pertanyaan user |
| `ground_truth` | `str` | ✅ | Jawaban referensi yang benar |
| `answer` | `str` | ✅ untuk fixture | Jawaban kandidat yang akan di-scoring |
| `contexts` | `list[object]` | ✅ untuk fixture | Chunk teks dengan flag relevansi |
| `metadata.source_file` | `str` | ✅ | Nama file sumber |
| `metadata.difficulty` | `easy\|medium\|hard` | ✅ | Tingkat kesulitan |
| `metadata.reasoning_type` | `single-hop\|multi-hop\|comparison\|inference` | ✅ | Jenis reasoning |
| `metadata.noise_level` | `low\|medium\|high` | ✅ | Level noise konteks |

### Validasi Dataset

Selalu validasi dataset sebelum dijalankan:

```bash
python scripts/build_ragas_dataset.py \
    --validate tests/fixtures/140726_kurikulum_01.jsonl \
    --stats
```

---

## Laporan Hasil

Laporan disimpan sebagai JSON di `src/eval/reports/`. Contoh isi laporan:

```json
{
  "evaluation_type": "ragas_rag_eval",
  "evaluation_date": "2026-07-15",
  "dataset_id": "140726_kurikulum_01.jsonl",
  "sample_count": 15,
  "metrics_tier": "core",
  "aggregate_scores": {
    "faithfulness": 0.8421,
    "answer_relevancy": 0.9103,
    "context_precision": 0.7654,
    "context_recall": 0.8012
  },
  "per_sample": [
    {
      "id": "ragas_q1",
      "question": "...",
      "scores": { "faithfulness": 1.0, "answer_relevancy": 0.95 },
      "metric_errors": {},
      "response_preview": "Total SKS yang harus ditempuh adalah 144 SKS.",
      "retrieved_context_count": 3,
      "metadata": { "difficulty": "easy", "reasoning_type": "single-hop" }
    }
  ],
  "openrouter_usage": {
    "request_count": 60,
    "prompt_tokens": 45230,
    "completion_tokens": 3120,
    "cost_usd_reported": 0.0312
  },
  "config": {
    "evaluator_llm": "deepseek/deepseek-chat",
    "k_eval": 5,
    "retrieval_strategy": "rrf",
    "mode": "fixture"
  }
}
```

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'ragas'`

Dependensi eval belum diinstal. Jalankan:

```bash
uv sync --extra eval
```

Lalu gunakan `uv run python` bukan `python` langsung, atau pastikan venv aktif.

---

### Proses hang / tidak ada progress lebih dari 5 menit

Evaluasi mungkin stuck menunggu response OpenRouter yang tidak kunjung datang.

**Hentikan proses:**

```cmd
# Temukan PID
wmic process where "name='python.exe'" get processid,commandline

# Kill by PID
taskkill /F /PID <pid>

# Atau kill semua python (hati-hati jika ada proses lain)
taskkill /F /IM python.exe
```

> Jika menggunakan `scripts/run_ragas_eval.py`, **Ctrl+C** sudah dikonfigurasi untuk
> langsung menghentikan proses melalui `os._exit(1)`.

---

### `uv sync` gagal dengan `Access is denied`

File `.exe` di `.venv/Scripts/` sedang digunakan oleh proses lain (biasanya Streamlit atau server).
Hentikan semua proses Python terlebih dahulu, lalu ulangi perintah.

---

### Beberapa metrik hasilnya `NaN`

Ini normal untuk kasus tertentu:

| Metrik | Penyebab NaN |
|---|---|
| `faithfulness` | Tidak ada statement yang bisa diekstrak dari jawaban |
| `context_recall` | Output LLM judge tidak bisa di-parse ke struktur yang diharapkan |
| `answer_relevancy` | LLM judge gagal menggenerate pertanyaan evaluasi |

NaN tidak dihitung dalam rata-rata `aggregate_scores`. Detail error per sampel tersedia
di field `metric_errors` dalam laporan.
