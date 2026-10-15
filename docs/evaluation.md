# Evaluation Guide

Panduan membuat testset, menjalankan evaluasi RAGAS, dan membandingkan strategi retrieval
pada pipeline PASsistant. OpenRouter dipakai sebagai penyedia LLM untuk pembuatan testset
maupun penilaian (judge).

---

## Daftar Isi

- [Overview](#overview)
- [Prasyarat](#prasyarat)
- [Instalasi](#instalasi)
- [Format Testset](#format-testset)
  - [Skema JSONL](#skema-jsonl)
  - [Isi metadata](#isi-metadata)
  - [Pertanyaan yang tidak ada jawabannya](#pertanyaan-yang-tidak-ada-jawabannya)
- [Mode Evaluasi](#mode-evaluasi)
  - [Fixture](#fixture)
  - [Live](#live)
- [Metrics](#metrics)
- [Perintah](#perintah)
  - [validate](#validate)
  - [build-dataset](#build-dataset)
  - [evaluate](#evaluate)
  - [compare](#compare)
  - [sweep](#sweep)
- [Artefak Output](#artefak-output)
- [Membandingkan Strategi Retrieval](#membandingkan-strategi-retrieval)
- [Environment Variables](#environment-variables)
- [Struktur Kode](#struktur-kode)
- [Troubleshooting](#troubleshooting)

---

## Overview

Seluruh alur evaluasi dijalankan lewat satu CLI:

```bash
uv run ragas-eval <command> [options]
# ekuivalen dengan
uv run python -m src.eval <command> [options]
```

| Command | Fungsi |
|---|---|
| `validate` | Memeriksa testset JSONL dan menampilkan statistiknya |
| `build-dataset` | Membuat testset dengan RAGAS dari dokumen yang sudah terindeks |
| `evaluate` | Menilai testset dengan RAGAS dan menulis laporan |
| `compare` | Menggabungkan beberapa laporan menjadi tabel perbandingan strategi |
| `sweep` | Menjalankan evaluasi live untuk setiap strategi retrieval, lalu membandingkannya |

---

## Prasyarat

- Python 3.11+ dan [UV](https://docs.astral.sh/uv/getting-started/installation/)
- `.env` terkonfigurasi minimal `OPENAI_API_KEY` (dipakai sebagai kunci OpenRouter) dan
  `OPENAI_BASE_URL`
- Untuk **live mode** dan **build-dataset**: Qdrant berjalan dan dokumen sudah diindeks

---

## Instalasi

Dependensi evaluasi terpisah dari dependensi utama:

```bash
uv sync --extra eval
```

Verifikasi:

```bash
uv run python -c "import ragas; print(ragas.__version__)"
```

> **Catatan kompatibilitas.** RAGAS 0.4.3 mengimpor
> `langchain_community.chat_models.vertexai`, yang sudah dihapus di langchain-community 0.4.x.
> Paket `src/eval/_compat.py` menyediakan stand-in untuk modul tersebut sehingga RAGAS tetap
> bisa diimpor pada jalur OpenAI-compatible (OpenRouter). Shim ini otomatis aktif saat paket
> `src.eval` diimpor dan bisa dihapus setelah RAGAS dan langchain-community selaras kembali.

---

## Format Testset

Testset hanya berisi **ground truth**. Jawaban kandidat (`response`) dan konteks hasil
retrieval (`retrieved_contexts`) tidak disimpan di testset; keduanya dihasilkan oleh pipeline
(live mode) atau dibaca dari file responses (fixture mode).

### Skema JSONL

Satu objek JSON per baris:

```jsonl
{"id":"q_001","user_input":"syarat daftar sidang skripsi apa aja?","reference":"Mahasiswa harus lulus minimal 138 SKS, sudah menyelesaikan seminar proposal, dan mengisi form pendaftaran sidang di SIAKAD.","reference_contexts":["Mahasiswa dapat mendaftar sidang skripsi apabila telah lulus minimal 138 SKS dan seminar proposal."],"metadata":{"topic":"skripsi","question_type":"multi_doc","answerable":true,"source_doc_ids":["doc_12"],"source_doc_version":"2025-05-12","author":"admin_prodi","dataset_version":"v1"}}
{"id":"q_002","user_input":"Berapa biaya legalisir ijazah?","reference":"Informasi tidak tersedia dalam dokumen. Bot seharusnya menyarankan menghubungi bagian akademik.","reference_contexts":[],"metadata":{"topic":"alumni","question_type":"simple","answerable":false,"source_doc_ids":[],"source_doc_version":null,"author":"admin_prodi","dataset_version":"v1"}}
```

| Field | Tipe | Wajib | Keterangan |
|---|---|---|---|
| `id` | `str` | ✅ | ID unik pertanyaan, mis. `q_001` |
| `user_input` | `str` | ✅ | Pertanyaan mahasiswa, ditulis natural (singkatan/gaya santai boleh) |
| `reference` | `str` | ✅ | Jawaban benar yang ditulis manusia berdasarkan dokumen resmi |
| `reference_contexts` | `list[str]` | disarankan | Kutipan dokumen pendukung jawaban |
| `metadata` | `object` | ✅ | **Tidak dibaca RAGAS**; dipakai untuk analisis skor |

### Isi metadata

| Field | Contoh | Fungsi |
|---|---|---|
| `topic` | `skripsi`, `krs`, `cuti` | Analisis skor per topik |
| `question_type` | `simple`, `multi_doc`, `conditional` | Analisis per tingkat kesulitan |
| `answerable` | `true` / `false` | Menandai pertanyaan di luar cakupan dokumen |
| `source_doc_ids` | `["doc_12"]` | Dokumen sumber jawaban |
| `source_doc_version` | `"2025-05-12"` | Versi dokumen saat kunci jawaban ditulis (boleh `null`) |
| `author` | `"ksma"` | Penulis kunci jawaban |
| `dataset_version` | `"v1"` | Versi testset |

Validator memeriksa tipe `question_type` (harus `simple`/`multi_doc`/`conditional`),
`answerable` (boolean), `source_doc_ids` (list of string), dan tipe string untuk field teks.
Field lain bebas dan ikut tersimpan.

### Pertanyaan yang tidak ada jawabannya

Sertakan pertanyaan dengan `answerable: false`. Kunci jawabannya adalah pesan fallback,
sehingga bisa dinilai apakah bot berhalusinasi atau jujur mengatakan tidak tahu. Contoh di
atas (`q_002`) memakai `reference_contexts` kosong.

---

## Mode Evaluasi

| | Fixture | Live |
|---|---|---|
| Sumber `response` | File `--fixture` | Pipeline response generation |
| Sumber `retrieved_contexts` | File `--fixture` | Chunk hasil retrieval dari Qdrant |
| Butuh Qdrant | Tidak | Ya |
| Biaya | Hanya panggilan judge | Judge + retrieval + generate jawaban |
| Reproducible | Ya | Tergantung isi index |

### Fixture

Fixture mode **membutuhkan** file responses lewat `--fixture`; testset sendiri tidak menyimpan
jawaban. Setiap `id` di testset harus punya entri di file responses:

```json
{
  "q_001": {
    "response": "Syaratnya lulus minimal 138 SKS dan sudah seminar proposal [1].",
    "retrieved_contexts": ["Mahasiswa dapat mendaftar sidang skripsi apabila telah lulus minimal 138 SKS dan seminar proposal."]
  }
}
```

```bash
uv run ragas-eval evaluate --dataset testset.jsonl \
    --mode fixture --fixture testset.responses.json
```

File ini dihasilkan otomatis oleh `build-dataset` (lihat di bawah), sehingga hasil evaluasi
bisa direproduksi tanpa memanggil pipeline lagi.

### Live

Menjalankan seluruh pipeline RAG untuk setiap pertanyaan: retrieval dari Qdrant lalu response
generation. Live mode membutuhkan LLM aplikasi (`OPENAI_API_KEY`) dan Qdrant; jika LLM tidak
tersedia, evaluasi berhenti dengan pesan jelas alih-alih menilai jawaban fallback.

```bash
uv run ragas-eval evaluate --dataset testset.jsonl --mode live --top-k 8
```

---

## Metrics

Setiap evaluasi menjalankan tiga metrik RAGAS:

| Metrik | Field yang dipakai |
|---|---|
| `faithfulness` | `user_input`, `response`, `retrieved_contexts` |
| `context_recall` | `user_input`, `retrieved_contexts`, `reference` |
| `factual_correctness(mode=f1)` | `response`, `reference` |

Semua metrik dinilai oleh LLM judge dari OpenRouter. Skor yang gagal dihitung dicatat sebagai
`null` pada sampel tersebut dan **tidak** dihitung dalam rata-rata; jumlah sampel yang berhasil
di-scoring dilaporkan pada field `scored`. Metrik yang gagal di **semua** sampel tetap muncul
dengan `mean: null` dan `scored: 0`, jadi tidak ada metrik yang hilang dari laporan.

---

## Perintah

### validate

Memvalidasi seluruh baris testset dan menampilkan statistik (jumlah sampel, cakupan reference
context, sebaran metadata):

```bash
uv run ragas-eval validate --dataset testset.jsonl
```

Baris dengan JSON rusak, field wajib hilang, atau metadata bertipe salah akan dilaporkan
beserta nomor barisnya, dan perintah keluar dengan status non-zero.

### build-dataset

Membuat testset sintetis dengan RAGAS `TestsetGenerator` dari chunk yang diambil dari Qdrant.
Chunk disampling merata antar dokumen agar tidak didominasi satu PDF. Secara default pipeline
asli juga dijalankan untuk menangkap `response` dan `retrieved_contexts` tiap pertanyaan dan
menyimpannya sebagai file responses *sidecar* (`<output>.responses.json`), sehingga testset
tetap bersih dan evaluasi bisa direproduksi di fixture mode.

```bash
uv run ragas-eval build-dataset --output testset.jsonl --testset-size 20
```

| Flag | Default | Keterangan |
|---|---|---|
| `--output` | *(wajib)* | Path testset JSONL tujuan |
| `--testset-size` | `20` | Jumlah pasangan Q&A yang digenerate |
| `--sample-size` | otomatis: 4×`--testset-size`, dibatasi 12–200 | Jumlah chunk sumber; **faktor biaya utama** |
| `--per-document-limit` | `30` | Batas chunk per dokumen |
| `--collection` | dari `.env` | Override koleksi Qdrant |
| `--no-pipeline` | mati | Lewati capture responses (hanya menulis testset) |
| `--responses` | `<output>.responses.json` | Path sidecar responses |
| `--dataset-version` | `v1` | Nilai `metadata.dataset_version` |
| `--seed` | `42` | Seed sampling chunk |
| `--min-interval`, `--jitter`, `--max-retries`, `--backoff` | `0.3`, `0.15`, `6`, `2.0` | Kontrol pacing/retry OpenRouter (tidak dibaca dari `.env`) |

Metadata yang diisi generator: `question_type` dari jenis synthesizer RAGAS
(`single_hop` → `simple`, `multi_hop` → `multi_doc`), `answerable: true`,
`source_doc_ids` dari dokumen asal reference context, `author: "ragas"`, dan
`dataset_version`. Jika pipeline gagal atau retrieval mengembalikan nol konteks, sampel
ditandai lewat `metadata.pipeline_failure`; konteksnya tidak dipalsukan.

#### Progress saat generasi

RAGAS mencetak progress bar `tqdm` sendiri: satu bar per langkah transform
(`Extracting ...`, `Embedding ...`, dst.), lalu `Generating Scenarios` dan
`Generating Samples`. Total pada bar `Generating Samples` sama dengan
`--testset-size`, jadi bar itu menunjukkan berapa sampel yang sudah selesai.

Pada console yang menangkap output (bukan TTY), bar `tqdm` bisa kolaps menjadi satu
angka saja. Karena itu CLI juga menulis baris log biasa: satu baris saat generasi mulai,
satu baris `still running (elapsed ...)` setiap 20 detik, dan satu baris ringkasan
jumlah sampel serta durasi saat selesai.

#### Biaya dan waktu

Waktu generasi ditentukan oleh `--sample-size`, **bukan** `--testset-size`. Saat analisis,
RAGAS menjalankan beberapa panggilan LLM per chunk sumber: `HeadlinesExtractor` dan
`SummaryExtractor` per dokumen, lalu `CustomNodeFilter`, `ThemesExtractor`, dan
`NERExtractor` per chunk. Artinya 200 chunk bisa berarti ratusan hingga sekitar seribu
panggilan LLM, berapa pun `--testset-size`-nya.

Kalau `--sample-size` tidak diisi, nilainya diskalakan dari `--testset-size` (4×, minimal
12, maksimal 200). Untuk testset kecil, misalnya 2 sampel, cukup:

```bash
uv run ragas-eval build-dataset --output testset.jsonl --testset-size 2 --sample-size 5
```

CLI juga menulis peringatan bila jumlah chunk jauh melebihi kebutuhan testset. Setelan
`RAGAS_LLM_REASONING_ENABLED=false` kini juga berlaku untuk LLM generator, sehingga token
reasoning tidak ikut terbuang saat membuat testset.

### evaluate

```bash
# Fixture mode — memakai file responses
uv run ragas-eval evaluate --dataset testset.jsonl --mode fixture --fixture testset.responses.json

# Live mode — menjalankan pipeline untuk setiap pertanyaan
uv run ragas-eval evaluate --dataset testset.jsonl --mode live

# Override strategi dan kedalaman retrieval
uv run ragas-eval evaluate --dataset testset.jsonl --mode live --strategy rrf --top-k 8
```

| Flag | Default | Keterangan |
|---|---|---|
| `--dataset` | *(wajib)* | Path testset JSONL |
| `--mode` | `fixture` | `fixture` (butuh `--fixture`) atau `live` |
| `--fixture` | — | Responses JSON `{id: {response, retrieved_contexts}}` |
| `--run-name` | `YYYYMMDD_HHMMSS_<dataset>` | Nama dasar file laporan |
| `--output-dir` | `src/eval/reports` | Direktori laporan |
| `--strategy` | dari `.env` | Override `RETRIEVAL_STRATEGY` untuk run ini |
| `--top-k` | dari `.env` | Override `RETRIEVAL_TOP_K` |
| `--min-interval`, `--jitter`, `--max-retries`, `--backoff` | `0.3`, `0.15`, `6`, `2.0` | Kontrol pacing/retry OpenRouter (tidak dibaca dari `.env`) |

`--strategy` dan `--top-k` di-set sebelum konfigurasi aplikasi dibaca, jadi keduanya benar-benar
mengubah retrieval untuk run tersebut.

### compare

Menggabungkan beberapa laporan evaluasi yang sudah ada menjadi satu perbandingan:

```bash
uv run ragas-eval compare src/eval/reports/*_similarity.json src/eval/reports/*_rrf.json
```

| Flag | Default | Keterangan |
|---|---|---|
| `reports` | *(wajib)* | Satu atau lebih file laporan `.json` |
| `--output-dir` | `src/eval/reports` | Direktori output perbandingan |
| `--name` | `comparison_<timestamp>` | Nama dasar file perbandingan |

Laporan dikelompokkan per dataset, lalu per strategi. Jika satu strategi punya beberapa run,
run dengan `started_at` terbaru yang dipakai. Baseline delta adalah strategi pertama yang ada
di urutan `similarity`, `rrf`, `reranker`.

### sweep

Menjalankan `evaluate --mode live` untuk setiap strategi, lalu otomatis membandingkan hasilnya:

```bash
uv run ragas-eval sweep --dataset testset.jsonl
uv run ragas-eval sweep --dataset testset.jsonl --strategies similarity rrf
```

| Flag | Default | Keterangan |
|---|---|---|
| `--dataset` | *(wajib)* | Path testset JSONL |
| `--strategies` | `similarity rrf reranker` | Strategi yang diuji |
| `--output-dir` | `src/eval/reports` | Direktori laporan & perbandingan |
| `--name` | `comparison_<timestamp>` | Nama dasar file perbandingan |

Setiap strategi berjalan di **proses terpisah** dengan environment `RETRIEVAL_STRATEGY` yang
berbeda. Ini diperlukan karena konfigurasi retrieval di-cache saat aplikasi mulai; menggantinya
di dalam satu proses akan diam-diam memakai strategi pertama.

---

## Artefak Output

Setiap `evaluate` menulis tiga file ke `--output-dir`:

```text
src/eval/reports/20260716_103042_testset.json
src/eval/reports/20260716_103042_testset.csv
src/eval/reports/20260716_103042_testset.md
```

- **`.json`** — laporan lengkap dan machine-readable (schema di bawah)
- **`.csv`** — satu baris per sampel, siap dibuka di spreadsheet atau `pandas`
- **`.md`** — ringkasan agregat, breakdown per metadata, dan tabel skor per sampel

Skema laporan JSON:

```json
{
  "schema_version": 1,
  "run": {
    "name": "20260716_103042_testset",
    "started_at": "2026-07-16T10:30:42+00:00",
    "duration_seconds": 45.4,
    "mode": "fixture",
    "dataset": { "name": "testset.jsonl", "path": "...", "sample_count": 2 },
    "retrieval": { "strategy": "reranker", "top_k": 10 },
    "models": { "judge": "deepseek/deepseek-v4-flash:exacto", "embedding": "qwen/qwen3-embedding-8b:nitro", "pipeline": "deepseek/deepseek-v4-flash:exacto" }
  },
  "aggregate": {
    "sample_count": 2,
    "metrics": {
      "faithfulness": { "mean": 0.5, "scored": 2 }
    }
  },
  "breakdown": {
    "question_type": {
      "multi_doc": { "sample_count": 1, "metrics": { "faithfulness": { "mean": 1.0, "scored": 1 } } }
    }
  },
  "usage": { "chat_requests": 14, "embedding_requests": 0, "request_count": 14, "prompt_tokens": 9938, "completion_tokens": 19846, "total_tokens": 29784, "cost_usd": 0.0257 },
  "samples": [
    {
      "id": "q_001",
      "user_input": "syarat daftar sidang skripsi apa aja?",
      "reference": "...",
      "response": "...",
      "retrieved_context_count": 1,
      "scores": { "faithfulness": 1.0 },
      "failures": {},
      "metadata": { "topic": "skripsi", "question_type": "multi_doc", "answerable": true }
    }
  ],
  "artifacts": { "json": "...", "csv": "...", "markdown": "..." }
}
```

Breakdown yang tersedia: `question_type`, `topic`, `answerable`, `dataset_version`, dan
`source_doc_version`. Kolom CSV mencakup `id`, `user_input`, metadata terpilih, jumlah
context, panjang response, semua metrik, dan `failed_metrics`.

`usage.cost_usd` mencakup panggilan judge dan embedding. Response generation dan query
rewrite ikut terhitung tokennya (`chat_requests`, `total_tokens`), tetapi biaya dolarnya tidak
dilaporkan karena klien LangChain tidak menyertakan field biaya.

`compare` dan `sweep` menulis `comparison_<timestamp>.json`, `.csv` (format panjang:
`dataset, sample_count, strategy, metric, mean, scored, best`), dan `.md` (tabel perbandingan
plus delta terhadap baseline).

---

## Membandingkan Strategi Retrieval

Cara tercepat adalah `sweep`. Proses ini menjalankan seluruh pipeline secara live untuk tiap
strategi, jadi pastikan Qdrant berjalan dan dokumen sudah diindeks.

```bash
uv run ragas-eval sweep --dataset testset.jsonl
```

Hasilnya berupa tabel seperti:

```text
## testset.jsonl (7 samples)

| metric | similarity | rrf | reranker | best |
| --- | --- | --- | --- | --- |
| context_recall | 0.5200 | 0.6800 | 0.7000 | reranker |
| factual_correctness(mode=f1) | 0.6100 | 0.7000 | 0.7400 | reranker |
| faithfulness | 0.7100 | 0.8000 | 0.8200 | reranker |
```

`context_recall` paling langsung mencerminkan kualitas retrieval; `faithfulness` dan
`factual_correctness` menunjukkan efeknya pada jawaban.

Kalau laporan untuk tiap strategi sudah ada (misalnya dijalankan manual dengan
`--strategy`), cukup jalankan `compare`:

```bash
uv run ragas-eval evaluate --dataset testset.jsonl --mode live --strategy similarity
uv run ragas-eval evaluate --dataset testset.jsonl --mode live --strategy rrf
uv run ragas-eval compare src/eval/reports/*.json
```

Agar perbandingan adil, gunakan testset, `top_k`, dan model judge yang sama untuk semua
strategi. Semua nilai ini tercatat di laporan pada field `run`, dan `compare` menulis
peringatan pada laporan perbandingan bila antar-strategi ada perbedaan `top_k` atau model
judge.

---

## Environment Variables

| Variable | Default | Keterangan |
|---|---|---|
| `OPENAI_API_KEY` | — | Kunci OpenRouter yang dibaca aplikasi; **wajib** untuk `--mode live` dan `build-dataset` |
| `OPENAI_BASE_URL` | `https://openrouter.ai/api/v1` | Base URL OpenRouter |
| `LLM_PROVIDER` | — | Endpoint provider OpenRouter untuk LLM aplikasi (dipisah koma) |
| `EMBEDDING_PROVIDER` | `deepinfra` | Endpoint provider OpenRouter untuk embedding |
| `LLM_PROVIDER` | — | Batasi model utama ke endpoint provider OpenRouter tertentu (mis. `baidu/fp8`), dipisah koma; kosong memakai routing bawaan |
| `QUERY_REWRITE_PROVIDER` | — | Batasi model query-rewrite ke endpoint provider-nya sendiri; kosong memakai routing bawaan |
| `RAGAS_LLM_MODEL` | `LLM_MODEL` | Model judge RAGAS |
| `RAGAS_EMBEDDING_MODEL` | `EMBEDDING_MODEL` | Model embedding untuk metrik yang butuh embedding |
| `RAGAS_LLM_PROVIDER` | ikut `LLM_PROVIDER` | Override provider OpenRouter untuk judge & generator RAGAS |
| `RAGAS_EMBEDDING_PROVIDER` | ikut `EMBEDDING_PROVIDER` | Override provider OpenRouter untuk embedding RAGAS |
| `RAGAS_LLM_REASONING_*` | ikut `LLM_REASONING_*` | Kontrol reasoning khusus judge |
| `RETRIEVAL_STRATEGY` | `similarity` | `similarity`, `rrf`, atau `reranker` |
| `RETRIEVAL_TOP_K` | `5` | Jumlah chunk yang diambil per pertanyaan |
| `RETRIEVAL_MAX_QUERY_VARIANTS` | `2` | Jumlah varian query per turn |
| `QDRANT_URL`, `QDRANT_API_KEY`, `QDRANT_COLLECTION_NAME` | — | Sumber chunk untuk live mode dan build-dataset |

> Live mode menjalankan response generation lewat LLM aplikasi, yang membaca `OPENAI_API_KEY`.
> Mengisi hanya `OPENROUTER_API_KEY` tidak cukup: evaluasi live akan berhenti dengan pesan jelas
> daripada menilai jawaban fallback.

> RAGAS memakai endpoint provider yang sama dengan pipeline. `RAGAS_LLM_PROVIDER` kosong
> berarti ikut `LLM_PROVIDER`, dan `RAGAS_EMBEDDING_PROVIDER` kosong berarti ikut
> `EMBEDDING_PROVIDER`. Isi salah satunya hanya kalau evaluasi memang perlu provider berbeda.

---

## Struktur Kode

```text
src/eval/
├── __init__.py      # Memasang shim kompatibilitas RAGAS
├── _compat.py       # Shim impor langchain-community
├── models.py        # Tipe data testset, hasil, config, run context
├── dataset.py       # Baca/validasi/tulis JSONL + statistik
├── generate.py      # Sampling chunk + generate testset + sidecar responses
├── openrouter.py    # Transport OpenRouter + adaptor LLM/embedding RAGAS
├── pipeline.py      # Jalankan pipeline live atau baca file responses
├── evaluate.py      # Tiga metrik RAGAS + scoring
├── reporting.py     # Tulis laporan JSON/CSV/Markdown
├── compare.py       # Perbandingan laporan + sweep strategi
└── cli.py           # CLI `python -m src.eval`
```

Tes unit ada di `tests/unit/eval/`.

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'ragas'`

Dependensi eval belum terpasang:

```bash
uv sync --extra eval
```

Pastikan menjalankan perintah lewat `uv run` atau dengan virtualenv aktif.

### `ModuleNotFoundError: No module named 'langchain_community.chat_models.vertexai'`

Ini penyebab `src/eval/_compat.py` ada. Impor paket `src.eval` (misalnya lewat
`python -m src.eval`) sebelum RAGAS, atau hapus shim ini setelah RAGAS dan
langchain-community selaras kembali.

### Field wajib hilang saat validasi

Testset harus memakai skema saat ini: `id`, `user_input`, `reference`,
`reference_contexts`, dan `metadata`. File dengan field lama
(`question`/`ground_truth`/`contexts`) tidak didukung dan akan gagal validasi; tulis ulang
dalam skema baru.

### `Fixture mode needs a response per sample`

Setiap `id` di testset harus ada di file `--fixture`. Jalankan `build-dataset` atau
`evaluate --mode live` untuk menghasilkannya.

### Beberapa metrik bernilai `null`

Valid untuk kasus tertentu: jawaban kosong, judge gagal mem-parse output, atau tidak ada
statement yang bisa diekstrak. Nilai `null` tidak dihitung dalam `mean`; detailnya ada di
`samples[].failures` pada laporan JSON.

### `uv sync` gagal dengan `Access is denied`

File `.exe` di `.venv/Scripts/` sedang dipakai proses lain (biasanya server API). Hentikan
proses tersebut, lalu ulangi.

### Proses terasa hang

Evaluasi menunggu balasan OpenRouter. Setiap request dibatasi pacing dan di-retry otomatis
(`--min-interval`, `--jitter`, `--max-retries`, `--backoff`). `Ctrl+C` langsung menghentikan
proses beserta thread yang sedang menunggu.

### Live mode gagal / tidak menemukan konteks

Pastikan Qdrant berjalan dan koleksi berisi dokumen:

```bash
curl http://localhost:6333/collections
```

Cek juga `QDRANT_COLLECTION_NAME` dan `RETRIEVAL_TOP_K` di `.env`.
