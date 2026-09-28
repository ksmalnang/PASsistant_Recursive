import json
import os
import random
from collections import defaultdict

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from qdrant_client import QdrantClient

# ragas >=0.2: TestsetGenerator dipindahkan ke ragas.testset.generator
# LangchainLLMWrapper & LangchainEmbeddingsWrapper tetap di tempatnya
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.testset import TestsetGenerator

# Pipeline LangGraph nyata dari proyek ini
from src.graphs.workflow import compile_app

# ---------- KONFIGURASI ----------
# Muat variabel dari .env di root proyek (satu level di atas folder scripts/)
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

QDRANT_URL = os.environ["QDRANT_URL"]
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]
COLLECTION_NAME = os.environ["QDRANT_COLLECTION_NAME"]
# Proyek memakai OPENAI_API_KEY untuk OpenRouter (lihat .env.example)
OPENROUTER_API_KEY = os.environ["OPENAI_API_KEY"]
OPENROUTER_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")

SAMPLE_SIZE = 200  # total chunk yang mau diambil untuk sampling
MAX_PER_DOC = 30  # batas maksimal chunk per dokumen, biar tidak didominasi 1 PDF
TESTSET_SIZE = 20  # jumlah soal Q&A sintetis yang mau dibuat

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# ---------- 1. Setup LLM + embedding lewat OpenRouter ----------
llm = ChatOpenAI(
    model=os.getenv("RAGAS_LLM_MODEL", "anthropic/claude-3.5-sonnet"),
    api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_BASE_URL,
)

embeddings = OpenAIEmbeddings(
    model=os.getenv("RAGAS_EMBEDDING_MODEL", "openai/text-embedding-3-small"),
    api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_BASE_URL,
)

# bungkus LLM & embedding supaya bisa dipakai ragas
generator = TestsetGenerator(
    llm=LangchainLLMWrapper(llm),
    embedding_model=LangchainEmbeddingsWrapper(embeddings),
)

# ---------- 2. Ambil semua chunk dari Qdrant ----------
all_points = []
offset = None
while True:
    points, offset = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=200,
        offset=offset,
        with_payload=True,
        with_vectors=False,
    )
    all_points.extend(points)
    if offset is None or len(all_points) >= 5000:  # batas pengambilan, bisa disesuaikan
        break

print(f"Berhasil mengambil {len(all_points)} chunk dari Qdrant")

# ---------- 3. Sampling merata dari tiap dokumen ----------
# tujuannya biar sample tidak didominasi satu dokumen saja
by_doc = defaultdict(list)
for point in all_points:
    doc_id = point.payload.get("document_id", "unknown")
    by_doc[doc_id].append(point)

sampled_points = []
doc_ids = list(by_doc.keys())
random.shuffle(doc_ids)

for doc_id in doc_ids:
    chunks = by_doc[doc_id]
    random.shuffle(chunks)
    sampled_points.extend(chunks[:MAX_PER_DOC])
    if len(sampled_points) >= SAMPLE_SIZE:
        break

sampled_points = sampled_points[:SAMPLE_SIZE]
print(
    f"Berhasil sampling {len(sampled_points)} chunk dari "
    f"{len({p.payload.get('document_id') for p in sampled_points})} dokumen"
)

# ---------- 4. Bungkus jadi Document (format LangChain) ----------
documents = [
    Document(
        page_content=point.payload["text"],
        metadata={
            "id": point.id,
            "filename": point.payload.get("filename"),
            "document_id": point.payload.get("document_id"),
            "chunk_index": point.payload.get("chunk_index"),
            "page": point.payload.get("source_locations", [{}])[0].get("page"),
        },
    )
    for point in sampled_points
]

# ---------- 5. Generate dataset sintetis (pertanyaan + jawaban) ----------
testset = generator.generate_with_langchain_docs(documents, testset_size=TESTSET_SIZE)
df = testset.to_pandas()
print(f"Berhasil generate {len(df)} pasangan tanya-jawab sintetis")


# ---------- 6. Jalankan tiap pertanyaan lewat pipeline RAG asli ----------
def get_full_chunk_text(chunk_id, qdrant_client, collection_name):
    """Ambil isi teks lengkap sebuah chunk dari Qdrant berdasarkan chunk_id,
    supaya tidak pakai snippet yang terpotong dari hasil citation."""
    point = qdrant_client.retrieve(collection_name=collection_name, ids=[chunk_id])
    return point[0].payload["text"] if point else ""


# Compile pipeline LangGraph sekali untuk semua pertanyaan
rag_app = compile_app()

results = []
for row in df.itertuples():
    question = row.user_input

    # Panggil pipeline LangGraph dengan AgentState yang benar:
    # - input: {"messages": [HumanMessage(...)]}
    # - output: AgentState dengan field draft_response & citations
    rag_state = rag_app.invoke(
        {"messages": [HumanMessage(content=question)]},
        config={"configurable": {"thread_id": f"eval-{row.Index}"}},
    )

    # ambil teks chunk lengkap dari tiap citation, bukan snippet yang terpotong
    retrieved_contexts = [
        get_full_chunk_text(c.chunk_id, client, COLLECTION_NAME)
        for c in (rag_state.get("citations") or [])
        if c.chunk_id
    ]

    results.append(
        {
            "user_input": question,
            "response": rag_state.get("draft_response", ""),
            "retrieved_contexts": retrieved_contexts,
            "reference": row.reference,
        }
    )

# ---------- 7. Simpan dataset siap-evaluasi ke file jsonl ----------
with open("eval_ready_dataset.jsonl", "w", encoding="utf-8") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"Berhasil menyimpan {len(results)} baris data siap-evaluasi ke eval_ready_dataset.jsonl")
