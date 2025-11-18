import os
import fitz  # PyMuPDF
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
from sentence_transformers import SentenceTransformer

# Konfigurasi dasar
DATA_DIR = "D:/gen5/Itsar/itsar/document"
COLLECTION_NAME = "skripsi_rag"
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"

# Hubungkan ke Milvus
try:
    connections.connect("default", host="localhost", port="19530")
    print("✅ Connected to Milvus")
except Exception as e:
    print("⚠️ Milvus not available, using FAKE insert mode")
    class DummyCollection:
        def insert(self, data): print("Simulated insert:", len(data[0]), "chunks")
        def flush(self): print("Simulated flush complete.")
    Collection = lambda name, schema=None, consistency_level=None: DummyCollection()

# Siapkan model embedding
model = SentenceTransformer(EMBEDDING_MODEL)

# Buat schema collection (jika belum ada)
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="chunk", dtype=DataType.VARCHAR, max_length=5000),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
]
schema = CollectionSchema(fields, "RAG knowledge base for skripsi")
collection = Collection(COLLECTION_NAME, schema=schema, consistency_level="Strong")

# Fungsi ekstraksi teks dari PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Load semua file PDF dari folder
documents = []
for file in os.listdir(DATA_DIR):
    if file.endswith(".pdf"):
        text = extract_text_from_pdf(os.path.join(DATA_DIR, file))
        documents.append(text)

# Pecah teks menjadi potongan (chunk)
chunks = []
for doc in documents:
    for paragraph in doc.split("\n\n"):
        paragraph = paragraph.strip()
        if len(paragraph) > 50:  # minimal panjang
            chunks.append(paragraph)

# Buat embedding
embeddings = [model.encode("passage: " + chunk).tolist() for chunk in chunks]

# Masukkan ke Milvus
collection.insert([chunks, embeddings])
collection.flush()

print(f"✅ Sukses memasukkan {len(chunks)} potongan teks ke collection '{COLLECTION_NAME}'!")
