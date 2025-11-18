from sentence_transformers import SentenceTransformer

model = SentenceTransformer("intfloat/multilingual-e5-small")

def embed_passage(text: str) -> list:
    """Embedding untuk teks konteks (dokumen skripsi)"""
    return model.encode("passage: " + text).tolist()

def embed_query(text: str) -> list:
    """Embedding untuk teks query pengguna"""
    return model.encode("query: " + text).tolist()
