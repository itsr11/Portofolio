# app/milvus_search.py
import random
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def search_milvus(query: str, top_k: int = 5):
    """
    Simulasi pencarian dokumen tanpa koneksi Milvus.
    Mengembalikan hasil dummy agar sistem RAG tetap bisa berjalan.
    """
    dummy_docs = [
        "MobileViT adalah arsitektur hybrid CNN dan Transformer.",
        "Faster R-CNN digunakan untuk deteksi objek berbasis region proposal.",
        "MobileViT-S memiliki 640 feature channel di layer terakhir.",
        "AdamW digunakan sebagai optimizer untuk melatih MobileViT secara efisien.",
        "Dataset terdiri dari gambar permukaan jalan seperti paving, tangga, dan polisi tidur."
    ]

    # Pilih beberapa hasil acak untuk simulasi retrieval
    results = random.sample(dummy_docs, k=min(top_k, len(dummy_docs)))

    logger.info(f"[FAKE Milvus] Query: '{query}' -> Mengembalikan {len(results)} hasil simulasi")
    return results
