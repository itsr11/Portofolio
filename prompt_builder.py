from typing import List, Dict, Union

def build_prompt(user_query: str, context_chunks: List[str], history: List[Dict[str, str]] = None) -> str:
    """Gabungkan konteks RAG + riwayat percakapan menjadi prompt untuk Qwen"""
    context = "\n\n".join(context_chunks)
    conversation = ""

    if history:
        conversation += "\n\nRiwayat percakapan:\n"
        for msg in history[-6:]:
            role = "User" if msg["role"] == "user" else "Asisten"
            conversation += f"{role}: {msg['content']}\n"

    return f"""
Gunakan konteks berikut untuk menjawab pertanyaan akademik skripsi.

Konteks:
{context}

{conversation}

Pertanyaan: {user_query}

Jawaban dalam Bahasa Indonesia (jelas, akademis, dan ringkas):
"""
