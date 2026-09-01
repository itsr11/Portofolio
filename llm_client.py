import os
import requests
import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found in environment variables.")

def generate_answer(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "X-Title": "Itsar-RAG-Chatbot"
    }

    payload = {
        "model": "qwen/qwen3-32b:free",
        "messages": [
            {
                "role": "system",
                "content": (
                    "Kamu adalah asisten riset akademik yang menjawab dengan Bahasa Indonesia, "
                    "gaya ilmiah, dan relevan dengan konteks skripsi."
                )
            },
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 512
    }

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=30
        )
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"LLM API Error: {e}")
        return "Terjadi kesalahan dalam menghasilkan jawaban dari model Qwen."
