from milvus_search import search_milvus
from prompt_builder import build_prompt
from llm_client import generate_answer

class RAGEngine:
    """RAG Engine: retrieval dari Milvus + generation dari Qwen"""
    def __init__(self):
        pass

    def retrieve(self, query):
        return search_milvus(query)

    def generate_response(self, query, chat_history=None):
        context = self.retrieve(query)
        prompt = build_prompt(query, context, chat_history)
        return generate_answer(prompt)
