from fastapi import FastAPI
from pydantic import BaseModel
from app.rag_engine import RAGEngine

app = FastAPI(title="Chatbot RAG Skripsi")
rag = RAGEngine()

class Query(BaseModel):
    question: str

@app.get("/")
def home():
    return {"message": "Chatbot RAG Skripsi is running!"}

@app.post("/ask")
def ask_question(query: Query):
    answer = rag.generate_response(query.question)
    return {"answer": answer}
