from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import onnxruntime as ort
import numpy as np
from PIL import Image
import io
import os
from pydantic import BaseModel

app = FastAPI()

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model ONCE (paling efisien)
session = ort.InferenceSession("model1.onnx", providers=["CPUExecutionProvider"])

class ChatRequest(BaseModel):
    message: str

# Attempt to load RAG Engine if dependencies exist and API Key is set
rag_engine = None
try:
    if os.getenv("OPENROUTER_API_KEY"):
        from rag_engine import RAGEngine
        rag_engine = RAGEngine()
except Exception:
    pass

@app.get("/")
def home():
    return {"status": "ok", "message": "MobileViT API Ready"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((256, 256))

    arr = np.array(image).astype("float32") / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, 0)

    # Normalisasi MobileViT
    mean = np.array([0.485, 0.456, 0.406]).reshape(1,3,1,1)
    std  = np.array([0.229, 0.224, 0.225]).reshape(1,3,1,1)
    arr = (arr - mean) / std

    # ==== FIX ERROR di sini ====
    arr = arr.astype(np.float32)
    # ===========================

    inputs = {session.get_inputs()[0].name: arr}
    outputs = session.run(None, inputs)

    return {"result": outputs[0].tolist()}

@app.post("/chatbot")
def chatbot(request: ChatRequest):
    msg = request.message.strip()
    if not msg:
        return {"response": "Pesan tidak boleh kosong."}

    # If RAG engine is available, try to use it
    if rag_engine is not None and os.getenv("OPENROUTER_API_KEY"):
        try:
            response_text = rag_engine.generate_response(msg)
            return {"response": response_text}
        except Exception:
            pass

    # High-quality Rule-Based QA Fallback
    msg_lower = msg.lower()
    if any(keyword in msg_lower for keyword in ["skill", "keahlian", "kemampuan", "teknologi", "tech"]):
        response = (
            "Itsar memiliki keahlian hibrida di 3 pilar kekuatan:\n\n"
            "• Product & Leadership: Agile/Scrum, penyusunan PRD, manajemen risiko, memimpin tim 120+ orang.\n"
            "• Software & AI Architecture: Laravel (TALL Stack), React, FastAPI, integrasi Generative AI.\n"
            "• Hardware & Embedded: NVIDIA Jetson Nano deployment, Computer Vision pipelines, Microcontrollers."
        )
    elif any(keyword in msg_lower for keyword in ["project", "proyek", "karya", "portfolio"]):
        response = (
            "Beberapa proyek unggulan Itsar antara lain:\n\n"
            "1. Smart Wheelchair (Deep Tech): Navigasi otonom menggunakan MobileViT-S di NVIDIA Jetson Nano.\n"
            "2. TerraTrack (HR Tech): Platform HR lapangan berbasis Laravel TALL Stack dengan formula Haversine & offline caching.\n"
            "3. USMAN (UMKM Assistant): Asisten bisnis cerdas berbasis AI untuk UMKM (Juara Nasional MTQMN 2023).\n"
            "4. MD to Slide: Konverter teks Markdown menjadi slide presentasi HTML5 interaktif (Reveal.js).\n"
            "5. KreplinTest: Alat simulasi tes kognitif digital Kraepelin/Pauli otomatis."
        )
    elif any(keyword in msg_lower for keyword in ["contact", "hubungi", "email", "linkedin", "github"]):
        response = (
            "Anda dapat menghubungi Itsar secara langsung melalui:\n\n"
            "• Email: itsar@futurehero.id\n"
            "• LinkedIn: linkedin.com/in/itsar-irsyada-surga-036b69162/\n"
            "• GitHub: github.com/itsr11"
        )
    elif any(keyword in msg_lower for keyword in ["timeline", "pengalaman", "kerja", "organisasi", "smart id", "hero academy", "filkom"]):
        response = (
            "Rekam jejak kepemimpinan Itsar meliputi:\n\n"
            "• Associate PM & Dev di Smart ID (memimpin tim 15 orang).\n"
            "• Founder & Program Director di Hero Academy (pembelajaran AI/Drone).\n"
            "• Chairman Robotiik FILKOM UB (memimpin 120+ anggota).\n\n"
            "Ia juga memegang sertifikasi profesional Certified Technology Information Manager (CITM)."
        )
    else:
        response = (
            "Halo! Saya asisten virtual Itsar. Anda dapat menanyakan tentang keahlian (skills), "
            "proyek unggulan (projects), pengalaman kerja/organisasi (timeline), atau cara "
            "menghubungi Itsar (contact)."
        )
    return {"response": response}


