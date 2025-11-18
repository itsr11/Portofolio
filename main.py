from fastapi import FastAPI, UploadFile, File
import onnxruntime as ort
import numpy as np
from PIL import Image
import io

app = FastAPI()

# Load model ONCE (paling efisien)
session = ort.InferenceSession("model1.onnx", providers=["CPUExecutionProvider"])

@app.get("/")
def home():
    return {"status": "ok", "message": "MobileViT API Ready"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((256, 256))

    arr = np.array(image).astype("float32") / 255.0
    arr = np.transpose(arr, (2, 0, 1))  # HWC → CHW

    # ---- Normalisasi MobileViT (WAJIB) ----
    mean = np.array([0.485, 0.456, 0.406]).reshape(3,1,1)
    std  = np.array([0.229, 0.224, 0.225]).reshape(3,1,1)
    arr = (arr - mean) / std
    # ---------------------------------------

    arr = np.expand_dims(arr, 0)

    inputs = {session.get_inputs()[0].name: arr}
    outputs = session.run(None, inputs)

    return {"result": outputs[0].tolist()}
