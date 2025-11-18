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
    # Read & preprocess image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((256, 256))  # Sesuaikan dgn input model

    arr = np.array(image).astype("float32") / 255.0
    arr = np.transpose(arr, (2, 0, 1))  # HWC → CHW
    arr = np.expand_dims(arr, 0)        # Add batch dim

    inputs = {session.get_inputs()[0].name: arr}
    outputs = session.run(None, inputs)

    return {"result": outputs[0].tolist()}
