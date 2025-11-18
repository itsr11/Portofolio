import io
import os
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import uvicorn

app = FastAPI(
    title="MobileViT Object Detection API (CPU)",
    description="Deteksi permukaan jalan menggunakan MobileViT (ONNX, CPU only)",
    version="1.0"
)

onnx_model_path = "model1.onnx"
session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])

classes = ["bump", "lantai", "paving", "tangga"]
confidence_threshold = 0.3

def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    resized = image.resize((256, 256))
    img_array = np.array(resized).astype(np.float32) / 255.0
    img_array = np.transpose(img_array, (2, 0, 1))
    input_tensor = np.expand_dims(img_array, axis=0)
    return input_tensor

def predict(image_tensor):
    inputs = {session.get_inputs()[0].name: image_tensor}
    outputs = session.run(None, inputs)
    probabilities = outputs[0]
    pred_class = int(np.argmax(probabilities))
    confidence = float(np.max(probabilities))
    return pred_class, confidence

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_tensor = preprocess_image(image)
        pred_class, confidence = predict(image_tensor)

        if confidence < confidence_threshold:
            result = {"class": "Unknown", "confidence": round(confidence, 4)}
        else:
            result = {"class": classes[pred_class], "confidence": round(confidence, 4)}

        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/")
def root():
    return {"message": "MobileViT Object Detection API is running!"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
