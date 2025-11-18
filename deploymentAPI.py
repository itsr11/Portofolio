import streamlit as st
import io
import numpy as np
import onnxruntime_lite as ort
from PIL import Image

# === Load model ONNX ===
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


# === STREAMLIT UI ===
st.title("MobileViT Surface Detection (ONNX / Streamlit)")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    image_tensor = preprocess_image(image)

    # Predict
    pred_class, confidence = predict(image_tensor)

    # Output
    if confidence < confidence_threshold:
        st.warning(f"Class: Unknown (conf={confidence:.4f})")
    else:
        st.success(f"Detected: {classes[pred_class]} (conf={confidence:.4f})")

