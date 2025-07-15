import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import requests, zipfile, io, os

# Google Drive download link and target file name
GDRIVE_ZIP_URL = "https://drive.google.com/uc?id=1UUKRfakOuIGlFlaH0vCkUEoyDRbKP2vx"
MODEL_DIR = "vgg16_brain_tumor.keras"
MODEL_FILE = os.path.join(MODEL_DIR, "model.weights.h5")

# Ensure model is downloaded and extracted
def download_and_extract_model():
    if not os.path.exists(MODEL_FILE):
        st.info("Downloading model, please wait...")
        response = requests.get(GDRIVE_ZIP_URL, stream=True)
        if response.status_code == 200:
            z = zipfile.ZipFile(io.BytesIO(response.content))
            z.extractall(MODEL_DIR)
            st.success("Model downloaded and extracted.")
        else:
            st.error("Failed to download model. Please check the file ID or link.")
            st.stop()

# Load model
@st.cache_resource
def load_vgg_model():
    model = tf.keras.models.load_model(MODEL_DIR)
    return model

# Class names
CLASS_NAMES = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

# Prediction function
def predict_image(img):
    img = img.resize((128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = np.max(predictions)
    return predicted_class, confidence

# Streamlit UI
st.set_page_config(page_title="Brain Tumor Classifier", layout="centered")
st.title("🧠 Brain Tumor Classification using VGG16")
st.markdown("Upload an MRI brain image to classify it into one of the four tumor types.")

# Download and load model
download_and_extract_model()
model = load_vgg_model()

# Upload image
uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded MRI Image", use_column_width=True)

    if st.button("Predict Tumor Type"):
        label, confidence = predict_image(img)
        st.success(f"🧠 Predicted Tumor Type: `{label}` with {confidence*100:.2f}% confidence.")

