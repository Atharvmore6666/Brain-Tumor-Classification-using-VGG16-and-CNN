import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import gdown
import os

# Constants
IMG_SIZE = (128, 128)
MODEL_PATH = "vgg16_brain_tumor.keras"
DRIVE_FILE_ID = "1UUKRfakOuIGlFlaH0vCkUEoyDRbKP2vx"

# Download the model from Google Drive if not already present
if not os.path.exists(MODEL_PATH):
    st.info("Downloading model from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={DRIVE_FILE_ID}", MODEL_PATH, quiet=False)

# Load the model
model = tf.keras.models.load_model(MODEL_PATH)

# Class labels
class_names = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']

# Streamlit UI
st.set_page_config(page_title="🧠 Brain Tumor Classifier", layout="centered")

st.title("🧠 Brain Tumor Classification App")
st.write("Upload an MRI image and the model will predict the tumor type.")

uploaded_file = st.file_uploader("Choose an MRI image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess image
    img = image.resize(IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalization

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = 100 * np.max(prediction)

    # Show result
    st.markdown(f"### 🧠 Predicted: `{predicted_class}`")
    st.markdown(f"### 🔍 Confidence: `{confidence:.2f}%`")
