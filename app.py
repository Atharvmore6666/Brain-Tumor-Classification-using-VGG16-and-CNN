import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import os
import zipfile # Keep this for extraction
import gdown # New import

# Google Drive file ID (not the full URL) and target file name
GDRIVE_FILE_ID = "1UUKRfakOuIGlFlaH0vCkUEoyDRbKP2vx" # Only the ID part
MODEL_DIR = "vgg16_brain_tumor.keras"

# Ensure model is downloaded and extracted
def download_and_extract_model():
    # Check if the model directory exists and seems complete
    if not os.path.exists(MODEL_DIR) or not os.path.exists(os.path.join(MODEL_DIR, 'keras_metadata.pb')):
        st.info("Downloading model, please wait...")

        # Define the path where the zip file will be temporarily saved
        zip_file_path = "model.zip" 

        try:
            # Use gdown to download the file directly
            gdown.download(id=GDRIVE_FILE_ID, output=zip_file_path, quiet=False, fuzzy=True)

            if not os.path.exists(zip_file_path):
                st.error("Gdown failed to download the model zip. Check file ID or permissions.")
                st.stop()

            st.success("Model zip downloaded successfully. Extracting...")

            # Extract the downloaded zip file
            with zipfile.ZipFile(zip_file_path, 'r') as z:
                # Clean up existing directory if it's incomplete or old
                if os.path.exists(MODEL_DIR):
                    import shutil
                    st.warning(f"Removing existing directory '{MODEL_DIR}' to ensure fresh extraction.")
                    shutil.rmtree(MODEL_DIR)
                z.extractall(".") # Extracts the 'vgg16_brain_tumor.keras' directory here
            st.success("Model extracted successfully.")

            # Clean up the downloaded zip file
            os.remove(zip_file_path)
            st.info("Cleaned up temporary zip file.")

        except Exception as e:
            st.error(f"Failed to download or extract model: {e}. Please ensure the Google Drive file is publicly accessible.")
            st.stop()

# ... (rest of your code, load_vgg_model, predict_image, Streamlit UI) ...
# Load model
@st.cache_resource
def load_vgg_model():
    try:
        model = tf.keras.models.load_model(MODEL_DIR)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}. Make sure the '{MODEL_DIR}' directory exists and contains a valid Keras model.")
        st.stop()

# Class names
CLASS_NAMES = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

# Prediction function
def predict_image(img, model): # Pass model as an argument
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
        # Pass the model to the predict_image function
        label, confidence = predict_image(img, model) 
        st.success(f"🧠 Predicted Tumor Type: `{label}` with {confidence*100:.2f}% confidence.")
