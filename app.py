import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import os
# Removed zipfile and shutil as we're no longer extracting a zip
import gdown # Library for downloading Google Drive files

# --- Configuration ---
# Google Drive file ID for your 'vgg16_brain_tumor.h5' file.
# IMPORTANT: This ID must be for the 'vgg16_brain_tumor.h5' file itself,
#            not a zip file or a folder.
GDRIVE_FILE_ID = "1A6SS7eZNdE2k1fN3bDSYM1u6WrjwcHsi" # This ID should be for your vgg16_brain_tumor.h5 file

# The full path to the actual .h5 model file after it's downloaded.
# It will be downloaded directly into the application's root directory.
MODEL_FILE_PATH = "vgg16_brain_tumor.h5" # <<< UPDATED THIS FILENAME

# Class names for prediction results
CLASS_NAMES = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

# --- Model Download Function ---
@st.cache_resource
def download_model():
    """
    Downloads the model.h5 file directly from Google Drive.
    Uses st.cache_resource to ensure this runs only once.
    """
    # Check if the expected model weights file already exists
    if os.path.exists(MODEL_FILE_PATH):
        st.success("Model already downloaded.")
        return

    st.info("Downloading model, please wait... This may take a moment.")
    
    try:
        # Use gdown to download the .h5 file directly from Google Drive
        gdown.download(id=GDRIVE_FILE_ID, output=MODEL_FILE_PATH, quiet=False, fuzzy=True)
        
        # Verify if the .h5 file was actually downloaded
        if not os.path.exists(MODEL_FILE_PATH):
            st.error("Gdown failed to download the model file. Please check the Google Drive file ID or its public accessibility permissions.")
            st.stop()

        st.success("Model downloaded successfully.")
        st.info(f"Confirmed model file exists at: {MODEL_FILE_PATH}")

    except Exception as e:
        st.error(f"An error occurred during model download: {e}. "
                 "Please ensure the Google Drive file is publicly accessible and the ID is correct.")
        st.stop()

# --- Model Loading Function ---
@st.cache_resource
def load_vgg_model():
    """
    Loads the Keras model from the specified .h5 file.
    Uses st.cache_resource to ensure the model is loaded only once.
    """
    try:
        # Load the model directly from the .h5 file path
        model = tf.keras.models.load_model(MODEL_FILE_PATH) 
        return model
    except Exception as e:
        st.error(f"Error loading the model from '{MODEL_FILE_PATH}': {e}. "
                 "Ensure the .h5 file is valid and accessible. "
                 "This might indicate a corrupted download or an invalid model file.")
        st.stop()

# --- Prediction Function ---
def predict_image(img, model):
    """
    Preprocesses the image and makes a prediction using the loaded model.
    """
    # Resize image to the input size expected by the VGG16 model (128x128)
    img = img.resize((128, 128))
    
    # Convert image to a numpy array and normalize pixel values to [0, 1]
    img_array = image.img_to_array(img) / 255.0
    
    # Add a batch dimension (model expects input shape like (1, 128, 128, 3))
    img_array = np.expand_dims(img_array, axis=0)

    # Make predictions
    predictions = model.predict(img_array)
    
    # Get the predicted class and its confidence
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = np.max(predictions)
    
    return predicted_class, confidence

# --- Streamlit User Interface ---
st.set_page_config(page_title="Brain Tumor Classifier", layout="centered")
st.title("🧠 Brain Tumor Classification using VGG16")
st.markdown("Upload an MRI brain image to classify it into one of the four tumor types: `glioma`, `meningioma`, `no_tumor`, or `pituitary`.")

# --- Main Application Flow ---
# 1. Download and load the model (this will run only once due to @st.cache_resource)
download_model()
model = load_vgg_model()

# 2. File Uploader for MRI Image
uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Open and display the uploaded image
    img = Image.open(uploaded_file).convert("RGB") # Ensure image is in RGB format
    st.image(img, caption="Uploaded MRI Image", use_column_width=True)

    # 3. Prediction Button
    if st.button("Predict Tumor Type"):
        # Show a spinner while predicting
        with st.spinner("Analyzing image..."):
            label, confidence = predict_image(img, model)
        
        # Display the prediction results
        st.success(f"🧠 Predicted Tumor Type: `{label}` with {confidence*100:.2f}% confidence.")
