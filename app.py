import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import os
import zipfile
import gdown # Library for downloading Google Drive files
import shutil # For removing directories

# --- Configuration ---
# Google Drive file ID for your vgg16_brain_tumor.keras.zip file.
# IMPORTANT: This ID must be for a zip file where 'vgg16_brain_tumor.keras'
#            is at the root level of the zip's contents.
#            Example: If you open the zip, you should immediately see
#            'vgg16_brain_tumor.keras' folder, and inside that, 'model.weights.h5'.
#            Or even better, if the zip contains directly 'model.weights.h5',
#            'config.json', 'metadata.json'.
#            The previous screenshot showed 'vgg16_brain_tumor.keras' as the directory
#            containing these files. So, the zip should contain the *contents*
#            of that 'vgg16_brain_tumor.keras' folder, or the folder itself at the root.
#            Given the error, it's likely the zip contains:
#            your_zip.zip/
#            └── vgg16_brain_tumor.keras/
#                ├── config.json
#                ├── metadata.json
#                └── model.weights.h5
#            This is what the code expects.
GDRIVE_FILE_ID = "1UUKRfakOuIGlFlaH0vCkUEoyDRbKP2vx" 

# The name of the directory that will be extracted (e.g., 'vgg16_brain_tumor.keras')
MODEL_ROOT_DIR = "vgg16_brain_tumor.keras" 

# The full path to the actual .h5 model file inside the extracted directory
MODEL_FILE_PATH = os.path.join(MODEL_ROOT_DIR, "model.weights.h5")

# Class names for prediction results
CLASS_NAMES = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

# --- Model Download and Extraction Function ---
@st.cache_resource
def download_and_extract_model():
    """
    Downloads the model zip from Google Drive and extracts it.
    Uses st.cache_resource to ensure this runs only once.
    """
    # Check if the expected model weights file already exists
    if os.path.exists(MODEL_FILE_PATH):
        st.success("Model already downloaded and extracted.")
        return

    st.info("Downloading model, please wait... This may take a moment.")
    
    # Define a temporary path for the downloaded zip file
    zip_file_path = "model.zip" 
    
    try:
        # Use gdown to download the file directly from Google Drive
        # 'quiet=False' shows progress, 'fuzzy=True' allows slight variations in ID
        gdown.download(id=GDRIVE_FILE_ID, output=zip_file_path, quiet=False, fuzzy=True)
        
        # Verify if the zip file was actually downloaded
        if not os.path.exists(zip_file_path):
            st.error("Gdown failed to download the model zip. Please check the Google Drive file ID or its public accessibility permissions.")
            st.stop()

        st.success("Model zip downloaded successfully. Extracting...")

        # Ensure the target directory for extraction is clean before extracting
        if os.path.exists(MODEL_ROOT_DIR):
            st.warning(f"Removing existing directory '{MODEL_ROOT_DIR}' to ensure a fresh extraction.")
            shutil.rmtree(MODEL_ROOT_DIR)

        # Extract the downloaded zip file
        with zipfile.ZipFile(zip_file_path, 'r') as z:
            # Extract all contents to the current directory ('.').
            # This assumes 'vgg16_brain_tumor.keras' folder is at the root of the zip.
            z.extractall(".") 
        st.success("Model extracted successfully.")
        
        # Verify that the expected model file exists after extraction
        if not os.path.exists(MODEL_FILE_PATH):
            st.error(f"Error: Expected model file '{MODEL_FILE_PATH}' not found after extraction. "
                     "Please check the zip's internal structure. The 'model.weights.h5' "
                     f"should be directly inside the '{MODEL_ROOT_DIR}' folder within the zip.")
            st.stop()
        else:
            st.info(f"Confirmed model weights file exists at: {MODEL_FILE_PATH}")

    except Exception as e:
        st.error(f"An error occurred during model download or extraction: {e}. "
                 "Please ensure the Google Drive file is publicly accessible and the ID is correct.")
        st.stop()
    finally:
        # Clean up the temporary zip file after extraction (or failure)
        if os.path.exists(zip_file_path):
            os.remove(zip_file_path)
            st.info("Cleaned up temporary zip file.")

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
download_and_extract_model()
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
