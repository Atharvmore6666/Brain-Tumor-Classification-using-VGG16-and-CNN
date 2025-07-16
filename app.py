import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import requests, zipfile, io, os

# --- Streamlit Page Configuration ---
# This must be called only once and as the very first Streamlit command.
st.set_page_config(page_title="Brain Tumor Classifier", layout="centered", icon="🧠")

# --- Configuration ---
# Google Drive download link for the .keras model file
GDRIVE_ZIP_URL = "https://drive.google.com/file/d/1UUKRfakOuIGlFlaH0vCkUEoyDRbKP2vx/view?usp=drive_link"
# The name of the model file after it's extracted from the zip
MODEL_FILE_NAME = "vgg16_brain_tumor.keras" 

# Class names for the tumor types
CLASS_NAMES = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

# --- Model Download and Loading Functions ---

def download_and_extract_model():
    """
    Downloads the model zip file from Google Drive and extracts the .keras model
    file to the current directory if it doesn't already exist.
    """
    if not os.path.exists(MODEL_FILE_NAME):
        st.info("Downloading model, please wait... This may take a moment.")
        try:
            response = requests.get(GDRIVE_ZIP_URL, stream=True)
            response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)

            # Check if the response content is a valid zip file
            if response.headers.get('Content-Type') == 'application/zip' or 'zip' in response.headers.get('Content-Type', ''):
                with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                    # Assuming the zip contains the 'vgg16_brain_tumor.keras' file directly at its root
                    z.extractall(".") # Extract to the current directory
                st.success("Model downloaded and extracted successfully!")
            else:
                st.error("Downloaded file is not a valid zip archive. Please check the Google Drive URL.")
                st.stop() # Stop execution if download fails or is not a zip
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to download model: {e}. Please check the file ID or link and your internet connection.")
            st.stop() # Stop execution if download fails
    else:
        st.success("Model already exists locally.")

@st.cache_resource
def load_vgg_model():
    """
    Loads the pre-trained VGG16 brain tumor classification model.
    Uses st.cache_resource to cache the model, so it's loaded only once.
    """
    st.info("Loading model...")
    try:
        # Load the model directly from the .keras file
        model = tf.keras.models.load_model(MODEL_FILE_NAME)
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}. Ensure '{MODEL_FILE_NAME}' is present and valid.")
        st.stop() # Stop execution if model loading fails

# --- Prediction Function ---

def predict_image(img: Image.Image, model: tf.keras.Model):
    """
    Preprocesses the input image and makes a prediction using the loaded model.

    Args:
        img (PIL.Image.Image): The input MRI image.
        model (tf.keras.Model): The loaded TensorFlow Keras model.

    Returns:
        tuple: A tuple containing the predicted class name (str) and confidence (float).
    """
    # Resize image to the model's expected input size (128x128)
    img = img.resize((128, 128))
    # Convert image to a NumPy array and normalize pixel values to [0, 1]
    img_array = image.img_to_array(img) / 255.0
    # Add a batch dimension (model expects input shape like (batch_size, height, width, channels))
    img_array = np.expand_dims(img_array, axis=0)

    # Make predictions
    predictions = model.predict(img_array)
    # Get the index of the class with the highest probability
    predicted_class_index = np.argmax(predictions)
    # Get the predicted class name
    predicted_class = CLASS_NAMES[predicted_class_index]
    # Get the confidence (probability) of the predicted class
    confidence = np.max(predictions)

    return predicted_class, confidence

# --- Streamlit UI ---

def main():
    """
    Main function to set up the Streamlit application UI and logic.
    """
    st.title("🧠 Brain Tumor Classification using VGG16")
    st.markdown("Upload an MRI brain image to classify it into one of the four tumor types: `glioma`, `meningioma`, `no_tumor`, or `pituitary`.")
    st.markdown("---")

    # Download and load model (these functions are cached, so they run only once)
    download_and_extract_model()
    model = load_vgg_model()

    # File uploader widget
    uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        try:
            # Open the uploaded image and convert to RGB (ensures consistency)
            img = Image.open(uploaded_file).convert("RGB")
            st.image(img, caption="Uploaded MRI Image", use_column_width=True)
            st.markdown("---")

            # Prediction button
            if st.button("Predict Tumor Type"):
                with st.spinner("Classifying image..."):
                    # Perform prediction using the loaded model
                    label, confidence = predict_image(img, model)
                
                st.success(f"**Predicted Tumor Type:** `{label}`")
                st.info(f"**Confidence:** `{confidence*100:.2f}%`")
                st.markdown("---")

        except Exception as e:
            st.error(f"An error occurred while processing the image: {e}")
            st.warning("Please ensure you upload a valid image file.")

if __name__ == "__main__":
    main()
