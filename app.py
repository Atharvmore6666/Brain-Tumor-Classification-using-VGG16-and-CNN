import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Set Streamlit page config
st.set_page_config(page_title="Brain Tumor Classifier", layout="centered")

# Define class labels (must match training order)
CLASS_NAMES = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

# Load the TFLite model
@st.cache_resource
def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
    return interpreter

# Prediction function
def predict_image(interpreter, image):
    # Preprocess image
    image = image.resize((128, 128))
    img_array = np.array(image).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Set input tensor
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()

    # Get prediction
    prediction = interpreter.get_tensor(output_details[0]['index'])[0]
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction)
    return predicted_class, confidence

# UI
st.title("🧠 Brain Tumor Classification using VGG16 (TFLite)")
st.markdown("Upload an MRI image to classify it into one of the four tumor types.")

# File uploader
uploaded_file = st.file_uploader("Choose an MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict Tumor Type"):
        interpreter = load_tflite_model()
        label, confidence = predict_image(interpreter, image)
        st.success(f"🧠 Predicted Tumor Type: `{label}`")
        st.info(f"Confidence: {confidence:.2%}")

