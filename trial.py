import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf  # or torch

# Load the model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model.h5")  # Change based on framework
    return model

model = load_model()

# Define a function to make predictions
def predict(image):
    img_array = np.array(image.resize((224, 224)))  # Resize to model input shape
    img_array = img_array / 255.0  # Normalize if required
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    predictions = model.predict(img_array)
    return predictions

# Streamlit App
st.title("Deep Learning Model Deployment with Streamlit")

# Upload Image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display Image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Predict
    st.write("Processing...")
    predictions = predict(image)
    st.write(f"Predictions: {predictions}")
