import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained ResNet50 model
@st.cache_resource
def load_trained_model():
    try:
        model = tf.keras.models.load_model("resenet50_model.keras")  # Ensure correct filename
        st.success("✅ Model Loaded Successfully!")
        return model
    except Exception as e:
        st.error(f"🚨 Model file not found at: resnet50_model.keras. Please check the file path!")
        return None

# Load the model
model = load_trained_model()

# Sidebar for navigation
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Image Classification"])

# Home Page
if app_mode == "Home":
    st.header("ResNet50 Image Classification Web App")
    st.image("b.jpg", use_container_width=True)
    st.markdown("""
    ### Welcome to the ResNet50 Image Classifier! 🎉  
    This application allows you to upload an image and get predictions based on the ResNet50 deep learning model.
    
    #### How It Works:
    1. Navigate to Image Classification from the sidebar.
    2. Upload an image of any object or scene.
    3. Our model will analyze the image and provide a classification result.
    
    👉 Click on the Image Classification page in the sidebar to get started!
    """)

# About Page
elif app_mode == "About":
    st.header("About the Project")
    st.markdown("""
    ### About This Model
    This image classifier is powered by ResNet50, a deep learning model pre-trained on a large dataset. It recognizes a wide variety of images and provides predictions with high accuracy.

    ### Features:
    - Deep Learning Based: Uses ResNet50 to classify images.
    - User-Friendly Interface: Simple image upload and prediction.
    - Fast Predictions: Get results in real-time.
    
    #### Dataset Used:
    The model has been trained using a dataset containing a diverse range of images, ensuring high performance.
    """)

# Image Classification Page
elif app_mode == "Image Classification":
    st.header("Image Classification with ResNet50")
    
    uploaded_file = st.file_uploader("Upload an Image:", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Ensure image is in RGB mode
        image = image.convert("RGB")  # Convert to RGB (Fix RGBA/Grayscale issues)
        image = image.resize((224, 224))  # Resize to ResNet50 input size

        # Convert image to numpy array
        image_array = np.array(image) / 255.0  # Normalize pixel values
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

        if st.button("Predict"):
            if model is None:
                st.error("🚨 Model not loaded. Please check the file path.")
            else:
                st.write("🔍 Analyzing the image...")
                predictions = model.predict(image_array)
                predicted_class = np.argmax(predictions)

                # Define class labels (Ensure these match your trained model)
                class_labels = ["E. coli", "Staphylococcus"]  # Update with actual labels

                st.success(f"🧬 Predicted Class: *{class_labels[predicted_class]}*")