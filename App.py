import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model('brain_tumor_classifier_mobilenet.h5')

# Function to predict the uploaded image
def predict_image(image):
    img = image.resize((224, 224))  # Resize to match the model's input size
    img_array = img_to_array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    prediction = model.predict(img_array)
    if prediction[0][0] > 0.5:
        return "Tumor", prediction[0][0]
    else:
        return "No Tumor", prediction[0][0]

# Streamlit application
def main():
    st.title("Brain Tumor Classification with MobileNet")
    st.write("Upload an MRI image to classify it as 'Tumor' or 'No Tumor'.")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an MRI image file", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Perform prediction
        st.write("Classifying...")
        label, confidence = predict_image(image)
        
        # Display the result
        st.write(f"Prediction: **{label}**")
        st.write(f"Confidence: **{confidence:.2f}**")

if __name__ == "__main__":
    main()
