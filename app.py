import streamlit as st
import os
from PIL import Image
import numpy as np
from utils import load_model, preprocess_image

def main():
    st.title("Building Type Classifier")
    st.write("Upload an image of a building to classify its type.")
    
    # Load the model and class names
    if os.path.exists('building_classifier.h5'):
        model = load_model('building_classifier.h5')
        with open('class_names.txt', 'r') as f:
            class_names = f.read().splitlines()
    else:
        st.error("Model file not found. Please train the model first.")
        return
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Preprocess the image
        processed_image = preprocess_image(uploaded_file)
        
        # Make prediction
        prediction = model.predict(processed_image)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100
        
        # Display results
        st.write("### Prediction Results:")
        st.write(f"Building Type: **{predicted_class}**")
        st.write(f"Confidence: **{confidence:.2f}%**")
        
        # Display all class probabilities
        st.write("### All Class Probabilities:")
        for class_name, prob in zip(class_names, prediction[0]):
            st.write(f"{class_name}: {prob*100:.2f}%")

if __name__ == '__main__':
    main() 