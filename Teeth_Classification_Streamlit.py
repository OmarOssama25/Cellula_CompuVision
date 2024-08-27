import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import requests
import os

# Function to download the model from Google Drive
def download_model(url, destination_file_name):
    response = requests.get(url, stream=True)
    with open(destination_file_name, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

# Path to save the downloaded model
model_path = r'C:\Users\HP\Desktop\Teeth_Classification_Streamlit\teeth_classification_model.h5'
model_url = 'https://drive.google.com/uc?export=download&id=1MSsULGhzoW8W1mpozWedAO-ABZdxMPxr'

# Download the model file
if not os.path.exists(model_path):
    st.write("Downloading the model...")
    download_model(model_url, model_path)

# Load the model
try:
    model = tf.keras.models.load_model(model_path)
    st.success('Model loaded successfully.')
except OSError as e:
    st.error(f'Error loading model: {e}')
    st.stop()  # Stop the app if the model cannot be loaded
    
# Preprocess function
def preprocess_image(image):
    image = image.resize((150, 150))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Class labels and descriptions
class_labels = ['CaS', 'CoS', 'Gum', 'MC', 'OC', 'OLP', 'OT']
class_descriptions = {
    'CaS': 'Caries (Cavities): Destruction of tooth enamel caused by bacteria producing acids. It often results in holes or decay in the teeth.',
    'CoS': 'Cosmetic Dentistry: Treatments aimed at improving the appearance of teeth, gums, and smile. This includes whitening, veneers, and bonding.',
    'Gum': 'Gum Disease (Periodontal Disease): Infection and inflammation of the gums and supporting structures of the teeth, which can lead to tooth loss if untreated.',
    'MC': 'Mouth Cancer: Malignancy that occurs in the mouth or throat. Symptoms include sores, lumps, or white/red patches in the mouth.',
    'OC': 'Oral Candidiasis: Fungal infection in the mouth caused by Candida species, characterized by white patches on the tongue and inside of the mouth.',
    'OLP': 'Oral Lichen Planus: A chronic inflammatory condition affecting the mucous membranes inside the mouth, presenting as white patches or painful sores.',
    'OT': 'Other Conditions: Miscellaneous dental conditions not classified under the above categories, including trauma, unusual growths, or rare diseases.'
}

# Streamlit app
st.title("Teeth Disease Classification")
st.write("Upload an image to classify and get detailed predictions.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Show a spinner while processing
    with st.spinner('Classifying image...'):
        processed_image = preprocess_image(image)
        
        # Predict
        predictions = model.predict(processed_image)
        predicted_class = np.argmax(predictions, axis=1)[0]
        
        # Display results
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Prediction Result")
            st.write(f"**Predicted Class:** {class_labels[predicted_class]}")
            
            # Display prediction probabilities
            st.write("**Prediction Probabilities:**")
            for i, label in enumerate(class_labels):
                st.write(f"{label}: {predictions[0][i]*100:.2f}%")
        
        with col2:
            st.subheader("Class Descriptions")
            st.write(class_descriptions[class_labels[predicted_class]])
            
            # Optional: Display images or additional information for each class
            # Example:
            # st.image('path_to_image_for_each_class', caption='Class Description')
else:
    st.info("Upload an image to classify.")

# Optional: Add a help section or documentation
st.sidebar.title("Help")
st.sidebar.write("""
    ## How to Use
    1. Upload an image of a tooth condition.
    2. The model will classify the image and provide the most likely disease classification.
    3. View the prediction result and probabilities for each class.
    4. Get a brief description of the predicted class.

    If you have any questions or need further assistance, feel free to reach out to us!
""")
