import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the model
model = tf.keras.models.load_model('teeth_classification_model.h5')

# Preprocess function
def preprocess_image(image):
    image = image.resize((100, 100))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Class labels and descriptions
class_labels = ['CaS', 'CoS', 'Gum', 'MC', 'OC', 'OLP', 'OT']
class_descriptions = {
    'CaS': 'Caries (Cavities): Cavities occur when tooth decay destroys the tooth\'s enamel and underlying layers. They are caused by bacteria that feed on sugars, producing acids that erode the tooth surface. Treatments include fluoride, fillings, crowns, or root canals for severe cases.',
    'CoS': 'Cosmetic Dentistry: This includes treatments such as whitening, veneers, and bonding aimed at improving the appearance of teeth, gums, and smile. Procedures vary depending on the desired aesthetic outcome.',
    'Gum': 'Gum Disease (Periodontal Disease): Gum disease ranges from gingivitis (mild inflammation) to periodontitis, where the infection damages tissues and bones supporting the teeth. Early detection and treatments like scaling, antibiotics, or surgery can prevent tooth loss.',
    'MC': 'Mouth Cancer: Oral cancer can develop in the mouth or throat, often manifesting as persistent sores, lumps, or abnormal patches. Early detection is key to effective treatment through surgery, radiation, or chemotherapy.',
    'OC': 'Oral Candidiasis (Thrush): A fungal infection caused by Candida species, oral thrush presents as white patches on the tongue or mouth lining. It can be treated with antifungal medications.',
    'OLP': 'Oral Lichen Planus: A chronic inflammatory condition that affects the mucous membranes inside the mouth, causing white patches or painful sores. Treatment focuses on managing symptoms and preventing flare-ups.',
    'OT': 'Other Conditions: This includes a range of other dental problems, such as dental abscesses (infection-induced pus pockets), impacted teeth (teeth stuck under the gum), and TMJ disorders (jaw pain).'
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
