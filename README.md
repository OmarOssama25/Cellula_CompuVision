# Teeth Classification Model

A deep learning model developed to classify teeth into 7 distinct categories, aimed at enhancing diagnostic precision in dental care. This project leverages TensorFlow and is deployed via a Streamlit web application.

## Introduction

This project is part of Cellula Technologies' AI-driven dental solutions. It involves creating a comprehensive teeth classification model to assist in accurate teeth identification and diagnosis.

## Objective

To preprocess, visualize, and train a robust computer vision model capable of classifying dental images into 7 categories. This model will be crucial in improving patient outcomes by supporting dental professionals in diagnosis.

## Dataset

The dataset consists of dental images that have been preprocessed through normalization and augmentation. Visualization of class distribution was performed to ensure dataset balance.

- **Categories**: 7 distinct tooth classes.

## Preprocessing and Data Augmentation

Dental images were preprocessed to ensure optimal model performance:

- **Normalization**: Adjusted pixel values for consistency.
- **Augmentation**: Applied transformations such as rotation, scaling, and flipping to increase dataset variety.

## Model Architecture

The model was developed using TensorFlow, with a custom architecture designed for image classification. Key highlights include:

- **Layers**: Conv2D, MaxPooling
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Callbacks**:
  - ReduceLROnPlateau
  - EarlyStopping

## Training Process

The model was trained with the following configuration:

- **Training Accuracy**: 94.5%
- **Validation Accuracy**: 99%
- **Test Accuracy**: 99.6%
- **Epochs**: 150
- **Batch Size**: 64

## Evaluation

The model achieved high performance, with test accuracy reaching 99.6%. Below are some sample predictions and their corresponding true labels:

[Insert images and descriptions of model performance, confusion matrix, etc.]

## Deployment

The model has been deployed using Streamlit. Users can upload dental images to receive real-time predictions.

To run the app locally:

1. Clone the repository.
2. Install the required dependencies: `pip install -r requirements.txt`.
3. Run the Streamlit app: `streamlit run app.py`.

## Usage

1. Clone the repository.
2. Install the dependencies.
3. Use the provided Streamlit app to classify dental images.
4. Optionally, retrain the model with new data.

## Future Improvements

- Expand the dataset to include more diverse dental images.
- Fine-tune the model for real-time deployment in dental clinics.
- Explore the use of advanced techniques like transfer learning to further enhance accuracy.

## Contributors

- **Omar Ossama**
