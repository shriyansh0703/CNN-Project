import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the model
model = load_model('image_classification_model.h5')

# Define class labels
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer',
          'dog', 'frog', 'horse', 'ship', 'truck']

# Streamlit app
def preprocess_image(image):
    """Preprocess the image to match the input shape of the model."""
    # Resize image to 32x32 (the input size of the model)
    image = image.resize((32, 32))
    # Convert image to array
    image_array = np.array(image)
    # Normalize pixel values to [0, 1]
    image_array = image_array / 255.0
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def predict_class(image_array):
    """Predict the class of the given image array."""
    predictions = model.predict(image_array)
    class_index = np.argmax(predictions)
    return labels[class_index]

# Streamlit UI
st.title('CIFAR-10 Image Classification')
st.write('Upload an image and the model will classify it.')

# Image upload
uploaded_image = st.file_uploader("Choose an image...", type="png")
if uploaded_image:
    # Open the image file
    image = Image.open(uploaded_image)
    
    # Preprocess the image
    image_array = preprocess_image(image)
    
    # Predict the class
    predicted_class = predict_class(image_array)
    
    # Display the image and prediction
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write(f'Prediction: {predicted_class}')
