import os
import tensorflow as tf
import gradio as gr
import numpy as np
from PIL import Image

# Disable all GPUS
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
current_directory = os.path.abspath(os.path.dirname(__file__))

# Load your pre-trained model
def load_model():
    model = tf.keras.models.load_model(os.path.join(current_directory, "model.h5")) # Replace with your model's path
    return model

model = load_model()

# Define the labels (categories)
labels = ['Water', 'Cloudy', 'Desert', 'Green Area']

# Function to preprocess the image and predict the class
def classify_image(image):
    # Ensure the image is in PIL format
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    
    img = image.resize((128, 128)) # Resize the image
    img = np.array(img) / 255.0 # Normalize the image
    img = np.expand_dims(img, axis=0) # Add batch dimension
    prediction = model.predict(img)
    predicted_class = labels[np.argmax(prediction)]

    # Prepare output with probabilities
    return {labels[i]: float(prediction[0][i]) for i in range(len(labels))}

# Define the Gradio interface
image_input = gr.Image(type="pil")  # Use "pil" as the type for PIL images
label_output = gr.Label(num_top_classes=4)

# Launch the interface
gr.Interface(fn=classify_image,
             inputs=image_input,
             outputs=label_output,
             title="Satellite Image Classification",
             description="Classify satellite images into four types: Water, Cloudy, Desert, Green Area").launch()
