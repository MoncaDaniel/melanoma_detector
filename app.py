import gradio as gr
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np

# Load the model (ensure melanoma_model.keras is uploaded in the Space)
model = load_model("melanoma_model.keras")

def predict_image(image):
    # Preprocess the image
    image = image.convert("RGB")
    image = image.resize((128, 128))  # Resize to model input size
    image = img_to_array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Predict
    prediction = model.predict(image)
    confidence = float(prediction[0][0])

    # Set threshold at 60%
    result = "Melanoma" if confidence >= 0.6 else "Benign"
    return f"Result: {result} - Confidence: {confidence:.2f}"

# Define the Gradio interface
iface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Melanoma Detector",
    description="Upload an image to analyze if it's likely melanoma or benign."
)

iface.launch()
