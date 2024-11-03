import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Starting application")
logger.info(f"Python version: {os.sys.version}")
logger.info(f"Installed packages: {os.popen('pip freeze').read()}")

try:
    # Place code for loading the model or initializing Gradio here
    logger.info("Model and Gradio initialized successfully")
except Exception as e:
    logger.error("Error during startup", exc_info=True)
    raise e


import gradio as gr
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np

# Load the model
model = load_model("melanoma_model.keras")

# Function for prediction with a threshold of 70%
def predict_image(image):
    # Preprocess the image
    image = image.convert("RGB")
    image = image.resize((128, 128))  # Ensure correct input size
    image = img_to_array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(image)
    confidence = float(prediction[0][0])

    # Updated threshold for melanoma classification
    result = "Melanoma" if confidence >= 0.7 else "Benign"
    return f"Result: {result} - Confidence: {confidence:.2f}"

# Gradio app with example image and updated interface
with gr.Blocks() as demo:
    gr.Markdown("# Melanoma Detection App")
    gr.Markdown("""
    This application analyzes skin lesion images and predicts whether the lesion is likely melanoma or benign.
    
    **Instructions**: Please upload a clear, zoomed-in image of the skin lesion, similar to the example image below. Ensure the lesion area fills most of the image frame for accurate analysis.
    
    ### Example Image
    Below is an example showing the ideal format and zoom level for your image.
    """)
    
    # Display example image as a guide
    example_image = gr.Image("example_melanoma.jpg", label="Example Image (Zoomed and Cropped)")

    gr.Markdown("### Upload Your Image for Analysis")
    
    # Image upload and result output section
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Your Image")
            submit_btn = gr.Button("Submit for Analysis")
        
        with gr.Column():
            result_output = gr.Textbox(label="Diagnosis Result", placeholder="The result will appear here")

    # Trigger prediction on button click
    submit_btn.click(predict_image, inputs=image_input, outputs=result_output)

demo.launch()
