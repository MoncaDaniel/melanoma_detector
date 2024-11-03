import os
import logging
import gradio as gr
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import traceback

# Disable GPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Starting application")
logger.info(f"Python version: {os.sys.version}")
logger.info(f"Installed packages: {os.popen('pip freeze').read()}")

try:
    # Load the model
    model = load_model("melanoma_model.keras")
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error("Error loading model", exc_info=True)
    raise e

# Prediction function with threshold
def predict_image(image):
    try:
        # Preprocess the image
        image = image.convert("RGB")
        image = image.resize((128, 128))  # Resize to model input size
        image = img_to_array(image) / 255.0  # Normalize
        image = np.expand_dims(image, axis=0)  # Add batch dimension

        # Make prediction
        prediction = model.predict(image)
        confidence = float(prediction[0][0])

        # Classification based on threshold
        result = "Melanoma" if confidence >= 0.7 else "Benign"
        return f"Result: {result} - Confidence: {confidence:.2f}"
    except Exception as e:
        # Log error and return message
        logger.error("Prediction error:", exc_info=True)
        return "An error occurred during prediction. Please contact support at your_email@example.com."

# Gradio app with example image and updated interface
with gr.Blocks() as demo:
    gr.Markdown("# 🩺 Melanoma Detection App")
    gr.Markdown("""
    This AI tool is designed to help detect melanoma from skin lesion images. However, since we're using only free resources for training and deployment, please follow these instructions to get the best results! 🌟

    ### How to Get the Best Results:
    1. **Crop your image** to look like our example below. This helps the AI focus on the lesion! ✂️
    2. **Avoid body hair** in the image (we know it's natural, but it confuses our model!). 🙆‍♂️
    3. **Upload a high-resolution photo** for better accuracy. 📸

    ⚠️ **Important**: If you're concerned about anything, see a medical professional immediately! This tool is here to support you, but it’s not a substitute for medical advice.  
    Stay healthy and take care! 🍀  
    -- Daniel
    
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

    # Footer with contact info
    gr.Markdown("""
    ---
    <p style="text-align: center; font-size: 16px;">
        Made with ❤️, data, and code by <span style="color: #228B22; font-weight: bold;">Daniel Moncada León</span>.<br>
        <a href="mailto:danielmoncada10@gmail.com">danielmoncada10@gmail.com</a>
    </p>
    """)

# Launch the Gradio app
demo.launch()
