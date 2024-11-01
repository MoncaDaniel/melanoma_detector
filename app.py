import gradio as gr
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image, ImageOps
import numpy as np

# Load the model
model = load_model("melanoma_model.keras")

# Function for prediction
def predict_image(image):
    image = image.convert("RGB")
    image = image.resize((128, 128))  # Resize to model input size
    image = img_to_array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    prediction = model.predict(image)
    confidence = float(prediction[0][0])

    result = "Melanoma" if confidence >= 0.6 else "Benign"
    return f"Result: {result} - Confidence: {confidence:.2f}"

# Update the interface layout and add instructional content
with gr.Blocks() as demo:
    # Title and introduction
    gr.Markdown("# Melanoma Detection App")
    gr.Markdown("""
    This application analyzes skin lesion images and predicts whether the lesion is likely melanoma or benign.  
    **Instructions**: Upload a clear, zoomed-in image of the skin lesion, crop it as needed, and click "Submit" for analysis.
    """)

    # Add a language selection dropdown if multi-language support is still needed
    language = gr.Dropdown(choices=["English", "Spanish", "French"], value="English", label="Select Language")

    # Image upload and cropping section
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload and Crop Your Image", tool="editor")
            submit_btn = gr.Button("Submit for Analysis")
        
        with gr.Column():
            result_output = gr.Textbox(label="Diagnosis Result", placeholder="Result will appear here")

    submit_btn.click(predict_image, inputs=image_input, outputs=result_output)

demo.launch()
