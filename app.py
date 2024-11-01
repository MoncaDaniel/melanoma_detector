import gradio as gr
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np

# Load the model
model = load_model("melanoma_model.keras")

# Function for prediction
def predict_image(image):
    # Preprocess the image to match model's input requirements
    image = image.convert("RGB")
    image = image.resize((128, 128))  # Resize to the model’s input size
    image = img_to_array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(image)
    confidence = float(prediction[0][0])

    # Set threshold for diagnosis
    result = "Melanoma" if confidence >= 0.6 else "Benign"
    return f"Result: {result} - Confidence: {confidence:.2f}"

# UI with Gradio Blocks
with gr.Blocks() as demo:
    gr.Markdown("# Melanoma Detection App")
    gr.Markdown("""
    This application analyzes skin lesion images and predicts whether the lesion is likely melanoma or benign.
    **Instructions**: Upload a clear, zoomed-in image of the skin lesion, crop it as needed, and click "Submit" for analysis.
    
    ### Example Image
    Below is an example image showing the ideal close-up, zoomed format.
    """)
    
    # Example image display
    example_image = gr.Image("example_melanoma.jpg", label="Example Image (Zoomed and Cropped)")

    # Image upload with cropping instructions
    gr.Markdown("### Upload and Crop Your Image")

    # Language selection dropdown
    language = gr.Dropdown(choices=["English", "Spanish", "French"], value="English", label="Select Language")

    # Image upload section
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Your Image")
            submit_btn = gr.Button("Submit for Analysis")
        
        with gr.Column():
            result_output = gr.Textbox(label="Diagnosis Result", placeholder="Result will appear here")

    submit_btn.click(predict_image, inputs=image_input, outputs=result_output)

demo.launch()
