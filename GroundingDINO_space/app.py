import gradio as gr
from fastapi import FastAPI
from groundingdino.util.inference import load_model, load_image, predict
from PIL import Image
import torch

# Load your model
model = load_model(
    "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", 
    "path_to_your_model_weights.pth", 
    device="cpu"
)

def predict_image(image: Image.Image, caption: str, box_threshold: float = 0.35, text_threshold: float = 0.25):
    # Preprocess image and make predictions
    transformed_image = load_image(image)
    
    # Run prediction
    boxes, logits, phrases = predict(
        model=model,
        image=transformed_image,
        caption=caption,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device="cpu"  # Explicitly pass CPU device
    )

    # Convert to list for JSON serialization
    boxes_list = boxes.tolist() if isinstance(boxes, torch.Tensor) else boxes

    return boxes_list, phrases

# Set up Gradio interface
iface = gr.Interface(
    fn=predict_image,
    inputs=[
        gr.Image(type="pil"),  # Image input
        gr.Textbox(label="Caption", placeholder="Enter caption", lines=2),  # Caption input
        gr.Slider(minimum=0, maximum=1, step=0.01, value=0.35, label="Box Threshold"),  # Slider for box threshold
        gr.Slider(minimum=0, maximum=1, step=0.01, value=0.25, label="Text Threshold")  # Slider for text threshold
    ],
    outputs=[
        gr.JSON(label="Boxes"),
        gr.JSON(label="Labels")
    ],
    live=True
)

# Launch the Gradio app
iface.launch()
