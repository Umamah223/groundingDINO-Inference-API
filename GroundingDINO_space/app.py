import gradio as gr
from fastapi import FastAPI
from groundingdino.util.inference import load_model, load_image, predict, annotate
from PIL import Image, ImageDraw, ImageFont
import torch
import numpy as np
import cv2  # Add this import

# Load your model
model = load_model(
    "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", 
    r"C:/Users/hp/Desktop/Grounding DINO Installation/GroundingDINO/weights/groundingdino_swint_ogc.pth", 
    device="cpu"
)

def predict_image(image: Image.Image, caption: str, box_threshold: float = 0.35, text_threshold: float = 0.25):
    original_image, transformed_image = load_image(image)
    
    # Run prediction
    boxes, logits, phrases = predict(
        model=model,
        image=transformed_image,
        caption=caption,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device="cpu"
    )

    # Annotate the image using Grounding DINO's built-in function
    annotated_image = annotate(
        image_source=np.array(original_image),  # Convert PIL to NumPy
        boxes=boxes,
        logits=logits,
        phrases=phrases
    )

    # Convert back to PIL for Gradio output
    annotated_image_pil = Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))

    return annotated_image_pil, boxes.tolist(), phrases

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
        gr.Image(label="Detection Result"),  # Add this to display the image with boxes
        gr.JSON(label="Boxes"),
        gr.JSON(label="Labels")
    ],
    live=True
)

# Launch the Gradio app
iface.launch()
