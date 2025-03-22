from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel
import torch
from PIL import Image
import io
import numpy as np
from groundingdino.util.inference import load_model, load_image, predict

app = FastAPI()

# Force CPU usage
device = torch.device("cpu")

# Load model once at startup
print("🔄 Loading Grounding DINO Model...")
model = load_model( "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",  r"C:/Users/hp/Desktop/Grounding DINO Installation/GroundingDINO/weights/groundingdino_swint_ogc.pth")
print("✅ Model loaded successfully!")

class PredictionResponse(BaseModel):
    boxes: list
    labels: list

@app.post("/predict/", response_model=PredictionResponse)
async def predict_dino(
    file: UploadFile = File(...),
    caption: str = Form(...),
    box_threshold: float = Form(0.35),
    text_threshold: float = Form(0.25)
):
    # Read the uploaded image
    contents = await file.read()
    image_pil = Image.open(io.BytesIO(contents)).convert("RGB")
    
    # Correctly unpack the tuple returned by load_image
    image, transformed_image = load_image(image_pil)
    
    # Call the prediction function with all required arguments
    boxes, logits, phrases = predict(
        model=model,
        image=transformed_image,  # Pass only the transformed image tensor
        caption=caption,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device=device  # Explicitly pass CPU device
    )
    
    # Convert tensors to lists for JSON serialization
    boxes_list = boxes.tolist() if isinstance(boxes, torch.Tensor) else boxes
    
    return {
        "boxes": boxes_list,
        "labels": phrases
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)