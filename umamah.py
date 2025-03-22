from groundingdino.util.inference import load_model, load_image, predict, annotate
import torch
import cv2
import os

# Force torch to use CPU only - add this before any other torch operations
os.environ["CUDA_VISIBLE_DEVICES"] = ""
torch.backends.cudnn.enabled = False

# Load the model
model = load_model(
    "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    r"C:/Users/hp/Desktop/Grounding DINO Installation/GroundingDINO/weights/groundingdino_swint_ogc.pth"
) 

# Explicitly move model to CPU
model = model.to('cpu')

# Image path
IMAGE_PATH = "GroundingDINO/.asset/6F2A5575-1024x683.jpg"
TEXT_PROMPT = "person . mask . hat . vest . gloves . boots ."

BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25

# Load the image
image_source, image = load_image(IMAGE_PATH)

# Override the default device in predict function
device = torch.device('cpu')

# Perform prediction with explicit CPU device
boxes, logits, phrases = predict(
    model=model,
    image=image,
    caption=TEXT_PROMPT,
    box_threshold=BOX_THRESHOLD,
    text_threshold=TEXT_THRESHOLD,
    device=device  # Explicitly pass CPU device if the function accepts it
)

# Annotate the image
annotated_frame = annotate(
    image_source=image_source,
    boxes=boxes,
    logits=logits,
    phrases=phrases
)

# Save the annotated image
cv2.imwrite("annotated_image.jpg", annotated_frame)

print("Inference complete! Check 'annotated_image.jpg'")