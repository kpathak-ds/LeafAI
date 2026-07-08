# src/inference_pipeline.py
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from ultralytics import YOLO

# Setup device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASSIFIER_PATH = os.path.join(BASE_DIR, "disease_classifier.pth")

# ── Load Models ───────────────────────────────────────────
# Load YOLOv11 leaf detector (using yolo11n.pt)
print("Loading YOLOv11 model...")
yolo_model = YOLO(os.path.join(BASE_DIR, "yolo11n.pt"))

# Load PyTorch EfficientNet Disease & Deficiency Classifier
print("Loading PyTorch disease classifier...")
classifier = models.efficientnet_b0()
num_ftrs = classifier.classifier[1].in_features
classifier.classifier[1] = nn.Linear(num_ftrs, 5)

if os.path.exists(CLASSIFIER_PATH):
    classifier.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=device))
    print(f"Loaded classifier weights from {CLASSIFIER_PATH}")
else:
    print(f"⚠️ Warning: Classifier weights not found at {CLASSIFIER_PATH}. Using untrained model.")

classifier = classifier.to(device)
classifier.eval()

# Transform for classifier input
val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

DISEASE_METADATA = {
    0: {
        "crop": "Snake Gourd",
        "disease": "Healthy",
        "cause": "Optimal soil health and nutrient balance.",
        "treatment": "Maintain current watering cycle and trellis support.",
        "severity": "None"
    },
    1: {
        "crop": "Snake Gourd",
        "disease": "Nitrogen Deficiency",
        "cause": "Depleted nitrogen levels in the soil, hindering vegetative growth.",
        "treatment": "Apply nitrogen-rich fertilizer like urea or organic compost. Increase watering by 20% to help nutrient absorption.",
        "severity": "Medium"
    },
    2: {
        "crop": "Tomato",
        "disease": "Healthy",
        "cause": "Optimal soil health and nutrient balance.",
        "treatment": "Maintain current irrigation and fertilizer schedule. Monitor weekly.",
        "severity": "None"
    },
    3: {
        "crop": "Tomato",
        "disease": "Nitrogen Deficiency",
        "cause": "Insufficient nitrogen in the soil, causing yellowing of older leaves.",
        "treatment": "Apply balanced NPK (19-19-19) fertilizer or nitrogen-rich urea. Add compost/manure.",
        "severity": "Low"
    },
    4: {
        "crop": "Tomato",
        "disease": "Potassium Deficiency / Leaf Blight",
        "cause": "Potassium depletion in soil or early fungal infection (Alternaria solani).",
        "treatment": "Apply Muriate of Potash (MOP) and spray fungicide like Mancozeb (2g/L) for early blight protection.",
        "severity": "Medium"
    }
}

def detect_and_classify_crop(img_rgb):
    """
    1. Detects leaf using YOLOv11.
    2. Crops to leaf bounding box (falls back to original if no detection).
    3. Classifies crop & disease using PyTorch EfficientNet.
    Returns:
        metadata: dict containing:
            crop: str
            disease: str
            confidence: float (0-100)
            cause: str
            treatment: str
            severity: str
        cropped_img_rgb: np.ndarray
    """
    # Stage 1: YOLOv11 Leaf/Object Detection
    # Run inference
    results = yolo_model(img_rgb, verbose=False)
    
    h, w, _ = img_rgb.shape
    x1, y1, x2, y2 = 0, 0, w, h # Default to full image bounds
    
    # Check if we got any detections
    if len(results) > 0 and len(results[0].boxes) > 0:
        # Find box with largest area
        max_area = 0
        for box in results[0].boxes:
            coords = box.xyxy[0].cpu().numpy().astype(int)
            bx1, by1, bx2, by2 = coords
            area = (bx2 - bx1) * (by2 - by1)
            if area > max_area:
                max_area = area
                x1, y1, x2, y2 = bx1, by1, bx2, by2
        
        # Ensure indices are within valid boundaries
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        print(f"YOLOv11: Leaf detected at [{x1}, {y1}, {x2}, {y2}]")
    else:
        print("YOLOv11: No prominent objects detected, falling back to full image.")

    # Crop image
    cropped_img_rgb = img_rgb[y1:y2, x1:x2]

    # Stage 2: PyTorch EfficientNet Disease Classification
    input_tensor = val_transform(cropped_img_rgb).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = classifier(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, predicted = probs.max(1)
        class_idx = predicted.item()
        confidence_pct = round(conf.item() * 100, 1)

    meta = DISEASE_METADATA.get(class_idx, {
        "crop": "Unknown",
        "disease": "Unknown",
        "cause": "Unknown",
        "treatment": "Unknown",
        "severity": "Unknown"
    }).copy()
    
    meta["confidence"] = confidence_pct
    print(f"EfficientNet: Crop={meta['crop']}, Disease={meta['disease']} ({confidence_pct}%)")

    return meta, cropped_img_rgb

if __name__ == "__main__":
    # Quick self-test script
    print("Self-testing inference pipeline...")
    # Create a random RGB image for test
    test_img = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    meta, cropped = detect_and_classify_crop(test_img)
    print("Self-test completed successfully!")
    print(meta)
