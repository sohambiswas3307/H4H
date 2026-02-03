import torch
import cv2
import numpy as np
from model import UNet

# ---------------- CONFIG ----------------
IMAGE_PATH = "benign-40.png"      # <-- path to image
MODEL_PATH = "unet_final.pth"      # <-- trained model
IMG_SIZE = 256
THRESHOLD = 0.3

# ---------------- DEVICE ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------------- LOAD MODEL ----------------
model = UNet().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ---------------- LOAD IMAGE ----------------
image = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)

if image is None:
    raise RuntimeError("Image not found!")

orig_h, orig_w = image.shape

image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
image = image / 255.0
image = torch.tensor(image).unsqueeze(0).unsqueeze(0).float().to(device)

# ---------------- PREDICT ----------------
with torch.no_grad():
    logits = model(image)
    probs = torch.sigmoid(logits)
    mask = (probs > THRESHOLD).float()

mask = mask.squeeze().cpu().numpy()
mask = cv2.resize(mask, (orig_w, orig_h))

kernel = np.ones((5, 5), np.uint8)
mask = cv2.dilate(mask, kernel, iterations=2)

mask = cv2.GaussianBlur(mask.astype(np.float32), (5, 5), 0)
mask = (mask > 0.2).astype(np.uint8)

# ---------------- TRANSLUCENT OVERLAY ----------------

# Reload original image (grayscale)
orig = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
orig = cv2.resize(orig, (orig_w, orig_h))

# Convert to BGR for color overlay
orig_bgr = cv2.cvtColor(orig, cv2.COLOR_GRAY2BGR)

# Create a colored mask (choose color here)
color_mask = np.zeros_like(orig_bgr)

# Green mask (medical standard)
color_mask[:, :, 1] = (mask * 255).astype(np.uint8)

# Blend original + mask
alpha = 0.35   # transparency (0.2–0.4 is best)
overlay = cv2.addWeighted(color_mask, alpha, orig_bgr, 1 - alpha, 0)

# Save results
cv2.imwrite("overlay.png", overlay)
print("Saved overlay.png")

# ---------------- SAVE OUTPUT ----------------
cv2.imwrite("predicted_mask.png", (mask * 255).astype(np.uint8))
print("Saved predicted_mask.png")
