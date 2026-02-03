import torch
import cv2
import numpy as np
from model import MultiTaskUNet

IMAGE_PATH = "sample.png"
MODEL_PATH = "unet_mtl_final.pth"
IMG_SIZE = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MultiTaskUNet().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

image = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
orig_h, orig_w = image.shape

image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
image = image / 255.0
image = torch.tensor(image).unsqueeze(0).unsqueeze(0).float().to(device)

with torch.no_grad():
    seg_logits, cls_logits = model(image)

seg_mask = torch.sigmoid(seg_logits).cpu().numpy()[0, 0]
seg_mask = (seg_mask > 0.3).astype(np.uint8)
seg_mask = cv2.resize(seg_mask, (orig_w, orig_h))

caries_prob = torch.sigmoid(cls_logits).item()
diagnosis = "Caries" if caries_prob > 0.5 else "Normal"

print(f"Diagnosis: {diagnosis} ({caries_prob:.3f})")

cv2.imwrite("predicted_mask.png", seg_mask * 255)
