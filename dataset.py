import os
import cv2
import torch
from torch.utils.data import Dataset

class DentalSegDataset(Dataset):
    def __init__(self, root_dir, img_size=256):
        self.samples = []
        self.img_size = img_size

        for folder in ["Carries", "Normal"]:
            folder_path = os.path.join(root_dir, folder)
            label = 1 if folder == "Carries" else 0

            for file in os.listdir(folder_path):
                if file.endswith(".png") and "mask" not in file:
                    img_path = os.path.join(folder_path, file)
                    mask_path = os.path.join(
                        folder_path,
                        file.replace(".png", "-mask.png")
                    )

                    if os.path.exists(mask_path):
                        self.samples.append((img_path, mask_path, label))
                    else:
                        print(f"[WARNING] Missing mask for {file}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, label = self.samples[idx]

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        image = cv2.resize(image, (self.img_size, self.img_size))
        mask = cv2.resize(mask, (self.img_size, self.img_size))

        image = image / 255.0
        mask = (mask > 127).astype("float32")

        image = torch.tensor(image).unsqueeze(0).float()
        mask = torch.tensor(mask).unsqueeze(0).float()
        label = torch.tensor(label).float()

        return image, mask, label
