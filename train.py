import torch
from torch.utils.data import DataLoader, random_split
from dataset import DentalSegDataset
from model import UNet
from utils import dice_score
from tqdm import tqdm

# ---------------- DEVICE ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------------- DATA ----------------
dataset = DentalSegDataset("dataset")

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_ds, test_ds = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(
    train_ds, batch_size=4, shuffle=True, num_workers=0
)
test_loader = DataLoader(
    test_ds, batch_size=4, shuffle=False, num_workers=0
)

# ---------------- MODEL ----------------
model = UNet().to(device)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# ---------------- TRAIN ----------------
EPOCHS = 20

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    train_dice = 0

    for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        imgs = imgs.to(device)
        masks = masks.to(device)

        preds = model(imgs)
        loss = criterion(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_dice += dice_score(preds, masks).item()

    train_loss /= len(train_loader)
    train_dice /= len(train_loader)

    # -------- TEST --------
    model.eval()
    test_dice = 0

    with torch.no_grad():
        for imgs, masks in test_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            preds = model(imgs)
            test_dice += dice_score(preds, masks).item()

    test_dice /= len(test_loader)

    print(
        f"Epoch {epoch+1} | "
        f"Loss: {train_loss:.4f} | "
        f"Train Dice: {train_dice:.4f} | "
        f"Test Dice: {test_dice:.4f}"
    )

# ---------------- SAVE ----------------
torch.save(model.state_dict(), "unet_final.pth")
print("Model saved as unet_final.pth")
