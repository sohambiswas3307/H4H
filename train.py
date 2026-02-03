import torch
from torch.utils.data import DataLoader, random_split
from dataset import DentalSegDataset
from model import MultiTaskUNet
from utils import dice_score
from tqdm import tqdm
import dataset
print("Dataset file in use:", dataset.__file__)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

dataset = DentalSegDataset("dataset")
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_ds, test_ds = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=4, shuffle=False)

model = MultiTaskUNet().to(device)

seg_loss_fn = torch.nn.BCEWithLogitsLoss()
cls_loss_fn = torch.nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

EPOCHS = 20

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    total_dice = 0

    for imgs, masks, labels in tqdm(train_loader):
        imgs = imgs.to(device)
        masks = masks.to(device)
        labels = labels.to(device).unsqueeze(1)

        seg_logits, cls_logits = model(imgs)

        seg_loss = seg_loss_fn(seg_logits, masks)
        cls_loss = cls_loss_fn(cls_logits, labels)

        loss = seg_loss + 0.5 * cls_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_dice += dice_score(seg_logits, masks).item()

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Loss: {total_loss/len(train_loader):.4f} | "
        f"Dice: {total_dice/len(train_loader):.4f}"
    )

torch.save(model.state_dict(), "unet_mtl_final.pth")
print("Saved unet_mtl_final.pth")
