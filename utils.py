import torch

def dice_score(logits, targets, smooth=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()

    intersection = (preds * targets).sum()
    return (2 * intersection + smooth) / (
        preds.sum() + targets.sum() + smooth
    )
