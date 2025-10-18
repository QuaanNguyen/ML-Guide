import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import trange

from loader import make_segmentation_loaders 
from models_unet import UNet 


# ---- Soft Dice (no threshold) as a metric ----
def soft_dice(probs, target, eps=1e-6):
    # probs, target: [B,1,H,W], target in {0,1}
    probs = probs.contiguous().view(probs.size(0), -1)
    target = target.float().contiguous().view(target.size(0), -1)
    inter = (probs * target).sum(dim=1)
    denom = probs.sum(dim=1) + target.sum(dim=1)
    dice = (2 * inter + eps) / (denom + eps)        # [B]
    return dice.mean()                               # scalar


def train(model, loader, criterion, optimizer, device, epoch):
    model.train()
    loss_sum, dice_sum, sample_count = 0.0, 0.0, 0

    for b, (x, y) in enumerate(loader, 1):
        # x: [B,3,H,W], y: [B,1,H,W] with {0,1}
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)                     # [B,1,H,W], raw logits
        loss = criterion(logits, y.float())   # BCE-with-logits
        loss.backward()
        optimizer.step()

        # Metric: soft dice on sigmoid probabilities (no threshold)
        probs = torch.sigmoid(logits)
        dice = soft_dice(probs, y)

        # accumulate weighted by batch size (so we average over images)
        bs = x.size(0)
        loss_sum += loss.item() * bs
        dice_sum += dice.item() * bs
        sample_count += bs

        print(f"Epoch {epoch+1} | Batch {b}/{len(loader)} | "
              f"Loss {loss.item():.4f} | Dice {dice.item():.4f}")

    avg_loss = loss_sum / max(sample_count, 1)
    avg_dice = dice_sum / max(sample_count, 1)
    return avg_loss, avg_dice


@torch.no_grad()
def test(model, loader, criterion, device):
    model.eval()
    loss_sum, dice_sum, sample_count = 0.0, 0.0, 0
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y.float())

        probs = torch.sigmoid(logits)
        dice = soft_dice(probs, y)

        bs = x.size(0)
        loss_sum += loss.item() * bs
        dice_sum += dice.item() * bs
        sample_count += bs

    avg_loss = loss_sum / max(sample_count, 1)
    avg_dice = dice_sum / max(sample_count, 1)
    return avg_loss, avg_dice


def main(root_dir, epochs, batch_size, img_size):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Expect masks as {0,1} with shape [B,1,H,W] from your loader
    train_loader, test_loader, _ = make_segmentation_loaders(
        root_dir, img_size=img_size, batch_size=batch_size
    )

    # Binary UNet
    model = UNet(in_channels=3, num_classes=1).to(device)

    # Binary loss on logits
    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print(f"Starting training for {epochs} epochs...")
    for epoch in trange(epochs, desc="Epochs"):
        train_loss, train_dice = train(model, train_loader, criterion, optimizer, device, epoch)
        test_loss, test_dice = test(model, test_loader, criterion, device)

        print(f"Epoch {epoch+1}: "
              f"train loss {train_loss:.4f} | train DICE {train_dice:.4f} | "
              f"test loss {test_loss:.4f} | test DICE {test_dice:.4f}")

    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train/Test U-Net for Segmentation01 (RGB)")
    parser.add_argument("--data_root", default="Segmentation01_RGB")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4) 
    parser.add_argument("--img_size", type=int, default=256) 
    args = parser.parse_args()
    main(args.data_root, args.epochs, args.batch_size, args.img_size)
