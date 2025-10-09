import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange, tqdm
from loader import make_loaders
from models import ResNet18


# ----------------- train & test funcs -----------------
def train(model, loader, criterion, optimizer, device, epoch):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for b, batch in enumerate(loader, 1):
        x, y = (batch[0], batch[1]) if isinstance(batch, (list, tuple)) else batch
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1} | Batch {b}/{len(loader)} | Loss {loss.item():.4f}")

        loss_sum += loss.item() * x.size(0)
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    avg_loss = loss_sum / max(total, 1)
    train_acc = correct / max(total, 1)
    return avg_loss, train_acc


@torch.no_grad()
def test(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total


# ----------------- main -----------------
def main(train_list, test_list, epochs, batch_size):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, test_loader, num_classes = make_loaders(
        train_list, test_list, batch_size=batch_size
    )

    model = ResNet18(num_classes=num_classes, in_channels=1).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in trange(epochs, desc="Epochs"):
        train_loss, train_acc = train(
            model, train_loader, criterion, optimizer, device, epoch
        )
        test_acc = test(model, test_loader, device)
        print(
            f"Epoch {epoch+1}: train loss {train_loss:.4f} | train acc {train_acc:.4f} | test acc {test_acc:.4f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train/Test ResNet18 (grayscale)")
    parser.add_argument("--train_list", default="Directions01/list_train.txt")
    parser.add_argument("--test_list", default="Directions01/list_test.txt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    main(args.train_list, args.test_list, args.epochs, args.batch_size)
