"""
Fast CNN training for deepfake detection on local GPU.
Uses simplified synthetic data for speed.
"""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torchvision.models import resnet18, ResNet18_Weights

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# Generate synthetic image data in memory (fast tensor ops)
def generate_synthetic_batch(n_samples, img_size=112):
    """Generate synthetic face-like feature tensors directly in PyTorch."""
    half = n_samples // 2
    # Real faces: smoother gradients, natural texture patterns
    # Fake faces: blocky, noisy, edge artifacts
    X = torch.randn(n_samples, 3, img_size, img_size) * 0.3
    y = torch.zeros(n_samples)

    for i in range(n_samples):
        is_real = i < half
        if is_real:
            # Real: smooth gradients with fine texture
            for c in range(3):
                base = torch.randn(1, img_size, img_size) * 0.5
                # Horizontal gradient (natural lighting)
                h_grad = torch.linspace(-0.5, 0.5, img_size).view(1, 1, -1).expand(1, img_size, img_size)
                # Vertical gradient
                v_grad = torch.linspace(-0.3, 0.3, img_size).view(1, -1, 1).expand(1, img_size, img_size)
                # Fine texture
                texture = torch.randn(1, img_size, img_size) * 0.08
                # Central face oval
                yy, xx = torch.meshgrid(torch.arange(img_size), torch.arange(img_size), indexing='ij')
                cx, cy = img_size // 2, img_size // 2
                dist = ((xx - cx) ** 2 / (img_size * 0.35) ** 2 + (yy - cy) ** 2 / (img_size * 0.42) ** 2)
                mask = torch.exp(-dist).unsqueeze(0)
                X[i, c] = (base + h_grad + v_grad + texture) * mask
            y[i] = 1.0
        else:
            # Fake: block artifacts, edge discontinuities, oversaturation
            fake_type = np.random.choice([0, 1, 2])
            block_size = np.random.choice([8, 16, 32])
            for c in range(3):
                if fake_type == 0:
                    # Block artifacts
                    blocks = torch.randn(img_size // block_size, img_size // block_size) * 0.8 + 0.2
                    b_up = torch.nn.functional.interpolate(
                        blocks.unsqueeze(0).unsqueeze(0), size=(img_size, img_size), mode='nearest'
                    ).squeeze()
                    X[i, c] = b_up + torch.randn(img_size, img_size) * 0.1
                elif fake_type == 1:
                    # Sharp edges / grid
                    grid = torch.zeros(img_size, img_size)
                    grid[::8, :] = torch.randn(img_size // 8, img_size) * 0.5
                    grid[:, ::8] += torch.randn(img_size, img_size // 8) * 0.5
                    X[i, c] = grid + torch.randn(img_size, img_size) * 0.15
                else:
                    # Washed out + JPEG-like noise
                    X[i, c] = torch.ones(img_size, img_size) * 0.1 + torch.randn(img_size, img_size) * 0.5
            y[i] = 0.0

    # Normalize like ImageNet
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    X = (X - X.mean(dim=(2, 3), keepdim=True)) / (X.std(dim=(2, 3), keepdim=True) + 1e-6)
    X = X * std + mean
    return X, y


class DeepfakeResNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        if pretrained:
            weights = ResNet18_Weights.DEFAULT
            self.backbone = resnet18(weights=weights)
        else:
            self.backbone = resnet18(weights=None)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        features = self.backbone(x)
        return torch.sigmoid(self.classifier(features)).squeeze(-1)


def train():
    batch_size = 64
    epochs = 40
    lr = 1e-3

    print("Generating synthetic data...")
    X, y = generate_synthetic_batch(4000, img_size=112)
    val_X, val_y = generate_synthetic_batch(1000, img_size=112)
    print(f"Train: {X.shape}, Val: {val_X.shape}")

    train_ds = TensorDataset(X, y)
    val_ds = TensorDataset(val_X, val_y)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = DeepfakeResNet(pretrained=True).to(DEVICE)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.BCELoss()
    optimizer = optim.Adam([
        {'params': model.backbone.parameters(), 'lr': lr * 0.1},
        {'params': model.classifier.parameters(), 'lr': lr},
    ], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float('inf')
    best_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                outputs = model(batch_x)
                val_loss += criterion(outputs, batch_y).item()
                preds = (outputs > 0.5).float()
                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        val_acc = correct / total
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | Train: {avg_train:.4f} | Val: {avg_val:.4f} | Acc: {val_acc:.3f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        final_correct = sum(
            ((model(bx.to(DEVICE)) > 0.5).float() == by.to(DEVICE)).sum().item()
            for bx, by in val_loader
        )
        final_acc = final_correct / len(val_ds)

    save_path = MODEL_DIR / "cnn_weights.pth"
    torch.save(best_state, save_path)
    size_mb = save_path.stat().st_size / 1024 / 1024

    print(f"\nTraining complete!")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Final accuracy: {final_acc:.3f}")
    print(f"Model saved: {save_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    train()
