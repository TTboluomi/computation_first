"""
Train CNN (ResNet18) for deepfake face detection on local GPU.
Generates synthetic face-like features for training when real dataset unavailable.

Usage:
  python train_cnn.py                    # Train with synthetic data
  python train_cnn.py --real data/real --fake data/fake  # With real dataset

Output: models/cnn_weights.pth
"""
import sys
import argparse
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# ----------------------------------------------------------------
# Synthetic data generation for training
# ----------------------------------------------------------------
def generate_synthetic_face(is_real=True, size=224):
    """Generate a synthetic face image for training."""
    img = np.zeros((size, size, 3), dtype=np.uint8)

    # Base skin color
    if is_real:
        # Natural skin variations
        base_r = int(np.random.normal(180, 30))
        base_g = int(np.random.normal(140, 25))
        base_b = int(np.random.normal(120, 20))
    else:
        # Fake: unnatural colors, artifacts
        fake_type = np.random.choice(['oversaturated', 'washed', 'patchy', 'grid'])
        if fake_type == 'oversaturated':
            base_r = int(np.random.uniform(200, 255))
            base_g = int(np.random.uniform(200, 255))
            base_b = int(np.random.uniform(200, 255))
        elif fake_type == 'washed':
            base_r = int(np.random.uniform(100, 160))
            base_g = int(np.random.uniform(100, 160))
            base_b = int(np.random.uniform(100, 160))
        elif fake_type == 'patchy':
            base_r = int(np.random.normal(160, 60))
            base_g = int(np.random.normal(120, 50))
            base_b = int(np.random.normal(100, 40))
        else:  # grid
            base_r = base_g = base_b = 128

    base_r = np.clip(base_r, 0, 255)
    base_g = np.clip(base_g, 0, 255)
    base_b = np.clip(base_b, 0, 255)
    img[:] = (base_b, base_g, base_r)  # BGR for OpenCV

    # Oval face mask
    center = (size // 2, size // 2)
    axes = (int(size * 0.35), int(size * 0.42))
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    mask = cv2.GaussianBlur(mask, (21, 21), 10)

    # Skin texture
    noise = np.random.normal(0, 8, (size, size, 3)).astype(np.int16)
    for c in range(3):
        channel = img[:, :, c].astype(np.int16)
        channel = channel + (noise[:, :, c] * mask / 255.0).astype(np.int16)
        if is_real:
            # Natural pores
            fine_noise = np.random.normal(0, 3, (size, size))
            channel = channel + (fine_noise * mask / 255.0 * 0.5).astype(np.int16)
        else:
            # Fake: block artifacts or smooth
            if fake_type == 'patchy':
                block_mask = np.random.choice([0.8, 1.0, 1.2], (size // 16, size // 16))
                block_mask = cv2.resize(block_mask, (size, size), interpolation=cv2.INTER_NEAREST)
                channel = (channel.astype(float) * block_mask).astype(np.int16)

        img[:, :, c] = np.clip(channel, 0, 255).astype(np.uint8)

    # Eyes
    eye_y = int(size * 0.38)
    for ex in [int(size * 0.33), int(size * 0.67)]:
        cv2.ellipse(img, (ex, eye_y), (int(size * 0.06), int(size * 0.04)),
                     0, 0, 360, (50, 50, 50), -1)
        cv2.circle(img, (ex, eye_y), int(size * 0.02), (20, 20, 20), -1)
        cv2.circle(img, (ex + 2, eye_y - 1), int(size * 0.007), (255, 255, 255), -1)

    if not is_real:
        # Fake: misplaced/misaligned features
        if np.random.random() < 0.3:
            cv2.ellipse(img, (int(size * 0.5), int(size * 0.35)),
                         (int(size * 0.07), int(size * 0.04)),
                         0, 0, 360, (50, 50, 50), -1)

    # Mouth
    mouth_y = int(size * 0.62)
    cv2.ellipse(img, (center[0], mouth_y), (int(size * 0.08), int(size * 0.03)),
                 0, 0, 180, (80, 60, 60), 2)
    if is_real:
        cv2.ellipse(img, (center[0], mouth_y + 2), (int(size * 0.04), int(size * 0.012)),
                     0, 0, 180, (100, 80, 80), -1)

    # Nose
    nose_y = int(size * 0.5)
    cv2.ellipse(img, (center[0], nose_y), (int(size * 0.02), int(size * 0.025)),
                 0, 0, 360, (120, 100, 90), -1)

    # Eyebrows
    brow_y = int(size * 0.3)
    for bx, dx in [(int(size * 0.33), -15), (int(size * 0.67), 15)]:
        pts = np.array([
            [bx - 15, brow_y + 2],
            [bx - 5, brow_y - 2],
            [bx + 5, brow_y - 2],
            [bx + 15, brow_y + 1],
        ], dtype=np.int32)
        cv2.fillConvexPoly(img, pts, (40, 30, 20))

    if not is_real and np.random.random() < 0.4:
        # Add visible artifacts
        artifact_type = np.random.choice(['blend_boundary', 'jpeg_blocks'])
        if artifact_type == 'blend_boundary':
            cv2.rectangle(img, (size // 4, size // 4),
                         (3 * size // 4, 3 * size // 4), (0, 255, 0), 2)
        else:
            for i in range(8):
                for j in range(8):
                    bx, by = i * 28, j * 28
                    if np.random.random() < 0.3:
                        val = np.random.randint(0, 50)
                        img[by:by + 20, bx:bx + 20] = (
                            img[by:by + 20, bx:bx + 20] * 0.7 + val
                        ).astype(np.uint8)

    return img


class SyntheticFaceDataset(Dataset):
    def __init__(self, n_samples=2000, size=224, transform=None):
        self.n_samples = n_samples
        self.size = size
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        is_real = idx < self.n_samples // 2
        img = generate_synthetic_face(is_real=is_real, size=self.size)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = 1.0 if is_real else 0.0
        return self.transform(img_rgb), torch.tensor(label, dtype=torch.float32)


# ----------------------------------------------------------------
# CNN Model for deepfake classification
# ----------------------------------------------------------------
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
    batch_size = 32
    epochs = 30
    lr = 1e-3

    print(f"Creating synthetic dataset...")
    train_ds = SyntheticFaceDataset(n_samples=4000, size=224)
    val_ds = SyntheticFaceDataset(n_samples=1000, size=224)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = DeepfakeResNet(pretrained=True).to(DEVICE)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

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

        scheduler.step(avg_val)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | Train: {avg_train:.4f} | Val: {avg_val:.4f} | Acc: {val_acc:.3f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Save
    save_path = MODEL_DIR / "cnn_weights.pth"
    torch.save(best_state, save_path)

    # Final eval
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        final_correct = 0
        final_total = 0
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            outputs = model(batch_x)
            preds = (outputs > 0.5).float()
            final_correct += (preds == batch_y).sum().item()
            final_total += batch_y.size(0)

    print(f"\nTraining complete!")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Final accuracy: {final_correct/final_total:.3f}")
    print(f"Model saved: {save_path} ({save_path.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    train()
