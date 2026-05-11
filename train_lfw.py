"""
Train CNN (ResNet18) for deepfake detection using LFW face dataset.
Real: LFW face images
Fake: Manipulated versions of the same images (blur, jpeg artifacts, face swaps, etc.)

Usage: python train_lfw.py
Output: models/cnn_weights.pth
"""
import sys
import random
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
IMG_SIZE = 112
BATCH_SIZE = 64
EPOCHS = 50
print(f"Device: {DEVICE}")


def read_image(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


# ----------------------------------------------------------------
# Face manipulation functions for generating "fake" samples
# ----------------------------------------------------------------
def apply_jpeg_artifact(img, quality=None):
    """Heavy JPEG compression artifacts."""
    if quality is None:
        quality = random.randint(5, 25)
    _, encoded = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def apply_blur_sharpen(img):
    """Blur then oversharpen - common in deepfakes."""
    k = random.choice([3, 5, 7])
    blurred = cv2.GaussianBlur(img, (k, k), 0)
    alpha = random.uniform(1.2, 2.0)
    return cv2.addWeighted(img, alpha, blurred, 1 - alpha, 0)


def apply_color_mismatch(img):
    """Unnatural color shifts - common blending artifact."""
    result = img.copy()
    if random.random() < 0.5:
        # Shift one channel
        c = random.randint(0, 2)
        shift = random.randint(-40, 40)
        result[:, :, c] = np.clip(result[:, :, c].astype(np.int32) + shift, 0, 255).astype(np.uint8)
    else:
        # Unnatural saturation
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1].astype(np.float32) * random.uniform(0.3, 2.5), 0, 255).astype(np.uint8)
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return result


def apply_face_boundary(img):
    """Simulate face-swap boundary artifacts."""
    h, w = img.shape[:2]
    result = img.copy()
    # Random ellipse boundary
    cx, cy = w // 2 + random.randint(-10, 10), h // 2 + random.randint(-10, 10)
    axes = (random.randint(w // 5, w // 3), random.randint(h // 4, h // 3))
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, (cx, cy), axes, 0, 0, 360, 255, -1)
    mask = cv2.GaussianBlur(mask, (11, 11), 5)
    mask_f = mask.astype(np.float32) / 255.0

    # Blend with a distorted version inside the ellipse
    distorted = cv2.GaussianBlur(img, (5, 5), 3)
    for c in range(3):
        result[:, :, c] = (img[:, :, c].astype(np.float32) * (1 - mask_f) +
                           distorted[:, :, c].astype(np.float32) * mask_f).astype(np.uint8)
    return result


def apply_noise_pattern(img):
    """Add structured noise patterns."""
    h, w = img.shape[:2]
    noise = np.random.normal(0, random.uniform(10, 35), (h, w, 3)).astype(np.int16)
    result = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Add grid artifacts
    if random.random() < 0.4:
        for i in range(0, w, 8):
            result[:, i:i + 1] = np.clip(result[:, i:i + 1].astype(np.int16) +
                                         random.randint(-15, 15), 0, 255).astype(np.uint8)
        for i in range(0, h, 8):
            result[i:i + 1, :] = np.clip(result[i:i + 1, :].astype(np.int16) +
                                         random.randint(-15, 15), 0, 255).astype(np.uint8)
    return result


def apply_face_swap(img, all_faces):
    """Paste another person's face region onto this one."""
    if not all_faces or len(all_faces) < 2:
        return img
    other_idx = random.randint(0, len(all_faces) - 1)
    other = read_image(all_faces[other_idx])
    if other is None:
        return img

    h, w = img.shape[:2]
    oh, ow = other.shape[:2]
    other = cv2.resize(other, (w, h))

    # Central face region mask
    cx, cy = w // 2, h // 2
    rw, rh = w // 3, h // 3
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(mask, (cx, cy), (rw, rh), 0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (15, 15), 8)

    result = img.copy()
    for c in range(3):
        result[:, :, c] = (img[:, :, c].astype(np.float32) * (1 - mask) +
                           other[:, :, c].astype(np.float32) * mask).astype(np.uint8)
    return result


def generate_fake(img, all_faces):
    """Apply a random manipulation to create a fake face."""
    manipulations = [
        apply_jpeg_artifact,
        apply_blur_sharpen,
        apply_color_mismatch,
        apply_noise_pattern,
    ]
    # Sometimes add face swap or boundary
    if random.random() < 0.3:
        manipulations.append(apply_face_boundary)
    if random.random() < 0.2:
        manipulations.append(lambda x: apply_face_swap(x, all_faces))

    # Apply 1-3 random manipulations
    n = random.randint(1, min(3, len(manipulations)))
    chosen = random.sample(manipulations, n)
    result = img.copy()
    for m in chosen:
        result = m(result)
    return result


# ----------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------
class LFWDataset(Dataset):
    def __init__(self, img_paths, all_paths, is_train=True):
        self.img_paths = img_paths
        self.all_paths = all_paths
        self.is_train = is_train
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.img_paths) * 2

    def __getitem__(self, idx):
        is_real = idx < len(self.img_paths)
        img_idx = idx if is_real else idx - len(self.img_paths)
        path = self.img_paths[img_idx]

        img = read_image(path)
        if img is None:
            img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

        if is_real:
            # Real: small augmentations (realistic variations)
            if self.is_train and random.random() < 0.5:
                img = apply_jpeg_artifact(img, quality=random.randint(60, 95))
            if self.is_train and random.random() < 0.3:
                img = cv2.GaussianBlur(img, (3, 3), 0)
        else:
            # Fake: aggressive manipulations
            img = generate_fake(img, self.all_paths)

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = 1.0 if is_real else 0.0
        return self.transform(img_rgb), torch.tensor(label, dtype=torch.float32)


# ----------------------------------------------------------------
# Model (must match _build_cnn_model in app.py)
# ----------------------------------------------------------------
class DeepfakeCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
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
        return self.classifier(features)


# ----------------------------------------------------------------
# Training
# ----------------------------------------------------------------
def train():
    # Collect all face images
    lfw_dir = BASE_DIR / "lfw"
    all_imgs = sorted(list(lfw_dir.rglob("*.jpg")))
    random.shuffle(all_imgs)

    split = int(len(all_imgs) * 0.9)
    train_imgs = all_imgs[:split]
    val_imgs = all_imgs[split:]
    print(f"Faces: {len(all_imgs)} total, {len(train_imgs)} train, {len(val_imgs)} val")

    # Reduce if too many for speed
    max_train = 3000
    train_imgs = train_imgs[:max_train]
    val_imgs = val_imgs[:max_train // 3]
    print(f"Using: {len(train_imgs)} train, {len(val_imgs)} val")

    train_ds = LFWDataset(train_imgs, train_imgs, is_train=True)
    val_ds = LFWDataset(val_imgs, val_imgs, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = DeepfakeCNN().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam([
        {'params': model.backbone.parameters(), 'lr': 1e-4},
        {'params': model.classifier.parameters(), 'lr': 1e-3},
    ], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_loss = float('inf')
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(batch_x).squeeze(-1)
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
                outputs = model(batch_x).squeeze(-1)
                val_loss += criterion(outputs, batch_y).item()
                preds = (torch.sigmoid(outputs) > 0.5).float()
                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        val_acc = correct / total
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{EPOCHS} | Train: {avg_train:.4f} | Val: {avg_val:.4f} | Acc: {val_acc:.3f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    save_path = MODEL_DIR / "cnn_weights.pth"
    torch.save(best_state, save_path)

    model.eval()
    with torch.no_grad():
        final_correct = 0
        final_total = 0
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            outputs = model(batch_x).squeeze(-1)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            final_correct += (preds == batch_y).sum().item()
            final_total += batch_y.size(0)

    print(f"\nTraining complete!")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Final accuracy: {final_correct/final_total:.3f}")
    print(f"Model saved: {save_path} ({save_path.stat().st_size/1024/1024:.1f} MB)")


if __name__ == "__main__":
    train()
