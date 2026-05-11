"""
训练 TemporalCNN：使用合成 rPPG 信号训练轻量级 1D CNN 二分类器。

=====================================================================
训练流程：
=====================================================================
1. 生成合成 rPPG 训练数据（真人类信号 vs 伪造随机噪声信号）
2. 训练 TemporalCNN (3 层 Conv1d + Linear + Sigmoid) 做二分类
3. 保存权重到 models/tcn_weights.pth
4. 后端启动时自动加载该权重

=====================================================================
后续升级：用 FaceForensics++ 等真实深伪数据集替换合成数据
=====================================================================
下载 FF++：
  git clone https://github.com/ondyari/FaceForensics.git
  cd FaceForensics
  # 下载真实视频
  python download-FaceForensicspp.py -d original_sequences -c c23 -t videos
  # 下载 Deepfakes 伪造视频
  python download-FaceForensicspp.py -d manipulated_sequences -c c23 -t videos -m Deepfakes

然后从 FF++ 视频中提取 rPPG 信号替换 generate_synthetic_rppg() 的数据生成。
=====================================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

torch.manual_seed(42)
np.random.seed(42)

# ----------------------------------------------------------------
# TemporalCNN — 与 app.py 中的定义保持完全一致
# ----------------------------------------------------------------
class TemporalCNN(nn.Module):
    def __init__(self, input_channels=1, seq_len=90):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, 16, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        return torch.sigmoid(self.fc(x)).squeeze(-1)


# ----------------------------------------------------------------
# 合成 rPPG 数据生成
# ----------------------------------------------------------------
def generate_synthetic_rppg(seq_len=90, is_real=True, hr_range=(60, 100)):
    """
    生成合成 rPPG 信号。
    真实人脸 rPPG：准周期性心率信号 + 谐波 + 呼吸调制 + 噪声
    伪造人脸：非结构化噪声/平坦信号
    """
    t = np.linspace(0, seq_len / 30.0, seq_len)

    if is_real:
        hr = np.random.uniform(*hr_range)
        freq = hr / 60.0

        signal = np.sin(2 * np.pi * freq * t + np.random.uniform(0, 2 * np.pi))
        signal += 0.3 * np.sin(4 * np.pi * freq * t + np.random.uniform(0, 2 * np.pi))
        signal += 0.15 * np.sin(6 * np.pi * freq * t + np.random.uniform(0, 2 * np.pi))

        # 呼吸调制 (~0.2 Hz)
        signal *= (1 + 0.15 * np.sin(2 * np.pi * 0.2 * t + np.random.uniform(0, 2 * np.pi)))

        signal += np.random.normal(0, 0.12, seq_len)
    else:
        fake_type = np.random.choice(['noise', 'flat', 'broken_periodic', 'step'])

        if fake_type == 'noise':
            signal = np.random.normal(0, 1, seq_len)
        elif fake_type == 'flat':
            signal = np.random.normal(0, 0.15, seq_len)
            signal += np.cumsum(np.random.normal(0, 0.03, seq_len))
        elif fake_type == 'broken_periodic':
            # 不稳定的假周期信号（频率漂移）
            freq_drift = 0.8 + 0.3 * np.sin(2 * np.pi * 0.5 * t)
            signal = np.sin(2 * np.pi * freq_drift * t + np.random.uniform(0, 2 * np.pi))
            signal += np.random.normal(0, 0.3, seq_len)
        else:  # step
            signal = np.zeros(seq_len)
            cut = seq_len // 2
            signal[:cut] = np.random.normal(0.8, 0.25, cut)
            signal[cut:] = np.random.normal(-0.8, 0.25, seq_len - cut)

    signal = signal - np.mean(signal)
    std = np.std(signal)
    if std > 1e-6:
        signal = signal / std

    return signal.astype(np.float32)


def generate_dataset(n_samples=2000, seq_len=90):
    """生成平衡的合成 rPPG 数据集"""
    X, y = [], []

    for _ in range(n_samples // 2):
        X.append(generate_synthetic_rppg(seq_len, is_real=True))
        y.append(1.0)

        X.append(generate_synthetic_rppg(seq_len, is_real=False))
        y.append(0.0)

    X = np.array(X)
    y = np.array(y)
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]


# ----------------------------------------------------------------
# 训练主函数
# ----------------------------------------------------------------
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    seq_len = 90
    model = TemporalCNN(input_channels=1, seq_len=seq_len).to(device)

    print("Generating synthetic rPPG training data...")
    X_train, y_train = generate_dataset(4000, seq_len)
    X_val, y_val = generate_dataset(1000, seq_len)

    X_train = torch.from_numpy(X_train).unsqueeze(1).to(device)
    y_train = torch.from_numpy(y_train).float().to(device)
    X_val = torch.from_numpy(X_val).unsqueeze(1).to(device)
    y_val = torch.from_numpy(y_val).float().to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    batch_size = 64
    epochs = 60
    best_val_loss = float("inf")
    best_state = None

    print(f"Training: {len(X_train)} train / {len(X_val)} val samples, {epochs} epochs")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_batches = 0

        perm = torch.randperm(len(X_train))
        for i in range(0, len(X_train), batch_size):
            idx = perm[i : i + batch_size]
            batch_X = X_train[idx]
            batch_y = y_train[idx]

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val).item()
            val_preds = (val_outputs > 0.5).float()
            val_acc = (val_preds == y_val).float().mean().item()

        scheduler.step(val_loss)

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:3d}/{epochs}  "
                f"Train Loss: {total_loss / n_batches:.4f}  "
                f"Val Loss: {val_loss:.4f}  "
                f"Val Acc: {val_acc:.3f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # 保存最佳权重
    model_dir = Path(__file__).resolve().parent / "models"
    model_dir.mkdir(exist_ok=True)
    save_path = model_dir / "tcn_weights.pth"
    torch.save(best_state, save_path)

    # 最终评估
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        final_outputs = model(X_val)
        final_acc = ((final_outputs > 0.5).float() == y_val).float().mean().item()

    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Val accuracy:  {final_acc:.3f}")
    print(f"  Model saved:   {save_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    train()
