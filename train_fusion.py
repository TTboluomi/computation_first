"""
训练融合层：用逻辑回归替代手写加权平均。

=====================================================================
前提：
  在 FaceForensics++ 或其他标注数据集上运行 app.py 的完整分析管线，
  收集每条样本的 6 个模型分数作为特征，以真/假标签训练逻辑回归。

训练流程：
=====================================================================
1. 准备标注视频目录：real_videos/ 和 fake_videos/
2. 运行 collect_scores() 遍历视频，通过 app.py 管线提取 6 维分数向量
3. 训练 sklearn LogisticRegression（6 维 → 二分类）
4. 保存 coefficients + intercept 到 models/fusion_weights.json

=====================================================================
使用示例：
=====================================================================
  # 1. 准备数据
  mkdir data/real  data/fake
  # 放入真实/伪造视频

  # 2. 收集分数（需后端运行中）
  python train_fusion.py --collect --real-dir data/real --fake-dir data/fake

  # 3. 训练融合层
  python train_fusion.py --train

  # 4. 单步完成（收集 + 训练）
  python train_fusion.py --all --real-dir data/real --fake-dir data/fake
=====================================================================
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_KEYS = ["cnn", "vit", "syncnet", "rppg", "flow", "geometry_model"]


def train_fusion_from_scores(scores_data, labels):
    """
    训练逻辑回归融合层。

    Args:
        scores_data: list of dicts, 每个 dict 包含 6 个模型分数
        labels: list of int, 1=real, 0=fake

    Returns:
        dict with coef, intercept
    """
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        print("scikit-learn 未安装。运行: pip install scikit-learn")
        print("使用默认均匀权重作为回退...")
        return {
            "coef": {k: 1.0 / len(MODEL_KEYS) for k in MODEL_KEYS},
            "intercept": 0.0,
            "method": "uniform_fallback",
        }

    X = np.array([[float(s.get(k, 0.0)) for k in MODEL_KEYS] for s in scores_data])
    y = np.array(labels)

    # 标准化
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    X_norm = (X - mean) / std

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_norm, y)

    train_acc = clf.score(X_norm, y)
    print(f"  Training accuracy: {train_acc:.3f}")
    print(f"  Coefficients: {dict(zip(MODEL_KEYS, clf.coef_[0].round(4)))}")
    print(f"  Intercept: {clf.intercept_[0]:.4f}")

    return {
        "coef": {k: float(v) for k, v in zip(MODEL_KEYS, clf.coef_[0])},
        "intercept": float(clf.intercept_[0]),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "method": "logistic_regression",
    }


def save_fusion_weights(weights_data):
    path = MODEL_DIR / "fusion_weights.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(weights_data, f, indent=2, ensure_ascii=False)
    print(f"Fusion weights saved to: {path}")


def load_fusion_weights():
    path = MODEL_DIR / "fusion_weights.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def collect_scores_from_videos(real_dir, fake_dir, server_url="http://localhost:5000"):
    """
    通过 HTTP API 收集模型分数。
    需要后端服务正在运行。
    """
    import requests

    scores_data = []
    labels = []

    for label, video_dir in [(1, real_dir), (0, fake_dir)]:
        video_files = list(Path(video_dir).glob("*.mp4")) + list(Path(video_dir).glob("*.avi"))
        print(f"Processing {len(video_files)} {'real' if label else 'fake'} videos...")

        for vf in video_files:
            try:
                with open(vf, "rb") as f:
                    response = requests.post(
                        f"{server_url}/api/analyze-video",
                        files={"file": (vf.name, f, "video/mp4")},
                        data={"session_id": f"train_{vf.stem}", "label": "training"},
                        timeout=120,
                    )
                if response.status_code == 200:
                    data = response.json()
                    scores_data.append(data.get("model_scores", {}))
                    labels.append(label)
                else:
                    print(f"  Failed: {vf.name} (HTTP {response.status_code})")
            except Exception as e:
                print(f"  Error processing {vf.name}: {e}")

            time.sleep(0.1)

    print(f"Collected {len(scores_data)} samples ({sum(labels)} real, {sum(1 - l for l in labels)} fake)")
    return scores_data, labels


def export_sample_scores_csv(scores_data, labels):
    """导出收集到的分数为 CSV，方便调试"""
    path = MODEL_DIR / "collected_scores.csv"
    with open(path, "w", encoding="utf-8") as f:
        f.write("label," + ",".join(MODEL_KEYS) + "\n")
        for s, l in zip(scores_data, labels):
            f.write(f"{l}," + ",".join(f"{s.get(k, 0):.4f}" for k in MODEL_KEYS) + "\n")
    print(f"Sample scores saved to: {path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train fusion layer for deepfake detection")
    parser.add_argument("--all", action="store_true", help="Run full pipeline: collect + train")
    parser.add_argument("--collect", action="store_true", help="Collect model scores from videos")
    parser.add_argument("--train", action="store_true", help="Train fusion from collected scores")
    parser.add_argument("--real-dir", type=str, help="Directory with real videos")
    parser.add_argument("--fake-dir", type=str, help="Directory with fake videos")
    parser.add_argument("--server-url", type=str, default="http://localhost:5000", help="Backend server URL")
    parser.add_argument("--scores-file", type=str, default=None, help="Load scores from JSON file instead of collecting")

    args = parser.parse_args()

    if args.all:
        if not args.real_dir or not args.fake_dir:
            print("Error: --all requires --real-dir and --fake-dir")
            sys.exit(1)
        scores_data, labels = collect_scores_from_videos(args.real_dir, args.fake_dir, args.server_url)
        export_sample_scores_csv(scores_data, labels)
        weights = train_fusion_from_scores(scores_data, labels)
        save_fusion_weights(weights)
        return

    if args.collect:
        if not args.real_dir or not args.fake_dir:
            print("Error: --collect requires --real-dir and --fake-dir")
            sys.exit(1)
        scores_data, labels = collect_scores_from_videos(args.real_dir, args.fake_dir, args.server_url)
        export_sample_scores_csv(scores_data, labels)
        # Save collected data for later training
        with open(MODEL_DIR / "collected_scores.json", "w", encoding="utf-8") as f:
            json.dump({"scores": scores_data, "labels": labels}, f, indent=2)
        print("Scores collected. Run --train to train the fusion layer.")
        return

    if args.train:
        if args.scores_file:
            with open(args.scores_file, "r") as f:
                data = json.load(f)
            scores_data = data["scores"]
            labels = data["labels"]
        else:
            scores_file = MODEL_DIR / "collected_scores.json"
            if not scores_file.exists():
                print("Error: No collected scores found. Run --collect first.")
                sys.exit(1)
            with open(scores_file, "r") as f:
                data = json.load(f)
            scores_data = data["scores"]
            labels = data["labels"]

        weights = train_fusion_from_scores(scores_data, labels)
        save_fusion_weights(weights)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
