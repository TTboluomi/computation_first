import base64
import io
import json
import sqlite3
import tempfile
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import av
except Exception:
    av = None

try:
    import librosa
except Exception:
    librosa = None

try:
    import mediapipe as mp
except Exception:
    mp = None

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from scipy.signal import butter, filtfilt

try:
    import typing
    if not hasattr(typing, "OrderedDict"):
        try:
            from typing_extensions import OrderedDict as _OD
            typing.OrderedDict = _OD
        except Exception:
            pass
    import torch
    import torch.nn as nn
except Exception:
    torch = None
    nn = None

try:
    if torch is not None:
        from torchvision.models import ResNet18_Weights, resnet18
    else:
        ResNet18_Weights = None
        resnet18 = None
except Exception:
    ResNet18_Weights = None
    resnet18 = None

try:
    if torch is not None:
        from torchvision.models import ViT_B_16_Weights, vit_b_16
    else:
        ViT_B_16_Weights = None
        vit_b_16 = None
except Exception:
    ViT_B_16_Weights = None
    vit_b_16 = None


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = str(BASE_DIR / "deepfake_platform.db")
MODEL_DIR = BASE_DIR / "models"

DEFAULT_CONFIG = {
    "weights": {
        "cnn": 0.18,
        "vit": 0.16,
        "syncnet": 0.18,
        "rppg": 0.16,
        "flow": 0.14,
        "geometry_model": 0.18,
    },
    "risk_thresholds": {"low": 0.35, "medium": 0.62},
    "analysis": {
        "window_size": 90,
        "min_samples_for_fft": 18,
        "max_video_seconds": 6,
        "sample_fps": 10,
        "audio_sr": 16000,
    },
}

MODEL_SCORE_KEYS = ["cnn", "vit", "syncnet", "rppg", "flow", "geometry_model"]

# ---------- Face Mesh landmark indices ----------
FACE_MESH_KEY_INDICES = {
    "nose_tip": 1,
    "forehead_center": 10,
    "left_cheek": 234,
    "right_cheek": 454,
    "chin": 152,
    "left_eye_outer": 33,
    "right_eye_outer": 263,
    "mouth_left": 61,
    "mouth_right": 291,
    "mouth_top": 13,
    "mouth_bottom": 14,
}
FACE_MESH_FOREHEAD_INDICES = [9, 107, 66, 105, 104, 103, 67, 69, 108, 109, 151, 337, 299, 333, 332, 298, 338, 297, 336, 10]
FACE_MESH_CHEEK_LEFT = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152]
FACE_MESH_CHEEK_RIGHT = [454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152]
FACE_MESH_MOUTH_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
FACE_MESH_LK_POINTS = [1, 33, 263, 61, 291, 13, 14, 10, 152, 234, 454]


class SessionCreatePayload(BaseModel):
    session_id: Optional[str] = None
    label: Optional[str] = "Web Session"


class AnalyzePayload(BaseModel):
    session_id: Optional[str] = None
    label: Optional[str] = "Web Session"
    image: Optional[str] = None
    image_data: Optional[str] = None
    audio_level: Optional[float] = 0.0
    fps: Optional[float] = 12.0


class ConfigUpdatePayload(BaseModel):
    weights: Optional[Dict[str, float]] = None
    risk_thresholds: Optional[Dict[str, float]] = None


app = FastAPI(title="Deepfake Platform")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal Server Error: {str(exc)}"},
    )
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
session_state: Dict[str, Dict[str, Any]] = {}
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
_MP_AVAILABLE = mp is not None

if _MP_AVAILABLE:
    _BaseOptions = mp.tasks.BaseOptions
    _FaceDetectorOptionsCls = mp.tasks.vision.FaceDetectorOptions
    _FaceLandmarkerOptionsCls = mp.tasks.vision.FaceLandmarkerOptions
    _FaceDetectorCls = mp.tasks.vision.FaceDetector
    _FaceLandmarkerCls = mp.tasks.vision.FaceLandmarker
    _FACE_DETECTOR_MODEL = BASE_DIR / "models" / "face_detector.tflite"
    _FACE_LANDMARKER_MODEL = BASE_DIR / "models" / "face_landmarker.task"
    if _FACE_DETECTOR_MODEL.exists():
        _FACE_DETECTOR_BYTES = _FACE_DETECTOR_MODEL.read_bytes()
    else:
        _FACE_DETECTOR_BYTES = None
    if _FACE_LANDMARKER_MODEL.exists():
        _FACE_LANDMARKER_BYTES = _FACE_LANDMARKER_MODEL.read_bytes()
    else:
        _FACE_LANDMARKER_BYTES = None


# ----------------------------------------------------------------
# 1. FaceDetector
# ----------------------------------------------------------------
if _MP_AVAILABLE:
    class FaceDetector:
        def __init__(self):
            base_options = _BaseOptions(model_asset_buffer=_FACE_DETECTOR_BYTES)
            self.options = _FaceDetectorOptionsCls(
                base_options=base_options,
                min_detection_confidence=0.45,
            )
            self.detector = _FaceDetectorCls.create_from_options(self.options)

        def detect(self, frame: np.ndarray) -> Tuple[Tuple[int, int, int, int], bool]:
            height, width = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self.detector.detect(mp_image)
            if result.detections:
                detection = max(
                    result.detections,
                    key=lambda item: item.bounding_box.width * item.bounding_box.height,
                )
                box = detection.bounding_box
                x = int(max(0, box.origin_x))
                y = int(max(0, box.origin_y))
                w = int(min(width - x, box.width))
                h = int(min(height - y, box.height))
                if w > 0 and h > 0:
                    return (x, y, w, h), True

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
            if len(faces) > 0:
                x, y, w, h = max(faces, key=lambda item: item[2] * item[3])
                return (int(x), int(y), int(w), int(h)), True

            size = int(min(height, width) * 0.48)
            x = int((width - size) / 2)
            y = max(0, int((height - size) / 2.8))
            return (x, y, size, size), False

    detector = FaceDetector()

    # ----------------------------------------------------------------
    # 2. FaceLandmarkDetector
    # ----------------------------------------------------------------
    class FaceLandmarkDetector:
        def __init__(self):
            base_options = _BaseOptions(model_asset_buffer=_FACE_LANDMARKER_BYTES)
            self.options = _FaceLandmarkerOptionsCls(
                base_options=base_options,
                num_faces=1,
                min_face_detection_confidence=0.45,
                min_tracking_confidence=0.45,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
            self.landmarker = _FaceLandmarkerCls.create_from_options(self.options)

        def detect(self, frame: np.ndarray) -> Optional[np.ndarray]:
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            results = self.landmarker.detect(mp_image)
            if not results.face_landmarks:
                return None
            landmarks = results.face_landmarks[0]
            pts = np.array([[lm.x * w, lm.y * h] for lm in landmarks], dtype=np.float32)
            return pts

        def extract_roi_rgb(self, frame: np.ndarray, landmarks: np.ndarray, indices: List[int]) -> np.ndarray:
            pts = landmarks[indices].astype(np.int32)
            x1, y1 = np.min(pts, axis=0)
            x2, y2 = np.max(pts, axis=0)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                return np.array([0, 0, 0], dtype=np.float32)
            roi = frame[y1:y2, x1:x2]
            mean_rgb = np.mean(roi.reshape(-1, 3).astype(np.float32), axis=0)
            return mean_rgb

        def align_face(self, frame: np.ndarray, landmarks: np.ndarray, target_size: int = 224) -> np.ndarray:
            left_eye = landmarks[33]
            right_eye = landmarks[263]
            dx = right_eye[0] - left_eye[0]
            dy = right_eye[1] - left_eye[1]
            angle = np.degrees(np.arctan2(dy, dx))
            eye_center = ((left_eye + right_eye) / 2.0).astype(np.float32)
            M = cv2.getRotationMatrix2D(tuple(eye_center), angle, 1.0)
            rotated = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))
            return cv2.resize(rotated, (target_size, target_size), interpolation=cv2.INTER_AREA)

    landmark_detector = FaceLandmarkDetector()

else:
    class FaceDetector:
        def detect(self, frame: np.ndarray) -> Tuple[Tuple[int, int, int, int], bool]:
            height, width = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
            if len(faces) > 0:
                x, y, w, h = max(faces, key=lambda item: item[2] * item[3])
                return (int(x), int(y), int(w), int(h)), True
            size = int(min(height, width) * 0.48)
            x = int((width - size) / 2)
            y = max(0, int((height - size) / 2.8))
            return (x, y, size, size), False

    detector = FaceDetector()

    class FaceLandmarkDetector:
        def detect(self, frame: np.ndarray) -> Optional[np.ndarray]:
            return None
        def extract_roi_rgb(self, frame, landmarks, indices):
            return np.array([0, 0, 0], dtype=np.float32)
        def align_face(self, frame, landmarks, target_size=224):
            return cv2.resize(frame, (target_size, target_size), interpolation=cv2.INTER_AREA)

    landmark_detector = FaceLandmarkDetector()


# ----------------------------------------------------------------
# 3. POS rPPG algorithm  –  replaces simple green-channel mean
# ----------------------------------------------------------------
def pos_rppg_signal(rgb_signals: List[np.ndarray], fps: float) -> np.ndarray:
    if len(rgb_signals) < 6:
        return np.array([], dtype=np.float32)

    X = np.array(rgb_signals, dtype=np.float32)  # (N, 3)
    X = X - np.mean(X, axis=0)
    X = X / (np.std(X, axis=0) + 1e-8)

    # POS projection
    eps = 1e-8
    # l = 32 frames window (paper default)
    l = min(32, len(X))
    h = np.zeros(len(X), dtype=np.float32)

    for i in range(l - 1, len(X)):
        window = X[i - l + 1 : i + 1]
        C = window.T @ window
        mean_rgb = np.mean(window, axis=0)
        proj_mat = np.array([[0, 1, -1], [-2, 1, 1]], dtype=np.float32)
        S = proj_mat @ window.T  # (2, l)
        std_s = np.array([np.std(S[0]), np.std(S[1])]) + eps
        alpha = std_s[0] / (std_s[0] + std_s[1])
        h[i] = S[0, -1] + (alpha - 1) * S[1, -1] / (alpha + eps)

    if np.std(h) > eps:
        h = (h - np.mean(h)) / np.std(h)
    return h


def compute_fft_stability(signal, fps):
    if len(signal) < 18:
        return 0.5, 0.0
    arr = np.array(signal, dtype=np.float32)
    arr = arr - np.mean(arr)
    if np.std(arr) < 1e-6:
        return 0.5, 0.0

    spectrum = np.fft.rfft(arr)
    freqs = np.fft.rfftfreq(len(arr), d=1.0 / fps)
    mask = (freqs >= 0.7) & (freqs <= 3.0)
    if not np.any(mask):
        return 0.5, 0.0

    band = np.abs(spectrum[mask])
    band_freqs = freqs[mask]
    dominant_idx = int(np.argmax(band))
    dominant_power = float(band[dominant_idx])
    total_power = float(np.sum(band) + 1e-6)
    heart_rate = float(band_freqs[dominant_idx] * 60.0)
    stability = dominant_power / total_power
    return float(np.clip(stability, 0.0, 1.0)), heart_rate


# ----------------------------------------------------------------
# 4. MFCC audio feature extraction  –  replaces simple RMS energy
# ----------------------------------------------------------------
def extract_mfcc_features(audio_chunk: np.ndarray, sr: int = 16000, n_mfcc: int = 13):
    if len(audio_chunk) < sr // 10:
        return np.zeros(n_mfcc, dtype=np.float32), 0.0

    try:
        mfcc = librosa.feature.mfcc(y=audio_chunk.astype(np.float32), sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc, axis=1).astype(np.float32)
        rms = float(np.clip(np.sqrt(np.mean(audio_chunk ** 2)) * 2.0, 0.0, 1.0))
        return mfcc_mean, rms
    except Exception:
        rms = float(np.clip(np.sqrt(np.mean(audio_chunk ** 2)) * 2.0, 0.0, 1.0))
        return np.zeros(n_mfcc, dtype=np.float32), rms


def mfcc_cosine_similarity(prev_mfcc: np.ndarray, curr_mfcc: np.ndarray) -> float:
    denom = (np.linalg.norm(prev_mfcc) * np.linalg.norm(curr_mfcc)) + 1e-8
    return float(np.dot(prev_mfcc, curr_mfcc) / denom)


# ----------------------------------------------------------------
# 5. Lucas-Kanade optical flow on face landmarks
# ----------------------------------------------------------------
def compute_optical_flow_score(
    prev_gray: Optional[np.ndarray],
    curr_gray: np.ndarray,
    prev_landmarks: Optional[np.ndarray],
    curr_landmarks: np.ndarray,
    lk_indices: List[int],
) -> Tuple[float, float, Optional[np.ndarray]]:

    if prev_gray is None or prev_landmarks is None or curr_landmarks is None:
        return 0.5, 0.0, curr_landmarks

    prev_pts = prev_landmarks[lk_indices].reshape(-1, 1, 2).astype(np.float32)
    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None,
                                                     winSize=(15, 15), maxLevel=3,
                                                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    if curr_pts is None or np.sum(status) < 3:
        return 0.5, 0.0, curr_landmarks

    valid_prev = prev_pts[status.flatten() == 1]
    valid_curr = curr_pts[status.flatten() == 1]

    displacements = np.linalg.norm(valid_curr - valid_prev, axis=1)
    mean_disp = float(np.mean(displacements))
    std_disp = float(np.std(displacements) + 1e-8)

    flow_magnitude = float(np.clip(mean_disp / 8.0, 0.0, 1.0))
    flow_inconsistency = float(np.clip(std_disp / (mean_disp + 1e-8) * 0.5, 0.0, 1.0))

    return flow_magnitude, flow_inconsistency, curr_pts.reshape(-1, 2)


# ----------------------------------------------------------------
# 6. Temporal CNN  –  lightweight 1D CNN for rPPG sequence
# ----------------------------------------------------------------
if nn is not None:

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

else:
    TemporalCNN = None


_tcn_model = None
_tcn_ready = False
_TCN_WEIGHTS_PATH = BASE_DIR / "models" / "tcn_weights.pth"


def get_tcn_model():
    global _tcn_model, _tcn_ready
    if _tcn_ready:
        return _tcn_model
    if torch is None or nn is None:
        _tcn_ready = True
        return None
    try:
        _tcn_model = TemporalCNN()
        if _TCN_WEIGHTS_PATH.exists():
            _tcn_model.load_state_dict(torch.load(_TCN_WEIGHTS_PATH, map_location="cpu"))
        _tcn_model.eval()
        _tcn_ready = True
    except Exception:
        _tcn_ready = True
        _tcn_model = None
    return _tcn_model


def temporal_cnn_score(rppg_sequence: List[float]) -> float:
    model = get_tcn_model()
    if model is None or len(rppg_sequence) < 18:
        return 0.5
    arr = np.array(rppg_sequence, dtype=np.float32)
    arr = (arr - np.mean(arr)) / (np.std(arr) + 1e-8)
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1, 1, T)
    with torch.inference_mode():
        score = float(model(tensor).item())
    return clamp01(score)


# ----------------------------------------------------------------
# Trained Deepfake CNN – uses locally GPU-trained weights
# ----------------------------------------------------------------
_trained_cnn = None
_CNN_WEIGHTS_PATH = BASE_DIR / "models" / "cnn_weights.pth"


def _build_cnn_model():
    global _trained_cnn
    if _trained_cnn is not None:
        return _trained_cnn
    if torch is None or nn is None or resnet18 is None:
        return None
    try:
        backbone = resnet18(weights=None)
        num_features = backbone.fc.in_features
        backbone.fc = nn.Identity()

        classifier = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

        class DeepfakeResNet(nn.Module):
            def __init__(self, bb, clf):
                super().__init__()
                self.backbone = bb
                self.classifier = clf

            def forward(self, x):
                features = self.backbone(x)
                return self.classifier(features)

        full_model = DeepfakeResNet(backbone, classifier)
        if _CNN_WEIGHTS_PATH.exists():
            full_model.load_state_dict(torch.load(_CNN_WEIGHTS_PATH, map_location="cpu"))
        full_model.eval()
        _trained_cnn = full_model
        return _trained_cnn
    except Exception:
        return None


def trained_cnn_score(face_roi):
    """Score a face crop using trained deepfake CNN."""
    model = _build_cnn_model()
    if model is None or face_roi is None or face_roi.size == 0:
        return None

    try:
        rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (112, 112), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(np.transpose(resized, (2, 0, 1))).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std
        tensor = tensor.unsqueeze(0)
        with torch.inference_mode():
            score = float(torch.sigmoid(model(tensor)).item())
        return clamp01(score)
    except Exception:
        return None


# ----------------------------------------------------------------
# 7. ViT Face Scorer
# ----------------------------------------------------------------
class ViTFaceScorer:
    def __init__(self):
        self.ready = False
        self.load_error = None
        self.preprocess = None
        self.backbone = None

    def _ensure_loaded(self):
        if self.ready or self.load_error:
            return
        if torch is None or vit_b_16 is None or ViT_B_16_Weights is None:
            self.load_error = "torchvision vit is unavailable"
            return
        try:
            weights = ViT_B_16_Weights.DEFAULT
            model = vit_b_16(weights=weights)
            model.eval()
            self.preprocess = weights.transforms()
            self.backbone = model
            self.ready = True
        except Exception as exc:
            self.load_error = str(exc)

    def _prepare_tensor(self, image: np.ndarray):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(np.transpose(resized, (2, 0, 1))).float() / 255.0
        return self.preprocess(tensor).unsqueeze(0)

    def _predict_probs(self, image: np.ndarray):
        with torch.inference_mode():
            tensor = self._prepare_tensor(image)
            logits = self.backbone(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0)
            confidence = float(torch.max(probs).item())
            entropy = float(
                (-torch.sum(probs * torch.log(probs + 1e-8)) / np.log(probs.shape[0])).item()
            )
        return probs, confidence, entropy

    def score(self, roi: np.ndarray):
        self._ensure_loaded()
        if not self.ready:
            raise RuntimeError(self.load_error or "vit model not ready")

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        equalized = cv2.equalizeHist(gray)
        equalized_bgr = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
        flipped = cv2.flip(roi, 1)

        base_probs, base_conf, base_entropy = self._predict_probs(roi)
        eq_probs, eq_conf, eq_entropy = self._predict_probs(equalized_bgr)
        flip_probs, flip_conf, flip_entropy = self._predict_probs(flipped)

        cosine_eq = float(
            torch.nn.functional.cosine_similarity(base_probs.unsqueeze(0), eq_probs.unsqueeze(0)).item()
        )
        cosine_flip = float(
            torch.nn.functional.cosine_similarity(base_probs.unsqueeze(0), flip_probs.unsqueeze(0)).item()
        )

        score = clamp01(
            (base_entropy * 0.36)
            + ((1.0 - base_conf) * 0.22)
            + ((1.0 - cosine_eq) * 0.22)
            + ((1.0 - cosine_flip) * 0.20)
        )
        return score, {
            "backend": "vit-b16-imagenet",
            "base_confidence": base_conf,
            "base_entropy": base_entropy,
            "equalized_confidence": eq_conf,
            "equalized_entropy": eq_entropy,
            "flipped_confidence": flip_conf,
            "flipped_entropy": flip_entropy,
            "equalized_cosine": cosine_eq,
            "flipped_cosine": cosine_flip,
        }


vit_face_scorer = ViTFaceScorer()


# ----------------------------------------------------------------
# 8. CNN Face Scorer (unchanged)
# ----------------------------------------------------------------
class CNNFaceScorer:
    def __init__(self):
        self.ready = False
        self.load_error = None
        self.device = "cpu"
        self.preprocess = None
        self.backbone = None
        self.feature_extractor = None

    def _ensure_loaded(self):
        if self.ready or self.load_error:
            return
        if torch is None or resnet18 is None or ResNet18_Weights is None or nn is None:
            self.load_error = "torchvision is unavailable"
            return
        try:
            weights = ResNet18_Weights.DEFAULT
            model = resnet18(weights=weights)
            model.eval()
            self.preprocess = weights.transforms()
            self.backbone = model
            self.feature_extractor = nn.Sequential(*list(model.children())[:-1])
            self.feature_extractor.eval()
            self.ready = True
        except Exception as exc:
            self.load_error = str(exc)

    def _prepare_tensor(self, image: np.ndarray):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(np.transpose(resized, (2, 0, 1))).float() / 255.0
        return self.preprocess(tensor).unsqueeze(0)

    def _forward(self, image: np.ndarray):
        with torch.inference_mode():
            tensor = self._prepare_tensor(image)
            logits = self.backbone(tensor)
            probs = torch.softmax(logits, dim=1)
            confidence = float(torch.max(probs).item())
            entropy = float(
                (-torch.sum(probs * torch.log(probs + 1e-8), dim=1) / np.log(probs.shape[1])).item()
            )
            features = self.feature_extractor(tensor).flatten(1)
        return probs.squeeze(0), confidence, entropy, features.squeeze(0)

    def score(self, face_roi: np.ndarray):
        self._ensure_loaded()
        if not self.ready:
            raise RuntimeError(self.load_error or "cnn model not ready")

        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        edge_map = cv2.Canny(gray, 80, 160)
        edge_rgb = cv2.cvtColor(edge_map, cv2.COLOR_GRAY2BGR)
        smoothed = cv2.bilateralFilter(face_roi, d=7, sigmaColor=35, sigmaSpace=35)

        _, base_conf, base_entropy, base_feat = self._forward(face_roi)
        _, edge_conf, edge_entropy, edge_feat = self._forward(edge_rgb)
        _, smooth_conf, smooth_entropy, smooth_feat = self._forward(smoothed)

        cos_edge = float(
            torch.nn.functional.cosine_similarity(base_feat.unsqueeze(0), edge_feat.unsqueeze(0)).item()
        )
        cos_smooth = float(
            torch.nn.functional.cosine_similarity(base_feat.unsqueeze(0), smooth_feat.unsqueeze(0)).item()
        )

        score = clamp01(
            (base_entropy * 0.28)
            + ((1.0 - base_conf) * 0.24)
            + ((1.0 - cos_edge) * 0.28)
            + ((1.0 - cos_smooth) * 0.20)
        )
        return score, {
            "backend": "resnet18-imagenet",
            "base_confidence": base_conf,
            "base_entropy": base_entropy,
            "edge_confidence": edge_conf,
            "edge_entropy": edge_entropy,
            "smooth_confidence": smooth_conf,
            "smooth_entropy": smooth_entropy,
            "edge_cosine": cos_edge,
            "smooth_cosine": cos_smooth,
        }


cnn_face_scorer = CNNFaceScorer()


# ----------------------------------------------------------------
# Database helpers (unchanged)
# ----------------------------------------------------------------
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS config (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            created_at REAL NOT NULL,
            last_seen REAL NOT NULL,
            label TEXT,
            last_risk_score REAL DEFAULT 0,
            last_risk_level TEXT DEFAULT 'LOW',
            snapshot_count INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS analysis_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            created_at REAL NOT NULL,
            risk_score REAL NOT NULL,
            risk_level TEXT NOT NULL,
            physiology_score REAL NOT NULL,
            geometry_score REAL NOT NULL,
            consistency_score REAL NOT NULL,
            texture_score REAL NOT NULL,
            temporal_score REAL NOT NULL,
            face_detected INTEGER NOT NULL,
            audio_level REAL DEFAULT 0,
            heart_rate REAL DEFAULT 0,
            sync_score REAL DEFAULT 0,
            details_json TEXT NOT NULL,
            FOREIGN KEY(session_id) REFERENCES sessions(session_id)
        );
        """
    )
    existing = conn.execute("SELECT value FROM config WHERE key='app_config'").fetchone()
    if existing is None:
        conn.execute(
            "INSERT INTO config(key, value) VALUES (?, ?)",
            ("app_config", json.dumps(DEFAULT_CONFIG)),
        )
    conn.commit()
    conn.close()


def load_config():
    conn = get_db()
    row = conn.execute("SELECT value FROM config WHERE key='app_config'").fetchone()
    conn.close()
    if row is None:
        return DEFAULT_CONFIG
    try:
        loaded = json.loads(row["value"])
        return merge_config_with_defaults(loaded)
    except Exception:
        return DEFAULT_CONFIG


def save_config(config):
    conn = get_db()
    conn.execute(
        "REPLACE INTO config(key, value) VALUES (?, ?)",
        ("app_config", json.dumps(config)),
    )
    conn.commit()
    conn.close()


def merge_config_with_defaults(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged = json.loads(json.dumps(DEFAULT_CONFIG))
    if not isinstance(config, dict):
        return merged
    for section, value in config.items():
        if isinstance(value, dict) and isinstance(merged.get(section), dict):
            merged[section].update(value)
        else:
            merged[section] = value
    return merged


# ----------------------------------------------------------------
# Session / helpers
# ----------------------------------------------------------------
def ensure_session(session_id, label=None):
    now = time.time()
    if session_id not in session_state:
        session_state[session_id] = {
            "green_signal": deque(maxlen=180),
            "pos_signal": deque(maxlen=180),
            "rppg_rgb": deque(maxlen=180),
            "bbox_history": deque(maxlen=120),
            "mouth_motion": deque(maxlen=120),
            "audio_levels": deque(maxlen=120),
            "mfcc_history": deque(maxlen=60),
            "prev_gray": None,
            "prev_landmarks": None,
            "optical_flow_magnitudes": deque(maxlen=120),
            "optical_flow_inconsistencies": deque(maxlen=120),
        }

    conn = get_db()
    row = conn.execute(
        "SELECT session_id FROM sessions WHERE session_id=?", (session_id,)
    ).fetchone()
    if row is None:
        conn.execute(
            """INSERT INTO sessions(session_id, created_at, last_seen, label)
               VALUES (?, ?, ?, ?)""",
            (session_id, now, now, label or "Web Session"),
        )
    else:
        conn.execute(
            "UPDATE sessions SET last_seen=?, label=COALESCE(?, label) WHERE session_id=?",
            (now, label, session_id),
        )
    conn.commit()
    conn.close()
    return session_state[session_id]


def decode_image(image_data: str):
    if "," in image_data:
        image_data = image_data.split(",", 1)[1]
    binary = base64.b64decode(image_data)
    np_data = np.frombuffer(binary, dtype=np.uint8)
    frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Unable to decode image frame.")
    return frame


def safe_crop(frame, box):
    x, y, w, h = box
    height, width = frame.shape[:2]
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    w = max(1, min(w, width - x))
    h = max(1, min(h, height - y))
    return frame[y : y + h, x : x + w]


def smooth_signal(values: List[float], cutoff: float = 0.2) -> np.ndarray:
    arr = np.array(values, dtype=np.float32)
    if len(arr) < 7:
        return arr
    try:
        b, a = butter(2, cutoff, btype="low")
        return filtfilt(b, a, arr)
    except Exception:
        kernel = np.ones(5, dtype=np.float32) / 5.0
        return np.convolve(arr, kernel, mode="same")


def clamp01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def correlation_score(values_a, values_b):
    if len(values_a) < 8 or len(values_b) < 8:
        return 0.5
    usable = min(len(values_a), len(values_b))
    a = np.array(list(values_a)[-usable:], dtype=np.float32)
    b = np.array(list(values_b)[-usable:], dtype=np.float32)
    a = a - np.mean(a)
    b = b - np.mean(b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-6
    corr = float(np.dot(a, b) / denom)
    return float(np.clip((corr + 1.0) / 2.0, 0.0, 1.0))


# ----------------------------------------------------------------
# Core: compute_frame_metrics  —  integrates ALL new modules
# ----------------------------------------------------------------
def compute_frame_metrics(frame, state, client_audio_level, fps, audio_chunk=None):
    frame_height, frame_width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_box, face_detected = detector.detect(frame)
    x, y, w, h = face_box

    # --- Face mesh landmarks ---
    landmarks = landmark_detector.detect(frame)
    has_landmarks = landmarks is not None

    # --- POS rPPG signal extraction ---
    if has_landmarks:
        forehead_rgb = landmark_detector.extract_roi_rgb(frame, landmarks, FACE_MESH_FOREHEAD_INDICES)
        cheek_l_rgb = landmark_detector.extract_roi_rgb(frame, landmarks, FACE_MESH_CHEEK_LEFT)
        cheek_r_rgb = landmark_detector.extract_roi_rgb(frame, landmarks, FACE_MESH_CHEEK_RIGHT)
        avg_rgb = (forehead_rgb + cheek_l_rgb + cheek_r_rgb) / 3.0
    else:
        forehead = safe_crop(
            frame,
            (x + int(w * 0.2), y + int(h * 0.08), int(w * 0.6), max(1, int(h * 0.18))),
        )
        avg_rgb = np.mean(forehead.reshape(-1, 3).astype(np.float32), axis=0)

    state["rppg_rgb"].append(avg_rgb)
    state["green_signal"].append(float(avg_rgb[1]))

    mouth_box = (x + int(w * 0.22), y + int(h * 0.58), int(w * 0.56), max(1, int(h * 0.2)))
    mouth = safe_crop(frame, mouth_box)
    state["bbox_history"].append([x, y, w, h])

    # --- MFCC audio features ---
    mfcc_vec = np.zeros(13, dtype=np.float32)
    mfcc_sim = 0.5
    if audio_chunk is not None and len(audio_chunk) > 0:
        mfcc_vec, audio_rms = extract_mfcc_features(audio_chunk, sr=16000)
        if len(state["mfcc_history"]) > 0:
            prev_mfcc = state["mfcc_history"][-1]
            mfcc_sim = mfcc_cosine_similarity(prev_mfcc, mfcc_vec)
        state["mfcc_history"].append(mfcc_vec)
        state["audio_levels"].append(audio_rms)
    else:
        state["audio_levels"].append(float(client_audio_level))

    # --- Texture scores ---
    blur_score = float(np.clip(cv2.Laplacian(gray, cv2.CV_64F).var() / 350.0, 0.0, 1.0))
    edges = cv2.Canny(gray, 80, 160)
    edge_density = float(np.mean(edges > 0))
    texture_score = float(np.clip((edge_density * 4.5) + (0.35 * (1.0 - blur_score)), 0.0, 1.0))

    # --- Optical flow on landmarks ---
    flow_magnitude, flow_inconsistency, tracked_pts = compute_optical_flow_score(
        state["prev_gray"], gray, state["prev_landmarks"],
        landmarks if has_landmarks else None, FACE_MESH_LK_POINTS,
    )
    state["optical_flow_magnitudes"].append(flow_magnitude)
    state["optical_flow_inconsistencies"].append(flow_inconsistency)
    state["prev_landmarks"] = landmarks

    # --- Motion / mouth ---
    if state["prev_gray"] is None or state["prev_gray"].shape != gray.shape:
        motion_score = 0.5
        mouth_motion = 0.0
    else:
        frame_delta = cv2.absdiff(gray, state["prev_gray"])
        motion_score = float(np.clip(np.mean(frame_delta) / 24.0, 0.0, 1.0))
        prev_mouth = safe_crop(state["prev_gray"], mouth_box)
        mouth_gray = cv2.cvtColor(mouth, cv2.COLOR_BGR2GRAY)
        prev_mouth_gray = cv2.resize(prev_mouth, (mouth_gray.shape[1], mouth_gray.shape[0]))
        mouth_motion = float(
            np.clip(np.mean(cv2.absdiff(mouth_gray, prev_mouth_gray)) / 24.0, 0.0, 1.0)
        )
    state["mouth_motion"].append(mouth_motion)
    state["prev_gray"] = gray

    # --- Bbox jitter ---
    bbox_jitter = 0.0
    if len(state["bbox_history"]) >= 6:
        boxes = np.array(state["bbox_history"], dtype=np.float32)
        centers = np.column_stack(
            [boxes[:, 0] + boxes[:, 2] / 2.0, boxes[:, 1] + boxes[:, 3] / 2.0]
        )
        smoothed_x = smooth_signal(centers[:, 0].tolist())
        smoothed_y = smooth_signal(centers[:, 1].tolist())
        jitter_x = centers[:, 0] - smoothed_x
        jitter_y = centers[:, 1] - smoothed_y
        scale = np.mean(boxes[:, 2] + boxes[:, 3]) / 2.0 + 1e-6
        bbox_jitter = float(np.clip(np.mean(np.sqrt(jitter_x ** 2 + jitter_y ** 2)) / scale, 0.0, 1.0))

    # --- rPPG analysis: POS + FFT ---
    pos_signal_val = 0.0
    physiology_stability = 0.5
    heart_rate = 0.0
    if len(state["rppg_rgb"]) >= 18:
        pos_raw = pos_rppg_signal(list(state["rppg_rgb"]), fps)
        if pos_raw.size > 0:
            pos_signal_val = float(pos_raw[-1])
            state["pos_signal"].append(pos_signal_val)
            green_signal = [s[1] for s in state["rppg_rgb"]]
            if False:
                stability_from_pos, hr_from_pos = compute_fft_stability(pos_raw, fps)
                physiology_stability = stability_from_pos
                heart_rate = hr_from_pos
            else:
                physiology_stability, heart_rate = compute_fft_stability(green_signal, fps)
        else:
            green_signal = [s[1] for s in state["rppg_rgb"]]
            physiology_stability, heart_rate = compute_fft_stability(green_signal, fps)
    else:
        green_signal = [s[1] for s in state["rppg_rgb"]]
        if len(green_signal) >= 18:
            physiology_stability, heart_rate = compute_fft_stability(green_signal, fps)

    if heart_rate < 1.0 and physiology_stability > 0.49:
        physiology_score = 0.5
    else:
        physiology_score = float(
            np.clip(1.0 - physiology_stability + abs(72.0 - heart_rate) / 180.0, 0.0, 1.0)
        )

    # --- Temporal CNN score on rPPG ---
    tcn_score = 0.5
    if len(state["pos_signal"]) >= 18:
        tcn_score = temporal_cnn_score(list(state["pos_signal"]))

    # --- AV sync: mouth motion vs audio with MFCC bonus ---
    sync_alignment = correlation_score(state["mouth_motion"], state["audio_levels"])
    mfcc_bonus = clamp01(mfcc_sim) if len(state["mfcc_history"]) >= 2 else 0.5
    consistency_score = float(np.clip((1.0 - sync_alignment) * 0.65 + (1.0 - mfcc_bonus) * 0.35, 0.0, 1.0))

    # --- Geometry score: bbox jitter + edge density + optical flow ---
    avg_flow_inconsistency = float(np.mean(list(state["optical_flow_inconsistencies"])[-20:])) if state["optical_flow_inconsistencies"] else flow_inconsistency
    geometry_score = float(
        np.clip(
            (bbox_jitter * 1.5)
            + (0.35 * edge_density)
            + (0.15 * (1.0 - blur_score))
            + (avg_flow_inconsistency * 0.8),
            0.0, 1.0,
        )
    )

    # --- Temporal score: motion mismatch + flow inconsistency ---
    temporal_score = float(np.clip(
        abs(motion_score - client_audio_level) * 0.5 + bbox_jitter * 0.3 + flow_inconsistency * 0.2,
        0.0, 1.0,
    ))

    # --- Face ROI for CNN / ViT ---
    face_roi = safe_crop(
        frame,
        (x - int(w * 0.08), y - int(h * 0.08), int(w * 1.16), int(h * 1.16)),
    )

    feature_inputs = {
        "physiology_feature": clamp01(1.0 - physiology_stability),
        "geometry_feature": clamp01(bbox_jitter + edge_density * 0.35 + avg_flow_inconsistency * 0.3),
        "audio_visual_feature": consistency_score,
        "image_feature": texture_score,
        "temporal_feature": temporal_score,
    }

    cnn_meta = {
        "backend": "heuristic-fallback",
        "base_confidence": 0.0, "base_entropy": 0.0,
        "edge_confidence": 0.0, "edge_entropy": 0.0,
        "smooth_confidence": 0.0, "smooth_entropy": 0.0,
        "edge_cosine": 1.0, "smooth_cosine": 1.0,
    }
    cnn_score = clamp01((texture_score * 0.58) + (geometry_score * 0.22) + ((1.0 - blur_score) * 0.20))
    if face_roi.size > 0:
        trained = trained_cnn_score(face_roi)
        if trained is not None:
            cnn_meta["real_score"] = trained
            cnn_score = 1.0 - trained
            cnn_meta["backend"] = "trained-deepfake-resnet18"
        else:
            try:
                cnn_score, cnn_meta = cnn_face_scorer.score(face_roi)
            except Exception as exc:
                cnn_meta["error"] = str(exc)

    # --- ViT score ---
    vit_score = clamp01((geometry_score * 0.44) + (temporal_score * 0.32) + (texture_score * 0.24))
    vit_meta = {"backend": "heuristic-fallback"}
    if face_roi.size > 0 and has_landmarks:
        try:
            aligned_face = landmark_detector.align_face(frame, landmarks, 224)
            vit_score, vit_meta = vit_face_scorer.score(aligned_face)
        except Exception as exc:
            vit_meta["error"] = str(exc)

    model_scores = {
        "cnn": cnn_score,
        "vit": vit_score,
        "syncnet": clamp01((consistency_score * 0.72) + (temporal_score * 0.18) + (motion_score * 0.10)),
        "rppg": clamp01((physiology_score * 0.62) + (tcn_score * 0.38)),
        "flow": clamp01(flow_inconsistency * 0.6 + flow_magnitude * 0.25 + bbox_jitter * 0.15),
        "geometry_model": clamp01((geometry_score * 0.74) + (bbox_jitter * 0.26)),
    }

    return {
        "physiology": physiology_score,
        "geometry": geometry_score,
        "consistency": consistency_score,
        "texture": texture_score,
        "temporal": temporal_score,
        "details": {
            "face_box": {"x": x, "y": y, "w": w, "h": h},
            "frame_width": int(frame_width),
            "frame_height": int(frame_height),
            "face_detected": face_detected,
            "has_landmarks": has_landmarks,
            "blur_score": blur_score,
            "edge_density": edge_density,
            "bbox_jitter": bbox_jitter,
            "mouth_motion": mouth_motion,
            "sync_alignment": sync_alignment,
            "mfcc_similarity": clamp01(mfcc_sim),
            "heart_rate": heart_rate,
            "physiology_stability": physiology_stability,
            "motion_score": motion_score,
            "optical_flow_magnitude": flow_magnitude,
            "optical_flow_inconsistency": flow_inconsistency,
            "avg_flow_inconsistency": avg_flow_inconsistency,
            "tcn_score": tcn_score,
            "audio_level": client_audio_level,
            "feature_inputs": feature_inputs,
            "model_scores": model_scores,
            "normalized_scores": model_scores.copy(),
            "cnn_meta": cnn_meta,
            "vit_meta": vit_meta,
        },
    }


# ----------------------------------------------------------------
# Video / aggregation (mostly unchanged)
# ----------------------------------------------------------------
def aggregate_video_metrics(frame_metrics: List[Dict[str, Any]], config: Dict[str, Any]):
    module_keys = ["physiology", "geometry", "consistency", "texture", "temporal"]
    aggregated = {
        key: float(np.mean([item[key] for item in frame_metrics])) for key in module_keys
    }
    detail_source = frame_metrics[-1]["details"] if frame_metrics else {}
    details = {
        **detail_source,
        "frame_count": len(frame_metrics),
        "face_detect_rate": float(
            np.mean([1.0 if item["details"]["face_detected"] else 0.0 for item in frame_metrics])
        ) if frame_metrics else 0.0,
    }
    details["feature_inputs"] = {
        key: float(np.mean([item["details"].get("feature_inputs", {}).get(key, 0.0) for item in frame_metrics]))
        for key in ["physiology_feature", "geometry_feature", "audio_visual_feature", "image_feature", "temporal_feature"]
    }
    details["model_scores"] = {
        key: float(np.mean([item["details"].get("model_scores", {}).get(key, 0.0) for item in frame_metrics]))
        for key in MODEL_SCORE_KEYS
    }
    details["normalized_scores"] = details["model_scores"].copy()
    risk_score, risk_level = fuse_scores(aggregated, config)
    return aggregated, details, risk_score, risk_level


def fuse_scores(metrics, config):
    detail_scores = metrics.get("details", {}).get("model_scores", {})

    fusion_path = BASE_DIR / "models" / "fusion_weights.json"
    if fusion_path.exists():
        try:
            fusion = json.loads(fusion_path.read_text(encoding="utf-8"))
            if fusion.get("method") == "logistic_regression":
                scores = np.array([float(detail_scores.get(k, 0.0)) for k in MODEL_SCORE_KEYS])
                mean = np.array(fusion.get("mean", [0.0] * len(MODEL_SCORE_KEYS)))
                std = np.array(fusion.get("std", [1.0] * len(MODEL_SCORE_KEYS)))
                scores_norm = (scores - mean) / (std + 1e-8)
                z = float(np.dot(scores_norm, [fusion["coef"][k] for k in MODEL_SCORE_KEYS]) + fusion["intercept"])
                score = clamp01(float(1.0 / (1.0 + np.exp(-z))))
                low = config["risk_thresholds"]["low"]
                medium = config["risk_thresholds"]["medium"]
                if score < low:
                    level = "LOW"
                elif score < medium:
                    level = "MEDIUM"
                else:
                    level = "HIGH"
                return float(np.clip(score, 0.0, 1.0)), level
        except Exception:
            pass

    if detail_scores:
        score = 0.0
        weights = config.get("weights", {})
        total_weight = 0.0
        for key in MODEL_SCORE_KEYS:
            weight = float(weights.get(key, 0.0))
            score += float(detail_scores.get(key, 0.0)) * weight
            total_weight += weight
        if total_weight > 0:
            score = score / total_weight
        score = clamp01(score)
    else:
        score = 0.0
        legacy_weights = {
            "physiology": 0.18, "geometry": 0.24, "consistency": 0.22,
            "texture": 0.20, "temporal": 0.16,
        }
        for key, weight in legacy_weights.items():
            score += metrics[key] * weight
    low = config["risk_thresholds"]["low"]
    medium = config["risk_thresholds"]["medium"]
    if score < low:
        level = "LOW"
    elif score < medium:
        level = "MEDIUM"
    else:
        level = "HIGH"
    return float(np.clip(score, 0.0, 1.0)), level


def persist_analysis(session_id, label, fused_score, fused_level, metrics):
    now = time.time()
    conn = get_db()
    conn.execute(
        """UPDATE sessions
           SET last_seen=?, label=COALESCE(?, label), last_risk_score=?, last_risk_level=?, snapshot_count=snapshot_count+1
           WHERE session_id=?""",
        (now, label, fused_score, fused_level, session_id),
    )
    conn.execute(
        """INSERT INTO analysis_logs(
               session_id, created_at, risk_score, risk_level, physiology_score, geometry_score,
               consistency_score, texture_score, temporal_score, face_detected, audio_level,
               heart_rate, sync_score, details_json
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            session_id, now, fused_score, fused_level,
            metrics["physiology"], metrics["geometry"],
            metrics["consistency"], metrics["texture"], metrics["temporal"],
            1 if metrics["details"].get("face_detected") else 0,
            metrics["details"].get("audio_level", 0.0),
            metrics["details"].get("heart_rate", 0.0),
            metrics["details"].get("sync_alignment", 0.0),
            json.dumps(metrics["details"]),
        ),
    )
    conn.commit()
    conn.close()


def delete_session_records(session_id):
    session_state.pop(session_id, None)
    conn = get_db()
    conn.execute("DELETE FROM analysis_logs WHERE session_id=?", (session_id,))
    deleted = conn.execute("DELETE FROM sessions WHERE session_id=?", (session_id,)).rowcount
    conn.commit()
    conn.close()
    return deleted > 0


def sample_video_metrics(video_path: Path, state, config: Dict[str, Any]):
    if av is None:
        raise ValueError("Video analysis unavailable: 'av' library not installed.")
    container = av.open(str(video_path))
    video_stream = next((stream for stream in container.streams if stream.type == "video"), None)
    audio_stream = next((stream for stream in container.streams if stream.type == "audio"), None)
    if video_stream is None:
        raise ValueError("No video stream found.")

    sample_fps = config["analysis"]["sample_fps"]
    audio_sr = config["analysis"]["audio_sr"]
    max_seconds = config["analysis"]["max_video_seconds"]
    duration = float(video_stream.duration * video_stream.time_base) if video_stream.duration else max_seconds
    analysis_seconds = min(max_seconds, duration if duration > 0 else max_seconds)

    audio_signal = np.array([], dtype=np.float32)
    if audio_stream is not None:
        container_audio = av.open(str(video_path))
        resampled = []
        for frame in container_audio.decode(audio=0):
            arr = frame.to_ndarray()
            if arr.ndim == 2:
                arr = np.mean(arr.astype(np.float32), axis=0)
            else:
                arr = arr.astype(np.float32)
            resampled.append(arr)
        if resampled:
            merged = np.concatenate(resampled)
            if np.max(np.abs(merged)) > 0:
                merged = merged / np.max(np.abs(merged))
            if librosa is not None:
                audio_signal = librosa.resample(merged, orig_sr=audio_stream.rate, target_sr=audio_sr)
            else:
                audio_signal = merged
        container_audio.close()

    audio_window = int(audio_sr / max(sample_fps, 1))
    frame_metrics = []
    container_video = av.open(str(video_path))
    sampled_every = max(int(round(float(video_stream.average_rate or sample_fps) / sample_fps)), 1)
    for index, frame in enumerate(container_video.decode(video=0)):
        if index % sampled_every != 0:
            continue
        frame_time = index / max(float(video_stream.average_rate or sample_fps), 1.0)
        if frame_time > analysis_seconds:
            break
        bgr = cv2.cvtColor(frame.to_ndarray(format="rgb24"), cv2.COR_RGB2BGR)
        audio_level = 0.0
        audio_chunk = np.array([], dtype=np.float32)
        if audio_signal.size:
            start = int(frame_time * audio_sr)
            end = min(start + audio_window, audio_signal.size)
            if end > start:
                audio_chunk = audio_signal[start:end]
                audio_level = float(np.clip(np.sqrt(np.mean(audio_chunk ** 2)) * 2.0, 0.0, 1.0))
        frame_metrics.append(compute_frame_metrics(bgr, state, audio_level, sample_fps, audio_chunk))
    container_video.close()
    if not frame_metrics:
        raise ValueError("Video is too short or could not be sampled.")
    return frame_metrics


# ----------------------------------------------------------------
# FastAPI routes (unchanged)
# ----------------------------------------------------------------
@app.on_event("startup")
def startup_event():
    init_db()
    cnn_face_scorer._ensure_loaded()
    vit_face_scorer._ensure_loaded()


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/admin", response_class=HTMLResponse)
def admin(request: Request):
    return templates.TemplateResponse("admin.html", {"request": request})


@app.post("/api/session")
def create_session(payload: SessionCreatePayload):
    session_id = payload.session_id or str(uuid.uuid4())
    label = payload.label or "Web Session"
    ensure_session(session_id, label)
    return {"session_id": session_id}


@app.delete("/api/session/{session_id}")
def delete_session(session_id: str):
    if not delete_session_records(session_id):
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"ok": True, "session_id": session_id}


@app.post("/api/analyze")
def analyze(payload: AnalyzePayload):
    session_id = payload.session_id or str(uuid.uuid4())
    label = payload.label or "Web Session"
    image_payload = payload.image or payload.image_data
    if not image_payload:
        raise HTTPException(status_code=400, detail="Missing image payload.")

    ensure_session(session_id, label)
    try:
        frame = decode_image(image_payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    config = load_config()
    metrics = compute_frame_metrics(
        frame,
        session_state[session_id],
        float(payload.audio_level or 0.0),
        float(payload.fps or 12.0),
    )
    risk_score, risk_level = fuse_scores(metrics, config)
    persist_analysis(session_id, label, risk_score, risk_level, metrics)
    return {
        "session_id": session_id,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "module_scores": {key: metrics[key] for key in ["physiology", "geometry", "consistency", "texture", "temporal"]},
        "model_scores": metrics["details"].get("model_scores", {}),
        "normalized_scores": metrics["details"].get("normalized_scores", {}),
        "feature_inputs": metrics["details"].get("feature_inputs", {}),
        "details": metrics["details"],
        "config": config,
        "mode": "image",
    }


@app.post("/api/analyze-video")
async def analyze_video(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    label: Optional[str] = Form("Video Session"),
):
    current_session_id = session_id or str(uuid.uuid4())
    ensure_session(current_session_id, label or "Video Session")
    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_path = Path(temp_file.name)
        temp_file.write(await file.read())

    config = load_config()
    try:
        frame_metrics = sample_video_metrics(temp_path, session_state[current_session_id], config)
        module_scores, details, risk_score, risk_level = aggregate_video_metrics(frame_metrics, config)
        persist_analysis(
            current_session_id, label or "Video Session", risk_score, risk_level,
            {**module_scores, "details": details},
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        temp_path.unlink(missing_ok=True)

    return {
        "session_id": current_session_id,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "module_scores": module_scores,
        "model_scores": details.get("model_scores", {}),
        "normalized_scores": details.get("normalized_scores", {}),
        "feature_inputs": details.get("feature_inputs", {}),
        "details": details,
        "config": config,
        "mode": "video",
    }


@app.get("/api/admin/summary")
def admin_summary():
    conn = get_db()
    config = load_config()
    sessions = [
        dict(row)
        for row in conn.execute(
            """SELECT session_id, created_at, last_seen, label, last_risk_score, last_risk_level, snapshot_count
               FROM sessions ORDER BY last_seen DESC LIMIT 25"""
        ).fetchall()
    ]
    recent_rows = conn.execute(
        """SELECT id, session_id, created_at, risk_score, risk_level, physiology_score, geometry_score,
                  consistency_score, texture_score, temporal_score, face_detected, audio_level, heart_rate, sync_score, details_json
           FROM analysis_logs ORDER BY created_at DESC LIMIT 60"""
    ).fetchall()
    stats_row = conn.execute(
        """SELECT COUNT(*) AS total_logs, AVG(risk_score) AS avg_risk,
                  SUM(CASE WHEN risk_level='HIGH' THEN 1 ELSE 0 END) AS high_count,
                  SUM(CASE WHEN risk_level='MEDIUM' THEN 1 ELSE 0 END) AS medium_count,
                  SUM(CASE WHEN risk_level='LOW' THEN 1 ELSE 0 END) AS low_count
           FROM analysis_logs"""
    ).fetchone()
    conn.close()
    recent = []
    model_average_seed = {key: [] for key in MODEL_SCORE_KEYS}
    for row in recent_rows:
        item = dict(row)
        details = {}
        try:
            details = json.loads(item.pop("details_json", "") or "{}")
        except Exception:
            details = {}
        model_scores = details.get("model_scores", {})
        for key in MODEL_SCORE_KEYS:
            model_average_seed[key].append(float(model_scores.get(key, 0.0)))
        item["model_scores"] = {key: float(model_scores.get(key, 0.0)) for key in MODEL_SCORE_KEYS}
        item["feature_inputs"] = details.get("feature_inputs", {})
        recent.append(item)

    return {
        "config": config,
        "sessions": sessions,
        "recent_logs": recent,
        "stats": dict(stats_row),
        "model_averages": {
            key: float(np.mean(values)) if values else 0.0
            for key, values in model_average_seed.items()
        },
    }


@app.delete("/api/admin/sessions/{session_id}")
def admin_delete_session(session_id: str):
    if not delete_session_records(session_id):
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"ok": True, "session_id": session_id}


@app.post("/api/admin/config")
def update_config(payload: ConfigUpdatePayload):
    config = load_config()
    if payload.weights:
        for key in MODEL_SCORE_KEYS:
            if key in payload.weights:
                config["weights"][key] = float(payload.weights[key])
    if payload.risk_thresholds:
        for key in config["risk_thresholds"]:
            if key in payload.risk_thresholds:
                config["risk_thresholds"][key] = float(payload.risk_thresholds[key])
    total_weight = sum(config["weights"].values()) or 1.0
    for key in config["weights"]:
        config["weights"][key] = config["weights"][key] / total_weight
    low = max(0.0, min(config["risk_thresholds"]["low"], 1.0))
    medium = max(low + 0.05, min(config["risk_thresholds"]["medium"], 1.0))
    config["risk_thresholds"]["low"] = low
    config["risk_thresholds"]["medium"] = medium
    save_config(config)
    return {"ok": True, "config": config}


@app.get("/health")
def health():
    return {"status": "ok", "service": "deepfake-platform"}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=False)
