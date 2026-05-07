const video = document.getElementById("video");
const previewImage = document.getElementById("previewImage");
const overlay = document.getElementById("overlay");
const overlayCtx = overlay.getContext("2d");
const captureCanvas = document.getElementById("captureCanvas");
const captureCtx = captureCanvas.getContext("2d");

const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const imageInput = document.getElementById("imageInput");
const videoInput = document.getElementById("videoInput");
const emptyState = document.getElementById("emptyState");
const SESSION_STORAGE_KEY = "deepfake-platform-session-id";

const MODEL_DEFS = [
  ["cnn", "CNN 分数", "#f35b4f"],
  ["vit", "ViT 分数", "#f39c12"],
  ["syncnet", "SyncNet 分数", "#3d8bfd"],
  ["rppg", "rPPG 分数", "#20b26b"],
  ["flow", "光流分数", "#7a57f4"],
  ["geometry_model", "几何分数", "#14a3a3"]
];

let mediaStream = null;
let audioContext = null;
let analyser = null;
let audioDataArray = null;
let sessionId = null;
let loopTimer = null;
let frameCounter = 0;
let lastTick = performance.now();

function setStatus(text) {
  document.getElementById("riskText").textContent = text;
}

function levelClass(score) {
  if (score < 0.35) return "risk-low";
  if (score < 0.62) return "risk-medium";
  return "risk-high";
}

function textForLevel(level) {
  if (level === "HIGH") return "高风险，建议人工复核";
  if (level === "MEDIUM") return "中风险，建议继续观察";
  return "低风险，当前相对稳定";
}

function renderModelPlaceholders() {
  const grid = document.getElementById("modelScoreGrid");
  const confidenceRow = document.getElementById("confidenceRow");

  grid.innerHTML = MODEL_DEFS.map(([key, label]) => `
    <div class="model-score-card">
      <span>${label}</span>
      <strong id="model-${key}">0.00</strong>
    </div>
  `).join("");

  confidenceRow.innerHTML = MODEL_DEFS.map(([key, label, color]) => `
    <div class="confidence-chip" style="--chip:${color}">
      <span>${label}</span>
      <strong id="confidence-${key}">0.00</strong>
    </div>
  `).join("");
}

function setStopState(enabled) {
  stopBtn.disabled = !enabled;
}

function showCameraPreview() {
  video.style.display = "block";
  previewImage.style.display = "none";
  emptyState.style.display = "none";
  overlay.style.display = "block";
  setStopState(true);
}

function showImagePreview(src) {
  previewImage.src = src;
  previewImage.style.display = "block";
  video.style.display = "none";
  emptyState.style.display = "none";
  overlay.style.display = "block";
  setStopState(true);
}

function showEmptyState() {
  previewImage.removeAttribute("src");
  previewImage.style.display = "none";
  video.style.display = "none";
  emptyState.style.display = "grid";
  overlay.style.display = "none";
  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
  setStopState(false);
}

async function createSession() {
  const existingSessionId = localStorage.getItem(SESSION_STORAGE_KEY);
  const response = await fetch("/api/session", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: existingSessionId, label: "Browser Session" })
  });
  const data = await response.json();
  sessionId = data.session_id;
  localStorage.setItem(SESSION_STORAGE_KEY, sessionId);
  document.getElementById("sessionLabel").textContent = `会话 ID: ${sessionId.slice(0, 8)}`;
}

async function ensureSession() {
  if (!sessionId) {
    await createSession();
  }
}

function getAudioLevel() {
  if (!analyser || !audioDataArray) return 0;
  analyser.getByteTimeDomainData(audioDataArray);
  let sum = 0;
  for (let i = 0; i < audioDataArray.length; i += 1) {
    const centered = (audioDataArray[i] - 128) / 128;
    sum += centered * centered;
  }
  return Math.min(Math.sqrt(sum / audioDataArray.length) * 2.2, 1);
}

function getDisplaySize() {
  if (previewImage.style.display !== "none" && previewImage.clientWidth > 0) {
    return { width: previewImage.clientWidth, height: previewImage.clientHeight };
  }
  return { width: video.clientWidth || 640, height: video.clientHeight || 480 };
}

function drawOverlay(faceBox, level, score, sourceWidth, sourceHeight) {
  const displaySize = getDisplaySize();
  overlay.width = displaySize.width;
  overlay.height = displaySize.height;
  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
  if (!faceBox || !sourceWidth || !sourceHeight) return;

  const ratioX = overlay.width / sourceWidth;
  const ratioY = overlay.height / sourceHeight;
  const color = level === "HIGH" ? "#ff6b57" : level === "MEDIUM" ? "#ffc53d" : "#57d37c";
  overlayCtx.strokeStyle = color;
  overlayCtx.lineWidth = 4;
  overlayCtx.strokeRect(faceBox.x * ratioX, faceBox.y * ratioY, faceBox.w * ratioX, faceBox.h * ratioY);
  overlayCtx.fillStyle = color;
  overlayCtx.font = "bold 18px Arial";
  overlayCtx.fillText(`${level} ${(score * 100).toFixed(1)}%`, faceBox.x * ratioX, Math.max(24, faceBox.y * ratioY - 10));
}

function updateModelViews(modelScores = {}, normalizedScores = {}, featureInputs = {}) {
  document.getElementById("featurePhysiology").textContent = Number(featureInputs.physiology_feature || 0).toFixed(2);
  document.getElementById("featureGeometry").textContent = Number(featureInputs.geometry_feature || 0).toFixed(2);
  document.getElementById("featureAudioVisual").textContent = Number(featureInputs.audio_visual_feature || 0).toFixed(2);
  document.getElementById("featureImage").textContent = Number(featureInputs.image_feature || 0).toFixed(2);
  document.getElementById("featureTemporal").textContent = Number(featureInputs.temporal_feature || 0).toFixed(2);

  MODEL_DEFS.forEach(([key]) => {
    document.getElementById(`model-${key}`).textContent = Number(modelScores[key] || 0).toFixed(2);
    document.getElementById(`confidence-${key}`).textContent = Number(normalizedScores[key] || 0).toFixed(2);
  });
}

function updateUI(result, audioLevel, fps) {
  const score = result.risk_score || 0;
  const level = result.risk_level || "LOW";
  const details = result.details || {};
  const moduleScores = result.module_scores || {};
  const modelScores = result.model_scores || {};
  const normalizedScores = result.normalized_scores || modelScores;
  const featureInputs = result.feature_inputs || {};
  const sourceWidth = Number(details.frame_width || captureCanvas.width || 640);
  const sourceHeight = Number(details.frame_height || captureCanvas.height || 480);

  updateModelViews(modelScores, normalizedScores, featureInputs);
  document.getElementById("modeValue").textContent = result.mode === "video" ? "短视频多模型分析" : "图片单帧分析";
  document.getElementById("riskBadgeText").textContent = level;
  const badge = document.getElementById("riskBadge");
  badge.className = `risk-pill ${levelClass(score)}`;
  badge.textContent = `${level} RISK`;

  document.getElementById("riskScoreValue").textContent = `${(score * 100).toFixed(1)}%`;
  setStatus(textForLevel(level));
  document.getElementById("heartRateValue").textContent = `${(details.heart_rate || 0).toFixed(1)} bpm`;
  document.getElementById("syncValue").textContent = Number(1 - (moduleScores.consistency || 0)).toFixed(2);
  document.getElementById("audioLevelValue").textContent = Number(details.audio_level || audioLevel || 0).toFixed(2);
  document.getElementById("mfccValue").textContent = Number(details.mfcc_similarity || 0).toFixed(2);
  document.getElementById("flowMagValue").textContent = Number(details.optical_flow_magnitude || 0).toFixed(2);
  document.getElementById("flowInconValue").textContent = Number(details.optical_flow_inconsistency || details.avg_flow_inconsistency || 0).toFixed(2);
  document.getElementById("tcnValue").textContent = Number(details.tcn_score || 0).toFixed(2);
  document.getElementById("landmarkValue").textContent = details.has_landmarks ? "已检测" : "未检测";
  document.getElementById("edgeValue").textContent = Number(details.edge_density || 0).toFixed(2);
  document.getElementById("motionValue").textContent = Number(details.bbox_jitter || 0).toFixed(2);
  document.getElementById("faceDetectValue").textContent = details.face_detected ? "是" : "否";
  document.getElementById("latencyLabel").textContent = `${fps.toFixed(1)} FPS`;

  drawOverlay(details.face_box, level, score, sourceWidth, sourceHeight);
}

async function analyzeFrame() {
  if (!mediaStream || !sessionId) return;
  captureCtx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);
  const image = captureCanvas.toDataURL("image/jpeg", 0.72);
  const audioLevel = getAudioLevel();

  frameCounter += 1;
  const now = performance.now();
  const elapsed = (now - lastTick) / 1000;
  const fps = elapsed > 0 ? frameCounter / elapsed : 0;
  if (elapsed >= 1) {
    frameCounter = 0;
    lastTick = now;
  }

  const response = await fetch("/api/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: sessionId,
      label: "Browser Live Session",
      image,
      audio_level: audioLevel,
      fps: Math.max(fps, 8)
    })
  });
  const result = await response.json();
  if (!response.ok) {
    throw new Error(result.detail || "实时分析失败");
  }
  updateUI(result, audioLevel, Math.max(fps, 8));
}

async function uploadVideo(file) {
  await ensureSession();
  previewImage.removeAttribute("src");
  previewImage.style.display = "none";
  video.style.display = "block";
  emptyState.style.display = "none";

  const objectUrl = URL.createObjectURL(file);
  video.srcObject = null;
  video.src = objectUrl;
  video.muted = true;
  video.loop = true;
  await video.play().catch(() => {});

  const formData = new FormData();
  formData.append("file", file);
  formData.append("session_id", sessionId);
  formData.append("label", "Uploaded Video Session");

  setStatus("短视频上传中，正在执行多模型独立判断");
  const response = await fetch("/api/analyze-video", {
    method: "POST",
    body: formData
  });
  const result = await response.json();
  if (!response.ok) {
    throw new Error(result.detail || "视频分析失败");
  }
  updateUI(result, 0, 0);
}

async function analyzeImageData(image, audioLevel = 0, fps = 8) {
  await ensureSession();
  const response = await fetch("/api/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: sessionId,
      label: "Uploaded Image Session",
      image,
      audio_level: audioLevel,
      fps
    })
  });
  const result = await response.json();
  if (!response.ok) {
    throw new Error(result.detail || "图片分析失败");
  }
  updateUI(result, audioLevel, fps);
}

function explainCameraFailure(error) {
  if (!window.isSecureContext) {
    return "当前页面不是安全上下文，浏览器会限制摄像头。建议优先使用短视频上传。";
  }
  if (error && error.name === "NotAllowedError") {
    return "浏览器拒绝了摄像头权限，请允许相机和麦克风访问。";
  }
  if (error && (error.name === "NotFoundError" || error.name === "DevicesNotFoundError")) {
    return "没有检测到可用摄像头，请检查设备或直接上传短视频。";
  }
  return `摄像头启动失败：${error && error.message ? error.message : "未知错误"}`;
}

async function start() {
  await ensureSession();
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    setStatus("当前浏览器不支持摄像头采集，建议直接上传短视频。");
    return;
  }

  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, frameRate: { ideal: 12, max: 15 } },
      audio: true
    });
  } catch (error) {
    console.error(error);
    setStatus(explainCameraFailure(error));
    showEmptyState();
    return;
  }

  video.src = "";
  video.srcObject = mediaStream;
  await video.play().catch(() => {});

  audioContext = new (window.AudioContext || window.webkitAudioContext)();
  const audioSource = audioContext.createMediaStreamSource(mediaStream);
  analyser = audioContext.createAnalyser();
  analyser.fftSize = 1024;
  audioDataArray = new Uint8Array(analyser.frequencyBinCount);
  audioSource.connect(analyser);

  startBtn.disabled = true;
  showCameraPreview();
  setStatus("摄像头已启动，当前按轻量多模型阵列方式分析");

  loopTimer = setInterval(() => {
    analyzeFrame().catch((error) => {
      console.error(error);
      setStatus(error.message || "分析请求失败，请检查后端服务");
    });
  }, 1000);
}

function stop() {
  if (loopTimer) {
    clearInterval(loopTimer);
    loopTimer = null;
  }
  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
    mediaStream = null;
  }
  if (audioContext) {
    audioContext.close();
    audioContext = null;
  }
  video.pause();
  video.srcObject = null;
  video.src = "";
  startBtn.disabled = false;
  showEmptyState();
}

async function handleImageUpload(event) {
  const [file] = event.target.files || [];
  if (!file) return;
  stop();
  const reader = new FileReader();
  reader.onload = async () => {
    try {
      showImagePreview(reader.result);
      await analyzeImageData(reader.result, 0, 8);
      setStatus("图片已进入多模型独立判断流程");
    } catch (error) {
      console.error(error);
      setStatus(error.message || "图片检测失败");
    }
  };
  reader.readAsDataURL(file);
  event.target.value = "";
}

async function handleVideoUpload(event) {
  const [file] = event.target.files || [];
  if (!file) return;
  stop();
  try {
    await uploadVideo(file);
    setStatus("短视频多模型检测完成");
  } catch (error) {
    console.error(error);
    setStatus(error.message || "视频检测失败");
  }
  event.target.value = "";
}

function initPage() {
  renderModelPlaceholders();
  setStatus("等待检测输入");
  showEmptyState();
}

startBtn.addEventListener("click", () => {
  start().catch((error) => {
    console.error(error);
    setStatus(error.message || "摄像头启动失败");
  });
});
stopBtn.addEventListener("click", stop);
imageInput.addEventListener("change", handleImageUpload);
videoInput.addEventListener("change", handleVideoUpload);

initPage();
