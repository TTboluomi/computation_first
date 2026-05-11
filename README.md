# 多模态深伪检测平台

这是一个基于 FastAPI 的深伪检测 Web 平台，支持图片和视频分析。项目融合了 CNN 人脸真伪分类、MediaPipe 人脸关键点、rPPG 生理信号、光流运动一致性、音画同步和纹理启发式特征，输出综合风险分数与风险等级。

## 功能概览

- 图片人脸真伪检测：上传或摄像头采集图片后返回风险分数。
- 视频深伪分析：抽帧分析人脸、关键点、光流、rPPG 和音画同步特征。
- CNN 真脸识别：使用 LFW 真人脸和增强伪造脸训练 ResNet18 分类器。
- MediaPipe 人脸检测与关键点：用于额头、脸颊、嘴部、眼部等区域定位。
- rPPG 生理检测：从脸部颜色变化估计生理信号稳定性。
- 光流和几何一致性：检测人脸框抖动、关键点运动和局部异常。
- 管理后台：查看会话、分析日志、模型分数和融合配置。

## 分数含义

- `details.cnn_meta.real_score`：CNN 真人置信度，越接近 `1` 越像真实人脸。
- `model_scores.cnn`：CNN 伪造风险，等于 `1 - real_score`。
- `risk_score`：多模块融合后的综合风险分数，越高越可疑。
- `risk_level`：风险等级，包含 `LOW`、`MEDIUM`、`HIGH`。

## 项目结构

```text
.
├── app.py                    # FastAPI 后端、模型加载、分析接口和后台接口
├── requirements.txt          # Python 运行依赖
├── static/                   # 前端 JS/CSS 资源
├── templates/                # 页面模板
├── models/
│   ├── cnn_weights.pth       # LFW 训练后的 CNN 权重
│   ├── tcn_weights.pth       # rPPG 时间序列模型权重
│   ├── face_detector.tflite  # MediaPipe 人脸检测模型
│   └── face_landmarker.task  # MediaPipe 人脸关键点模型
├── train_lfw.py              # LFW 真人脸 + 伪造增强 CNN 训练脚本
├── train_tcn.py              # rPPG/时序 CNN 训练脚本
├── train_cnn.py              # CNN 训练脚本
├── train_cnn_fast.py         # 快速合成数据 CNN 训练脚本
├── train_fusion.py           # 融合模型训练脚本
├── lfw.zip                   # LFW 数据集压缩包
└── lfw/                      # 已解压的 LFW 数据集
```

## 本地运行

推荐 Python 3.10。

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m uvicorn app:app --host 0.0.0.0 --port 5000 --ws none
```

打开页面：

- 用户页面：`http://127.0.0.1:5000/`
- 管理页面：`http://127.0.0.1:5000/admin`
- 健康检查：`http://127.0.0.1:5000/health`

## 虚拟机部署

当前虚拟机部署路径为：

```text
/opt/deepfake-platform
```

建议使用独立 Python 3.10 环境，避免破坏系统 Python：

```bash
cd /opt/deepfake-platform
/opt/miniconda3/bin/python -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 5000 --ws none
```

当前已验证的虚拟机运行环境：

- Python `3.10.14`
- torch `2.4.1+cpu`
- torchvision `0.19.1+cpu`
- mediapipe `0.10.14`
- av `14.2.0`

## CNN 训练逻辑

`train_lfw.py` 使用 LFW 数据集训练 CNN：

- 真实样本：LFW 真人脸图片。
- 伪造样本：基于 LFW 生成的增强图，包括 JPEG 压缩、模糊锐化、颜色偏移、噪声、网格伪影、边界融合和简单换脸。
- 输出语义：真人接近 `1`，伪造/处理图接近 `0`。
- 权重保存：`models/cnn_weights.pth`。

训练命令：

```bash
python train_lfw.py
```

已验证训练结果：

- LFW 验证准确率约 `94.6%`
- 真人样例 `real_score` 约 `0.98`
- 真人样例综合风险为 `LOW`

## API

### 健康检查

```http
GET /health
```

### 图片分析

```http
POST /api/analyze
Content-Type: application/json
```

请求字段：

- `session_id`：可选，会话 ID。
- `label`：可选，会话标签。
- `image` 或 `image_data`：base64 图片数据。
- `audio_level`：可选，音频能量。
- `fps`：可选，采样帧率。

### 视频分析

```http
POST /api/analyze-video
Content-Type: multipart/form-data
```

请求字段：

- `file`：视频文件。
- `session_id`：可选，会话 ID。
- `label`：可选，会话标签。

## 注意事项

- MediaPipe 在无 GPU 的虚拟机中会使用 CPU/XNNPACK，日志中出现 `GPU support is not available` 属于正常情况。
- `av` 用于视频解码，缺失时视频分析不可用；当前 Python 3.10 环境已补齐。
- `models/cnn_weights.pth` 是当前图片真伪判断的核心权重。
- `lfw/` 和 `lfw.zip` 体积较大，仅在需要重新训练时使用。
