# 多模态深伪检测平台

这是一个可直接运行的 Web 平台原型，实现了文档中的核心链路：

- 浏览器采集摄像头和麦克风
- 后端执行多模块风险分析
- 融合决策输出实时风险等级
- 提供管理员页面查看会话、日志和调参

## 技术栈

- 后端：Flask
- 分析：OpenCV + NumPy
- 前端：原生 HTML/CSS/JavaScript
- 存储：SQLite

## 启动

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
gunicorn -k uvicorn.workers.UvicornWorker -b 0.0.0.0:5000 app:app
```

打开：

- 用户页面：`http://<server-ip>:5000/`
- 管理员页面：`http://<server-ip>:5000/admin`

## 模块说明

- 生理信号层：基于 forehead ROI 的绿色通道序列与 FFT 稳定性评估
- 几何特征层：基于人脸框抖动、边缘密度与清晰度的异常分数
- 音视频一致性：使用嘴部区域运动与音频能量相关性作为同步代理
- 纹理伪影：基于 Laplacian 与边缘密度的伪影估计
- 时序漂移：帧间运动与音频变化的差异分析
- 决策引擎：可调权重的多模块融合
