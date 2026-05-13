# Jetson Nano 部署 + 手机 MQTT 报警

把 YOLOv5 火灾烟雾检测部署到 Jetson Nano，做实时推理；检测到火焰/烟雾后通过 MQTT 把
报警 JSON 发送给配套的 Android App（仓库 `Flame_Smoke_Android/`）实现手机推送。

## 目录结构

```
jetson_deploy/
├── README.md            ← 本文件，部署+联调步骤
├── config.yaml          ← 所有运行参数（broker、推理、防误报…）
├── detect_alarm.py      ← 主程序：取流 → YOLOv5 推理 → MQTT 上报
├── mqtt_publisher.py    ← MQTT 客户端封装（自动重连）
├── export_trt.py        ← 把 .pt 转换为 TensorRT .engine（推理加速）
├── test_publish.py      ← 不依赖摄像头，手动发一条测试消息验证 App
└── requirements_jetson.txt
```

## 总体数据流

```
[CSI / USB / RTSP 摄像头]
        │
        ▼
[Jetson Nano · YOLOv5(TensorRT)]
        │ 检测到 fire/smoke 连续 N 帧
        ▼
[MQTT broker · broker.hivemq.com:1883]
        │ topic: flame_smoke/alarm/<device_id>
        ▼
[Android App] → 高优先级通知 + 全屏报警 + 现场截图
```

消息格式（JSON）：
```json
{
  "device_id": "demo-AbC123Xy",
  "timestamp": 1730000000000,
  "event_type": "fire",
  "confidence": 0.93,
  "bbox": [100, 80, 400, 360],
  "location": "客厅摄像头",
  "image_base64": "/9j/4AAQSk..."
}
```

---

## 一、Jetson Nano 环境准备

> 假定你已经刷好 JetPack 4.6+（自带 CUDA / cuDNN / TensorRT / OpenCV）。

```bash
# 1) 安装 PyTorch（必须用 NVIDIA 提供的 wheel，不要 pip install torch）
#    参考 https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048
#    例如 JetPack 4.6 + Python3.6：torch-1.10.0-cp36-cp36m-linux_aarch64.whl
sudo apt install -y python3-pip libopenblas-base libopenmpi-dev
pip3 install <下载的 torch wheel>
pip3 install <下载的 torchvision wheel>

# 2) 安装本仓库需要的 Python 包
cd ~/yolov5_fire_smoke_detection
pip3 install -r yolov5/requirements.txt
pip3 install -r jetson_deploy/requirements_jetson.txt

# 3) 验证
python3 -c "import torch, cv2; print(torch.__version__, torch.cuda.is_available(), cv2.__version__)"
```

## 二、（可选但推荐）TensorRT 加速

`.pt` 模型在 Jetson Nano CPU/GPU 直推大概 5–10 FPS；转 TensorRT FP16 后可达 20–30 FPS。
**必须在 Jetson Nano 本机执行，不能跨机复制 .engine 文件。**

```bash
python3 jetson_deploy/export_trt.py --weights model/best.pt --imgsz 640 --half
# 产物: model/best.engine
```

然后改 `jetson_deploy/config.yaml`：

```yaml
inference:
  weights: model/best.engine
  half: true
```

## 三、配置 MQTT + 设备 ID

1. 打开 `jetson_deploy/config.yaml`。
2. 把 `mqtt.device_id` 改成一个**难以猜测的字符串**，例如 `demo-7K8f3aZx`。
   - 因为我们使用的是公共 broker `broker.hivemq.com`，topic 不加密；用随机 device_id 可避免别人订阅到你的告警。
3. 在手机 App 的“设置”里：
   - `MQTT Broker` 填 `broker.hivemq.com`，端口 `1883`，TLS 关。
   - `订阅 Topic` 填 `flame_smoke/alarm/demo-7K8f3aZx`（与 Jetson 完全一致）。
   - `设备 ID` 填 `demo-7K8f3aZx`。
   - 保存后会自动重连，状态条变绿色 = 已连接。

## 四、最小化联调（先不接摄像头）

1. 手机 App 安装完成、显示 “已连接”。
2. 在 Jetson 或任何台 PC 上运行：
   ```bash
   python3 jetson_deploy/test_publish.py --device-id demo-7K8f3aZx
   ```
3. 手机应立即收到 “⚠️ 火警提醒” 通知 + 进入历史列表。
4. 若收不到：检查两边 device_id / 端口是否一致；APP 关闭电池优化避免被杀后台。

## 五、跑通完整摄像头管线

```bash
# CSI 摄像头（Jetson Nano IMX219）
# config.yaml 把 source.uri 改为 "csi"
python3 jetson_deploy/detect_alarm.py --config jetson_deploy/config.yaml

# USB 摄像头：把 source.uri 改为 0
# 视频文件回放：把 source.uri 改为 "/path/to/video.mp4"
# RTSP：把 source.uri 改为 "rtsp://user:pass@ip:554/stream"
```

## 六、防误报参数调优

`config.yaml` → `alarm`：

| 参数 | 含义 | 调高的效果 |
| --- | --- | --- |
| `min_confidence` | 单帧最低置信度 | 误报↓ 漏报↑ |
| `consecutive_frames` | 连续命中帧数才上报 | 误报↓ 延迟↑ |
| `cooldown_seconds` | 同类报警冷却 | 减少消息风暴 |
| `image_max_width` / `_jpeg_quality` | 截图大小 | 网络流量↓ |

## 七、开机自启（systemd）

```bash
sudo tee /etc/systemd/system/flame-smoke.service > /dev/null <<'EOF'
[Unit]
Description=Flame Smoke Detection
After=network-online.target

[Service]
User=jetson
WorkingDirectory=/home/jetson/yolov5_fire_smoke_detection
ExecStart=/usr/bin/python3 jetson_deploy/detect_alarm.py --config jetson_deploy/config.yaml
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable --now flame-smoke
journalctl -u flame-smoke -f
```

## 八、常见问题

- **`Illegal instruction (core dumped)`**：numpy 版本与 JetPack 不兼容，`pip3 install "numpy<1.20"` 试试。
- **CSI pipeline 报错**：`gst-inspect-1.0 nvarguscamerasrc` 验证插件存在；摄像头排线可能装反。
- **`.engine` 加载报错 `Cuda failure`**：engine 必须在本机生成，重新跑 `export_trt.py`。
- **手机收不到消息**：(1) 两端 device_id/topic 完全一致；(2) Jetson 控制台是否打印 `[ALARM]`；(3) 用 `mosquitto_sub -h broker.hivemq.com -t 'flame_smoke/alarm/#' -v` 旁路验证；(4) Android 关闭后台限制。

### 8.1 CSI 摄像头 `Failed to create CaptureSession` / `Argus Error AlreadyAllocated`

**典型现象**：

```
Error generated. .../gstnvarguscamerasrc.cpp, execute:751 Failed to create CaptureSession
WARNING detect_alarm | 视频读取失败，1 秒后重试
```

或者 `journalctl -u nvargus-daemon` 在每秒刷：

```
(Argus) Error AlreadyAllocated: Device 0 (of 1) is in use
```

**根因**：上一次 `detect_alarm.py` 没有正常退出（Ctrl+C 中断时 GStreamer 的 nvargus pipeline 经常释放不彻底），残留进程握着 `/dev/video0`，导致 nvargus daemon 一直返回「设备已被占用」。

**一键清场**（推荐做成 alias）：

```bash
sudo pkill -9 -f detect_alarm.py
sudo pkill -9 -f gst-launch
sudo pkill -9 -f nvgstcapture
sudo pkill -9 -f argus_camera
sleep 2
sudo systemctl restart nvargus-daemon
sleep 3
```

加到 `~/.bashrc` 里方便复用：

```bash
cat >> ~/.bashrc <<'EOF'

# Flame Smoke: 一键释放 CSI 摄像头
alias cam-reset='sudo pkill -9 -f detect_alarm.py 2>/dev/null; sudo pkill -9 -f gst-launch 2>/dev/null; sudo systemctl restart nvargus-daemon; sleep 2; echo "[cam-reset] done"'
EOF
source ~/.bashrc
```

以后只要遇到该报错：

```bash
cam-reset
python3 jetson_deploy/detect_alarm.py --config jetson_deploy/config.yaml
```

**验证摄像头本身是否健康**（清场之后跑，不应再报错）：

```bash
gst-launch-1.0 nvarguscamerasrc num-buffers=30 ! \
  'video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1' ! \
  fakesink
```

输出 `Got EOS from element "pipeline0"` + `Done Success` = 硬件 + daemon 都正常，可放心重跑主程序。

**仍然失败的排查顺序**：

1. `sudo journalctl -u nvargus-daemon -n 30 --no-pager --output=cat` 看具体错误码
2. `ls /dev/video*`：没有 `video0` → 排线松了或插反，断电重新插 IMX219 排线（金属触点朝向 SoC 模块，黑色压条要压紧）
3. `sudo i2cdetect -y -r 6`（或 7/8）：地址 `0x10` 没出现 → 硬件层就没认到摄像头
4. `dmesg | grep -i imx219`：内核是否识别到 sensor
5. 都不行 → `sudo reboot`，重启后**先单独跑上面的 gst-launch**，确认通了再启 detect_alarm

