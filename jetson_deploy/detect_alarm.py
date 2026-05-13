import argparse
import base64
import json
import logging
import os
import sys
import threading
import time
import uuid
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Union

import cv2
import numpy as np
import torch
import yaml

# 让 yolov5/ 可被 import
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
YOLOV5_DIR = ROOT / "yolov5"
if str(YOLOV5_DIR) not in sys.path:
    sys.path.insert(0, str(YOLOV5_DIR))

# yolov5 自带工具
from models.common import DetectMultiBackend  # noqa: E402
from utils.augmentations import letterbox  # noqa: E402
from utils.general import non_max_suppression, scale_boxes  # noqa: E402
from utils.torch_utils import select_device  # noqa: E402

from mqtt_publisher import MqttConfig, MqttPublisher  # noqa: E402
from cmd_handler import CommandHandler, RuntimeState  # noqa: E402


log = logging.getLogger("detect_alarm")


# --------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------- #
def load_config(path: Union[str, os.PathLike]) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------------- #
# 视频源
# --------------------------------------------------------------------- #
def open_capture(src_cfg: dict) -> cv2.VideoCapture:
    uri = src_cfg["uri"]
    if isinstance(uri, str) and uri.lower() == "csi":
        # Jetson CSI 摄像头（IMX219），通过 GStreamer 取流
        pipeline = (
            f"nvarguscamerasrc ! video/x-raw(memory:NVMM), "
            f"width=(int){src_cfg['csi_width']}, height=(int){src_cfg['csi_height']}, "
            f"format=(string)NV12, framerate=(fraction){src_cfg['csi_fps']}/1 ! "
            f"nvvidconv flip-method={src_cfg['csi_flip_method']} ! "
            f"video/x-raw, format=(string)BGRx ! videoconvert ! "
            f"video/x-raw, format=(string)BGR ! appsink drop=true max-buffers=1"
        )
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    else:
        cap = cv2.VideoCapture(uri)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频源: {uri}")
    return cap


# --------------------------------------------------------------------- #
# 推理封装
# --------------------------------------------------------------------- #
class YoloRunner:
    def __init__(self, cfg: dict) -> None:
        inf = cfg["inference"]
        self.imgsz = int(inf["imgsz"])
        self.conf = float(inf["conf_thres"])
        self.iou = float(inf["iou_thres"])
        self.device = select_device(str(inf["device"]))
        self.half = bool(inf["half"]) and self.device.type != "cpu"

        weights = inf["weights"]
        if not Path(weights).is_absolute():
            weights = str((ROOT / weights).resolve())
        data = inf["data"]
        if not Path(data).is_absolute():
            data = str((ROOT / data).resolve())

        log.info("加载权重 %s on %s (half=%s)", weights, self.device, self.half)
        self.model = DetectMultiBackend(
            weights=weights, device=self.device, dnn=False, data=data, fp16=self.half
        )
        self.names: Dict[int, str] = self.model.names if isinstance(self.model.names, dict) \
            else {i: n for i, n in enumerate(self.model.names)}
        self.stride = int(self.model.stride)
        # warmup
        self.model.warmup(imgsz=(1, 3, self.imgsz, self.imgsz))

    @torch.no_grad()
    def infer(self, frame_bgr: np.ndarray) -> List[dict]:
        """返回检测结果列表 [{cls, conf, xyxy(原图坐标)}, ...]"""
        img = letterbox(frame_bgr, self.imgsz, stride=self.stride, auto=True)[0]
        img = img.transpose((2, 0, 1))[::-1]  # BGR -> RGB, HWC -> CHW
        img = np.ascontiguousarray(img)
        t = torch.from_numpy(img).to(self.device)
        t = t.half() if self.half else t.float()
        t /= 255.0
        if t.ndim == 3:
            t = t.unsqueeze(0)

        pred = self.model(t, augment=False, visualize=False)
        pred = non_max_suppression(pred, self.conf, self.iou, classes=None, agnostic=False)

        results: List[dict] = []
        det = pred[0]
        if det is not None and len(det):
            det[:, :4] = scale_boxes(t.shape[2:], det[:, :4], frame_bgr.shape).round()
            for *xyxy, conf_t, cls_t in det.tolist():
                cls_id = int(cls_t)
                results.append({
                    "cls_id": cls_id,
                    "cls_name": self.names.get(cls_id, str(cls_id)).lower(),
                    "conf": float(conf_t),
                    "xyxy": [float(x) for x in xyxy],
                })
        return results


# --------------------------------------------------------------------- #
# 报警触发器（防误报 + 冷却）
# --------------------------------------------------------------------- #
class AlarmTrigger:
    """对每个目标类别独立维护：连续命中计数 + 上次上报时间戳。"""

    def __init__(
        self,
        trigger_classes: List[str],
        min_conf: float,
        consecutive: int,
        cooldown_s: float,
    ) -> None:
        self.classes = {c.lower() for c in trigger_classes}
        self.min_conf = min_conf
        self.consecutive = max(1, int(consecutive))
        self.cooldown = float(cooldown_s)
        self._streak: Dict[str, int] = defaultdict(int)
        self._last_emit: Dict[str, float] = defaultdict(lambda: 0.0)

    def update(self, detections: List[dict]) -> List[dict]:
        """根据这一帧的检测结果，返回需要立即上报的目标列表（每类至多一个）。"""
        # 找到这一帧每个关注类别的最高置信度结果
        best: Dict[str, dict] = {}
        for d in detections:
            name = d["cls_name"]
            if name not in self.classes or d["conf"] < self.min_conf:
                continue
            cur = best.get(name)
            if cur is None or d["conf"] > cur["conf"]:
                best[name] = d

        emit: List[dict] = []
        now = time.time()
        # 对每个关注的类别更新 streak
        for cls in self.classes:
            if cls in best:
                self._streak[cls] += 1
                if self._streak[cls] >= self.consecutive \
                        and now - self._last_emit[cls] >= self.cooldown:
                    emit.append(best[cls])
                    self._last_emit[cls] = now
            else:
                self._streak[cls] = 0
        return emit


# --------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------- #
def encode_jpeg_b64(frame: np.ndarray, max_w: int, quality: int) -> str:
    h, w = frame.shape[:2]
    if w > max_w:
        new_h = int(h * max_w / w)
        frame = cv2.resize(frame, (max_w, new_h), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")


def encode_jpeg_bytes(frame: np.ndarray, max_w: int, quality: int) -> bytes:
    """缩放并编码为 JPEG 字节（用于视频流推送，不做 base64）。"""
    h, w = frame.shape[:2]
    if w > max_w:
        new_h = int(h * max_w / w)
        frame = cv2.resize(frame, (max_w, new_h), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return b""
    return buf.tobytes()


def draw_overlay(frame: np.ndarray, dets: List[dict]) -> None:
    for d in dets:
        x1, y1, x2, y2 = (int(v) for v in d["xyxy"])
        color = (0, 0, 255) if d["cls_name"] == "fire" else (0, 165, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{d['cls_name']} {d['conf']:.2f}", (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


# --------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------- #
def run(cfg: dict) -> None:
    mqtt_cfg = cfg["mqtt"]
    alarm_cfg = cfg["alarm"]
    src_cfg = cfg["source"]
    disp_cfg = cfg.get("display", {})
    video_cfg = cfg.get("video_stream", {}) or {}

    device_id = mqtt_cfg["device_id"]
    topic = mqtt_cfg["topic_template"].format(device_id=device_id)
    video_topic = (
        mqtt_cfg.get("video_topic_template", "flame_smoke/video/{device_id}")
        .format(device_id=device_id)
    )
    meta_topic = (
        mqtt_cfg.get("meta_topic_template", "flame_smoke/meta/{device_id}")
        .format(device_id=device_id)
    )
    status_topic = (
        mqtt_cfg.get("status_topic_template", "flame_smoke/status/{device_id}")
        .format(device_id=device_id)
    )
    cmd_topic = (
        mqtt_cfg.get("cmd_topic_template", "flame_smoke/cmd/{device_id}")
        .format(device_id=device_id)
    )
    ack_topic = (
        mqtt_cfg.get("ack_topic_template", "flame_smoke/ack/{device_id}")
        .format(device_id=device_id)
    )

    # 视频流参数（部分通过 RuntimeState 在线可调）
    video_max_w = int(video_cfg.get("max_width", 480))
    video_quality = int(video_cfg.get("jpeg_quality", 60))
    video_draw = bool(video_cfg.get("draw_overlay", True))
    video_qos = int(video_cfg.get("qos", 0))
    last_video_pub = 0.0

    # 运行时可变状态（被 cmd_handler 修改）
    runtime = RuntimeState(
        video_enabled=bool(video_cfg.get("enabled", False)),
        video_fps=max(1, int(video_cfg.get("fps", 4))),
        min_confidence=float(alarm_cfg["min_confidence"]),
    )

    # 设备元数据 + 心跳
    device_cfg = cfg.get("device", {}) or {}
    device_name = device_cfg.get("name") or device_id
    loc_cfg = device_cfg.get("location", {}) or {}
    device_address = loc_cfg.get("address", "") or alarm_cfg.get("location", "")
    device_lat = float(loc_cfg.get("lat", 0) or 0)
    device_lng = float(loc_cfg.get("lng", 0) or 0)
    heartbeat_s = max(5, int(device_cfg.get("status_heartbeat_seconds", 30)))

    # LWT：失联时 broker 发布 offline retained 到 status_topic
    will_payload_dict = {
        "device_id": device_id,
        "online": False,
        "reason": "lwt",
    }
    will_payload = json.dumps(will_payload_dict, ensure_ascii=False).encode("utf-8")

    publisher = MqttPublisher(MqttConfig(
        host=mqtt_cfg["broker_host"],
        port=int(mqtt_cfg["broker_port"]),
        use_tls=bool(mqtt_cfg.get("use_tls", False)),
        username=mqtt_cfg.get("username", "") or "",
        password=mqtt_cfg.get("password", "") or "",
        client_id=f"jetson-{device_id}-{uuid.uuid4().hex[:6]}",
        qos=int(mqtt_cfg.get("qos", 1)),
        will_topic=status_topic,
        will_payload=will_payload,
        will_qos=1,
        will_retain=True,
    ))

    def read_cpu_temp_c():
        paths = (
            "/sys/devices/virtual/thermal/thermal_zone0/temp",
            "/sys/class/thermal/thermal_zone0/temp",
        )
        for p in paths:
            try:
                with open(p, "r") as f:
                    raw = int(f.read().strip())
                return round(raw / 1000.0, 1) if raw > 1000 else float(raw)
            except Exception:
                continue
        return None

    # 在线 status 快照（cmd ping / 心跳都用）
    def make_status() -> Dict[str, Any]:
        s = runtime.snapshot_dict()
        s.update({
            "device_id": device_id,
            "online": True,
            "ts": int(time.time() * 1000),
            "cpu_temp_c": read_cpu_temp_c(),
        })
        return s

    # 连接成功后立刻发 retained meta + online status
    def on_connected() -> None:
        meta = {
            "device_id": device_id,
            "name": device_name,
            "address": device_address,
            "lat": device_lat,
            "lng": device_lng,
            "fw_version": "1.0.0",
            "ts": int(time.time() * 1000),
        }
        publisher.publish_json(meta_topic, meta, retain=True)
        publisher.publish_json(status_topic, make_status(), retain=True)
        log.info("published retained meta and online status for %s", device_id)

    publisher.add_on_connected(on_connected)

    # 远程指令
    cmd_handler = CommandHandler(publisher, ack_topic, runtime, make_status,
                                 status_topic=status_topic)
    publisher.subscribe(cmd_topic, cmd_handler.handle, qos=1)

    publisher.start()

    # 心跳线程
    stop_event = threading.Event()

    def heartbeat_loop() -> None:
        while not stop_event.is_set():
            if publisher.is_connected():
                publisher.publish_json(status_topic, make_status(), retain=True)
            stop_event.wait(heartbeat_s)

    hb_thread = threading.Thread(target=heartbeat_loop, daemon=True)
    hb_thread.start()

    runner = YoloRunner(cfg)
    trigger = AlarmTrigger(
        trigger_classes=alarm_cfg["trigger_classes"],
        min_conf=float(alarm_cfg["min_confidence"]),
        consecutive=int(alarm_cfg["consecutive_frames"]),
        cooldown_s=float(alarm_cfg["cooldown_seconds"]),
    )
    # 让 trigger 的 min_conf 跟随 RuntimeState
    def sync_trigger_threshold() -> None:
        trigger.min_conf = runtime.snapshot_dict()["min_confidence"]

    cap = open_capture(src_cfg)
    log.info("视频源就绪，alarm topic=%s, video topic=%s (enabled=%s) cmd topic=%s",
             topic, video_topic, runtime.snapshot_dict()["video_enabled"], cmd_topic)

    frame_count = 0
    fps_window: deque = deque(maxlen=30)
    last_log = time.time()
    show_window = bool(disp_cfg.get("show_window", False))

    try:
        while True:
            t0 = time.time()
            ok, frame = cap.read()
            if not ok or frame is None:
                log.warning("视频读取失败，1 秒后重试")
                time.sleep(1.0)
                cap.release()
                cap = open_capture(src_cfg)
                continue

            detections = runner.infer(frame)
            sync_trigger_threshold()
            to_emit = trigger.update(detections)

            # 主动抓拍指令：把当前帧作为 snapshot 上报
            if runtime.take_snapshot_request():
                annotated = frame.copy()
                draw_overlay(annotated, detections)
                snapshot_payload = {
                    "device_id": device_id,
                    "timestamp": int(time.time() * 1000),
                    "event_type": "snapshot",
                    "confidence": 1.0,
                    "bbox": [],
                    "location": device_address,
                    "image_base64": encode_jpeg_b64(
                        annotated,
                        int(alarm_cfg.get("image_max_width", 640)),
                        int(alarm_cfg.get("image_jpeg_quality", 75)),
                    ),
                }
                publisher.publish_json(topic, snapshot_payload, retain=False)
                log.info("[SNAPSHOT] published to %s", topic)

            for d in to_emit:
                payload: Dict[str, Any] = {
                    "device_id": device_id,
                    "timestamp": int(time.time() * 1000),
                    "event_type": d["cls_name"],
                    "confidence": round(d["conf"], 4),
                    "bbox": [round(v, 1) for v in d["xyxy"]],
                    "location": alarm_cfg.get("location", ""),
                }
                if alarm_cfg.get("send_image", True):
                    annotated = frame.copy()
                    draw_overlay(annotated, [d])
                    payload["image_base64"] = encode_jpeg_b64(
                        annotated,
                        int(alarm_cfg.get("image_max_width", 640)),
                        int(alarm_cfg.get("image_jpeg_quality", 75)),
                    )
                ok_pub = publisher.publish_json(topic, payload, retain=False)
                log.warning("[ALARM] %s conf=%.2f published=%s topic=%s",
                            d["cls_name"], d["conf"], ok_pub, topic)

            # 视频流推送（节流到 RuntimeState.video_fps；可热切）
            rt = runtime.snapshot_dict()
            if rt["video_enabled"]:
                interval = 1.0 / max(1, rt["video_fps"])
                if (t0 - last_video_pub) >= interval:
                    stream_frame = frame.copy() if video_draw else frame
                    if video_draw:
                        draw_overlay(stream_frame, detections)
                    jpg = encode_jpeg_bytes(stream_frame, video_max_w, video_quality)
                    if jpg:
                        publisher.publish_bytes(video_topic, jpg, qos=video_qos, retain=False)
                    last_video_pub = t0

            if show_window:
                draw_overlay(frame, detections)
                cv2.imshow("flame_smoke", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_count += 1
            fps_window.append(time.time() - t0)
            if frame_count % int(disp_cfg.get("print_fps_every", 30)) == 0:
                avg = sum(fps_window) / len(fps_window)
                fps = 1.0 / avg if avg > 0 else 0
                conn = "ON" if publisher.is_connected() else "OFF"
                log.info("frame=%d fps=%.1f mqtt=%s det=%d", frame_count, fps, conn, len(detections))
                last_log = time.time()
    except KeyboardInterrupt:
        log.info("Ctrl-C, exiting")
    finally:
        stop_event.set()
        # 主动发一条 offline retained （仅限干净退出场景）
        if publisher.is_connected():
            publisher.publish_json(status_topic, {
                "device_id": device_id,
                "online": False,
                "reason": "shutdown",
                "ts": int(time.time() * 1000),
            }, retain=True)
        cap.release()
        if show_window:
            cv2.destroyAllWindows()
        publisher.stop()


# --------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(HERE / "config.yaml"),
                   help="YAML 配置文件路径")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    cfg = load_config(args.config)
    run(cfg)


if __name__ == "__main__":
    main()
