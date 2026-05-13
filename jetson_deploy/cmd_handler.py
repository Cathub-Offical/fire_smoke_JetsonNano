import json
import logging
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Optional


log = logging.getLogger("cmd_handler")


@dataclass
class RuntimeState:
    """检测主循环共享的可变运行参数。所有读写都要持锁。"""
    video_enabled: bool
    video_fps: int
    min_confidence: float
    snapshot_requested: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def snapshot_dict(self) -> dict:
        with self._lock:
            return {
                "video_enabled": self.video_enabled,
                "video_fps": self.video_fps,
                "min_confidence": round(self.min_confidence, 3),
            }

    def set_video(self, enabled: Optional[bool] = None,
                  fps: Optional[int] = None) -> None:
        with self._lock:
            if enabled is not None:
                self.video_enabled = bool(enabled)
            if fps is not None:
                self.video_fps = max(1, min(30, int(fps)))

    def set_threshold(self, value: float) -> None:
        with self._lock:
            self.min_confidence = max(0.0, min(1.0, float(value)))

    def request_snapshot(self) -> None:
        with self._lock:
            self.snapshot_requested = True

    def take_snapshot_request(self) -> bool:
        with self._lock:
            if self.snapshot_requested:
                self.snapshot_requested = False
                return True
            return False


class CommandHandler:
    def __init__(self, publisher, ack_topic: str, state: RuntimeState,
                 status_provider, status_topic=None) -> None:
        """
        publisher:       MqttPublisher
        ack_topic:       回执 topic
        state:           RuntimeState
        status_provider: 无参函数，返回当前 status dict（在线快照）
        status_topic:    可选；若提供，状态类指令执行后会立刻 retained 发一条 status
                         让后来/其他订阅端也立即看到最新状态
        """
        self._pub = publisher
        self._ack_topic = ack_topic
        self._state = state
        self._status_provider = status_provider
        self._status_topic = status_topic

    def _broadcast_status(self) -> None:
        if self._status_topic:
            try:
                self._pub.publish_json(
                    self._status_topic, self._status_provider(), retain=True
                )
            except Exception:
                log.exception("broadcast status failed")

    def handle(self, topic: str, payload: bytes) -> None:
        try:
            text = payload.decode("utf-8", errors="replace")
            data = json.loads(text)
        except Exception as e:
            log.warning("bad cmd payload on %s: %s", topic, e)
            self._ack(None, False, f"bad json: {e}", ts=None)
            return

        cmd = str(data.get("cmd", "")).strip().lower()
        ts = data.get("ts")
        log.info("CMD received: %s data=%s", cmd, data)

        try:
            if cmd == "ping":
                snap = self._status_provider()
                self._ack(cmd, True, "pong", ts=ts, extra=snap)

            elif cmd == "set_video":
                self._state.set_video(
                    enabled=data.get("enabled"),
                    fps=data.get("fps"),
                )
                self._ack(cmd, True, "ok", ts=ts,
                          extra=self._state.snapshot_dict())
                self._broadcast_status()

            elif cmd == "set_threshold":
                v = data.get("value")
                if v is None:
                    self._ack(cmd, False, "missing 'value'", ts=ts)
                    return
                self._state.set_threshold(float(v))
                self._ack(cmd, True, "ok", ts=ts,
                          extra=self._state.snapshot_dict())
                self._broadcast_status()

            elif cmd == "snapshot":
                self._state.request_snapshot()
                self._ack(cmd, True, "queued", ts=ts)

            elif cmd == "restart_service":
                self._ack(cmd, True, "restarting", ts=ts)
                # 让外层守护进程（systemd / 启动脚本）拉起
                threading.Thread(target=_delayed_exit, args=(1.0,),
                                 daemon=True).start()

            elif cmd == "reboot":
                self._ack(cmd, True, "rebooting", ts=ts)
                threading.Thread(target=_delayed_reboot, args=(1.0,),
                                 daemon=True).start()

            else:
                self._ack(cmd or None, False, f"unknown cmd: {cmd}", ts=ts)
        except Exception as e:
            log.exception("cmd %s failed", cmd)
            self._ack(cmd, False, f"exception: {e}", ts=ts)

    def _ack(self, cmd: Optional[str], ok: bool, msg: str,
             ts=None, extra: Optional[dict] = None) -> None:
        payload = {
            "cmd": cmd,
            "ok": ok,
            "msg": msg,
            "ts": ts,
            "reply_ts": int(time.time() * 1000),
        }
        if extra:
            payload.update(extra)
        self._pub.publish_json(self._ack_topic, payload, retain=False)


def _delayed_exit(delay: float) -> None:
    time.sleep(delay)
    log.warning("Exiting process for restart_service")
    os._exit(0)


def _delayed_reboot(delay: float) -> None:
    time.sleep(delay)
    log.warning("Calling system reboot")
    try:
        subprocess.run(["sudo", "-n", "reboot"], check=False, timeout=5)
    except Exception:
        log.exception("reboot command failed")
