import json
import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import paho.mqtt.client as mqtt


log = logging.getLogger("flame_mqtt")


@dataclass
class MqttConfig:
    host: str
    port: int = 1883
    use_tls: bool = False
    username: str = ""
    password: str = ""
    client_id: str = ""
    qos: int = 1
    # LWT（Last Will and Testament）：失联时 broker 自动发布该消息到 will_topic
    will_topic: str = ""
    will_payload: bytes = b""
    will_qos: int = 1
    will_retain: bool = True


class MqttPublisher:
    def __init__(self, cfg: MqttConfig) -> None:
        self._cfg = cfg
        self._client = mqtt.Client(client_id=cfg.client_id or f"jetson-{int(time.time())}",
                                   clean_session=True)
        if cfg.username:
            self._client.username_pw_set(cfg.username, cfg.password)
        if cfg.use_tls:
            self._client.tls_set()
        if cfg.will_topic:
            self._client.will_set(cfg.will_topic, cfg.will_payload,
                                  qos=cfg.will_qos, retain=cfg.will_retain)
        self._client.reconnect_delay_set(min_delay=1, max_delay=60)
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message = self._on_message
        self._connected = threading.Event()
        # topic -> callback(topic, payload_bytes)
        self._subscriptions: dict = {}
        # 连接成功后的回调（用于发 retained meta/status）
        self._on_connected_cbs: list = []

    # ---- lifecycle ---------------------------------------------------
    def start(self) -> None:
        log.info("MQTT connecting to %s:%s tls=%s", self._cfg.host, self._cfg.port, self._cfg.use_tls)
        self._client.connect_async(self._cfg.host, self._cfg.port, keepalive=20)
        self._client.loop_start()

    def stop(self) -> None:
        try:
            self._client.disconnect()
        finally:
            self._client.loop_stop()

    # ---- callbacks ---------------------------------------------------
    def _on_connect(self, _client, _userdata, _flags, rc, _props=None) -> None:
        if rc == 0:
            log.info("MQTT connected")
            self._connected.set()
            # 重连后重新订阅所有 topic
            for topic, (qos, _cb) in list(self._subscriptions.items()):
                try:
                    self._client.subscribe(topic, qos=qos)
                    log.info("MQTT re-subscribed %s qos=%s", topic, qos)
                except Exception as e:
                    log.warning("re-subscribe %s failed: %s", topic, e)
            for cb in list(self._on_connected_cbs):
                try:
                    cb()
                except Exception:
                    log.exception("on_connected callback raised")
        else:
            log.warning("MQTT connect failed rc=%s", rc)
            self._connected.clear()

    def _on_disconnect(self, _client, _userdata, rc, _props=None) -> None:
        log.warning("MQTT disconnected rc=%s, will auto-reconnect", rc)
        self._connected.clear()

    def _on_message(self, _client, _userdata, msg) -> None:
        entry = self._subscriptions.get(msg.topic)
        if entry is None:
            # 也尝试通配符（paho 已按订阅 filter 派发，这里精确命中即可）
            return
        _qos, cb = entry
        try:
            cb(msg.topic, msg.payload)
        except Exception:
            log.exception("subscription callback raised for %s", msg.topic)

    # ---- subscribe ---------------------------------------------------
    def subscribe(self, topic: str, callback: Callable[[str, bytes], None],
                  qos: int = 1) -> None:
        """注册一个订阅。重连后会自动重新订阅。"""
        self._subscriptions[topic] = (qos, callback)
        if self._connected.is_set():
            self._client.subscribe(topic, qos=qos)
            log.info("MQTT subscribed %s qos=%s", topic, qos)

    def add_on_connected(self, callback: Callable[[], None]) -> None:
        """注册连接成功后的回调（每次重连都触发）。"""
        self._on_connected_cbs.append(callback)
        if self._connected.is_set():
            try:
                callback()
            except Exception:
                log.exception("on_connected immediate callback raised")

    # ---- publish -----------------------------------------------------
    def is_connected(self) -> bool:
        return self._connected.is_set()

    def publish_json(self, topic: str, payload: dict, retain: bool = False) -> bool:
        try:
            data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError) as exc:
            log.error("payload not JSON-serializable: %s", exc)
            return False
        info = self._client.publish(topic, data, qos=self._cfg.qos, retain=retain)
        # paho 会在断线时把消息写入 outgoing queue（QoS>=1）。
        ok = info.rc == mqtt.MQTT_ERR_SUCCESS
        if not ok:
            log.warning("publish rc=%s topic=%s", info.rc, topic)
        return ok

    def publish_bytes(self, topic: str, payload: bytes, qos: int = 0,
                      retain: bool = False) -> bool:

        info = self._client.publish(topic, payload, qos=qos, retain=retain)
        ok = info.rc == mqtt.MQTT_ERR_SUCCESS
        if not ok:
            # debug 级别，避免视频帧丢一帧就刷屏
            log.debug("publish_bytes rc=%s topic=%s size=%d", info.rc, topic, len(payload))
        return ok
