"""手动发一条假报警消息，用于在没有 Jetson 摄像头的情况下验证手机 App。

用法：
    python jetson_deploy/test_publish.py --device-id demo-XXXXXXXX
    python jetson_deploy/test_publish.py --device-id demo-XXXXXXXX --image some.jpg

App 在“设置”里把 device_id / topic 改为同一个后即可收到此消息。
"""

import argparse
import base64
import json
import sys
import time
from pathlib import Path

import paho.mqtt.client as mqtt


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="broker.hivemq.com")
    p.add_argument("--port", type=int, default=1883)
    p.add_argument("--device-id", required=True, help="必须与 App 里的 device_id 一致")
    p.add_argument("--topic-template", default="flame_smoke/alarm/{device_id}")
    p.add_argument("--event-type", default="fire", choices=["fire", "smoke"])
    p.add_argument("--confidence", type=float, default=0.92)
    p.add_argument("--location", default="测试位置")
    p.add_argument("--image", help="可选：本地图片路径，会以 base64 一起上报")
    args = p.parse_args()

    payload: dict = {
        "device_id": args.device_id,
        "timestamp": int(time.time() * 1000),
        "event_type": args.event_type,
        "confidence": args.confidence,
        "bbox": [100.0, 80.0, 400.0, 360.0],
        "location": args.location,
    }
    if args.image:
        data = Path(args.image).read_bytes()
        payload["image_base64"] = base64.b64encode(data).decode("ascii")

    topic = args.topic_template.format(device_id=args.device_id)
    client = mqtt.Client(client_id=f"test-pub-{int(time.time())}")
    client.connect(args.host, args.port, keepalive=10)
    client.loop_start()
    info = client.publish(topic, json.dumps(payload, ensure_ascii=False), qos=1)
    info.wait_for_publish(timeout=5)
    print(f"published rc={info.rc} topic={topic} bytes={len(json.dumps(payload))}")
    client.loop_stop()
    client.disconnect()


if __name__ == "__main__":
    sys.exit(main())
