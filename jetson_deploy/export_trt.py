"""把 YOLOv5 .pt 权重转换为 TensorRT .engine（FP16），用于 Jetson Nano 加速。

实质是对 yolov5/export.py 的一层薄封装，固定常用参数。建议在 Jetson Nano 本机
上执行，因为 .engine 与 GPU 架构强相关，不可跨机使用。

用法：
    python jetson_deploy/export_trt.py \
        --weights model/best.pt --imgsz 640 --half

成功后会在 weights 同目录得到 best.engine，把 config.yaml 的
inference.weights 改成它即可。
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
YOLOV5_DIR = ROOT / "yolov5"
if str(YOLOV5_DIR) not in sys.path:
    sys.path.insert(0, str(YOLOV5_DIR))

from export import run as export_run  # noqa: E402  (yolov5/export.py)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True, help=".pt 权重路径")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--half", action="store_true", help="FP16，Jetson Nano 推荐")
    p.add_argument("--workspace", type=int, default=2, help="TensorRT 构建工作区(GiB)")
    args = p.parse_args()

    export_run(
        weights=args.weights,
        imgsz=(args.imgsz, args.imgsz),
        batch_size=args.batch_size,
        device="0",
        half=args.half,
        include=("engine",),
        workspace=args.workspace,
        verbose=False,
    )


if __name__ == "__main__":
    main()
