"""金属缺陷检测：YOLO 权重加载、单图推理与命令行入口。"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_TRAINED_WEIGHTS = BASE_DIR / "best.pt"
DEFAULT_FALLBACK_WEIGHTS = BASE_DIR / "yolov8n.pt"
DEFAULT_IMAGE = BASE_DIR / "demo_input.jpg"


def resolve_weights(
    weights_path: str | Path | None = None,
    *,
    base_dir: Path = BASE_DIR,
) -> Path:
    """解析推理权重。

    未显式指定时优先使用训练权重 ``best.pt``，缺失后回退到
    ``yolov8n.pt``。显式路径不存在时直接报错，避免静默切换模型。
    """
    base_dir = Path(base_dir)
    if weights_path is None:
        trained = base_dir / "best.pt"
        weights = trained if trained.exists() else base_dir / "yolov8n.pt"
    else:
        candidate = Path(weights_path)
        weights = candidate if candidate.is_absolute() else base_dir / candidate

    if not weights.exists():
        raise FileNotFoundError(f"找不到模型权重: {weights}")
    return weights.resolve()


def parse_args() -> argparse.Namespace:
    """解析图片、权重、置信度和输出目录参数。"""
    parser = argparse.ArgumentParser(description="YOLOv8 钢材表面缺陷检测")
    parser.add_argument("--weights", help="自定义权重路径；省略时自动选择 best.pt/yolov8n.pt")
    parser.add_argument("--image", default=str(DEFAULT_IMAGE), help="待检测图片路径")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--project", default="runs/detect", help="结果保存根目录")
    parser.add_argument("--name", default="predict", help="本次运行子目录名")
    parser.add_argument("--no-save", action="store_true", help="只推理，不保存可视化结果")
    return parser.parse_args()

class DefectDetector:
    """封装 Ultralytics YOLO 的加载与单图推理。"""

    def __init__(self, weights_path):
        """加载指定权重并保存底层 YOLO 模型。"""
        self.weights_path = Path(weights_path)
        self.model = YOLO(str(self.weights_path))

    def detect(
        self,
        image_path,
        conf=0.25,
        save_dir="runs/detect",
        *,
        name="predict",
        save=True,
    ):
        """对单张图片执行推理，并返回第一张图的结果对象。"""
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"找不到输入图片: {image_path}")

        project = Path(save_dir)
        if not project.is_absolute():
            project = BASE_DIR / project

        results = self.model.predict(
            source=str(image_path),
            conf=conf,
            save=save,
            project=str(project),
            name=name,
        )
        if not results:
            raise RuntimeError("YOLO 未返回推理结果")
        return results[0]


def main() -> int:
    """解析命令行参数、执行检测并打印每个预测框。"""
    args = parse_args()
    weights = resolve_weights(args.weights)
    detector = DefectDetector(weights)
    result = detector.detect(
        args.image,
        conf=args.conf,
        save_dir=args.project,
        name=args.name,
        save=not args.no_save,
    )

    print(f"权重: {weights}")
    print(f"检测到 {len(result.boxes)} 个目标")
    for box in result.boxes:
        label = result.names[int(box.cls)]
        confidence = box.conf.item()
        coordinates = box.xyxy.tolist()
        print(f"{label}: {confidence:.4f} -> {coordinates}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
