"""权重解析和 YOLO 封装行为测试。"""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from defect_detector import DefectDetector, resolve_weights


class FakeYolo:
    """记录 predict 参数，避免加载真实模型。"""

    def __init__(self):
        """初始化空结果列表。"""
        self.calls = []

    def predict(self, **kwargs):
        """保存调用参数并返回固定结果。"""
        self.calls.append(kwargs)
        return [SimpleNamespace(boxes=[])]


class WeightResolutionTests(unittest.TestCase):
    """验证训练权重、回退权重和显式路径的优先级。"""

    def test_prefers_trained_weights(self):
        """best.pt 存在时应优先选择。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "best.pt").touch()
            (root / "yolov8n.pt").touch()

            weights = resolve_weights(base_dir=root)

        self.assertEqual(weights.name, "best.pt")

    def test_falls_back_to_pretrained_weights(self):
        """训练权重缺失时应回退到 yolov8n.pt。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "yolov8n.pt").touch()

            weights = resolve_weights(base_dir=root)

        self.assertEqual(weights.name, "yolov8n.pt")

    def test_raises_for_missing_explicit_path(self):
        """显式权重不存在时不能静默换模型。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(FileNotFoundError):
                resolve_weights("missing.pt", base_dir=Path(temp_dir))


class DetectorTests(unittest.TestCase):
    """验证推理路径和相对输出目录。"""

    def test_detect_passes_absolute_project_and_returns_first_result(self):
        """相对输出目录应基于仓库根目录，且返回首张结果。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            image = Path(temp_dir) / "demo.jpg"
            image.write_bytes(b"fake-image")
            detector = DefectDetector.__new__(DefectDetector)
            detector.model = FakeYolo()

            result = detector.detect(image, save_dir="runs/custom", name="case")

        self.assertTrue(Path(detector.model.calls[0]["project"]).is_absolute())
        self.assertEqual(detector.model.calls[0]["name"], "case")
        self.assertEqual(result.boxes, [])


if __name__ == "__main__":
    unittest.main()
