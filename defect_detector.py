# defect_detector.py —— 金属缺陷检测：模型封装
from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent

class DefectDetector:
    def __init__(self, weights_path):
        # 填空1：用 YOLO() 加载权重，保存到 self.model
        self.model = YOLO(weights_path)

    def detect(self, image_path, conf=0.25, save_dir="runs/detect"):
        # 填空2：调用 predict()，参数参考 ultralytics 文档
        results = self.model.predict(source=image_path, conf=conf, save=True, project=str(BASE_DIR / save_dir))
        return results[0]   # 返回第一张图的结果

if __name__ == "__main__":
    weights_path = BASE_DIR / "best.pt"
    if not weights_path.exists():
        weights_path = BASE_DIR / "yolov8n.pt"  # 回退到预训练模型
    detector = DefectDetector(str(weights_path))
    result = detector.detect(str(BASE_DIR / "demo_input.jpg"))
    print("检测到", len(result.boxes), "个目标")
    for box in result.boxes:
        print(result.names[int(box.cls)], box.conf.item(), box.xyxy.tolist())
