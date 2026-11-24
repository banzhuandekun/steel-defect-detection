# 模型说明

## 用途

基于 YOLOv8n 的钢材表面缺陷目标检测，支持 NEU-DET 的六类缺陷框识别：
crazing、inclusion、patches、pitted_surface、rolled-in_scale、scratches。

## 数据与训练

| 项目 | 配置 |
|---|---|
| 数据集 | NEU-DET，1800 张，训练 / 验证 = 1620 / 180 |
| 输入尺寸 | 640×640 letterbox |
| 初始化 | COCO 预训练 YOLOv8n |
| 训练轮数 | 100 |
| batch size | 16 |

## 指标

| 指标 | 训练记录 | 当前权重在 180 张验证图上复核 |
|---|---:|---:|
| mAP50 | 0.739 | 0.756 |
| mAP50-95 | 0.390 | 0.405 |

各类别 mAP50：patches 0.934、scratches 0.869、inclusion 0.842、
pitted_surface 0.782、rolled-in_scale 0.613、crazing 0.497。

## 性能口径

- 原训练环境 RTX 4060 上，模型单步推理约 5–7 ms；
- 当前环境完整验证流程平均约 23.5 ms 推理、26.2 ms 前后处理；
- CPU 单图推理约 50 ms。

5–7 ms 不包含预处理、NMS 后处理和结果落盘，不能当作端到端系统延迟。

## 适用边界

- 适合“发现并定位”缺陷，不应直接用于尺寸或面积测量；
- mAP50-95 明显低于 mAP50，说明严格定位精度不足；
- crazing 类别表现明显较弱，实际部署应单独设置验收阈值并补充难例；
- 更换产线、钢种、光照或相机后必须重新采集验证集。

## 复现

```powershell
.\.venv\Scripts\python.exe defect_detector.py
.\.venv\Scripts\python.exe defect_detector.py --image path\to\image.jpg --conf 0.3
.\.venv\Scripts\python.exe defect_detector.py --weights yolov8n.pt --no-save
```
