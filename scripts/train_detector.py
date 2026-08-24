"""
训练 YOLOv8 背鳍检测器。

数据：datasets/dorsal_fin/（由 build_yolo_det_dataset.py 构建）
模型：YOLOv8n 预训练初始化（迁移学习，小数据集足够）
输出：models/detectors/yolov8n_dorsalfin.pt + 训练指标

用法：
    python scripts/train_yolo_detector.py --epochs 100
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    base = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="训练 YOLOv8 背鳍检测器")
    parser.add_argument("--data", type=Path, default=base / "datasets" / "dorsal_fin" / "data.yaml")
    parser.add_argument("--weights", type=Path,
                        default=base / "models" / "detectors" / "yolov8n_pretrained.pt",
                        help="预训练权重（models/detectors/ 下）")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, default=base / "models" / "detectors")
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    args.out.mkdir(parents=True, exist_ok=True)
    results = model.train(
        data=str(args.data.resolve()),  # 绝对路径：规避 ultralytics 相对 path 基于 cwd 的解析
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(args.out),
        name="dorsalfin",
        exist_ok=True,
        patience=20,
        seed=42,
        # 小数据集保守设置：无预训练增强破坏背鳍几何；标签噪声容忍
        hsv_h=0.02, hsv_s=0.4, hsv_v=0.4,
        degrees=5.0, translate=0.05, scale=0.3, shear=2.0,
        fliplr=0.0,  # 默认不水平翻转：左右侧特征不同（全局约束 4）
    )
    # 训练完成后把最佳权重软链到约定路径
    best = args.out / "dorsalfin" / "weights" / "best.pt"
    target = args.out / "yolov8n_dorsalfin.pt"
    import shutil
    if best.exists() and target != best:
        shutil.copy2(best, target)
        print(f"[train] 最佳权重 → {target}")
    else:
        print(f"[train] 权重位于 {best}")


if __name__ == "__main__":
    main()
