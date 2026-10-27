"""Fine-tune a YOLO11 detector on a custom ball dataset.

Expected dataset YAML format is the Ultralytics YOLO detection format, for example:
path: C:/datasets/ball_dataset
train: images/train
val: images/val
names:
  0: ball

Run from your project environment:
python train_custom_ball_model.py --data C:/datasets/ball_dataset/data.yaml --model yolo11n.pt --epochs 50 --imgsz 640
The output weights are usually saved under runs/detect/ball_finetune/weights/best.pt.
"""
import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLO11 for a custom ball dataset")
    parser.add_argument("--data", required=True, help="Path to dataset YAML")
    parser.add_argument("--model", default="yolo11n.pt", help="Starting checkpoint: yolo11n.pt, yolo11s.pt, etc.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="", help="Optional device, e.g. 0 for CUDA GPU or cpu")
    args = parser.parse_args()

    model = YOLO(args.model)
    train_args = dict(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project="runs/detect",
        name="ball_finetune",
        exist_ok=True,
    )
    if args.device:
        train_args["device"] = args.device
    results = model.train(**train_args)
    print("Training finished. Inspect runs/detect/ball_finetune/weights/best.pt and validate it on held-out videos.")
    print("Do not report accuracy until validation/test metrics have been measured.")


if __name__ == "__main__":
    main()
