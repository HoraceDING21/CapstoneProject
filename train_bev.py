"""Training script for YOLOv11 Pose BEV position estimation.

Adapted from mmpose sskit task: predicts 2 keypoints (pelvis + pelvis_ground)
per player for Bird's Eye View localization on Spiideo SoccerNet dataset.

Key modifications from standard YOLO11 pose training:
  - kpt_shape: [2, 3] instead of [17, 3]
  - Custom OKS sigmas: [0.089, 0.089] (from sskit metainfo)
  - Backbone frozen (layers 0-10) to retain pretrained features
  - Lower learning rate (1e-4) matching mmpose BEV config
  - nc=1 (person class only)

Usage:
    # First convert COCO annotations to YOLO format (one-time):
    python tools/convert_coco_to_yolo_pose.py \
        --coco-dir /path/to/soccernet-dataset \
        --output-dir /path/to/soccernet-synloc-yolo

    # Then update 'path' in ultralytics/cfg/datasets/soccernet-synloc.yaml

    # Train (pick a model size: n/s/m/l/x):
    python train_bev.py --model-size n --epochs 300 --batch 64 --imgsz 640
"""

import argparse

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv11 Pose for BEV position estimation")
    parser.add_argument("--model-size", type=str, default="m", choices=["n", "s", "m", "l", "x"],
                        help="YOLO11 model size variant")
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Path to pretrained weights (e.g. yolo11m-pose.pt). "
                             "If not specified, uses COCO pretrained for the chosen size.")
    parser.add_argument("--data", type=str, default="soccernet-synloc.yaml",
                        help="Path to dataset YAML config")
    parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=64, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--lr0", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--freeze", type=int, default=11,
                        help="Freeze first N backbone layers (11 = full YOLO11 backbone)")
    parser.add_argument("--device", type=str, default=None, help="Device: 0, '0,1', 'cpu', etc.")
    parser.add_argument("--workers", type=int, default=8, help="Dataloader workers")
    parser.add_argument("--project", type=str, default="runs/pose-bev", help="Project directory")
    parser.add_argument("--name", type=str, default=None, help="Experiment name")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--no-freeze", action="store_true", help="Train without freezing backbone")
    return parser.parse_args()


def main():
    args = parse_args()

    model_yaml = f"yolo11{args.model_size}-pose-bev.yaml"

    if args.pretrained:
        model = YOLO(args.pretrained)
    else:
        model = YOLO(model_yaml)

    exp_name = args.name or f"yolo11{args.model_size}-pose-bev-{args.imgsz}"

    train_args = dict(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        lr0=args.lr0,
        lrf=0.05,
        workers=args.workers,
        project=args.project,
        name=exp_name,
        resume=args.resume,
        patience=100,
        close_mosaic=10,
        pose=12.0,
        kobj=1.0,
    )

    if not args.no_freeze:
        train_args["freeze"] = args.freeze

    if args.device is not None:
        train_args["device"] = args.device

    results = model.train(**train_args)
    return results


if __name__ == "__main__":
    main()
