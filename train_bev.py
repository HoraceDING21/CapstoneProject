"""Two-stage training script for YOLOv11 Pose BEV position estimation.

Training strategy (recommended):
  Stage 1 — Frozen backbone:  Only neck + head are trained with a larger
      learning rate so the new head converges quickly on the BEV task.
  Stage 2 — Full fine-tune:   All layers are unfrozen and trained with a
      much smaller learning rate to adapt the backbone to the soccer BEV domain.

Input resolution notes:
  - YOLO training accepts a single integer for imgsz (always square).
  - Use --rect to enable rectangular batching: images are padded to their
    actual aspect ratio (e.g. 16:9 → ~1280×736) instead of square 1280×1280.
    This saves ~42% compute for typical broadcast soccer footage.
  - Larger imgsz (960/1280) is more effective than scaling up model size
    for this task, because players occupy a small fraction of the image.
  - Memory scales quadratically: 1280² ≈ 4× pixels of 640², so reduce
    batch size accordingly (e.g. batch 64 @ 640 → batch 16 @ 1280).

Usage examples:
    # Two-stage training (recommended), 640 resolution:
    python train_bev.py --model-size m --imgsz 640

    # Two-stage training, 1280 resolution with rectangular batching:
    python train_bev.py --model-size m --imgsz 1280 --rect \
        --stage1-batch 16 --stage2-batch 8

    # Run only stage 2 from a completed stage 1 checkpoint:
    python train_bev.py --stage 2 --stage1-weights runs/pose-bev/.../weights/best.pt \
        --model-size m --imgsz 1280 --rect

    # Legacy single-stage training (backward compatible):
    python train_bev.py --single-stage --model-size m --epochs 300 --batch 64

    # Resume an interrupted stage:
    python train_bev.py --stage 1 --resume --model-size m
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="Two-stage YOLOv11 Pose BEV training")

    # ── model / data ──────────────────────────────────────────────────
    p.add_argument("--model-size", type=str, default="m",
                   choices=["n", "s", "m", "l", "x"])
    p.add_argument("--pretrained", type=str, default=None,
                   help="Pretrained weights for stage 1 init (e.g. yolo11m-pose.pt)")
    p.add_argument("--data", type=str, default="soccernet-synloc.yaml")

    # ── resolution ────────────────────────────────────────────────────
    p.add_argument("--imgsz", type=int, default=640,
                   help="Input image size (square). Use with --rect for non-square.")
    p.add_argument("--rect", action="store_true",
                   help="Rectangular batching — pads to actual aspect ratio, not square")

    # ── stage control ─────────────────────────────────────────────────
    p.add_argument("--stage", type=int, default=0, choices=[0, 1, 2],
                   help="0 = run both stages sequentially, 1 = stage 1 only, 2 = stage 2 only")
    p.add_argument("--stage1-weights", type=str, default=None,
                   help="Path to stage 1 best.pt (required when --stage 2)")

    # ── stage 1 hyperparams (frozen backbone) ─────────────────────────
    p.add_argument("--stage1-epochs", type=int, default=100)
    p.add_argument("--stage1-lr", type=float, default=1e-3,
                   help="Larger LR for head-only training")
    p.add_argument("--stage1-batch", type=int, default=64)
    p.add_argument("--freeze", type=int, default=11,
                   help="Freeze first N backbone layers (11 = full YOLO11 backbone)")

    # ── stage 2 hyperparams (full fine-tune) ──────────────────────────
    p.add_argument("--stage2-epochs", type=int, default=200)
    p.add_argument("--stage2-lr", type=float, default=1e-5,
                   help="Smaller LR for full network fine-tuning")
    p.add_argument("--stage2-batch", type=int, default=None,
                   help="Batch size for stage 2 (defaults to stage1-batch // 2)")

    # ── common training args ──────────────────────────────────────────
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--project", type=str, default="runs/pose-bev")
    p.add_argument("--name", type=str, default=None)
    p.add_argument("--resume", action="store_true",
                   help="Resume training from last checkpoint of the current stage")
    p.add_argument("--patience", type=int, default=50,
                   help="Early stopping patience (epochs without improvement)")
    p.add_argument("--cos-lr", action="store_true",
                   help="Use cosine learning rate schedule")

    # ── legacy single-stage mode (backward compatible) ────────────────
    p.add_argument("--single-stage", action="store_true",
                   help="Fall back to single-stage training (original behavior)")
    p.add_argument("--epochs", type=int, default=300,
                   help="Total epochs for single-stage mode")
    p.add_argument("--batch", type=int, default=64,
                   help="Batch size for single-stage mode")
    p.add_argument("--lr0", type=float, default=1e-4,
                   help="Learning rate for single-stage mode")
    p.add_argument("--no-freeze", action="store_true",
                   help="Disable backbone freezing in single-stage mode")

    return p.parse_args()


def _shared_train_args(args):
    """Training arguments common to both stages."""
    cfg = dict(
        data=args.data,
        imgsz=args.imgsz,
        workers=args.workers,
        project=args.project,
        pose=12.0,
        kobj=1.0,
        close_mosaic=10,
        rect=args.rect,
        cos_lr=args.cos_lr,
    )
    if args.device is not None:
        cfg["device"] = args.device
    return cfg


def run_stage1(args):
    """Stage 1: frozen backbone — train neck + head with large LR."""
    model_yaml = f"yolo11{args.model_size}-pose-bev.yaml"
    if args.pretrained:
        model = YOLO(args.pretrained)
    else:
        model = YOLO(model_yaml)

    exp_name = args.name or f"yolo11{args.model_size}-bev-{args.imgsz}"
    stage1_name = f"{exp_name}-stage1"

    cfg = _shared_train_args(args)
    cfg.update(
        epochs=args.stage1_epochs,
        batch=args.stage1_batch,
        lr0=args.stage1_lr,
        lrf=0.1,
        freeze=args.freeze,
        patience=args.patience,
        name=stage1_name,
        resume=args.resume,
    )

    print(f"\n{'='*60}")
    print(f"  STAGE 1: Frozen backbone (layers 0-{args.freeze - 1})")
    print(f"  LR={args.stage1_lr}, epochs={args.stage1_epochs}, "
          f"batch={args.stage1_batch}, imgsz={args.imgsz}")
    print(f"{'='*60}\n")

    results = model.train(**cfg)

    best_pt = Path(args.project) / stage1_name / "weights" / "best.pt"
    print(f"\nStage 1 complete. Best weights: {best_pt}")
    return str(best_pt)


def run_stage2(args, stage1_weights):
    """Stage 2: full fine-tune — unfreeze all layers, small LR."""
    model = YOLO(stage1_weights)

    exp_name = args.name or f"yolo11{args.model_size}-bev-{args.imgsz}"
    stage2_name = f"{exp_name}-stage2"
    stage2_batch = args.stage2_batch or max(args.stage1_batch // 2, 1)

    cfg = _shared_train_args(args)
    cfg.update(
        epochs=args.stage2_epochs,
        batch=stage2_batch,
        lr0=args.stage2_lr,
        lrf=0.01,
        patience=args.patience,
        name=stage2_name,
        resume=args.resume,
    )
    # No freeze → all parameters trainable

    print(f"\n{'='*60}")
    print(f"  STAGE 2: Full fine-tuning (all layers unfrozen)")
    print(f"  LR={args.stage2_lr}, epochs={args.stage2_epochs}, "
          f"batch={stage2_batch}, imgsz={args.imgsz}")
    print(f"  Loading weights from: {stage1_weights}")
    print(f"{'='*60}\n")

    results = model.train(**cfg)

    best_pt = Path(args.project) / stage2_name / "weights" / "best.pt"
    print(f"\nStage 2 complete. Best weights: {best_pt}")
    return results


def run_single_stage(args):
    """Legacy single-stage training (original train_bev.py behavior)."""
    model_yaml = f"yolo11{args.model_size}-pose-bev.yaml"
    if args.pretrained:
        model = YOLO(args.pretrained)
    else:
        model = YOLO(model_yaml)

    exp_name = args.name or f"yolo11{args.model_size}-pose-bev-{args.imgsz}"

    cfg = _shared_train_args(args)
    cfg.update(
        epochs=args.epochs,
        batch=args.batch,
        lr0=args.lr0,
        lrf=0.05,
        patience=100,
        name=exp_name,
        resume=args.resume,
    )
    if not args.no_freeze:
        cfg["freeze"] = args.freeze

    print(f"\n{'='*60}")
    print(f"  SINGLE-STAGE training")
    print(f"  LR={args.lr0}, epochs={args.epochs}, "
          f"batch={args.batch}, imgsz={args.imgsz}")
    print(f"{'='*60}\n")

    return model.train(**cfg)


def main():
    args = parse_args()

    if args.single_stage:
        run_single_stage(args)
        return

    if args.stage == 1:
        run_stage1(args)
    elif args.stage == 2:
        if not args.stage1_weights:
            raise ValueError("--stage1-weights is required when --stage 2")
        run_stage2(args, args.stage1_weights)
    else:
        stage1_best = run_stage1(args)
        run_stage2(args, stage1_best)


if __name__ == "__main__":
    main()
