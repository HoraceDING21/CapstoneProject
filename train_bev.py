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
import json
from copy import copy
from pathlib import Path

from ultralytics import YOLO
from ultralytics.models.yolo.pose.train import PoseTrainer
from ultralytics.models.yolo.pose.val_bev import BEVPoseValidator
from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import unwrap_model


class BEVTrainingValidator(BEVPoseValidator):
    """BEV validator that also computes LocSim metrics during training."""

    def __call__(self, trainer=None, model=None):
        """Run validation and expose BEV LocSim as training fitness."""
        results = super().__call__(trainer=trainer, model=model)
        if trainer is None or results is None:
            return results

        if self.args.save_json and self.jdict:
            pred_json = self.save_dir / "predictions.json"
            with open(pred_json, "w", encoding="utf-8") as f:
                json.dump(self.jdict, f)

            # BaseValidator skips eval_json() during training. Run it here so
            # LocSim metrics are available for checkpoint selection.
            results = self.eval_json(results)

        fitness = self._select_bev_fitness(results)
        results["fitness"] = fitness
        LOGGER.info(f"BEV-aware model selection fitness (locsim/AP): {fitness:.5f}")
        return self._round_results(results)

    @staticmethod
    def _select_bev_fitness(results):
        """Select the primary BEV metric used for best.pt / early stopping."""
        for key in ("locsim/AP", "locsim_bbox/AP", "metrics/mAP50-95(P)", "metrics/mAP50-95(B)"):
            value = results.get(key)
            if value is not None:
                return float(value)
        return 0.0

    @staticmethod
    def _round_results(results):
        """Round numeric validation outputs to match Ultralytics defaults."""
        rounded = {}
        for key, value in results.items():
            if hasattr(value, "item"):
                value = value.item()
            if isinstance(value, (int, float)):
                rounded[key] = round(float(value), 5)
            else:
                rounded[key] = value
        return rounded


class BEVPoseTrainer(PoseTrainer):
    """Pose trainer that validates with BEV LocSim metrics."""

    mmpose_val_interval = 10
    mmpose_dense_val_epochs = 10
    mmpose_dense_val_interval = 2
    bev_metric_defaults = {
        "locsim_bbox/AP": 0.0,
        "locsim_bbox/AP .5": 0.0,
        "locsim_bbox/AP .75": 0.0,
        "locsim_bbox/AP (S)": 0.0,
        "locsim_bbox/AP (M)": 0.0,
        "locsim_bbox/AP (L)": 0.0,
        "locsim_bbox/AR": 0.0,
        "locsim_bbox/AR .5": 0.0,
        "locsim_bbox/AR .75": 0.0,
        "locsim_bbox/AR (S)": 0.0,
        "locsim_bbox/AR (M)": 0.0,
        "locsim_bbox/AR (L)": 0.0,
        "locsim_bbox/precision": 0.0,
        "locsim_bbox/recall": 0.0,
        "locsim_bbox/f1": 0.0,
        "locsim_bbox/score_threshold": 0.0,
        "locsim_bbox/frame_accuracy": 0.0,
        "locsim/AP": 0.0,
        "locsim/AP .5": 0.0,
        "locsim/AP .75": 0.0,
        "locsim/AP (S)": 0.0,
        "locsim/AP (M)": 0.0,
        "locsim/AP (L)": 0.0,
        "locsim/AR": 0.0,
        "locsim/AR .5": 0.0,
        "locsim/AR .75": 0.0,
        "locsim/AR (S)": 0.0,
        "locsim/AR (M)": 0.0,
        "locsim/AR (L)": 0.0,
        "locsim/precision": 0.0,
        "locsim/recall": 0.0,
        "locsim/f1": 0.0,
        "locsim/score_threshold": 0.0,
        "locsim/frame_accuracy": 0.0,
    }

    def get_validator(self):
        """Use BEV validator so training-time validation matches test-time task."""
        self.loss_names = "box_loss", "pose_loss", "kobj_loss", "cls_loss", "dfl_loss"
        if getattr(unwrap_model(self.model).model[-1], "flow_model", None) is not None:
            self.loss_names += ("rle_loss",)
        return BEVTrainingValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
            phase="val",
        )

    def validate(self):
        """Run validation on an mmpose-aligned cadence.

        Match the mmpose YOLOXPose schedule:
        - validate every 10 epochs during the main training phase
        - validate every 2 epochs for the final dense-validation phase
        """
        current_epoch = self.epoch + 1
        dense_start_epoch = max(self.epochs - self.mmpose_dense_val_epochs, 1)
        is_dense_epoch = current_epoch >= dense_start_epoch
        should_validate = (
            current_epoch >= self.epochs
            or (
                is_dense_epoch
                and (
                    (current_epoch - dense_start_epoch) % self.mmpose_dense_val_interval == 0
                    or current_epoch == dense_start_epoch
                )
            )
            or (not is_dense_epoch and current_epoch % self.mmpose_val_interval == 0)
        )
        if should_validate:
            metrics, fitness = super().validate()
            self.metrics = self._with_bev_metric_defaults(metrics)
            return self.metrics, fitness

        if is_dense_epoch:
            next_val_epoch = min(
                dense_start_epoch
                + (((current_epoch - dense_start_epoch) // self.mmpose_dense_val_interval) + 1)
                * self.mmpose_dense_val_interval,
                self.epochs,
            )
        else:
            next_sparse_epoch = ((current_epoch // self.mmpose_val_interval) + 1) * self.mmpose_val_interval
            next_val_epoch = min(next_sparse_epoch, self.epochs)
        LOGGER.info(
            "Skipping validation to match mmpose cadence: "
            f"epoch {current_epoch}/{self.epochs}, next val at epoch {next_val_epoch}."
        )
        self.metrics = self._with_bev_metric_defaults(self.metrics)
        return self.metrics, None

    @classmethod
    def _with_bev_metric_defaults(cls, metrics):
        """Ensure skipped-validation epochs keep a stable results.csv schema."""
        merged = dict(cls.bev_metric_defaults)
        if metrics:
            merged.update(metrics)
        return merged


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
    p.add_argument("--project", type=str, default=None,
                   help="Project dir (default: YOLO saves to runs/pose/<name>)")
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
        pose=12.0,
        kobj=1.0,
        close_mosaic=10,
        rect=args.rect,
        cos_lr=args.cos_lr,
    )
    if args.project is not None:
        cfg["project"] = args.project
    if args.device is not None:
        cfg["device"] = args.device
    cfg["save_json"] = True
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

    print("  Validation/selection: BEV-aware (best.pt selected by val locsim/AP)\n")
    results = model.train(trainer=BEVPoseTrainer, **cfg)

    # Use the actual save_dir from training results — YOLO may prepend
    # 'runs/<task>/' to args.project internally, so do not hardcode the path.
    best_pt = Path(results.save_dir) / "weights" / "best.pt"
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

    print("  Validation/selection: BEV-aware (best.pt selected by val locsim/AP)\n")
    results = model.train(trainer=BEVPoseTrainer, **cfg)

    best_pt = Path(results.save_dir) / "weights" / "best.pt"
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

    print("  Validation/selection: BEV-aware (best.pt selected by val locsim/AP)\n")
    return model.train(trainer=BEVPoseTrainer, **cfg)


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
