# Ultralytics BEV Pose Validator with sskit LocSim evaluation support.
# Mirrors the mmpose CocoMetric locsim/locsim_bbox evaluation used in:
#   mmpose/mmpose/evaluation/metrics/coco_metric.py
#   mmpose/configs/body_bev_position/spiideo_soccernet/synloc.py
#
# Key evaluation behavior reproduced from mmpose:
#   - Runs standard val first to find score_threshold (saved as *_val_stats.json)
#   - Uses sskit.coco.LocSimCOCOeval for LocSim mAP (position_from_keypoint_index=1)
#   - Uses sskit.coco.BBoxLocSimCOCOeval for BBox LocSim mAP
#   - Applies the val score_threshold when evaluating on test / challenge splits
#   - Writes results.json + metadata.json and zips for challenge submission

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import torch
from xtcocotools.coco import COCO

from ultralytics.utils import LOGGER

from .val import PoseValidator


class BEVPoseValidator(PoseValidator):
    """Extends PoseValidator with sskit LocSim evaluation.

    Mirrors the mmpose CocoMetric with iou_type='locsim'/'locsim_bbox'.

    Usage (programmatic):
        from ultralytics import YOLO
        model = YOLO("runs/pose-bev/yolo11m-pose-bev-640/weights/best.pt")
        validator = BEVPoseValidator(args=dict(
            data="soccernet-synloc.yaml",
            split="val",
            save_json=True,
        ))
        validator(model=model.model)

    Usage (via test_bev.py script):
        python test_bev.py --weights best.pt --data soccernet-synloc.yaml --split val
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None):
        """Initialize BEVPoseValidator."""
        super().__init__(dataloader, save_dir, args, _callbacks)
        # phase will be set to 'val', 'test', or 'challenge' by test_bev.py
        self.phase = getattr(self.args, "phase", "val")

    def init_metrics(self, model: torch.nn.Module) -> None:
        """Initialize metrics, reading custom sigmas from dataset config."""
        super().init_metrics(model)
        # Ensure save_json is on so predictions.json is written for LocSim eval
        self.args.save_json = True

    def eval_json(self, stats: dict[str, Any]) -> dict[str, Any]:
        """Run LocSim evaluation via sskit after standard validation finishes.

        Mirrors mmpose CocoMetric._do_python_keypoint_eval() with
        iou_type='locsim' and iou_type='locsim_bbox'.

        Args:
            stats: Existing stats dict from YOLO validation loop.

        Returns:
            Updated stats dict with LocSim AP/AR metrics added.
        """
        try:
            from sskit.coco import BBoxLocSimCOCOeval, LocSimCOCOeval
        except ImportError:
            LOGGER.warning("sskit not installed — skipping LocSim evaluation. Install with: pip install sskit")
            return stats

        pred_json = self.save_dir / "predictions.json"
        anno_json = self._get_anno_json()

        if not pred_json.exists():
            LOGGER.warning(f"predictions.json not found at {pred_json}; skipping LocSim eval.")
            return stats

        if not anno_json.exists():
            LOGGER.warning(f"Annotation file not found at {anno_json}; skipping LocSim eval.")
            return stats

        sigmas = list(self.sigma)  # [0.089, 0.089] for BEV task

        LOGGER.info(f"\nRunning sskit LocSim evaluation (phase='{self.phase}')...")

        # ---- build predictions in sskit-compatible COCO format ----
        coco_gt = COCO(str(anno_json))
        coco_dt = coco_gt.loadRes(str(pred_json))

        # ---- BBox LocSim ----
        stats = self._run_locsim(
            stats,
            coco_gt,
            coco_dt,
            sigmas,
            iou_type="locsim_bbox",
            EvalClass=BBoxLocSimCOCOeval,
            prefix="locsim_bbox",
        )

        # ---- Keypoint LocSim (primary metric) ----
        stats = self._run_locsim(
            stats,
            coco_gt,
            coco_dt,
            sigmas,
            iou_type="locsim",
            EvalClass=LocSimCOCOeval,
            prefix="locsim",
        )

        return stats

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_anno_json(self) -> Path:
        """Return annotation JSON path for the current split."""
        data_path = Path(self.data["path"])
        split = getattr(self.args, "split", "val")
        # Try challenge annotation file first
        if split == "challenge":
            candidates = [
                data_path / "annotations" / "challenge_public.json",
                data_path / "annotations" / "challenge.json",
            ]
        elif split == "test":
            candidates = [data_path / "annotations" / "test.json"]
        else:
            candidates = [data_path / "annotations" / "val.json"]

        for c in candidates:
            if c.exists():
                return c

        # Final fallback: whatever the dataset YAML says
        return data_path / "annotations" / f"{split}.json"

    def _run_locsim(
        self,
        stats: dict,
        coco_gt: COCO,
        coco_dt,
        sigmas: list,
        iou_type: str,
        EvalClass,
        prefix: str,
    ) -> dict:
        """Run a single LocSim evaluation pass and merge results into stats.

        Mirrors mmpose coco_metric._do_python_keypoint_eval() lines 565-609.
        """
        stats_file = self.save_dir / f"{prefix}_{self.phase}_stats.json"
        val_stats_file = self.save_dir / f"{prefix}_val_stats.json"

        coco_eval = EvalClass(coco_gt, coco_dt, "bbox", sigmas, use_area=True)

        # For test/challenge: load score_threshold from val run (mirrors mmpose test.py line 177)
        if self.phase in ("test", "challenge"):
            if not val_stats_file.exists():
                LOGGER.warning(
                    f"{val_stats_file} not found. "
                    "Run --split val first to obtain the optimal score threshold, "
                    "then re-run with --split test/challenge."
                )
            else:
                with open(val_stats_file) as f:
                    th = json.load(f)["stats"]["score_threshold"]
                LOGGER.info(f"Loaded score_threshold={th:.4f} from {val_stats_file}")
                coco_eval.params.score_threshold = th

        # position_from_keypoint_index=1 → pelvis_ground is the BEV position
        # (matches mmpose coco_metric.py line 581)
        coco_eval.params.position_from_keypoint_index = 1
        coco_eval.params.useSegm = None

        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # LocSim stat names (from mmpose coco_metric.py lines 594-596)
        stats_names = [
            "AP",
            "AP .5",
            "AP .75",
            "AP (S)",
            "AP (M)",
            "AP (L)",
            "AR",
            "AR .5",
            "AR .75",
            "AR (S)",
            "AR (M)",
            "AR (L)",
            "precision",
            "recall",
            "f1",
            "score_threshold",
            "frame_accuracy",
        ]
        assert len(stats_names) == len(coco_eval.stats), (
            f"LocSim stats length mismatch: expected {len(stats_names)}, got {len(coco_eval.stats)}"
        )

        result = dict(zip(stats_names, coco_eval.stats))

        # Save per-phase stats JSON so test phase can load score_threshold from val
        with open(stats_file, "w") as f:
            json.dump({"stats": result}, f, indent=4)

        # Also save as val_stats so test_bev.py can find it by the canonical name
        if self.phase == "val":
            shutil.copyfile(stats_file, val_stats_file)
            LOGGER.info(f"Saved val stats → {val_stats_file}")

        LOGGER.info(f"\n{prefix.upper()} LocSim results ({self.phase}):")
        for k, v in result.items():
            LOGGER.info(f"  {k:>20s}: {v:.4f}" if isinstance(v, float) else f"  {k:>20s}: {v}")

        # Merge into main stats dict under namespaced keys
        for k, v in result.items():
            stats[f"{prefix}/{k}"] = v

        return stats
