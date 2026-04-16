#!/usr/bin/env python3
"""Build val-calibrated y-band score-threshold calibration for BEV inference."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from xtcocotools.coco import COCO

STATS_NAMES = [
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


def log_progress(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build BEV y-band calibration from val predictions.")
    parser.add_argument("--data", type=str, required=True, help="Dataset YAML path (e.g. soccernet-synloc.yaml).")
    parser.add_argument(
        "--ann",
        type=str,
        default=None,
        help="Optional explicit val annotation JSON path. Defaults to <data.path>/annotations/val.json.",
    )
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Baseline val predictions.json path used for y-band threshold calibration.",
    )
    parser.add_argument(
        "--val-stats",
        type=str,
        default=None,
        help="Optional locsim_val_stats.json path to reuse the global val score_threshold.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path. Defaults to <data.path>/annotations/bev_band_calibration.json.",
    )
    parser.add_argument(
        "--position-keypoint-index",
        type=int,
        default=1,
        help="Keypoint index used as player position (default: pelvis_ground=1).",
    )
    parser.add_argument(
        "--y-bands",
        type=int,
        default=4,
        help="Number of vertical image bands used for threshold calibration.",
    )
    parser.add_argument(
        "--bounds-lower-percentile",
        type=float,
        default=0.5,
        help="Lower percentile over per-frame min position (x/y).",
    )
    parser.add_argument(
        "--bounds-upper-percentile",
        type=float,
        default=99.5,
        help="Upper percentile over per-frame max position (x/y).",
    )
    parser.add_argument(
        "--min-gt-per-band",
        type=int,
        default=25,
        help="Minimum GT annotations required to calibrate one band directly from val predictions.",
    )
    parser.add_argument(
        "--min-pred-per-band",
        type=int,
        default=25,
        help="Minimum predictions required to calibrate one band directly from val predictions.",
    )
    parser.add_argument(
        "--min-threshold-ratio",
        type=float,
        default=0.70,
        help="Lower clamp for learned per-band threshold ratio relative to the global val threshold.",
    )
    parser.add_argument(
        "--max-threshold-ratio",
        type=float,
        default=1.10,
        help="Upper clamp for learned per-band threshold ratio relative to the global val threshold.",
    )
    parser.add_argument(
        "--fallback-threshold-ratio",
        type=float,
        default=1.00,
        help="Fallback ratio for sparse or uncalibrated bands.",
    )
    parser.add_argument("--sweep-steps", type=int, default=31, help="Maximum number of candidate thresholds per band.")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def ann_position_norm_from_keypoint(
    ann: dict[str, Any],
    image_wh: tuple[float, float],
    keypoint_index: int,
) -> tuple[float, float] | None:
    if "keypoints" not in ann:
        return None
    kpts = ann["keypoints"]
    if len(kpts) <= keypoint_index or len(kpts[keypoint_index]) < 2:
        return None
    width, height = image_wh
    if width <= 0 or height <= 0:
        return None
    x = float(kpts[keypoint_index][0]) / float(width)
    y = float(kpts[keypoint_index][1]) / float(height)
    return x, y


def image_position_norm_from_prediction(
    pred: dict[str, Any],
    keypoint_index: int,
    image_wh: tuple[float, float],
) -> tuple[float, float] | None:
    kpts = pred.get("keypoints")
    if not isinstance(kpts, list):
        return None
    start = keypoint_index * 3
    if len(kpts) < start + 2:
        return None
    width, height = image_wh
    if width <= 0 or height <= 0:
        return None
    x = float(kpts[start]) / float(width)
    y = float(kpts[start + 1]) / float(height)
    return x, y


def infer_bounds_from_frame_extrema(
    points_by_image: dict[Any, list[tuple[float, float]]],
    lower_p: float,
    upper_p: float,
) -> dict[str, float]:
    frame_x_min: list[float] = []
    frame_x_max: list[float] = []
    frame_y_min: list[float] = []
    frame_y_max: list[float] = []
    for pts in points_by_image.values():
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        frame_x_min.append(float(min(xs)))
        frame_x_max.append(float(max(xs)))
        frame_y_min.append(float(min(ys)))
        frame_y_max.append(float(max(ys)))
    if not frame_x_min:
        return {"x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0}
    x_min = float(np.percentile(frame_x_min, lower_p))
    x_max = float(np.percentile(frame_x_max, upper_p))
    y_min = float(np.percentile(frame_y_min, lower_p))
    y_max = float(np.percentile(frame_y_max, upper_p))
    if x_max <= x_min:
        x_min, x_max = 0.0, 1.0
    if y_max <= y_min:
        y_min, y_max = 0.0, 1.0
    return {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max}


def band_index(y: float, y_min: float, y_max: float, y_bands: int) -> int:
    y_span = max(y_max - y_min, 1e-6)
    band = int(np.floor((y - y_min) / y_span * y_bands))
    return min(max(band, 0), y_bands - 1)


def coco_from_dataset(dataset: dict[str, Any]) -> COCO:
    coco = COCO()
    coco.dataset = dataset
    coco.createIndex()
    return coco


def subset_coco_payload(
    ann: dict[str, Any],
    image_ids: set[Any],
    ann_ids: set[int],
) -> dict[str, Any]:
    images = [im for im in ann.get("images", []) if im.get("id") in image_ids]
    anns: list[dict[str, Any]] = []
    for a in ann.get("annotations", []):
        if int(a.get("id")) not in ann_ids:
            continue
        item = dict(a)
        item.setdefault("iscrowd", 0)
        if "area" not in item and "bbox" in item and len(item["bbox"]) >= 4:
            item["area"] = float(max(0.0, item["bbox"][2] * item["bbox"][3]))
        if "num_keypoints" not in item and "keypoints" in item:
            kpts = item["keypoints"]
            if kpts and isinstance(kpts[0], list):
                item["num_keypoints"] = int(sum(1 for kp in kpts if len(kp) >= 3 and kp[2] > 0))
            elif isinstance(kpts, list):
                item["num_keypoints"] = int(sum(1 for i in range(2, len(kpts), 3) if kpts[i] > 0))
        anns.append(item)
    return {
        "images": images,
        "annotations": anns,
        "categories": ann.get("categories", []),
    }


def compute_locsim_stats(
    ann_payload: dict[str, Any],
    preds: list[dict[str, Any]],
    position_keypoint_index: int,
    score_threshold: float | None = None,
) -> dict[str, float] | None:
    try:
        from sskit.coco import LocSimCOCOeval
    except ImportError as exc:
        raise ImportError("sskit is required to calibrate y-band thresholds") from exc

    if not ann_payload.get("annotations") or not preds:
        return None

    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = coco_from_dataset(ann_payload)
        coco_dt = coco_gt.loadRes(preds)
        coco_eval = LocSimCOCOeval(coco_gt, coco_dt, "bbox", [0.089, 0.089], use_area=True)
        if score_threshold is not None:
            coco_eval.params.score_threshold = float(score_threshold)
        coco_eval.params.position_from_keypoint_index = position_keypoint_index
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
    stats = coco_eval.stats
    if len(stats) < len(STATS_NAMES):
        return None
    result = dict(zip(STATS_NAMES, (float(x) for x in stats)))
    return result if np.isfinite(result["f1"]) and np.isfinite(result["score_threshold"]) else None


def compute_locsim_score_threshold(ann_payload: dict[str, Any], preds: list[dict[str, Any]], position_keypoint_index: int) -> float | None:
    stats = compute_locsim_stats(ann_payload, preds, position_keypoint_index)
    if not stats:
        return None
    threshold = float(stats["score_threshold"])
    return threshold if np.isfinite(threshold) else None


def candidate_thresholds(
    preds: list[dict[str, Any]],
    lower: float,
    upper: float,
    max_steps: int,
    include_threshold: float | None = None,
) -> list[float]:
    scores = []
    for pred in preds:
        score = pred.get("score")
        if score is None:
            continue
        score = float(score)
        if np.isfinite(score) and lower <= score <= upper:
            scores.append(score)

    candidates: set[float] = {float(lower), float(upper)}
    if include_threshold is not None and lower <= include_threshold <= upper:
        candidates.add(float(include_threshold))

    if scores:
        uniq = np.unique(np.asarray(scores, dtype=np.float64))
        if uniq.size > max_steps:
            q = np.linspace(0.0, 1.0, num=max_steps)
            uniq = np.quantile(uniq, q, method="nearest")
            uniq = np.unique(uniq)
        candidates.update(float(x) for x in uniq.tolist())

    return sorted(candidates)


def filter_predictions_by_band_thresholds(
    preds: list[dict[str, Any]],
    image_wh_by_id: dict[Any, tuple[float, float]],
    position_keypoint_index: int,
    pos_bounds: dict[str, float],
    y_bands: int,
    band_thresholds: dict[int, float],
    fallback_score_threshold: float,
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for pred in preds:
        image_id = pred.get("image_id")
        image_wh = image_wh_by_id.get(image_id)
        threshold = float(fallback_score_threshold)
        if image_wh is not None:
            pos2d = image_position_norm_from_prediction(pred, position_keypoint_index, image_wh)
            if pos2d is not None:
                band = band_index(pos2d[1], pos_bounds["y_min"], pos_bounds["y_max"], y_bands)
                threshold = float(band_thresholds.get(band, fallback_score_threshold))
        score = pred.get("score")
        if score is not None and float(score) >= threshold:
            filtered.append(pred)
    return filtered


def sweep_best_band_threshold(
    ann_payload: dict[str, Any],
    preds: list[dict[str, Any]],
    position_keypoint_index: int,
    global_threshold: float,
    min_ratio: float,
    max_ratio: float,
    max_steps: int,
    band_name: str,
) -> tuple[float | None, dict[str, float] | None]:
    lower = float(global_threshold * min_ratio)
    upper = float(global_threshold * max_ratio)
    candidates = candidate_thresholds(preds, lower, upper, max_steps, include_threshold=global_threshold)
    log_progress(
        f"  [{band_name}] sweeping {len(candidates)} candidate thresholds "
        f"in [{lower:.4f}, {upper:.4f}]"
    )

    best_threshold: float | None = None
    best_stats: dict[str, float] | None = None
    best_key: tuple[float, float, float, float] | None = None
    for idx, threshold in enumerate(candidates, start=1):
        stats = compute_locsim_stats(ann_payload, preds, position_keypoint_index, score_threshold=threshold)
        if not stats:
            continue
        # Mimic baseline intent: choose the working point that gives the best
        # thresholded quality, with mild tie-breakers for AP/recall and then
        # prefer thresholds closer to the global operating point.
        key = (
            float(stats["f1"]),
            float(stats["AP"]),
            float(stats["recall"]),
            -abs(float(threshold) - float(global_threshold)),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_stats = stats
        if idx == 1 or idx == len(candidates) or idx % max(1, len(candidates) // 5) == 0:
            log_progress(
                f"    [{band_name}] {idx}/{len(candidates)} "
                f"thr={threshold:.4f} f1={stats['f1']:.4f} ap={stats['AP']:.4f}"
            )
    if best_threshold is not None and best_stats is not None:
        log_progress(
            f"  [{band_name}] best threshold={best_threshold:.4f} "
            f"(f1={best_stats['f1']:.4f}, ap={best_stats['AP']:.4f})"
        )
    else:
        log_progress(f"  [{band_name}] no valid threshold found during sweep")
    return best_threshold, best_stats


def main() -> None:
    t0 = time.time()
    args = parse_args()
    if args.y_bands < 1:
        raise ValueError("y-bands must be >= 1")
    if args.sweep_steps < 2:
        raise ValueError("sweep-steps must be >= 2")
    if args.min_threshold_ratio <= 0 or args.max_threshold_ratio <= 0:
        raise ValueError("threshold ratio clamps must be positive")
    if args.max_threshold_ratio < args.min_threshold_ratio:
        raise ValueError("max-threshold-ratio must be >= min-threshold-ratio")

    log_progress("Loading dataset config...")
    with open(args.data) as f:
        data_cfg = yaml.safe_load(f)
    data_root = Path(data_cfg["path"])

    ann_path = Path(args.ann) if args.ann else data_root / "annotations" / "val.json"
    pred_path = Path(args.predictions)
    out_path = Path(args.output) if args.output else data_root / "annotations" / "bev_band_calibration.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log_progress(f"Loading val annotations: {ann_path}")
    ann = load_json(ann_path)
    log_progress(f"Loading val predictions: {pred_path}")
    preds = load_json(pred_path)
    if not isinstance(preds, list):
        raise ValueError("predictions JSON must be a list of detection dicts")

    anns = ann.get("annotations", [])
    images = ann.get("images", [])
    image_wh_by_id = {im.get("id"): (float(im.get("width", 0.0)), float(im.get("height", 0.0))) for im in images}

    log_progress("Collecting normalized GT positions...")
    pos2d_by_image: dict[Any, list[tuple[float, float]]] = defaultdict(list)
    for a in anns:
        image_id = a.get("image_id")
        image_wh = image_wh_by_id.get(image_id)
        if image_wh is None:
            continue
        pos2d = ann_position_norm_from_keypoint(a, image_wh=image_wh, keypoint_index=args.position_keypoint_index)
        if pos2d is None:
            continue
        pos2d_by_image[image_id].append(pos2d)

    log_progress("Inferring normalized image-position bounds...")
    pos_bounds = infer_bounds_from_frame_extrema(
        points_by_image=pos2d_by_image,
        lower_p=args.bounds_lower_percentile,
        upper_p=args.bounds_upper_percentile,
    )

    log_progress(f"Assigning GT/predictions into {args.y_bands} y-bands...")
    ann_ids_by_band: dict[int, set[int]] = defaultdict(set)
    pred_entries_by_band: dict[int, list[dict[str, Any]]] = defaultdict(list)
    gt_count_by_band: dict[int, int] = defaultdict(int)
    pred_count_by_band: dict[int, int] = defaultdict(int)

    for a in anns:
        image_id = a.get("image_id")
        image_wh = image_wh_by_id.get(image_id)
        if image_wh is None:
            continue
        pos2d = ann_position_norm_from_keypoint(a, image_wh=image_wh, keypoint_index=args.position_keypoint_index)
        if pos2d is None:
            continue
        band = band_index(pos2d[1], pos_bounds["y_min"], pos_bounds["y_max"], args.y_bands)
        ann_ids_by_band[band].add(int(a.get("id")))
        gt_count_by_band[band] += 1

    for pred in preds:
        image_id = pred.get("image_id")
        image_wh = image_wh_by_id.get(image_id)
        if image_wh is None:
            continue
        pos2d = image_position_norm_from_prediction(pred, args.position_keypoint_index, image_wh)
        if pos2d is None:
            continue
        band = band_index(pos2d[1], pos_bounds["y_min"], pos_bounds["y_max"], args.y_bands)
        pred_entries_by_band[band].append(pred)
        pred_count_by_band[band] += 1

    if args.val_stats:
        log_progress(f"Loading global val threshold from: {args.val_stats}")
        val_stats = load_json(Path(args.val_stats))
        global_threshold = float(val_stats["stats"]["score_threshold"])
    else:
        log_progress("Computing global val threshold from full predictions...")
        global_threshold = compute_locsim_score_threshold(ann, preds, args.position_keypoint_index)
        if global_threshold is None:
            raise RuntimeError("Failed to compute global val score_threshold from predictions")
    if global_threshold <= 0:
        raise RuntimeError(f"Invalid global score_threshold: {global_threshold}")
    log_progress(f"Global score_threshold = {global_threshold:.4f}")

    band_threshold_lookup: dict[str, dict[str, float | int | str]] = {}
    band_thresholds: dict[int, float] = {}
    band_threshold_ratios: dict[int, float] = {}
    calibrated_bands = 0
    fallback_bands = 0
    skip_low_gt = 0
    skip_low_pred = 0
    skip_no_threshold = 0

    for band in range(args.y_bands):
        band_name = f"band {band}"
        gt_count = int(gt_count_by_band.get(band, 0))
        pred_count = int(pred_count_by_band.get(band, 0))
        threshold_ratio = float(args.fallback_threshold_ratio)
        score_threshold = float(global_threshold * threshold_ratio)
        learned_threshold = None
        source = "fallback"

        log_progress(f"Processing {band_name}: gt={gt_count}, pred={pred_count}")

        if gt_count < args.min_gt_per_band:
            skip_low_gt += 1
            log_progress(f"  [{band_name}] skipped: gt_count < min_gt_per_band ({args.min_gt_per_band})")
        elif pred_count < args.min_pred_per_band:
            skip_low_pred += 1
            log_progress(f"  [{band_name}] skipped: pred_count < min_pred_per_band ({args.min_pred_per_band})")
        else:
            ann_ids = ann_ids_by_band.get(band, set())
            pred_subset = pred_entries_by_band.get(band, [])
            pred_image_ids = {p.get("image_id") for p in pred_subset}
            gt_image_ids = {a.get("image_id") for a in anns if int(a.get("id")) in ann_ids}
            subset = subset_coco_payload(ann, pred_image_ids | gt_image_ids, ann_ids)
            learned_threshold, _ = sweep_best_band_threshold(
                subset,
                pred_subset,
                args.position_keypoint_index,
                global_threshold,
                args.min_threshold_ratio,
                args.max_threshold_ratio,
                args.sweep_steps,
                band_name,
            )
            if learned_threshold is not None and learned_threshold > 0:
                score_threshold = float(learned_threshold)
                threshold_ratio = float(learned_threshold / global_threshold)
                threshold_ratio = min(max(threshold_ratio, args.min_threshold_ratio), args.max_threshold_ratio)
                score_threshold = float(global_threshold * threshold_ratio)
                source = "val_calibrated"
                calibrated_bands += 1
                log_progress(f"  [{band_name}] calibrated threshold={score_threshold:.4f} ratio={threshold_ratio:.4f}")
            else:
                skip_no_threshold += 1
                log_progress(f"  [{band_name}] fallback: no usable threshold learned")

        if source == "fallback":
            fallback_bands += 1

        band_thresholds[band] = float(score_threshold)
        band_threshold_ratios[band] = float(threshold_ratio)
        band_threshold_lookup[str(band)] = {
            "score_threshold": float(score_threshold),
            "threshold_ratio": float(threshold_ratio),
            "gt_count": int(gt_count),
            "pred_count": int(pred_count),
            "source": source,
        }
        if learned_threshold is not None:
            band_threshold_lookup[str(band)]["learned_threshold"] = float(learned_threshold)

    fallback_score_threshold = float(global_threshold * args.fallback_threshold_ratio)
    filtered_preds = filter_predictions_by_band_thresholds(
        preds=preds,
        image_wh_by_id=image_wh_by_id,
        position_keypoint_index=args.position_keypoint_index,
        pos_bounds=pos_bounds,
        y_bands=args.y_bands,
        band_thresholds=band_thresholds,
        fallback_score_threshold=fallback_score_threshold,
    )
    filtered_stats = compute_locsim_stats(ann, filtered_preds, args.position_keypoint_index)
    if filtered_stats:
        log_progress(
            "Band-filtered val summary: "
            f"AP={filtered_stats['AP']:.4f}, "
            f"precision={filtered_stats['precision']:.4f}, "
            f"recall={filtered_stats['recall']:.4f}, "
            f"f1={filtered_stats['f1']:.4f}, "
            f"frame_accuracy={filtered_stats['frame_accuracy']:.4f}"
        )
    else:
        log_progress("Band-filtered val summary: unavailable")

    payload = {
        "version": 7,
        "source_annotation": str(ann_path),
        "position_keypoint_index": int(args.position_keypoint_index),
        "global_score_threshold": float(global_threshold),
        "base_score_threshold": float(global_threshold),
        "image_position_bounds": pos_bounds,
        "banding": {"type": "image_norm_y", "y_bands": int(args.y_bands)},
        "band_thresholds": {str(k): float(v) for k, v in band_thresholds.items()},
        "band_threshold_ratios": {str(k): float(v) for k, v in band_threshold_ratios.items()},
        "band_threshold_lookup": band_threshold_lookup,
        "fallback_score_threshold": float(fallback_score_threshold),
        "fallback_threshold_ratio": float(args.fallback_threshold_ratio),
        "min_gt_per_band": int(args.min_gt_per_band),
        "min_pred_per_band": int(args.min_pred_per_band),
        "min_threshold_ratio": float(args.min_threshold_ratio),
        "max_threshold_ratio": float(args.max_threshold_ratio),
        "optimization_metric": "val_locsim_f1_with_bandwise_instance_filtering",
        "val_prediction_count_before": int(len(preds)),
        "val_prediction_count_after": int(len(filtered_preds)),
        "val_filtered_locsim_stats": filtered_stats,
    }

    with out_path.open("w") as f:
        json.dump(payload, f, indent=2)

    ratios = [float(v["threshold_ratio"]) for v in band_threshold_lookup.values()]
    thresholds = [float(v["score_threshold"]) for v in band_threshold_lookup.values()]
    elapsed = time.time() - t0
    print(f"Saved BEV band calibration: {out_path}")
    print(f"  occupied bands: {sum(1 for i in range(args.y_bands) if gt_count_by_band.get(i, 0) > 0 or pred_count_by_band.get(i, 0) > 0)} / {args.y_bands}")
    print(
        "  inferred image-position bounds: "
        f"x=[{pos_bounds['x_min']:.4f}, {pos_bounds['x_max']:.4f}], "
        f"y=[{pos_bounds['y_min']:.4f}, {pos_bounds['y_max']:.4f}]"
    )
    print(f"  global score_threshold: {global_threshold:.4f}")
    print(
        "  learned threshold ratios: "
        f"min={min(ratios) if ratios else args.fallback_threshold_ratio:.3f}, "
        f"max={max(ratios) if ratios else args.fallback_threshold_ratio:.3f}, "
        f"calibrated_bands={calibrated_bands}, fallback_bands={fallback_bands}"
    )
    print(
        "  learned score thresholds: "
        f"min={min(thresholds) if thresholds else fallback_score_threshold:.4f}, "
        f"max={max(thresholds) if thresholds else fallback_score_threshold:.4f}, "
        f"fallback={fallback_score_threshold:.4f}"
    )
    print(
        "  fallback reasons: "
        f"low_gt={skip_low_gt}, low_pred={skip_low_pred}, no_threshold={skip_no_threshold}"
    )
    print(f"  val prediction count: before={len(preds)}, after={len(filtered_preds)}")
    if filtered_stats:
        print(
            "  band-filtered val locsim: "
            f"AP={filtered_stats['AP']:.4f}, "
            f"precision={filtered_stats['precision']:.4f}, "
            f"recall={filtered_stats['recall']:.4f}, "
            f"f1={filtered_stats['f1']:.4f}, "
            f"frame_accuracy={filtered_stats['frame_accuracy']:.4f}"
        )
    print(f"  elapsed_sec: {elapsed:.1f}")


if __name__ == "__main__":
    main()
