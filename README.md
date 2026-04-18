# PitchPose

Efficient bird's-eye-view athlete localisation for soccer broadcast images with YOLO11-Pose.

This repository is the codebase for my capstone project at The Hong Kong Polytechnic University:

- Author: Honghe DING
- Department: Department of Computing, The Hong Kong Polytechnic University
- Project type: COMP4913 Capstone Project

This project started from the official Ultralytics repository and was adapted for the Spiideo SoccerNet SynLoc bird's-eye-view (BEV) athlete localisation task. The current codebase includes task-specific model, training, evaluation, and demo components for predicting:

- the pelvis keypoint in image space
- the pelvis ground-projection keypoint in image space
- the corresponding athlete position on the soccer pitch

Results in this README are taken from `capstone_report.pdf`. Some additional experiments are still running and may update these numbers later.

## Project Overview

The goal of PitchPose is single-frame world-coordinate athlete localisation from static soccer broadcast cameras. Given one RGB image, the system detects each athlete, predicts two keypoints per player, and maps the ground-projection keypoint to pitch coordinates.

The project is built around four main ideas:

1. Replace the original YOLOX-pose + MMPose baseline with YOLO11x-pose inside the Ultralytics framework.
2. Use a two-stage training schedule: frozen backbone warm-up, then full fine-tuning.
3. Introduce an occlusion-aware keypoint loss using the segmentation-area / bounding-box-area ratio.
4. Use validation-calibrated y-band inference thresholds to improve far-field recall.

## Benchmark Setting

The project targets the Spiideo SoccerNet SynLoc benchmark for BEV athlete localisation.

Dataset summary from the report:

| Split | Images | Arenas |
| --- | ---: | ---: |
| Train | 42,504 | 13 |
| Validation | 6,777 | 13 |
| Test | 9,309 | 15 |
| Challenge | 11,352 | 15 |
| Total | 69,942 | 17 |

The dataset contains roughly 1.1 million annotated athletes. Each instance includes a bounding box, segmentation area, and two 3D keypoints: pelvis and pelvis ground projection.

## Main Results

Primary metric: `mAP-LocSim`, where localisation is evaluated directly in world coordinates rather than image-space IoU.

### Test-set comparison

| Method | Epochs | mAP-LocSim | Precision | Recall | F1 | Frame Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLOX-m pose baseline | 300 | 76.2 | 92.8 | 89.0 | 90.9 | 31.6 |
| YOLO11x-pose, single-stage, global threshold | 50 | 68.6 | 85.9 | 84.0 | 84.9 | 19.9 |
| YOLO11x-pose, two-stage + occlusion-aware loss, global threshold | 50 | 81.7 | 94.1 | 92.0 | 93.0 | 42.9 |
| YOLO11x-pose, two-stage + occlusion-aware loss, y-band inference | 50 | 80.5 | 91.2 | 93.0 | 92.1 | 47.1 |

### Efficiency summary

- Best global-threshold result: `81.7 mAP-LocSim`
- Improvement over published YOLOX-m pose baseline: `+5.5 mAP-LocSim`
- Training budget reduction: `50 epochs` vs `300 epochs`
- Checkpoint size reduction: about `110 MB` vs about `400 MB`
- Reported training time on one RTX PRO 6000: about `6.8 hours` total

## Method Summary

### 1. YOLO11-Pose BEV architecture

The original benchmark baseline used YOLOX-pose with MMPose. PitchPose migrates the task to Ultralytics YOLO11x-pose and adapts it to a two-keypoint BEV setup:

- one class: `person`
- two keypoints: `pelvis`, `pelvis_ground`
- custom OKS sigmas matching the benchmark protocol
- support for out-of-frame ground-projection keypoints

### 2. Two-stage training

The recommended training schedule is:

- Stage 1: freeze backbone, train neck + head for 30 epochs with a higher learning rate
- Stage 2: unfreeze all layers and fine-tune for 20 epochs with a much smaller learning rate

This was the largest single contributor in the ablation study, improving mAP-LocSim by `+10.9`.

### 3. Occlusion-aware keypoint loss

The project uses the ratio

`r = A_seg / A_bbox`

as a proxy for occlusion severity. More occluded athletes receive larger keypoint-loss weight during training, improving localisation quality under crowding and partial occlusion.

### 4. y-band calibrated inference

Because distant players appear smaller and tend to receive lower confidence scores, inference can be calibrated with separate confidence thresholds for horizontal image bands. In the report, a simple near-field / far-field split improved frame accuracy from `42.9` to `47.1`.

## Repository Structure

Key files and folders:

- `train_bev.py`: two-stage training entry point
- `test_bev.py`: LocSim-oriented evaluation script
- `run_experiments.sh`: experiment runner for dataset prep, training, validation, and post-processing
- `bev_demo/`: local Flask demo for image-space and pitch-space visualisation
- `tools/convert_coco_to_yolo_pose.py`: COCO-to-YOLO pose label conversion
- `tools/build_bev_postprocess_stats.py`: validation-calibrated y-band threshold generation
- `ultralytics/cfg/datasets/soccernet-synloc.yaml`: SynLoc dataset config
- `ultralytics/cfg/models/11/yolo11-pose-bev.yaml`: BEV pose model config
- `ultralytics/models/yolo/pose/val_bev.py`: custom validator with SSKit LocSim evaluation
- `ultralytics/models/yolo/pose/postprocess_bev.py`: BEV y-band post-processing

## Training

Example:

```bash
python train_bev.py --model-size x --imgsz 960 --rect
```

Two-stage training is the recommended mode. The script supports:

- frozen-backbone warm-up
- full fine-tuning
- rectangular batching
- custom BEV LocSim validation
- resume support

## Evaluation

Example validation:

```bash
python test_bev.py \
  --weights runs/pose/.../weights/best.pt \
  --data ultralytics/cfg/datasets/soccernet-synloc.yaml \
  --split val \
  --imgsz 960
```

The evaluation pipeline mirrors the benchmark protocol:

- run validation with a permissive confidence threshold
- select an operating threshold from validation
- evaluate test predictions with either a global threshold or y-band thresholds

## Local Demo

A lightweight local Flask demo is provided in `bev_demo/`.

Current demo features:

- choose from predefined sample images
- run the BEV pose model on one image
- visualise thin player bounding boxes
- visualise the two predicted keypoints
- assign consistent player IDs across the image and pitch views
- show per-player confidence and coordinates

Run locally:

```bash
python -m bev_demo.app
```

Then open:

```text
http://127.0.0.1:5000
```

## Notes on Projection

The benchmark task is defined through camera calibration and ground-plane projection. In the current demo, the pitch view is intended for presentation and qualitative inspection. The local demo uses the available sample annotation pairs to render a stable world-view visualisation for chosen examples.

## Current Status

- Core method and ablation results are documented in `capstone_report.pdf`
- Local demo is available for presentation
- Additional experiments are still in progress

## Acknowledgement

This repository is based on the official Ultralytics codebase and extends it for BEV soccer athlete localisation research.
