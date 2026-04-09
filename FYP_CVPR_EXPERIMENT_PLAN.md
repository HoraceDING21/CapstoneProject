# FYP CVPR-Style Experiment Plan

This note turns your four innovations into a paper-friendly experiment section and maps them to the runnable script `run_fyp_cvpr_experiments.sh`.

## Core Principle

Use one main setting for the paper body:

- model: `YOLO11x`
- image size: `960`
- rectangular batching: `--rect`
- tuning budget: `30 + 20` epochs for two-stage
- evaluation: `LocSim` / `BBox LocSim`

This gives you a clean, reviewer-friendly story:

1. `YOLOXPose-M` baseline from your existing reproduced logs
2. swap backbone to `YOLO11x`
3. replace head-only tuning with two-stage tuning
4. add occlusion-aware loss
5. add validation-calibrated y-band inference

## Recommended Tables

### Table 1: Main Comparison

Use your existing reproduced `YOLOXPose-M` logs as the baseline rows, then add your best final model:

- `YOLOXPose-M`, `imgsz=960`, 300 epochs, existing log
- `YOLO11x + two-stage + occ loss + y-band inference`, `imgsz=960`, `rect`

Suggested metrics:

- `LocSim AP`
- `LocSim precision`
- `LocSim recall`
- `LocSim F1`
- `BBox LocSim AP`

### Table 2: Training Ablation

Keep everything fixed except the training strategy / loss:

- `YOLO11x`, single-stage, 50 epochs
- `YOLO11x`, two-stage `(30 + 20)`, no occlusion loss
- `YOLO11x`, two-stage `(30 + 20)`, occlusion-aware loss

This is the cleanest way to isolate innovation 2 and innovation 3 under the same total budget.

### Table 3: Inference Ablation

Use the same best checkpoint from Table 2 and only change inference:

- baseline global threshold
- y-band inference with `2` bands
- y-band inference with `3` bands
- y-band inference with `4` bands

This isolates innovation 4.

## Why The Script Generates Two Dataset YAMLs

This is important:

- the plain dataset must use `use_occ_weights: false`
- the occlusion-weighted dataset must use `use_occ_weights: true`

If you reuse the current default YAML directly, the "no occlusion loss" ablation will not be clean because the visibility channel would still be interpreted as weights. The script fixes this automatically by generating:

- `experiments/fyp_cvpr/generated_cfg/soccernet-synloc-base.yaml`
- `experiments/fyp_cvpr/generated_cfg/soccernet-synloc-occ.yaml`

## Recommended Run Order

### Fast Track

This is the one you should run first.

```bash
export COCO_ROOT=/path/to/your/coco_style_dataset
export PRETRAINED_X=/path/to/yolo11x-pose.pt
export EXISTING_OCC_STAGE2_WEIGHTS=/path/to/your/already_trained/best.pt  # optional

bash run_fyp_cvpr_experiments.sh fast
```

What it does:

- prepares base and occlusion-weighted datasets
- trains `YOLO11x` single-stage 50-epoch baseline
- trains `YOLO11x` two-stage `(30 + 20)` without occlusion loss
- reuses or trains `YOLO11x` two-stage `(30 + 20)` with occlusion loss
- runs baseline test evaluation for each trained model
- runs y-band inference ablation for the best occlusion-loss model
- writes CSV summaries

### Inference Only

If your weighted-loss model is already trained and you only want the inference ablation:

```bash
export EXISTING_OCC_STAGE2_WEIGHTS=/path/to/your/already_trained/best.pt
bash run_fyp_cvpr_experiments.sh inference
```

## Where Results Are Saved

Everything is now grouped under a more distinctive folder tree so it does not mix with your old `runs/pose` experiments.

- generated dataset YAMLs: `experiments/fyp_cvpr_50ep_960/generated_cfg`
- CSV summaries: `experiments/fyp_cvpr_50ep_960/results`
- all training and evaluation outputs: `runs/pose/fyp_cvpr_50ep_960/main_study`

The main experiment folders are:

- `runs/pose/fyp_cvpr_50ep_960/main_study/yolo11x_single_960_rect_50ep`
- `runs/pose/fyp_cvpr_50ep_960/main_study/yolo11x_twostage_960_rect_30p20_base`
- `runs/pose/fyp_cvpr_50ep_960/main_study/yolo11x_twostage_960_rect_30p20_occ`

Inside each experiment folder:

- training checkpoint: `weights/best.pt`
- baseline validation outputs: `eval/baseline/val`
- baseline test outputs: `eval/baseline/test`

For the best occlusion-loss model, y-band inference ablations are saved in:

- `runs/pose/fyp_cvpr_50ep_960/main_study/yolo11x_twostage_960_rect_30p20_occ/eval/postprocess/yband_2/test`
- `runs/pose/fyp_cvpr_50ep_960/main_study/yolo11x_twostage_960_rect_30p20_occ/eval/postprocess/yband_3/test`
- `runs/pose/fyp_cvpr_50ep_960/main_study/yolo11x_twostage_960_rect_30p20_occ/eval/postprocess/yband_4/test`

The summary files are:

- `training_ablation.csv`
- `inference_ablation.csv`

## Suggested Writing Logic For The Paper

A good CVPR-style narrative is:

1. Start from your reproduced `YOLOXPose-M`, `imgsz=960`, 300-epoch baseline logs.
2. Show that switching to `YOLO11x` already beats the old baseline even under only 50 epochs.
3. Show that the gain is not only from the backbone: two-stage tuning further improves the same `YOLO11x` model under the same 50-epoch budget.
4. Show that occlusion-aware weighting gives an additional gain on top of two-stage tuning.
5. Finally show that the y-band inference calibration further improves the same checkpoint without retraining.

That gives you a clean additive story rather than four disconnected tricks.
