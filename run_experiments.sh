#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

PYTHON="${PYTHON:-python3}"
DEVICE="${DEVICE:-0}"
WORKERS="${WORKERS:-8}"

EVAL_BATCH="${EVAL_BATCH:-32}"
EVAL_CONF="${EVAL_CONF:-0.01}"
EVAL_IOU="${EVAL_IOU:-0.65}"
EVAL_MAX_DET="${EVAL_MAX_DET:-100000}"

MODEL_SIZE="${MODEL_SIZE:-x}"
PRETRAINED_WEIGHTS="${PRETRAINED_WEIGHTS:-yolo11x-pose.pt}"

COCO_ROOT="${COCO_ROOT:-}"
DATA_TEMPLATE="${DATA_TEMPLATE:-$REPO_DIR/ultralytics/cfg/datasets/soccernet-synloc.yaml}"
BASE_YOLO_ROOT="${BASE_YOLO_ROOT:-$REPO_DIR/datasets/soccernet-synloc-yolo-base}"
OCC_YOLO_ROOT="${OCC_YOLO_ROOT:-$REPO_DIR/datasets/soccernet-synloc-yolo-occ}"
OCC_ALPHA="${OCC_ALPHA:-2.0}"
FORCE_REBUILD_DATA="${FORCE_REBUILD_DATA:-0}"

EXP_TAG="${EXP_TAG:-fyp_bev_100ep}"
EXP_ROOT="${EXP_ROOT:-$REPO_DIR/experiments/$EXP_TAG}"
CFG_DIR="$EXP_ROOT/generated_cfg"
RESULTS_DIR="$EXP_ROOT/results"
LOG_ROOT="$EXP_ROOT/logs"
RUN_ROOT="${RUN_ROOT:-$REPO_DIR/runs/pose/$EXP_TAG}"

MAIN_IMGSZ="${MAIN_IMGSZ:-960}"
MAIN_RECT="${MAIN_RECT:-1}"
TRAIN_PATIENCE="${TRAIN_PATIENCE:-20}"

BASELINE_EPOCHS="${BASELINE_EPOCHS:-100}"
BASELINE_BATCH="${BASELINE_BATCH:-16}"
BASELINE_LR="${BASELINE_LR:-1e-4}"

HEADONLY_EPOCHS="${HEADONLY_EPOCHS:-100}"
HEADONLY_BATCH="${HEADONLY_BATCH:-16}"
HEADONLY_LR="${HEADONLY_LR:-1e-4}"
HEADONLY_FREEZE="${HEADONLY_FREEZE:-11}"

STAGE1_EPOCHS="${STAGE1_EPOCHS:-40}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-60}"
STAGE1_BATCH="${STAGE1_BATCH:-16}"
STAGE2_BATCH="${STAGE2_BATCH:-8}"
STAGE1_LR="${STAGE1_LR:-1e-3}"
STAGE2_LR="${STAGE2_LR:-1e-5}"

YBANDS_SPEC="${YBANDS_SPEC:-3}"
YBAND_MIN_GT="${YBAND_MIN_GT:-25}"
YBAND_MIN_PRED="${YBAND_MIN_PRED:-25}"
YBAND_MIN_RATIO="${YBAND_MIN_RATIO:-0.70}"
YBAND_MAX_RATIO="${YBAND_MAX_RATIO:-1.10}"
YBAND_FALLBACK_RATIO="${YBAND_FALLBACK_RATIO:-1.00}"
YBAND_SWEEP_STEPS="${YBAND_SWEEP_STEPS:-31}"

BASE_DATA_CFG="$CFG_DIR/soccernet-synloc-base.yaml"
OCC_DATA_CFG="$CFG_DIR/soccernet-synloc-occ.yaml"

EXP_SINGLE_BASE="yolo11${MODEL_SIZE}_single_full_${MAIN_IMGSZ}_rect${MAIN_RECT}_${BASELINE_EPOCHS}ep_base"
EXP_SINGLE_OCC="yolo11${MODEL_SIZE}_single_full_${MAIN_IMGSZ}_rect${MAIN_RECT}_${BASELINE_EPOCHS}ep_occ"
EXP_HEADONLY_BASE="yolo11${MODEL_SIZE}_single_head_${MAIN_IMGSZ}_rect${MAIN_RECT}_${HEADONLY_EPOCHS}ep_base"
EXP_TWOSTAGE_BASE="yolo11${MODEL_SIZE}_twostage_${MAIN_IMGSZ}_rect${MAIN_RECT}_s1e${STAGE1_EPOCHS}_s2e${STAGE2_EPOCHS}_base"
EXP_TWOSTAGE_OCC="yolo11${MODEL_SIZE}_twostage_${MAIN_IMGSZ}_rect${MAIN_RECT}_s1e${STAGE1_EPOCHS}_s2e${STAGE2_EPOCHS}_occ"

mkdir -p "$CFG_DIR" "$RESULTS_DIR" "$LOG_ROOT" "$RUN_ROOT"

timestamp() {
  date '+%F %T'
}

log() {
  printf '\n[%s] %s\n' "$(timestamp)" "$*" >&2
}

die() {
  printf '\n[ERROR] %s\n' "$*" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}

require_file() {
  [[ -f "$1" ]] || die "Missing file: $1"
}

require_dir() {
  [[ -d "$1" ]] || die "Missing directory: $1"
}

rect_args() {
  if [[ "$MAIN_RECT" == "1" ]]; then
    printf '%s\n' "--rect"
  fi
}

run_logged() {
  local log_file="$1"
  shift
  mkdir -p "$(dirname "$log_file")"
  log "Logging to $log_file"
  "$@" 2>&1 | tee "$log_file" >&2
}

sync_annotations_dir() {
  local src_root="$1"
  local dst_root="$2"
  local src_ann="$src_root/annotations"
  local dst_ann="$dst_root/annotations"
  require_dir "$src_ann"
  mkdir -p "$dst_ann"
  cp -f "$src_ann"/*.json "$dst_ann/" 2>/dev/null || true
  local copied=0
  local f
  for f in "$dst_ann"/*.json; do
    if [[ -f "$f" ]]; then
      copied=1
      break
    fi
  done
  [[ "$copied" == "1" ]] || die "No annotation json files copied from $src_ann to $dst_ann"
}

write_dataset_cfg() {
  local output_path="$1"
  local dataset_root="$2"
  local use_occ_weights="$3"
  "$PYTHON" - "$DATA_TEMPLATE" "$output_path" "$dataset_root" "$use_occ_weights" <<'PY'
from pathlib import Path
import sys

template = Path(sys.argv[1])
output = Path(sys.argv[2])
dataset_root = sys.argv[3]
use_occ = sys.argv[4].lower() == "true"

text = template.read_text()
lines = text.splitlines()
out = []
replaced_path = False
replaced_occ = False
for line in lines:
    stripped = line.strip()
    if stripped.startswith("path:"):
        out.append(f"path: {dataset_root}")
        replaced_path = True
    elif stripped.startswith("use_occ_weights:"):
        out.append(f"use_occ_weights: {'true' if use_occ else 'false'}")
        replaced_occ = True
    else:
        out.append(line)

if not replaced_path:
    out.append(f"path: {dataset_root}")
if not replaced_occ:
    out.append(f"use_occ_weights: {'true' if use_occ else 'false'}")

output.parent.mkdir(parents=True, exist_ok=True)
output.write_text("\n".join(out) + "\n")
PY
}

prepare_datasets() {
  require_cmd "$PYTHON"
  require_file "$DATA_TEMPLATE"

  if [[ -n "$COCO_ROOT" ]]; then
    require_dir "$COCO_ROOT"
    if [[ ! -d "$BASE_YOLO_ROOT/labels/train" || "$FORCE_REBUILD_DATA" == "1" ]]; then
      log "Building base YOLO dataset at $BASE_YOLO_ROOT"
      rm -rf "$BASE_YOLO_ROOT"
      run_logged "$LOG_ROOT/prepare/convert_base.log" \
        "$PYTHON" tools/convert_coco_to_yolo_pose.py \
        --coco-dir "$COCO_ROOT" \
        --output-dir "$BASE_YOLO_ROOT" \
        --splits train val test
    else
      log "Reusing base YOLO dataset at $BASE_YOLO_ROOT"
    fi

    if [[ ! -d "$OCC_YOLO_ROOT/labels/train" || "$FORCE_REBUILD_DATA" == "1" ]]; then
      log "Building occlusion-weighted YOLO dataset at $OCC_YOLO_ROOT"
      rm -rf "$OCC_YOLO_ROOT"
      run_logged "$LOG_ROOT/prepare/convert_occ.log" \
        "$PYTHON" tools/convert_coco_to_yolo_pose.py \
        --coco-dir "$COCO_ROOT" \
        --output-dir "$OCC_YOLO_ROOT" \
        --splits train val test \
        --occ-alpha "$OCC_ALPHA"
    else
      log "Reusing occlusion-weighted YOLO dataset at $OCC_YOLO_ROOT"
    fi

    log "Syncing annotations into YOLO dataset roots"
    sync_annotations_dir "$COCO_ROOT" "$BASE_YOLO_ROOT"
    sync_annotations_dir "$COCO_ROOT" "$OCC_YOLO_ROOT"
  else
    log "COCO_ROOT is empty, reusing existing YOLO-format datasets"
  fi

  require_dir "$BASE_YOLO_ROOT"
  require_dir "$OCC_YOLO_ROOT"
  require_dir "$BASE_YOLO_ROOT/annotations"
  require_dir "$OCC_YOLO_ROOT/annotations"

  write_dataset_cfg "$BASE_DATA_CFG" "$BASE_YOLO_ROOT" false
  write_dataset_cfg "$OCC_DATA_CFG" "$OCC_YOLO_ROOT" true

  log "Generated dataset configs:"
  printf '  %s\n' "$BASE_DATA_CFG" "$OCC_DATA_CFG"
}

resolve_weights_path() {
  local exp_name="$1"
  local candidates=(
    "$RUN_ROOT/$exp_name/weights/best.pt"
    "$REPO_DIR/runs/pose/$EXP_TAG/$exp_name/weights/best.pt"
    "$REPO_DIR/runs/pose/$exp_name/weights/best.pt"
  )
  local p
  for p in "${candidates[@]}"; do
    if [[ -f "$p" ]]; then
      printf '%s\n' "$p"
      return 0
    fi
  done
  die "Could not locate best.pt for experiment: $exp_name"
}

run_val_eval() {
  local exp_name="$1"
  local weights="$2"
  local data_cfg="$3"
  local save_dir="$RUN_ROOT/$exp_name/eval/baseline/val"
  local log_file="$LOG_ROOT/eval/${exp_name}_baseline_val.log"
  local extra_rect=()
  if [[ "$MAIN_RECT" == "1" ]]; then
    extra_rect+=(--rect)
  fi
  mkdir -p "$save_dir"
  run_logged "$log_file" \
    "$PYTHON" test_bev.py \
    --weights "$weights" \
    --data "$data_cfg" \
    --split val \
    --imgsz "$MAIN_IMGSZ" \
    --batch "$EVAL_BATCH" \
    --conf "$EVAL_CONF" \
    --iou "$EVAL_IOU" \
    --max-det "$EVAL_MAX_DET" \
    --workers "$WORKERS" \
    --device "$DEVICE" \
    "${extra_rect[@]}" \
    --save-dir "$save_dir"
}

run_test_eval() {
  local exp_name="$1"
  local weights="$2"
  local data_cfg="$3"
  local save_dir="$4"
  local log_file="$5"
  local extra_rect=()
  if [[ "$MAIN_RECT" == "1" ]]; then
    extra_rect+=(--rect)
  fi
  mkdir -p "$save_dir"
  run_logged "$log_file" \
    "$PYTHON" test_bev.py \
    --weights "$weights" \
    --data "$data_cfg" \
    --split test \
    --imgsz "$MAIN_IMGSZ" \
    --batch "$EVAL_BATCH" \
    --conf "$EVAL_CONF" \
    --iou "$EVAL_IOU" \
    --max-det "$EVAL_MAX_DET" \
    --workers "$WORKERS" \
    --device "$DEVICE" \
    "${extra_rect[@]}" \
    --save-dir "$save_dir"
}

run_test_eval_with_yband() {
  local exp_name="$1"
  local weights="$2"
  local data_cfg="$3"
  local save_dir="$4"
  local log_file="$5"
  local stats_json="$6"
  local extra_rect=()
  if [[ "$MAIN_RECT" == "1" ]]; then
    extra_rect+=(--rect)
  fi
  mkdir -p "$save_dir"
  run_logged "$log_file" \
    "$PYTHON" test_bev.py \
    --weights "$weights" \
    --data "$data_cfg" \
    --split test \
    --imgsz "$MAIN_IMGSZ" \
    --batch "$EVAL_BATCH" \
    --conf "$EVAL_CONF" \
    --iou "$EVAL_IOU" \
    --max-det "$EVAL_MAX_DET" \
    --workers "$WORKERS" \
    --device "$DEVICE" \
    "${extra_rect[@]}" \
    --bev-postprocess-stats "$stats_json" \
    --save-dir "$save_dir"
}

evaluate_baseline_model() {
  local exp_name="$1"
  local weights="$2"
  local data_cfg="$3"
  run_val_eval "$exp_name" "$weights" "$data_cfg"
  run_test_eval \
    "$exp_name" \
    "$weights" \
    "$data_cfg" \
    "$RUN_ROOT/$exp_name/eval/baseline/test" \
    "$LOG_ROOT/eval/${exp_name}_baseline_test.log"
}

build_yband_stats() {
  local exp_name="$1"
  local data_cfg="$2"
  local ybands="$3"
  local val_dir="$RUN_ROOT/$exp_name/eval/baseline/val"
  local pp_root="$RUN_ROOT/$exp_name/eval/postprocess/yband_${ybands}"
  local stats_json="$pp_root/bev_band_calibration.json"
  local log_file="$LOG_ROOT/postprocess/${exp_name}_yband_${ybands}_build.log"
  require_file "$val_dir/predictions.json"
  require_file "$val_dir/locsim_val_stats.json"
  mkdir -p "$pp_root"

  run_logged "$log_file" \
    "$PYTHON" tools/build_bev_postprocess_stats.py \
    --data "$data_cfg" \
    --predictions "$val_dir/predictions.json" \
    --val-stats "$val_dir/locsim_val_stats.json" \
    --output "$stats_json" \
    --y-bands "$ybands" \
    --min-gt-per-band "$YBAND_MIN_GT" \
    --min-pred-per-band "$YBAND_MIN_PRED" \
    --min-threshold-ratio "$YBAND_MIN_RATIO" \
    --max-threshold-ratio "$YBAND_MAX_RATIO" \
    --fallback-threshold-ratio "$YBAND_FALLBACK_RATIO" \
    --sweep-steps "$YBAND_SWEEP_STEPS"

  printf '%s\n' "$stats_json"
}

run_yband_ablation() {
  local exp_name="$1"
  local weights="$2"
  local data_cfg="$3"
  local ybands=()
  read -r -a ybands <<< "$YBANDS_SPEC"
  local yb
  for yb in "${ybands[@]}"; do
    local stats_json
    stats_json="$(build_yband_stats "$exp_name" "$data_cfg" "$yb")"
    run_test_eval_with_yband \
      "$exp_name" \
      "$weights" \
      "$data_cfg" \
      "$RUN_ROOT/$exp_name/eval/postprocess/yband_${yb}/test" \
      "$LOG_ROOT/eval/${exp_name}_yband_${yb}_test.log" \
      "$stats_json"
  done
}

train_single_full() {
  local exp_name="$1"
  local data_cfg="$2"
  local epochs="$3"
  local batch="$4"
  local lr="$5"
  local weights_path="$RUN_ROOT/$exp_name/weights/best.pt"
  local extra_rect=()
  if [[ "$MAIN_RECT" == "1" ]]; then
    extra_rect+=(--rect)
  fi

  if [[ -f "$weights_path" ]]; then
    log "Reusing existing weights: $weights_path"
    printf '%s\n' "$weights_path"
    return 0
  fi

  run_logged "$LOG_ROOT/train/${exp_name}.log" \
    "$PYTHON" train_bev.py \
    --single-stage \
    --no-freeze \
    --model-size "$MODEL_SIZE" \
    --pretrained "$PRETRAINED_WEIGHTS" \
    --data "$data_cfg" \
    --imgsz "$MAIN_IMGSZ" \
    --epochs "$epochs" \
    --batch "$batch" \
    --lr0 "$lr" \
    --workers "$WORKERS" \
    --device "$DEVICE" \
    --project "$RUN_ROOT" \
    --name "$exp_name" \
    --patience "$TRAIN_PATIENCE" \
    --cos-lr \
    "${extra_rect[@]}"

  resolve_weights_path "$exp_name"
}

train_single_head_only() {
  local exp_name="$1"
  local data_cfg="$2"
  local weights_path="$RUN_ROOT/$exp_name/weights/best.pt"
  local extra_rect=()
  if [[ "$MAIN_RECT" == "1" ]]; then
    extra_rect+=(--rect)
  fi

  if [[ -f "$weights_path" ]]; then
    log "Reusing existing weights: $weights_path"
    printf '%s\n' "$weights_path"
    return 0
  fi

  run_logged "$LOG_ROOT/train/${exp_name}.log" \
    "$PYTHON" train_bev.py \
    --single-stage \
    --model-size "$MODEL_SIZE" \
    --pretrained "$PRETRAINED_WEIGHTS" \
    --data "$data_cfg" \
    --imgsz "$MAIN_IMGSZ" \
    --epochs "$HEADONLY_EPOCHS" \
    --batch "$HEADONLY_BATCH" \
    --lr0 "$HEADONLY_LR" \
    --freeze "$HEADONLY_FREEZE" \
    --workers "$WORKERS" \
    --device "$DEVICE" \
    --project "$RUN_ROOT" \
    --name "$exp_name" \
    --patience "$TRAIN_PATIENCE" \
    --cos-lr \
    "${extra_rect[@]}"

  resolve_weights_path "$exp_name"
}

train_two_stage() {
  local exp_name="$1"
  local data_cfg="$2"
  local stage2_weights="$RUN_ROOT/${exp_name}-stage2/weights/best.pt"
  local extra_rect=()
  if [[ "$MAIN_RECT" == "1" ]]; then
    extra_rect+=(--rect)
  fi

  if [[ -f "$stage2_weights" ]]; then
    log "Reusing existing weights: $stage2_weights"
    printf '%s\n' "$stage2_weights"
    return 0
  fi

  run_logged "$LOG_ROOT/train/${exp_name}.log" \
    "$PYTHON" train_bev.py \
    --model-size "$MODEL_SIZE" \
    --pretrained "$PRETRAINED_WEIGHTS" \
    --data "$data_cfg" \
    --imgsz "$MAIN_IMGSZ" \
    --stage1-epochs "$STAGE1_EPOCHS" \
    --stage2-epochs "$STAGE2_EPOCHS" \
    --stage1-batch "$STAGE1_BATCH" \
    --stage2-batch "$STAGE2_BATCH" \
    --stage1-lr "$STAGE1_LR" \
    --stage2-lr "$STAGE2_LR" \
    --workers "$WORKERS" \
    --device "$DEVICE" \
    --project "$RUN_ROOT" \
    --name "$exp_name" \
    --patience "$TRAIN_PATIENCE" \
    --cos-lr \
    "${extra_rect[@]}"

  resolve_weights_path "${exp_name}-stage2"
}

run_training_ablation() {
  prepare_datasets

  local single_base_weights
  single_base_weights="$(train_single_full "$EXP_SINGLE_BASE" "$BASE_DATA_CFG" "$BASELINE_EPOCHS" "$BASELINE_BATCH" "$BASELINE_LR")"
  evaluate_baseline_model "$EXP_SINGLE_BASE" "$single_base_weights" "$BASE_DATA_CFG"

  local headonly_base_weights
  headonly_base_weights="$(train_single_head_only "$EXP_HEADONLY_BASE" "$BASE_DATA_CFG")"
  evaluate_baseline_model "$EXP_HEADONLY_BASE" "$headonly_base_weights" "$BASE_DATA_CFG"

  local twostage_base_weights
  twostage_base_weights="$(train_two_stage "$EXP_TWOSTAGE_BASE" "$BASE_DATA_CFG")"
  evaluate_baseline_model "${EXP_TWOSTAGE_BASE}-stage2" "$twostage_base_weights" "$BASE_DATA_CFG"

  local single_occ_weights
  single_occ_weights="$(train_single_full "$EXP_SINGLE_OCC" "$OCC_DATA_CFG" "$BASELINE_EPOCHS" "$BASELINE_BATCH" "$BASELINE_LR")"
  evaluate_baseline_model "$EXP_SINGLE_OCC" "$single_occ_weights" "$OCC_DATA_CFG"

  local twostage_occ_weights
  twostage_occ_weights="$(train_two_stage "$EXP_TWOSTAGE_OCC" "$OCC_DATA_CFG")"
  evaluate_baseline_model "${EXP_TWOSTAGE_OCC}-stage2" "$twostage_occ_weights" "$OCC_DATA_CFG"

  write_summaries
}

run_inference_ablation() {
  prepare_datasets
  local weights
  weights="$(resolve_weights_path "${EXP_TWOSTAGE_OCC}-stage2")"
  evaluate_baseline_model "${EXP_TWOSTAGE_OCC}-stage2" "$weights" "$OCC_DATA_CFG"
  run_yband_ablation "${EXP_TWOSTAGE_OCC}-stage2" "$weights" "$OCC_DATA_CFG"
  write_summaries
}

write_summaries() {
  log "Writing CSV summaries into $RESULTS_DIR"
  "$PYTHON" - "$RUN_ROOT" "$RESULTS_DIR" \
    "$EXP_SINGLE_BASE" "$EXP_HEADONLY_BASE" "${EXP_TWOSTAGE_BASE}-stage2" "$EXP_SINGLE_OCC" "${EXP_TWOSTAGE_OCC}-stage2" \
    "$YBANDS_SPEC" <<'PY'
import csv
import json
import sys
from pathlib import Path

run_root = Path(sys.argv[1])
results_dir = Path(sys.argv[2])
exp_single_base = sys.argv[3]
exp_headonly_base = sys.argv[4]
exp_twostage_base = sys.argv[5]
exp_single_occ = sys.argv[6]
exp_twostage_occ = sys.argv[7]
yband_spec = sys.argv[8].split()
results_dir.mkdir(parents=True, exist_ok=True)


def load_stats(directory: Path):
    candidates = [
        directory / "locsim_test_band_stats.json",
        directory / "locsim_test_stats.json",
        directory / "locsim_val_stats.json",
    ]
    bbox_candidates = [
        directory / "locsim_bbox_test_band_stats.json",
        directory / "locsim_bbox_test_stats.json",
        directory / "locsim_bbox_val_stats.json",
    ]
    stats = {}
    bbox_stats = {}
    for path in candidates:
        if path.exists():
            stats = json.loads(path.read_text()).get("stats", {})
            break
    for path in bbox_candidates:
        if path.exists():
            bbox_stats = json.loads(path.read_text()).get("stats", {})
            break
    return stats, bbox_stats


def make_row(experiment: str, directory: Path):
    stats, bbox_stats = load_stats(directory)
    if not stats:
        return None
    return {
        "experiment": experiment,
        "dir": str(directory),
        "locsim_AP": stats.get("AP"),
        "locsim_precision": stats.get("precision"),
        "locsim_recall": stats.get("recall"),
        "locsim_f1": stats.get("f1"),
        "locsim_score_threshold": stats.get("score_threshold"),
        "locsim_frame_accuracy": stats.get("frame_accuracy"),
        "bbox_locsim_AP": bbox_stats.get("AP"),
    }


training_specs = [
    ("single_full_base", run_root / exp_single_base / "eval" / "baseline" / "test"),
    ("single_headonly_base", run_root / exp_headonly_base / "eval" / "baseline" / "test"),
    ("twostage_base", run_root / exp_twostage_base / "eval" / "baseline" / "test"),
    ("single_full_occ", run_root / exp_single_occ / "eval" / "baseline" / "test"),
    ("twostage_occ", run_root / exp_twostage_occ / "eval" / "baseline" / "test"),
]

inference_specs = [
    ("baseline", run_root / exp_twostage_occ / "eval" / "baseline" / "test"),
]
for yb in yband_spec:
    inference_specs.append(
        (f"yband_{yb}", run_root / exp_twostage_occ / "eval" / "postprocess" / f"yband_{yb}" / "test")
    )

for filename, specs in (
    ("training_ablation.csv", training_specs),
    ("inference_ablation.csv", inference_specs),
):
    rows = []
    for name, directory in specs:
        row = make_row(name, directory)
        if row is not None:
            rows.append(row)
    if not rows:
        continue
    keys = list(rows[0].keys())
    with (results_dir / filename).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
PY
}

usage() {
  cat <<EOF
Usage:
  bash run_fyp_cvpr_experiments.sh prepare
  bash run_fyp_cvpr_experiments.sh train
  bash run_fyp_cvpr_experiments.sh inference
  bash run_fyp_cvpr_experiments.sh summarize

Default 100-epoch study:
  - Training ablation:
    1. single-stage full fine-tune, base loss
    2. single-stage head-only, base loss
    3. two-stage, base loss
    4. single-stage full fine-tune, occlusion-aware loss
    5. two-stage, occlusion-aware loss
  - Inference ablation:
    6. best two-stage + occlusion-aware model, baseline inference
    7. same model + y-band inference

Recommended setup:
  export COCO_ROOT=/path/to/coco_style_dataset
  export PRETRAINED_WEIGHTS=/path/to/yolo11x-pose.pt
  bash run_fyp_cvpr_experiments.sh train
  bash run_fyp_cvpr_experiments.sh inference

Outputs:
  - Dataset YAMLs:  $CFG_DIR
  - Logs:           $LOG_ROOT
  - Runs:           $RUN_ROOT
  - CSV summaries:  $RESULTS_DIR
EOF
}

main() {
  require_cmd "$PYTHON"
  local mode="${1:-}"
  case "$mode" in
    prepare)
      prepare_datasets
      ;;
    train)
      run_training_ablation
      ;;
    inference)
      run_inference_ablation
      ;;
    summarize)
      write_summaries
      ;;
    *)
      usage
      ;;
  esac
}

main "${1:-}"
