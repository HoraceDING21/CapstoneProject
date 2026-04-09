#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

PYTHON="${PYTHON:-python3}"
DEVICE="${DEVICE:-0}"
WORKERS="${WORKERS:-8}"
EVAL_BATCH="${EVAL_BATCH:-64}"
EVAL_CONF="${EVAL_CONF:-0.001}"
EVAL_IOU="${EVAL_IOU:-0.7}"

MODEL_SIZE="${MODEL_SIZE:-x}"
PRETRAINED_X="${PRETRAINED_X:-yolo11x-pose.pt}"

COCO_ROOT="${COCO_ROOT:-}"
DATA_TEMPLATE="${DATA_TEMPLATE:-$REPO_DIR/ultralytics/cfg/datasets/soccernet-synloc.yaml}"
BASE_YOLO_ROOT="${BASE_YOLO_ROOT:-$REPO_DIR/datasets/soccernet-synloc-yolo-base}"
OCC_YOLO_ROOT="${OCC_YOLO_ROOT:-$REPO_DIR/datasets/soccernet-synloc-yolo-occ}"
OCC_ALPHA="${OCC_ALPHA:-2.0}"
FORCE_REBUILD_DATA="${FORCE_REBUILD_DATA:-0}"

EXP_ROOT="${EXP_ROOT:-$REPO_DIR/experiments/fyp_cvpr_50ep_960}"
CFG_DIR="$EXP_ROOT/generated_cfg"
RESULTS_DIR="$EXP_ROOT/results"
RUN_ROOT="${RUN_ROOT:-$REPO_DIR/runs/pose/fyp_cvpr_50ep_960/main_study}"

MAIN_IMGSZ="${MAIN_IMGSZ:-960}"
MAIN_RECT="${MAIN_RECT:-1}"

SINGLE_FAST_EPOCHS="${SINGLE_FAST_EPOCHS:-50}"
SINGLE_FAST_BATCH="${SINGLE_FAST_BATCH:-16}"
SINGLE_FAST_LR="${SINGLE_FAST_LR:-1e-4}"

STAGE1_EPOCHS="${STAGE1_EPOCHS:-30}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-20}"
STAGE1_BATCH="${STAGE1_BATCH:-16}"
STAGE2_BATCH="${STAGE2_BATCH:-8}"
STAGE1_LR="${STAGE1_LR:-1e-3}"
STAGE2_LR="${STAGE2_LR:-1e-5}"
TRAIN_PATIENCE="${TRAIN_PATIENCE:-15}"

YBANDS_SPEC="${YBANDS_SPEC:-2 3 4}"
YBAND_MIN_GT="${YBAND_MIN_GT:-5}"
YBAND_MIN_PRED="${YBAND_MIN_PRED:-5}"
YBAND_MIN_RATIO="${YBAND_MIN_RATIO:-0.70}"
YBAND_MAX_RATIO="${YBAND_MAX_RATIO:-1.10}"
YBAND_FALLBACK_RATIO="${YBAND_FALLBACK_RATIO:-1.00}"
YBAND_RESCUE_RATIO="${YBAND_RESCUE_RATIO:-0.90}"
YBAND_SWEEP_STEPS="${YBAND_SWEEP_STEPS:-10}"

EXISTING_OCC_STAGE2_WEIGHTS="${EXISTING_OCC_STAGE2_WEIGHTS:-}"

BASE_DATA_CFG="$CFG_DIR/soccernet-synloc-base.yaml"
OCC_DATA_CFG="$CFG_DIR/soccernet-synloc-occ.yaml"

FAST_SINGLE_NAME="yolo11x_single_960_rect_50ep"
FAST_TWO_STAGE_BASE_NAME="yolo11x_twostage_960_rect_30p20_base"
FAST_TWO_STAGE_OCC_NAME="yolo11x_twostage_960_rect_30p20_occ"

mkdir -p "$CFG_DIR" "$RESULTS_DIR" "$RUN_ROOT"

log() {
  printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"
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
      "$PYTHON" tools/convert_coco_to_yolo_pose.py \
        --coco-dir "$COCO_ROOT" \
        --output-dir "$BASE_YOLO_ROOT" \
        --splits train val test
    else
      log "Reusing existing base YOLO dataset at $BASE_YOLO_ROOT"
    fi

    if [[ ! -d "$OCC_YOLO_ROOT/labels/train" || "$FORCE_REBUILD_DATA" == "1" ]]; then
      log "Building occlusion-weighted YOLO dataset at $OCC_YOLO_ROOT"
      rm -rf "$OCC_YOLO_ROOT"
      "$PYTHON" tools/convert_coco_to_yolo_pose.py \
        --coco-dir "$COCO_ROOT" \
        --output-dir "$OCC_YOLO_ROOT" \
        --splits train val test \
        --occ-alpha "$OCC_ALPHA"
    else
      log "Reusing existing occlusion-weighted YOLO dataset at $OCC_YOLO_ROOT"
    fi

    log "Syncing COCO annotations into YOLO dataset roots"
    sync_annotations_dir "$COCO_ROOT" "$BASE_YOLO_ROOT"
    sync_annotations_dir "$COCO_ROOT" "$OCC_YOLO_ROOT"
  else
    log "COCO_ROOT is empty, skipping conversion and reusing existing YOLO-format datasets"
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

copy_val_stats() {
  local val_dir="$1"
  local dst_dir="$2"
  mkdir -p "$dst_dir"
  local copied=0
  local f
  for f in "$val_dir"/locsim*_val_stats.json; do
    if [[ -f "$f" ]]; then
      cp "$f" "$dst_dir/"
      copied=1
    fi
  done
  [[ "$copied" == "1" ]] || die "No val stats found in $val_dir"
}

resolve_weights_path() {
  local exp_name="$1"
  local candidates=(
    "$RUN_ROOT/$exp_name/weights/best.pt"
    "$RUN_ROOT/pose/$exp_name/weights/best.pt"
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
  local weights="$1"
  local data_cfg="$2"
  local imgsz="$3"
  local rect_flag="$4"
  local save_dir="$5"
  local rect_args=()
  if [[ "$rect_flag" == "1" ]]; then
    rect_args+=(--rect)
  fi
  mkdir -p "$save_dir"
  log "VAL -> $save_dir"
  "$PYTHON" test_bev.py \
    --weights "$weights" \
    --data "$data_cfg" \
    --split val \
    --imgsz "$imgsz" \
    --batch "$EVAL_BATCH" \
    --conf "$EVAL_CONF" \
    --iou "$EVAL_IOU" \
    --workers "$WORKERS" \
    --device "$DEVICE" \
    "${rect_args[@]}" \
    --save-dir "$save_dir"
}

run_test_eval() {
  local weights="$1"
  local data_cfg="$2"
  local imgsz="$3"
  local rect_flag="$4"
  local val_dir="$5"
  local save_dir="$6"
  local bev_stats="${7:-}"
  local rect_args=()
  if [[ "$rect_flag" == "1" ]]; then
    rect_args+=(--rect)
  fi
  mkdir -p "$save_dir"
  copy_val_stats "$val_dir" "$save_dir"
  log "TEST -> $save_dir"
  if [[ -n "$bev_stats" ]]; then
    "$PYTHON" test_bev.py \
      --weights "$weights" \
      --data "$data_cfg" \
      --split test \
      --imgsz "$imgsz" \
      --batch "$EVAL_BATCH" \
      --conf "$EVAL_CONF" \
      --iou "$EVAL_IOU" \
      --workers "$WORKERS" \
      --device "$DEVICE" \
      "${rect_args[@]}" \
      --bev-postprocess-stats "$bev_stats" \
      --save-dir "$save_dir"
  else
    "$PYTHON" test_bev.py \
      --weights "$weights" \
      --data "$data_cfg" \
      --split test \
      --imgsz "$imgsz" \
      --batch "$EVAL_BATCH" \
      --conf "$EVAL_CONF" \
      --iou "$EVAL_IOU" \
      --workers "$WORKERS" \
      --device "$DEVICE" \
      "${rect_args[@]}" \
      --save-dir "$save_dir"
  fi
}

evaluate_baseline_model() {
  local exp_name="$1"
  local weights="$2"
  local data_cfg="$3"
  local imgsz="$4"
  local rect_flag="$5"
  local val_dir="$RUN_ROOT/$exp_name/eval/baseline/val"
  local test_dir="$RUN_ROOT/$exp_name/eval/baseline/test"
  run_val_eval "$weights" "$data_cfg" "$imgsz" "$rect_flag" "$val_dir"
  run_test_eval "$weights" "$data_cfg" "$imgsz" "$rect_flag" "$val_dir" "$test_dir"
}

run_yband_ablation() {
  local exp_name="$1"
  local weights="$2"
  local data_cfg="$3"
  local imgsz="$4"
  local rect_flag="$5"
  local baseline_val_dir="$RUN_ROOT/$exp_name/eval/baseline/val"
  require_file "$baseline_val_dir/predictions.json"
  require_file "$baseline_val_dir/locsim_val_stats.json"

  local ybands=()
  read -r -a ybands <<< "$YBANDS_SPEC"

  local yb
  for yb in "${ybands[@]}"; do
    local tag="yband_${yb}"
    local pp_root="$RUN_ROOT/$exp_name/eval/postprocess/$tag"
    local stats_json="$pp_root/bev_yband_stats.json"
    local test_dir="$pp_root/test"
    mkdir -p "$pp_root"

    log "Building y-band stats ($tag)"
    "$PYTHON" tools/build_bev_postprocess_stats.py \
      --data "$data_cfg" \
      --predictions "$baseline_val_dir/predictions.json" \
      --val-stats "$baseline_val_dir/locsim_val_stats.json" \
      --output "$stats_json" \
      --y-bands "$yb" \
      --min-gt-per-band "$YBAND_MIN_GT" \
      --min-pred-per-band "$YBAND_MIN_PRED" \
      --min-threshold-ratio "$YBAND_MIN_RATIO" \
      --max-threshold-ratio "$YBAND_MAX_RATIO" \
      --fallback-threshold-ratio "$YBAND_FALLBACK_RATIO" \
      --rescue-keypoint-ratio "$YBAND_RESCUE_RATIO" \
      --sweep-steps "$YBAND_SWEEP_STEPS"

    run_test_eval "$weights" "$data_cfg" "$imgsz" "$rect_flag" "$baseline_val_dir" "$test_dir" "$stats_json"
  done
}

train_single_stage_fast() {
  prepare_datasets
  local exp_name="$FAST_SINGLE_NAME"
  local rect_args=()
  if [[ "$MAIN_RECT" == "1" ]]; then
    rect_args+=(--rect)
  fi

  log "Training $exp_name"
  "$PYTHON" train_bev.py \
    --single-stage \
    --model-size "$MODEL_SIZE" \
    --pretrained "$PRETRAINED_X" \
    --data "$BASE_DATA_CFG" \
    --imgsz "$MAIN_IMGSZ" \
    --epochs "$SINGLE_FAST_EPOCHS" \
    --batch "$SINGLE_FAST_BATCH" \
    --lr0 "$SINGLE_FAST_LR" \
    --freeze 11 \
    --workers "$WORKERS" \
    --device "$DEVICE" \
    --project "$RUN_ROOT" \
    --name "$exp_name" \
    --patience "$TRAIN_PATIENCE" \
    --cos-lr \
    "${rect_args[@]}"

  local weights
  weights="$(resolve_weights_path "$exp_name")"
  evaluate_baseline_model "$exp_name" "$weights" "$BASE_DATA_CFG" "$MAIN_IMGSZ" "$MAIN_RECT"
}

train_two_stage_base() {
  prepare_datasets
  local exp_name="$FAST_TWO_STAGE_BASE_NAME"
  local rect_args=()
  if [[ "$MAIN_RECT" == "1" ]]; then
    rect_args+=(--rect)
  fi

  log "Training $exp_name"
  "$PYTHON" train_bev.py \
    --model-size "$MODEL_SIZE" \
    --pretrained "$PRETRAINED_X" \
    --data "$BASE_DATA_CFG" \
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
    "${rect_args[@]}"

  local weights
  weights="$(resolve_weights_path "$exp_name-stage2")"
  evaluate_baseline_model "$exp_name" "$weights" "$BASE_DATA_CFG" "$MAIN_IMGSZ" "$MAIN_RECT"
}

train_or_reuse_two_stage_occ() {
  prepare_datasets
  local exp_name="$FAST_TWO_STAGE_OCC_NAME"
  local weights=""
  local rect_args=()
  if [[ "$MAIN_RECT" == "1" ]]; then
    rect_args+=(--rect)
  fi

  if [[ -n "$EXISTING_OCC_STAGE2_WEIGHTS" ]]; then
    require_file "$EXISTING_OCC_STAGE2_WEIGHTS"
    weights="$EXISTING_OCC_STAGE2_WEIGHTS"
    log "Reusing existing occlusion-loss weights: $weights"
  else
    log "Training $exp_name"
    "$PYTHON" train_bev.py \
      --model-size "$MODEL_SIZE" \
      --pretrained "$PRETRAINED_X" \
      --data "$OCC_DATA_CFG" \
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
      "${rect_args[@]}"
    weights="$(resolve_weights_path "$exp_name-stage2")"
  fi

  evaluate_baseline_model "$exp_name" "$weights" "$OCC_DATA_CFG" "$MAIN_IMGSZ" "$MAIN_RECT"
  run_yband_ablation "$exp_name" "$weights" "$OCC_DATA_CFG" "$MAIN_IMGSZ" "$MAIN_RECT"
}

run_fast_protocol() {
  prepare_datasets
  train_single_stage_fast
  train_two_stage_base
  train_or_reuse_two_stage_occ
  write_summaries
}

run_only_inference() {
  prepare_datasets
  local weights=""
  if [[ -n "$EXISTING_OCC_STAGE2_WEIGHTS" ]]; then
    require_file "$EXISTING_OCC_STAGE2_WEIGHTS"
    weights="$EXISTING_OCC_STAGE2_WEIGHTS"
  else
    weights="$(resolve_weights_path "$FAST_TWO_STAGE_OCC_NAME-stage2")"
  fi
  evaluate_baseline_model "$FAST_TWO_STAGE_OCC_NAME" "$weights" "$OCC_DATA_CFG" "$MAIN_IMGSZ" "$MAIN_RECT"
  run_yband_ablation "$FAST_TWO_STAGE_OCC_NAME" "$weights" "$OCC_DATA_CFG" "$MAIN_IMGSZ" "$MAIN_RECT"
  write_summaries
}

write_summaries() {
  log "Writing CSV summaries into $RESULTS_DIR"
  "$PYTHON" - "$RUN_ROOT" "$RESULTS_DIR" <<'PY'
import csv
import json
import sys
from pathlib import Path

run_root = Path(sys.argv[1])
results_dir = Path(sys.argv[2])
results_dir.mkdir(parents=True, exist_ok=True)


def load_stats(directory: Path):
    candidates = [
        directory / "locsim_test_postprocess_stats.json",
        directory / "locsim_test_filtered_stats.json",
        directory / "locsim_test_stats.json",
        directory / "locsim_val_stats.json",
    ]
    bbox_candidates = [
        directory / "locsim_bbox_test_postprocess_stats.json",
        directory / "locsim_bbox_test_filtered_stats.json",
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


training_rows = []
training_specs = [
    ("single_stage_50ep", run_root / "yolo11x_single_960_rect_50ep" / "eval" / "baseline" / "test"),
    ("two_stage_30p20", run_root / "yolo11x_twostage_960_rect_30p20_base" / "eval" / "baseline" / "test"),
    ("two_stage_30p20_occ", run_root / "yolo11x_twostage_960_rect_30p20_occ" / "eval" / "baseline" / "test"),
]
for name, directory in training_specs:
    if not directory.exists():
        continue
    stats, bbox_stats = load_stats(directory)
    if not stats:
        continue
    training_rows.append(
        {
            "experiment": name,
            "dir": str(directory),
            "locsim_AP": stats.get("AP"),
            "locsim_precision": stats.get("precision"),
            "locsim_recall": stats.get("recall"),
            "locsim_f1": stats.get("f1"),
            "locsim_score_threshold": stats.get("score_threshold"),
            "locsim_frame_accuracy": stats.get("frame_accuracy"),
            "bbox_locsim_AP": bbox_stats.get("AP"),
        }
    )

inference_rows = []
inference_specs = [
    ("baseline", run_root / "yolo11x_twostage_960_rect_30p20_occ" / "eval" / "baseline" / "test"),
    ("yband_2", run_root / "yolo11x_twostage_960_rect_30p20_occ" / "eval" / "postprocess" / "yband_2" / "test"),
    ("yband_3", run_root / "yolo11x_twostage_960_rect_30p20_occ" / "eval" / "postprocess" / "yband_3" / "test"),
    ("yband_4", run_root / "yolo11x_twostage_960_rect_30p20_occ" / "eval" / "postprocess" / "yband_4" / "test"),
]
for name, directory in inference_specs:
    if not directory.exists():
        continue
    stats, bbox_stats = load_stats(directory)
    if not stats:
        continue
    inference_rows.append(
        {
            "experiment": name,
            "dir": str(directory),
            "locsim_AP": stats.get("AP"),
            "locsim_precision": stats.get("precision"),
            "locsim_recall": stats.get("recall"),
            "locsim_f1": stats.get("f1"),
            "locsim_score_threshold": stats.get("score_threshold"),
            "locsim_frame_accuracy": stats.get("frame_accuracy"),
            "bbox_locsim_AP": bbox_stats.get("AP"),
        }
    )

for filename, rows in (
    ("training_ablation.csv", training_rows),
    ("inference_ablation.csv", inference_rows),
):
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
  bash run_fyp_cvpr_experiments.sh fast
  bash run_fyp_cvpr_experiments.sh inference
  bash run_fyp_cvpr_experiments.sh summarize

Recommended:
  1) export COCO_ROOT=/path/to/your/coco_style_dataset
  2) export PRETRAINED_X=/path/to/yolo11x-pose.pt
  3) bash run_fyp_cvpr_experiments.sh fast

Optional reuse:
  export EXISTING_OCC_STAGE2_WEIGHTS=/path/to/your/already_trained/best.pt

Outputs:
  - Generated dataset YAMLs: $CFG_DIR
  - Training/eval runs:       $RUN_ROOT
  - CSV summaries:            $RESULTS_DIR
EOF
}

main() {
  require_cmd "$PYTHON"
  local mode="${1:-}"
  case "$mode" in
    prepare)
      prepare_datasets
      ;;
    fast)
      run_fast_protocol
      ;;
    inference)
      run_only_inference
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
