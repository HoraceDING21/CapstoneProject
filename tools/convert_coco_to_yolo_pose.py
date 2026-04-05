"""Convert COCO-style pose annotations (as used in mmpose/sskit SoccerNet) to YOLO pose format.

Usage:
    python tools/convert_coco_to_yolo_pose.py \
        --coco-dir /path/to/soccernet-dataset \
        --output-dir /path/to/soccernet-synloc-yolo \
        --splits train val test

The COCO dataset is expected to have:
    coco-dir/
    ├── annotations/
    │   ├── train.json
    │   ├── val.json
    │   └── test.json
    └── train/  val/  test/  (image directories)

Output YOLO format:
    output-dir/
    ├── images/
    │   ├── train/  (symlinks or copies)
    │   ├── val/
    │   └── test/
    └── labels/
        ├── train/  (.txt files, one per image)
        ├── val/
        └── test/
"""

import argparse
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path


def convert_coco_to_yolo_pose(coco_dir: str, output_dir: str, splits: list, num_keypoints: int = 2, copy_images: bool = False):
    coco_dir = Path(coco_dir)
    output_dir = Path(output_dir)

    for split in splits:
        ann_file = coco_dir / "annotations" / f"{split}.json"
        if not ann_file.exists():
            print(f"Warning: {ann_file} not found, skipping split '{split}'")
            continue

        with open(ann_file) as f:
            coco = json.load(f)

        images = {img["id"]: img for img in coco["images"]}
        ann_by_image = defaultdict(list)
        for ann in coco["annotations"]:
            ann_by_image[ann["image_id"]].append(ann)

        img_out_dir = output_dir / "images" / split
        lbl_out_dir = output_dir / "labels" / split
        img_out_dir.mkdir(parents=True, exist_ok=True)
        lbl_out_dir.mkdir(parents=True, exist_ok=True)

        img_src_dir = coco_dir / split

        for img_id, img_info in images.items():
            img_w = img_info["width"]
            img_h = img_info["height"]
            file_name = img_info["file_name"]

            src_img = img_src_dir / file_name
            dst_img = img_out_dir / file_name

            if src_img.exists() and not dst_img.exists():
                if copy_images:
                    shutil.copy2(src_img, dst_img)
                else:
                    os.symlink(src_img.resolve(), dst_img)

            label_name = Path(file_name).stem + ".txt"
            label_path = lbl_out_dir / label_name

            anns = ann_by_image.get(img_id, [])
            lines = []
            for ann in anns:
                if ann.get("iscrowd", 0):
                    continue

                cat_id = 0  # single class (person)

                x, y, w, h = ann["bbox"]
                x_center = (x + w / 2) / img_w
                y_center = (y + h / 2) / img_h
                w_norm = w / img_w
                h_norm = h / img_h

                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                w_norm = max(0, min(1, w_norm))
                h_norm = max(0, min(1, h_norm))

                raw_kpts = ann.get("keypoints", [])

                # Normalise to a list of (x, y, v) tuples regardless of storage format:
                #   - Nested format (sskit/SoccerNet): [[x0,y0,v0], [x1,y1,v1], ...]
                #   - Flat format (standard COCO):      [x0, y0, v0, x1, y1, v1, ...]
                if raw_kpts and isinstance(raw_kpts[0], (list, tuple)):
                    kpt_triples = [tuple(kp) for kp in raw_kpts]
                else:
                    kpt_triples = [
                        (raw_kpts[i], raw_kpts[i + 1], raw_kpts[i + 2])
                        for i in range(0, len(raw_kpts) - 2, 3)
                    ]

                kpt_values = []
                for ki in range(min(nk, num_keypoints)):
                    kx = keypoints[ki * 3] / img_w
                    ky = keypoints[ki * 3 + 1] / img_h
                    kv = keypoints[ki * 3 + 2]  # visibility: 0=not labeled, 1=labeled not visible, 2=visible
                    kpt_values.extend([kx, ky, kv])

                while len(kpt_values) < num_keypoints * 3:
                    kpt_values.extend([0.0, 0.0, 0])

                parts = [str(cat_id)]
                parts.extend([f"{v:.6f}" for v in [x_center, y_center, w_norm, h_norm]])
                parts.extend([f"{v:.6f}" if isinstance(v, float) else str(int(v)) for v in kpt_values])
                lines.append(" ".join(parts))

            with open(label_path, "w") as f:
                f.write("\n".join(lines))
                if lines:
                    f.write("\n")

        print(f"Converted {split}: {len(images)} images, {sum(len(v) for v in ann_by_image.values())} annotations")

    print(f"\nDone! YOLO dataset saved to: {output_dir}")
    print(f"Update 'path' in soccernet-synloc.yaml to: {output_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert COCO pose annotations to YOLO pose format")
    parser.add_argument("--coco-dir", type=str, required=True, help="Path to COCO-style dataset root")
    parser.add_argument("--output-dir", type=str, required=True, help="Path to output YOLO-format dataset")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], help="Dataset splits to convert")
    parser.add_argument("--num-keypoints", type=int, default=2, help="Number of keypoints per annotation")
    parser.add_argument("--copy-images", action="store_true", help="Copy images instead of creating symlinks")
    args = parser.parse_args()

    convert_coco_to_yolo_pose(args.coco_dir, args.output_dir, args.splits, args.num_keypoints, args.copy_images)
