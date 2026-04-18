# BEV Demo

Local Flask demo for single-image BEV pose visualization.

## What it does

- Select one sample image from your local SynLoc-style dataset.
- Run the fine-tuned YOLO BEV pose model on that image.
- Visualize:
  - thin player bounding boxes
  - `pelvis` and `pelvis_ground` keypoints
  - stable player ids shared across both views
- Show per-player score and coordinates in a side table.
- Render a top-down pitch view with matching player ids.

## Setup

1. Copy the example config:

```bash
cp bev_demo/demo_config.example.yaml bev_demo/demo_config.yaml
```

2. Edit `bev_demo/demo_config.yaml`:

- set `weights` to your local `.pt`
- set `images_dir`
- set `annotations_json`
- optionally set `bev_postprocess_stats`

3. Start the demo:

```bash
python -m bev_demo.app
```

4. Open:

```text
http://127.0.0.1:5000
```

## Notes

- This demo is intended for local presentation only.
- It does not store user data or require a database.
- The world-coordinate view currently uses a per-sample homography fitted from the sample annotation pairs:
  `pelvis_ground (image)` -> `position_on_pitch (world)`.
- If later you confirm the exact SynLoc camera projection definition, that part can be replaced with a stricter camera-geometry module.
