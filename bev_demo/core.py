from __future__ import annotations

import base64
import json
import os
from io import BytesIO
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont


DEFAULT_FIELD_LENGTH_M = 105.0
DEFAULT_FIELD_WIDTH_M = 68.0
DEFAULT_TITLE = "BEV Pose Demo"
DEFAULT_CONFIG_FILENAMES = (
    "demo_config.yaml",
    "demo_config.yml",
)
PLAYER_COLORS_BGR = (
    (34, 87, 255),
    (0, 159, 230),
    (38, 191, 136),
    (0, 200, 255),
    (103, 80, 255),
    (0, 120, 255),
    (52, 168, 83),
    (17, 138, 178),
    (255, 140, 66),
    (94, 92, 230),
    (0, 180, 216),
    (245, 158, 11),
)


@dataclass
class SampleSourceConfig:
    name: str
    images_dir: Path
    annotations_json: Path | None = None
    glob: str = "*.jpg"
    recursive: bool = False
    limit: int | None = None


@dataclass
class DemoConfig:
    title: str = DEFAULT_TITLE
    weights: Path | None = None
    device: str | None = None
    imgsz: int = 960
    conf: float = 0.01
    iou: float = 0.65
    max_det: int = 300
    bev_postprocess_stats: Path | None = None
    field_length_m: float = DEFAULT_FIELD_LENGTH_M
    field_width_m: float = DEFAULT_FIELD_WIDTH_M
    sample_sources: list[SampleSourceConfig] = field(default_factory=list)
    config_path: Path | None = None
    errors: list[str] = field(default_factory=list)


@dataclass
class AnnotationIndex:
    path: Path
    images_by_id: dict[int, dict[str, Any]]
    annotations_by_image_id: dict[int, list[dict[str, Any]]]

    @classmethod
    def from_file(cls, path: Path) -> AnnotationIndex:
        with path.open() as f:
            payload = json.load(f)

        images = payload.get("images", [])
        annotations = payload.get("annotations", [])
        images_by_id = {int(image["id"]): image for image in images}
        annotations_by_image_id: dict[int, list[dict[str, Any]]] = {}
        for annotation in annotations:
            image_id = int(annotation["image_id"])
            annotations_by_image_id.setdefault(image_id, []).append(annotation)

        return cls(path=path, images_by_id=images_by_id, annotations_by_image_id=annotations_by_image_id)


@dataclass
class SampleEntry:
    id: str
    source_name: str
    display_name: str
    image_path: Path
    image_record: dict[str, Any] | None = None
    annotations: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class SampleCatalog:
    samples: list[SampleEntry]
    errors: list[str] = field(default_factory=list)

    def first(self) -> SampleEntry | None:
        return self.samples[0] if self.samples else None

    def get(self, sample_id: str | None) -> SampleEntry | None:
        if not sample_id:
            return None
        for sample in self.samples:
            if sample.id == sample_id:
                return sample
        return None


@dataclass
class PlayerDetection:
    player_id: int
    score: float
    bbox_xyxy: tuple[float, float, float, float]
    pelvis_xy: tuple[float, float]
    pelvis_ground_xy: tuple[float, float]
    world_xy: tuple[float, float] | None
    color_bgr: tuple[int, int, int]
    color_hex: str


@dataclass
class DemoRunResult:
    annotated_image_data_uri: str
    pitch_image_data_uri: str
    players: list[PlayerDetection]
    projection_note: str
    prediction_count: int


@dataclass
class RuntimeContext:
    config: DemoConfig
    catalog: SampleCatalog
    service: BEVDemoService


def _resolve_candidate_path(path_value: str | os.PathLike[str] | None, base_dir: Path) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def resolve_config_path() -> Path | None:
    env_path = os.getenv("BEV_DEMO_CONFIG")
    if env_path:
        candidate = Path(env_path).expanduser()
        if candidate.exists():
            return candidate.resolve()

    demo_dir = Path(__file__).resolve().parent
    for filename in DEFAULT_CONFIG_FILENAMES:
        candidate = demo_dir / filename
        if candidate.exists():
            return candidate.resolve()
    return None


def load_demo_config() -> DemoConfig:
    config_path = resolve_config_path()
    if config_path is None:
        demo_dir = Path(__file__).resolve().parent
        return DemoConfig(
            config_path=demo_dir / DEFAULT_CONFIG_FILENAMES[0],
            errors=[
                (
                    "Config file not found. Copy "
                    f"{(demo_dir / 'demo_config.example.yaml').name} to "
                    f"{DEFAULT_CONFIG_FILENAMES[0]} and update the local dataset and weights paths."
                )
            ],
        )

    with config_path.open() as f:
        payload = yaml.safe_load(f) or {}

    base_dir = config_path.parent
    config = DemoConfig(
        title=str(payload.get("title") or DEFAULT_TITLE),
        weights=_resolve_candidate_path(payload.get("weights"), base_dir),
        device=str(payload.get("device")).strip() if payload.get("device") is not None else None,
        imgsz=int(payload.get("imgsz", 960)),
        conf=float(payload.get("conf", 0.01)),
        iou=float(payload.get("iou", 0.65)),
        max_det=int(payload.get("max_det", 300)),
        bev_postprocess_stats=_resolve_candidate_path(payload.get("bev_postprocess_stats"), base_dir),
        field_length_m=float(payload.get("field_length_m", DEFAULT_FIELD_LENGTH_M)),
        field_width_m=float(payload.get("field_width_m", DEFAULT_FIELD_WIDTH_M)),
        config_path=config_path,
    )

    for idx, item in enumerate(payload.get("sample_sources", []) or []):
        if not item:
            config.errors.append(f"sample_sources[{idx}] is empty.")
            continue
        images_dir = _resolve_candidate_path(item.get("images_dir"), base_dir)
        annotations_json = _resolve_candidate_path(item.get("annotations_json"), base_dir)
        if images_dir is None:
            config.errors.append(f"sample_sources[{idx}] is missing images_dir.")
            continue
        config.sample_sources.append(
            SampleSourceConfig(
                name=str(item.get("name") or images_dir.name),
                images_dir=images_dir,
                annotations_json=annotations_json,
                glob=str(item.get("glob") or "*.jpg"),
                recursive=bool(item.get("recursive", False)),
                limit=int(item["limit"]) if item.get("limit") is not None else None,
            )
        )

    if config.weights is None:
        config.errors.append("weights is not configured.")
    elif not config.weights.exists():
        config.errors.append(f"Weights file not found: {config.weights}")

    if config.bev_postprocess_stats and not config.bev_postprocess_stats.exists():
        config.errors.append(
            f"BEV postprocess stats file not found: {config.bev_postprocess_stats}. Band filtering will be disabled."
        )

    if not config.sample_sources:
        config.errors.append("No sample_sources configured.")

    return config


def _resolve_image_path(images_dir: Path, file_name: str) -> Path | None:
    candidates = [images_dir / file_name, images_dir / Path(file_name).name]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def build_sample_catalog(config: DemoConfig) -> SampleCatalog:
    samples: list[SampleEntry] = []
    errors = list(config.errors)

    for source_idx, source in enumerate(config.sample_sources):
        if not source.images_dir.exists():
            errors.append(f"Sample images directory not found: {source.images_dir}")
            continue

        if source.annotations_json:
            if not source.annotations_json.exists():
                errors.append(f"Annotations JSON not found: {source.annotations_json}")
                continue

            try:
                annotation_index = AnnotationIndex.from_file(source.annotations_json)
            except Exception as exc:
                errors.append(f"Failed to read {source.annotations_json}: {exc}")
                continue

            image_records = sorted(
                annotation_index.images_by_id.values(),
                key=lambda record: str(record.get("file_name", "")),
            )
            count = 0
            for image_record in image_records:
                image_path = _resolve_image_path(source.images_dir, str(image_record.get("file_name", "")))
                if image_path is None:
                    continue
                image_id = int(image_record["id"])
                samples.append(
                    SampleEntry(
                        id=f"s{source_idx}-i{image_id}",
                        source_name=source.name,
                        display_name=f"{source.name} / {Path(image_path).name}",
                        image_path=image_path,
                        image_record=image_record,
                        annotations=annotation_index.annotations_by_image_id.get(image_id, []),
                    )
                )
                count += 1
                if source.limit is not None and count >= source.limit:
                    break

            if count == 0:
                errors.append(
                    "No sample images were resolved for "
                    f"{source.name}. Check images_dir and annotations_json."
                )
            continue

        iterator = source.images_dir.rglob(source.glob) if source.recursive else source.images_dir.glob(source.glob)
        image_paths = sorted(path.resolve() for path in iterator if path.is_file())
        if source.limit is not None:
            image_paths = image_paths[: source.limit]
        if not image_paths:
            errors.append(f"No sample images found in {source.images_dir} with pattern {source.glob}")
            continue

        for image_idx, image_path in enumerate(image_paths):
            samples.append(
                SampleEntry(
                    id=f"s{source_idx}-f{image_idx}",
                    source_name=source.name,
                    display_name=f"{source.name} / {image_path.name}",
                    image_path=image_path,
                )
            )

    samples.sort(key=lambda sample: sample.display_name.lower())
    return SampleCatalog(samples=samples, errors=errors)


def _normalize_keypoints(raw_keypoints: Any) -> list[tuple[float, float, float]]:
    if not raw_keypoints:
        return []
    if isinstance(raw_keypoints[0], (list, tuple)):
        output = []
        for kp in raw_keypoints:
            if len(kp) >= 3:
                output.append((float(kp[0]), float(kp[1]), float(kp[2])))
        return output

    output = []
    values = list(raw_keypoints)
    for idx in range(0, len(values) - 2, 3):
        output.append((float(values[idx]), float(values[idx + 1]), float(values[idx + 2])))
    return output


def _color_for_player(player_id: int) -> tuple[int, int, int]:
    return PLAYER_COLORS_BGR[(player_id - 1) % len(PLAYER_COLORS_BGR)]


def _bgr_to_rgb(color: tuple[int, int, int]) -> tuple[int, int, int]:
    b, g, r = color
    return r, g, b


def _bgr_to_hex(color: tuple[int, int, int]) -> str:
    b, g, r = color
    return f"#{r:02x}{g:02x}{b:02x}"


def _encode_image_data_uri(image: Image.Image, suffix: str = ".jpg") -> str:
    ext = ".png" if suffix.lower() == ".png" else ".jpg"
    mime = "image/png" if ext == ".png" else "image/jpeg"
    buffer = BytesIO()
    if ext == ".png":
        image.save(buffer, format="PNG", compress_level=2)
    else:
        image.save(buffer, format="JPEG", quality=90)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _clip_point_to_canvas(point: tuple[int, int], width: int, height: int) -> tuple[int, int]:
    return max(0, min(width - 1, point[0])), max(0, min(height - 1, point[1]))


def _load_font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    for candidate in ("Avenir Next.ttc", "Arial.ttf", "Helvetica.ttc", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _compute_homography_dlt(src_points: np.ndarray, dst_points: np.ndarray) -> np.ndarray | None:
    if src_points.shape[0] < 4 or src_points.shape[0] != dst_points.shape[0]:
        return None

    rows = []
    for (x, y), (u, v) in zip(src_points, dst_points):
        rows.append([-x, -y, -1.0, 0.0, 0.0, 0.0, x * u, y * u, u])
        rows.append([0.0, 0.0, 0.0, -x, -y, -1.0, x * v, y * v, v])

    a = np.asarray(rows, dtype=np.float64)
    try:
        _, _, vh = np.linalg.svd(a)
    except np.linalg.LinAlgError:
        return None

    h = vh[-1, :]
    if np.isclose(h[-1], 0.0):
        return None
    h = h / h[-1]
    homography = h.reshape(3, 3)
    if not np.all(np.isfinite(homography)):
        return None
    return homography


def _project_point_with_homography(homography: np.ndarray, point: tuple[float, float]) -> tuple[float, float] | None:
    x, y = point
    vector = homography @ np.asarray([x, y, 1.0], dtype=np.float64)
    if np.isclose(vector[2], 0.0):
        return None
    world = vector[:2] / vector[2]
    if not np.all(np.isfinite(world)):
        return None
    return float(world[0]), float(world[1])


class BEVDemoService:
    def __init__(self, config: DemoConfig) -> None:
        self.config = config
        self._model: Any | None = None
        self._postprocessor: Any | None = None
        self._homography_cache: dict[str, np.ndarray | None] = {}
        self._empty_pitch_image_data_uri: str | None = None

    def _load_model(self):
        if self._model is None:
            if self.config.weights is None:
                raise RuntimeError("weights is not configured.")
            if not self.config.weights.exists():
                raise RuntimeError(f"Weights file not found: {self.config.weights}")
            from ultralytics import YOLO

            self._model = YOLO(str(self.config.weights))
        return self._model

    def _load_postprocessor(self):
        if self._postprocessor is not None:
            return self._postprocessor
        if not self.config.bev_postprocess_stats or not self.config.bev_postprocess_stats.exists():
            return None

        from ultralytics.models.yolo.pose.postprocess_bev import PositionBandPostProcessor

        postprocessor = PositionBandPostProcessor.from_file(self.config.bev_postprocess_stats)
        postprocessor.set_base_score_threshold(self.config.conf)
        self._postprocessor = postprocessor
        return self._postprocessor

    def _compute_homography(self, sample: SampleEntry) -> np.ndarray | None:
        if sample.id in self._homography_cache:
            return self._homography_cache[sample.id]

        image_points: list[list[float]] = []
        world_points: list[list[float]] = []
        for annotation in sample.annotations:
            keypoints = _normalize_keypoints(annotation.get("keypoints"))
            position = annotation.get("position_on_pitch")
            if len(keypoints) <= 1 or position is None or len(position) < 2:
                continue
            pelvis_ground = keypoints[1]
            image_points.append([float(pelvis_ground[0]), float(pelvis_ground[1])])
            world_points.append([float(position[0]), float(position[1])])

        homography: np.ndarray | None = None
        if len(image_points) >= 4:
            homography = _compute_homography_dlt(
                np.asarray(image_points, dtype=np.float64),
                np.asarray(world_points, dtype=np.float64),
            )

        self._homography_cache[sample.id] = homography
        return homography

    def _project_world_point(self, sample: SampleEntry, image_point: tuple[float, float]) -> tuple[float, float] | None:
        homography = self._compute_homography(sample)
        if homography is None:
            return None
        return _project_point_with_homography(homography, image_point)

    def _apply_bev_postprocess(
        self,
        scores: np.ndarray,
        keypoints: np.ndarray,
        image_shape: tuple[int, int],
    ) -> np.ndarray:
        postprocessor = self._load_postprocessor()
        if postprocessor is None or len(scores) == 0:
            return np.arange(len(scores), dtype=np.int64)

        import torch

        predn_scaled = {
            "conf": torch.as_tensor(scores, dtype=torch.float32),
            "kpts": torch.as_tensor(keypoints, dtype=torch.float32),
        }
        keep = postprocessor.apply(predn_scaled, image_shape).keep_indices
        return keep.cpu().numpy().astype(np.int64)

    def _run_model(self, image_path: Path):
        model = self._load_model()
        kwargs: dict[str, Any] = {
            "source": str(image_path),
            "imgsz": self.config.imgsz,
            "conf": self.config.conf,
            "iou": self.config.iou,
            "max_det": self.config.max_det,
            "verbose": False,
        }
        if self.config.device:
            kwargs["device"] = self.config.device
        return model.predict(**kwargs)[0]

    def get_empty_pitch_image_data_uri(self) -> str:
        if self._empty_pitch_image_data_uri is None:
            self._empty_pitch_image_data_uri = _encode_image_data_uri(self._draw_pitch_view([]), ".png")
        return self._empty_pitch_image_data_uri

    def run(self, sample: SampleEntry) -> DemoRunResult:
        result = self._run_model(sample.image_path)
        orig_image = result.orig_img.copy()

        if result.boxes is None or result.keypoints is None:
            raise RuntimeError("The loaded model did not return both boxes and keypoints.")

        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        keypoints = result.keypoints.data.cpu().numpy()

        keep_indices = self._apply_bev_postprocess(scores, keypoints, orig_image.shape[:2])
        boxes_xyxy = boxes_xyxy[keep_indices]
        scores = scores[keep_indices]
        keypoints = keypoints[keep_indices]

        players = self._build_players(sample, boxes_xyxy, scores, keypoints)
        annotated_image = self._draw_image_view(orig_image, players)
        pitch_image = self._draw_pitch_view(players)

        projection_note = (
            "World coordinates are projected with a per-sample homography fitted from the sample's annotation pairs."
            if self._compute_homography(sample) is not None
            else "World projection unavailable for this sample."
        )

        return DemoRunResult(
            annotated_image_data_uri=_encode_image_data_uri(annotated_image, ".jpg"),
            pitch_image_data_uri=_encode_image_data_uri(pitch_image, ".png"),
            players=players,
            projection_note=projection_note,
            prediction_count=len(players),
        )

    def _build_players(
        self,
        sample: SampleEntry,
        boxes_xyxy: np.ndarray,
        scores: np.ndarray,
        keypoints: np.ndarray,
    ) -> list[PlayerDetection]:
        raw_players: list[dict[str, Any]] = []
        for idx in range(len(scores)):
            if keypoints.shape[1] < 2:
                continue
            pelvis = keypoints[idx, 0, :2]
            pelvis_ground = keypoints[idx, 1, :2]
            raw_players.append(
                {
                    "score": float(scores[idx]),
                    "bbox_xyxy": tuple(float(v) for v in boxes_xyxy[idx].tolist()),
                    "pelvis_xy": (float(pelvis[0]), float(pelvis[1])),
                    "pelvis_ground_xy": (float(pelvis_ground[0]), float(pelvis_ground[1])),
                    "world_xy": self._project_world_point(sample, (float(pelvis_ground[0]), float(pelvis_ground[1]))),
                }
            )

        raw_players.sort(key=lambda item: (item["pelvis_ground_xy"][0], item["pelvis_ground_xy"][1]))
        players: list[PlayerDetection] = []
        for player_id, item in enumerate(raw_players, start=1):
            color_bgr = _color_for_player(player_id)
            players.append(
                PlayerDetection(
                    player_id=player_id,
                    score=item["score"],
                    bbox_xyxy=item["bbox_xyxy"],
                    pelvis_xy=item["pelvis_xy"],
                    pelvis_ground_xy=item["pelvis_ground_xy"],
                    world_xy=item["world_xy"],
                    color_bgr=color_bgr,
                    color_hex=_bgr_to_hex(color_bgr),
                )
            )
        return players

    def _draw_image_view(self, image: np.ndarray, players: list[PlayerDetection]) -> np.ndarray:
        image_rgb = Image.fromarray(image[:, :, ::-1])
        draw = ImageDraw.Draw(image_rgb)
        width, height = image_rgb.size
        line_width = max(1, int(round(min(height, width) / 900)))
        radius = max(4, line_width + 3)
        font = _load_font(max(14, int(round(min(height, width) / 72))))

        for player in players:
            color_rgb = _bgr_to_rgb(player.color_bgr)
            raw_x1, raw_y1, raw_x2, raw_y2 = (int(round(v)) for v in player.bbox_xyxy)
            x1 = min(raw_x1, raw_x2)
            y1 = min(raw_y1, raw_y2)
            x2 = max(raw_x1, raw_x2)
            y2 = max(raw_y1, raw_y2)
            draw.rectangle((x1, y1, x2, y2), outline=color_rgb, width=line_width)

            pelvis = (int(round(player.pelvis_xy[0])), int(round(player.pelvis_xy[1])))
            pelvis_ground = (
                int(round(player.pelvis_ground_xy[0])),
                int(round(player.pelvis_ground_xy[1])),
            )
            draw.ellipse(
                (pelvis[0] - radius, pelvis[1] - radius, pelvis[0] + radius, pelvis[1] + radius),
                fill=color_rgb,
            )
            draw.ellipse(
                (
                    pelvis_ground[0] - radius,
                    pelvis_ground[1] - radius,
                    pelvis_ground[0] + radius,
                    pelvis_ground[1] + radius,
                ),
                fill=(255, 255, 255),
                outline=color_rgb,
                width=max(1, line_width),
            )

            label = str(player.player_id)
            anchor_x = x1 + 4
            anchor_y = max(22, y1 - 10)
            bbox = draw.textbbox((anchor_x, anchor_y), label, font=font)
            box = (bbox[0] - 6, bbox[1] - 4, bbox[2] + 6, bbox[3] + 4)
            draw.rounded_rectangle(box, radius=8, fill=color_rgb)
            draw.text((anchor_x, anchor_y), label, fill=(255, 255, 255), font=font)

        return image_rgb

    def _draw_pitch_view(self, players: list[PlayerDetection]) -> Image.Image:
        field_length = float(self.config.field_length_m)
        field_width = float(self.config.field_width_m)
        canvas_width = 1120
        padding = 40
        drawable_width = canvas_width - 2 * padding
        drawable_height = int(round(drawable_width * field_width / field_length))
        canvas_height = drawable_height + 2 * padding

        pitch = Image.new("RGB", (canvas_width, canvas_height), (74, 111, 31))
        draw = ImageDraw.Draw(pitch)
        line_color = (245, 245, 245)
        line_width = 2
        label_font = _load_font(20)

        def field_to_canvas(x: float, y: float) -> tuple[int, int]:
            px = padding + int(round((field_length / 2.0 - y) / field_length * drawable_width))
            py = padding + int(round((field_width / 2.0 - x) / field_width * drawable_height))
            return _clip_point_to_canvas((px, py), canvas_width, canvas_height)

        corner_a = field_to_canvas(field_width / 2.0, -field_length / 2.0)
        corner_b = field_to_canvas(-field_width / 2.0, field_length / 2.0)
        x0 = min(corner_a[0], corner_b[0])
        y0 = min(corner_a[1], corner_b[1])
        x1 = max(corner_a[0], corner_b[0])
        y1 = max(corner_a[1], corner_b[1])
        draw.rectangle((x0, y0, x1, y1), outline=line_color, width=line_width)

        center_top = field_to_canvas(field_width / 2.0, 0.0)
        center_bottom = field_to_canvas(-field_width / 2.0, 0.0)
        draw.line((center_top[0], center_top[1], center_bottom[0], center_bottom[1]), fill=line_color, width=line_width)

        center = field_to_canvas(0.0, 0.0)
        circle_radius = int(round(9.15 / field_length * drawable_width))
        draw.ellipse(
            (center[0] - circle_radius, center[1] - circle_radius, center[0] + circle_radius, center[1] + circle_radius),
            outline=line_color,
            width=line_width,
        )
        draw.ellipse((center[0] - 4, center[1] - 4, center[0] + 4, center[1] + 4), fill=line_color)

        def draw_box(y_outer: float, y_inner: float, half_box_width: float) -> None:
            corner_a = field_to_canvas(half_box_width, y_outer)
            corner_b = field_to_canvas(-half_box_width, y_inner)
            x0 = min(corner_a[0], corner_b[0])
            y0 = min(corner_a[1], corner_b[1])
            x1 = max(corner_a[0], corner_b[0])
            y1 = max(corner_a[1], corner_b[1])
            draw.rectangle(
                (x0, y0, x1, y1),
                outline=line_color,
                width=line_width,
            )

        half_penalty_width = 40.32 / 2.0
        half_goal_area_width = 18.32 / 2.0
        left_penalty_box_y = -field_length / 2.0 + 16.5
        right_penalty_box_y = field_length / 2.0 - 16.5
        left_goal_area_y = -field_length / 2.0 + 5.5
        right_goal_area_y = field_length / 2.0 - 5.5
        draw_box(-field_length / 2.0, left_penalty_box_y, half_penalty_width)
        draw_box(field_length / 2.0, right_penalty_box_y, half_penalty_width)
        draw_box(-field_length / 2.0, left_goal_area_y, half_goal_area_width)
        draw_box(field_length / 2.0, right_goal_area_y, half_goal_area_width)

        left_penalty_spot = field_to_canvas(0.0, -field_length / 2.0 + 11.0)
        right_penalty_spot = field_to_canvas(0.0, field_length / 2.0 - 11.0)
        draw.ellipse(
            (left_penalty_spot[0] - 4, left_penalty_spot[1] - 4, left_penalty_spot[0] + 4, left_penalty_spot[1] + 4),
            fill=line_color,
        )
        draw.ellipse(
            (
                right_penalty_spot[0] - 4,
                right_penalty_spot[1] - 4,
                right_penalty_spot[0] + 4,
                right_penalty_spot[1] + 4,
            ),
            fill=line_color,
        )

        for player in players:
            if player.world_xy is None:
                continue

            draw_x = float(np.clip(player.world_xy[0], -field_width / 2.0, field_width / 2.0))
            draw_y = float(np.clip(player.world_xy[1], -field_length / 2.0, field_length / 2.0))
            center_pt = field_to_canvas(draw_x, draw_y)
            color_rgb = _bgr_to_rgb(player.color_bgr)
            draw.ellipse(
                (center_pt[0] - 8, center_pt[1] - 8, center_pt[0] + 8, center_pt[1] + 8),
                fill=(255, 255, 255),
            )
            draw.ellipse(
                (center_pt[0] - 6, center_pt[1] - 6, center_pt[0] + 6, center_pt[1] + 6),
                fill=color_rgb,
            )
            draw.text((center_pt[0] + 10, center_pt[1] - 14), str(player.player_id), fill=(255, 255, 255), font=label_font)

        return pitch


@lru_cache(maxsize=1)
def get_runtime() -> RuntimeContext:
    config = load_demo_config()
    catalog = build_sample_catalog(config)
    service = BEVDemoService(config)
    return RuntimeContext(config=config, catalog=catalog, service=service)
