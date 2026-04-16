from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass
class BEVPostprocessResult:
    """Container for position-band post-process selection."""

    keep_indices: torch.Tensor
    stage_counts: dict[str, int] | None = None
    stage_thresholds: dict[str, float] | None = None


class PositionBandPostProcessor:
    """Per-y-band calibrated BEV post-processing.

    Runtime behavior is intentionally simple:
      1) map the predicted player position keypoint to a vertical image band,
      2) read the learned score threshold assigned to that band,
      3) compare the detection score against that threshold,
      4) keep or drop the whole prediction instance.

    The keypoint itself is only used to decide which band a prediction belongs
    to. Filtering always happens at the detection-instance level.
    """

    def __init__(self, stats: dict[str, Any]) -> None:
        self.stats = stats

        self.position_kpt_index = int(stats.get("position_keypoint_index", 1))
        self.base_score_threshold = float(stats.get("base_score_threshold", stats.get("global_score_threshold", 0.0)))
        self.global_score_threshold = float(stats.get("global_score_threshold", self.base_score_threshold))

        image_bounds = stats.get("image_position_bounds") or {"x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0}
        self.pos_y_min = float(image_bounds["y_min"])
        self.pos_y_max = float(image_bounds["y_max"])

        banding = stats.get("banding", {})
        self.y_bands = max(int(banding.get("y_bands", stats.get("y_bands", 4))), 1)

        fallback_ratio = float(stats.get("fallback_threshold_ratio", 1.0))
        self.fallback_score_threshold = float(
            stats.get("fallback_score_threshold", self.global_score_threshold * fallback_ratio)
        )

        band_thresholds = stats.get("band_thresholds", {})
        if band_thresholds:
            self.band_thresholds = {str(key): float(value) for key, value in band_thresholds.items()}
        else:
            # Backward compatibility with older calibration artifacts.
            lookup = stats.get("band_threshold_lookup", {})
            ratios = stats.get("band_threshold_ratios", {})
            self.band_thresholds = {}
            for band in range(self.y_bands):
                key = str(band)
                if key in lookup and isinstance(lookup[key], dict):
                    if "learned_threshold" in lookup[key]:
                        self.band_thresholds[key] = float(lookup[key]["learned_threshold"])
                        continue
                    if "threshold_ratio" in lookup[key]:
                        self.band_thresholds[key] = float(self.global_score_threshold) * float(lookup[key]["threshold_ratio"])
                        continue
                if key in ratios:
                    self.band_thresholds[key] = float(self.global_score_threshold) * float(ratios[key])

        self.band_threshold_ratios = {
            key: (float(value) / float(self.global_score_threshold) if self.global_score_threshold > 0 else 0.0)
            for key, value in self.band_thresholds.items()
        }

    @classmethod
    def from_file(cls, path: str | Path) -> PositionBandPostProcessor:
        with Path(path).open() as f:
            payload = json.load(f)
        return cls(payload)

    def set_base_score_threshold(self, score_threshold: float) -> None:
        score_threshold = float(score_threshold)
        self.base_score_threshold = score_threshold
        if self.global_score_threshold <= 0:
            self.global_score_threshold = score_threshold
        if self.fallback_score_threshold <= 0:
            self.fallback_score_threshold = score_threshold

    def apply(
        self,
        predn_scaled: dict[str, torch.Tensor],
        ori_shape: tuple[int, int],
    ) -> BEVPostprocessResult:
        """Apply y-band thresholds and return selected indices."""
        n = predn_scaled["conf"].shape[0]
        if n == 0:
            empty = torch.empty((0,), dtype=torch.long, device=predn_scaled["conf"].device)
            return BEVPostprocessResult(
                empty,
                stage_counts={"input": 0, "valid_position": 0, "kept": 0},
                stage_thresholds={
                    "global_score_threshold": float(self.global_score_threshold),
                    "fallback_score_threshold": float(self.fallback_score_threshold),
                    "band_low_threshold": float(min(self.band_thresholds.values(), default=self.fallback_score_threshold)),
                    "band_high_threshold": float(max(self.band_thresholds.values(), default=self.fallback_score_threshold)),
                    "num_bands": float(self.y_bands),
                    "local_threshold_min": 0.0,
                    "local_threshold_max": 0.0,
                },
            )

        kpts = predn_scaled["kpts"].detach().cpu().numpy()
        conf_scores = predn_scaled["conf"].detach().cpu().numpy().astype(np.float32)
        pos_y = self._normalized_image_positions_y(kpts, ori_shape)
        valid_pos = np.isfinite(pos_y)
        local_thresholds = np.full(n, float(self.fallback_score_threshold), dtype=np.float32)
        for i in np.where(valid_pos)[0]:
            local_thresholds[i] = self._threshold_for_position_y(float(pos_y[i]))
        keep = conf_scores >= local_thresholds

        kept_indices = np.where(keep)[0]
        keep_tensor = torch.as_tensor(kept_indices, dtype=torch.long, device=predn_scaled["conf"].device)
        return BEVPostprocessResult(
            keep_tensor,
            stage_counts={
                "input": int(n),
                "valid_position": int(np.count_nonzero(valid_pos)),
                "kept": int(len(kept_indices)),
            },
            stage_thresholds={
                "global_score_threshold": float(self.global_score_threshold),
                "fallback_score_threshold": float(self.fallback_score_threshold),
                "band_low_threshold": float(min(self.band_thresholds.values(), default=self.fallback_score_threshold)),
                "band_high_threshold": float(max(self.band_thresholds.values(), default=self.fallback_score_threshold)),
                "num_bands": float(self.y_bands),
                "local_threshold_min": float(local_thresholds.min()) if local_thresholds.size else 0.0,
                "local_threshold_max": float(local_thresholds.max()) if local_thresholds.size else 0.0,
            },
        )

    def _normalized_image_positions_y(self, kpts: np.ndarray, ori_shape: tuple[int, int]) -> np.ndarray:
        out = np.full((kpts.shape[0],), np.nan, dtype=np.float32)
        if kpts.ndim != 3 or kpts.shape[1] <= self.position_kpt_index:
            return out
        h = float(ori_shape[0])
        if h <= 0:
            return out
        pts = kpts[:, self.position_kpt_index, :2]
        out[:] = pts[:, 1] / h
        return out

    def _band_index(self, y: float) -> int:
        y_span = max(self.pos_y_max - self.pos_y_min, 1e-6)
        by = int(np.floor((y - self.pos_y_min) / y_span * self.y_bands))
        return min(max(by, 0), self.y_bands - 1)

    def _threshold_for_position_y(self, y: float) -> float:
        key = str(self._band_index(y))
        return float(self.band_thresholds.get(key, self.fallback_score_threshold))
