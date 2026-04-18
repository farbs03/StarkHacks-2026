from __future__ import annotations

from dataclasses import dataclass

from vision.detectors import Detection


@dataclass(slots=True)
class TargetSelection:
    detection: Detection
    center_norm_xy: tuple[float, float]


class TargetSelector:
    def __init__(self, class_priority: dict[str, int] | None = None) -> None:
        self.class_priority = class_priority or {
            "fire": 100,
            "smoke": 95,
            "gas": 90,
            "hazard": 85,
            "person": 70,
            "face": 60,
        }

    def select(
        self,
        detections: list[Detection],
        frame_shape: tuple[int, int, int],
        previous_center_norm: tuple[float, float] | None = None,
    ) -> TargetSelection | None:
        if not detections:
            return None

        frame_h, frame_w = frame_shape[:2]

        def score_key(det: Detection) -> tuple[int, float, float, int]:
            cx, cy = det.center_xy
            center_norm = (cx / frame_w, cy / frame_h)
            continuity_bonus = 0.0
            if previous_center_norm is not None:
                dx = center_norm[0] - previous_center_norm[0]
                dy = center_norm[1] - previous_center_norm[1]
                continuity_bonus = -(dx * dx + dy * dy)
            return (
                self.class_priority.get(det.label, 10),
                float(det.score),
                continuity_bonus,
                det.area,
            )

        best = max(detections, key=score_key)
        cx, cy = best.center_xy
        return TargetSelection(
            detection=best,
            center_norm_xy=(cx / frame_w, cy / frame_h),
        )
