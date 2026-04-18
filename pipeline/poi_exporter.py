from __future__ import annotations

import json
import math
import time
import uuid
from pathlib import Path
from typing import Any

from pipeline.capture_pipeline import CaptureRecord
from sensors.sensor_bridge import SensorSnapshot
from vision.detectors import Detection


class POIExporter:
    def __init__(self, *, output_jsonl: str) -> None:
        self.output_path = Path(output_jsonl)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def export(
        self,
        *,
        record: CaptureRecord,
        arm_state: dict[str, float],
        event_type: str,
        confidence: float,
        selected_detection: Detection | None,
        frame_shape: tuple[int, int, int],
        sensor_snapshot: SensorSnapshot | None,
        default_depth_m: float,
        horizontal_fov_deg: float,
        vertical_fov_deg: float,
    ) -> dict[str, Any]:
        local_point = self._estimate_local_xyz(
            selected_detection=selected_detection,
            frame_shape=frame_shape,
            sensor_snapshot=sensor_snapshot,
            default_depth_m=default_depth_m,
            horizontal_fov_deg=horizontal_fov_deg,
            vertical_fov_deg=vertical_fov_deg,
        )
        poi = {
            "poi_id": str(uuid.uuid4()),
            "timestamp": time.time(),
            "event_type": event_type,
            "confidence": confidence,
            "world_or_local_pose": {
                "joint_state": arm_state,
                "local_xyz_m": {
                    "x": local_point["x_m"],
                    "y": local_point["y_m"],
                    "z": local_point["z_m"],
                },
                "depth_source": local_point["depth_source"],
            },
            "image_path": record.image_path,
            "overlay_path": record.overlay_path,
            "metadata_path": record.metadata_path,
            "description": record.metadata.get("description", ""),
        }
        with self.output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(poi) + "\n")
        return poi

    @staticmethod
    def _estimate_local_xyz(
        *,
        selected_detection: Detection | None,
        frame_shape: tuple[int, int, int],
        sensor_snapshot: SensorSnapshot | None,
        default_depth_m: float,
        horizontal_fov_deg: float,
        vertical_fov_deg: float,
    ) -> dict[str, float | str]:
        frame_h, frame_w = frame_shape[:2]
        if selected_detection is None or frame_h <= 0 or frame_w <= 0:
            return {
                "x_m": 0.0,
                "y_m": 0.0,
                "z_m": default_depth_m,
                "depth_source": "fallback_default",
            }

        depth_m = default_depth_m
        depth_source = "fallback_default"
        if sensor_snapshot is not None and sensor_snapshot.dist not in (0, 999):
            depth_m = max(0.05, float(sensor_snapshot.dist) / 100.0)
            depth_source = "ultrasonic_dist_cm"

        cx, cy = selected_detection.center_xy
        norm_x = (cx / frame_w) - 0.5
        norm_y = (cy / frame_h) - 0.5

        angle_x = norm_x * math.radians(horizontal_fov_deg)
        angle_y = norm_y * math.radians(vertical_fov_deg)

        x_m = depth_m * math.tan(angle_x)
        y_m = depth_m * math.tan(angle_y)
        z_m = depth_m
        return {
            "x_m": round(x_m, 4),
            "y_m": round(y_m, 4),
            "z_m": round(z_m, 4),
            "depth_source": depth_source,
        }
