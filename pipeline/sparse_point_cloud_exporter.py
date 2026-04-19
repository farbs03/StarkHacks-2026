"""
Append-only sparse 3D observations for each capture (JSONL).

Each line is one point in a camera-centric local frame (meters), using the same
projection and depth rules as :class:`pipeline.poi_exporter.POIExporter`:
ultrasonic ``DIST`` when valid, otherwise ``default_depth_m``. Severity comes
from the capture decision (``risk_level``, ``trigger_score``); full Arduino
telemetry is stored under ``sensor_snapshot`` when available.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from pipeline.capture_pipeline import CaptureRecord
from pipeline.poi_exporter import POIExporter
from sensors.sensor_bridge import SensorSnapshot
from vision.detectors import Detection


class SparsePointCloudExporter:
    """Stream sparse hazard / POI points to a JSONL file for fusion or visualization."""

    def __init__(self, *, output_jsonl: str) -> None:
        self.output_path = Path(output_jsonl)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def append_observation(
        self,
        *,
        record: CaptureRecord,
        arm_state: dict[str, float],
        event_type: str,
        risk_level: str,
        trigger_score: float,
        selected_detection: Detection | None,
        frame_shape: tuple[int, int, int],
        sensor_snapshot: SensorSnapshot | None,
        default_depth_m: float,
        horizontal_fov_deg: float,
        vertical_fov_deg: float,
    ) -> dict[str, Any]:
        local = POIExporter._estimate_local_xyz(
            selected_detection=selected_detection,
            frame_shape=frame_shape,
            sensor_snapshot=sensor_snapshot,
            default_depth_m=default_depth_m,
            horizontal_fov_deg=horizontal_fov_deg,
            vertical_fov_deg=vertical_fov_deg,
        )
        row: dict[str, Any] = {
            "point_id": str(uuid.uuid4()),
            "capture_id": record.capture_id,
            "timestamp": record.timestamp,
            "frame_shape_hw": [int(frame_shape[0]), int(frame_shape[1])],
            "position_m": {
                "x": local["x_m"],
                "y": local["y_m"],
                "z": local["z_m"],
            },
            "depth_source": local["depth_source"],
            "severity": {
                "risk_level": risk_level,
                "trigger_score": float(trigger_score),
                "event_type": event_type,
            },
            "sensor_snapshot": asdict(sensor_snapshot) if sensor_snapshot else None,
            "detection": (
                {
                    "label": selected_detection.label,
                    "score": float(selected_detection.score),
                    "bbox_xyxy": list(selected_detection.bbox_xyxy),
                    "center_xy": [
                        float(selected_detection.center_xy[0]),
                        float(selected_detection.center_xy[1]),
                    ],
                }
                if selected_detection is not None
                else None
            ),
            "arm_state": arm_state,
            "image_path": record.image_path,
            "overlay_path": record.overlay_path,
            "metadata_path": record.metadata_path,
        }
        with self.output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        return row
