from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from sensors.sensor_bridge import SensorSnapshot
from vision.detectors import Detection


@dataclass(slots=True)
class CaptureRecord:
    capture_id: str
    timestamp: float
    image_path: str
    overlay_path: str | None
    metadata_path: str
    metadata: dict[str, Any]


class CapturePipeline:
    def __init__(self, *, output_dir: str, write_overlay: bool = True) -> None:
        self.output_root = Path(output_dir)
        self.write_overlay = write_overlay
        self.output_root.mkdir(parents=True, exist_ok=True)

    def capture(
        self,
        *,
        frame: np.ndarray,
        detection: Detection | None,
        detections: list[Detection],
        arm_state: dict[str, float],
        sensor_snapshot: SensorSnapshot | None,
        trigger_reason: str,
        risk_level: str,
        trigger_score: float,
    ) -> CaptureRecord:
        ts = time.time()
        capture_id = f"{int(ts * 1000)}"
        folder = self.output_root / time.strftime("%Y%m%d")
        folder.mkdir(parents=True, exist_ok=True)

        image_path = folder / f"{capture_id}.jpg"
        overlay_path = folder / f"{capture_id}.overlay.jpg"
        metadata_path = folder / f"{capture_id}.json"

        cv2.imwrite(str(image_path), frame)
        if self.write_overlay:
            overlay = self._draw_overlay(frame.copy(), detections, detection)
            cv2.imwrite(str(overlay_path), overlay)
            overlay_ref: str | None = str(overlay_path)
        else:
            overlay_ref = None

        metadata = {
            "capture_id": capture_id,
            "timestamp": ts,
            "photo_path": str(image_path),
            "overlay_path": overlay_ref,
            "event_label": detection.label if detection else trigger_reason,
            "trigger_reason": trigger_reason,
            "risk_level": risk_level,
            "trigger_score": trigger_score,
            "selected_detection": asdict(detection) if detection else None,
            "detections": [asdict(det) for det in detections],
            "arm_state": arm_state,
            "sensor_snapshot": asdict(sensor_snapshot) if sensor_snapshot else None,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        return CaptureRecord(
            capture_id=capture_id,
            timestamp=ts,
            image_path=str(image_path),
            overlay_path=overlay_ref,
            metadata_path=str(metadata_path),
            metadata=metadata,
        )

    def update_metadata(self, record: CaptureRecord, patch: dict[str, Any]) -> CaptureRecord:
        merged = dict(record.metadata)
        merged.update(patch)
        Path(record.metadata_path).write_text(
            json.dumps(merged, indent=2), encoding="utf-8"
        )
        record.metadata = merged
        return record

    @staticmethod
    def _draw_overlay(
        frame: np.ndarray,
        detections: list[Detection],
        selected: Detection | None,
    ) -> np.ndarray:
        selected_bbox = selected.bbox_xyxy if selected else None
        for det in detections:
            x1, y1, x2, y2 = det.bbox_xyxy
            color = (40, 220, 60) if det.bbox_xyxy == selected_bbox else (255, 160, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"{det.label}:{det.score:.2f}",
                (x1, max(15, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
        return frame
