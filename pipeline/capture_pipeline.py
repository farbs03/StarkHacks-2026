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
        raw_image_path = folder / f"{capture_id}.raw.jpg"
        overlay_path = folder / f"{capture_id}.overlay.jpg"
        metadata_path = folder / f"{capture_id}.json"

        grouped = self._group_detections(detections)
        primary_group = self._pick_primary_group(grouped, detection)
        grouped_detections = primary_group if primary_group else (
            [detection] if detection else []
        )
        event_classes = sorted({det.label for det in grouped_detections})
        event_bboxes = [det.bbox_xyxy for det in grouped_detections]

        # Save raw image and a boxed event image for quick inspection.
        cv2.imwrite(str(raw_image_path), frame)
        boxed = self._draw_overlay(frame.copy(), grouped_detections, detection)
        cv2.imwrite(str(image_path), boxed)
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
            "raw_photo_path": str(raw_image_path),
            "overlay_path": overlay_ref,
            "event_label": detection.label if detection else trigger_reason,
            "event_classes": event_classes,
            "event_bboxes": event_bboxes,
            "trigger_reason": trigger_reason,
            "risk_level": risk_level,
            "trigger_score": trigger_score,
            "selected_detection": asdict(detection) if detection else None,
            "grouped_detections": [asdict(det) for det in grouped_detections],
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

    @staticmethod
    def _box_iou(
        box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]
    ) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter = inter_w * inter_h
        if inter == 0:
            return 0.0
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        denom = max(1, area_a + area_b - inter)
        return inter / denom

    @classmethod
    def _is_connected(cls, a: Detection, b: Detection) -> bool:
        iou = cls._box_iou(a.bbox_xyxy, b.bbox_xyxy)
        if iou >= 0.12:
            return True

        # Treat strong containment as the same event cluster.
        ax1, ay1, ax2, ay2 = a.bbox_xyxy
        bx1, by1, bx2, by2 = b.bbox_xyxy
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter = inter_w * inter_h
        area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(1, (bx2 - bx1) * (by2 - by1))
        contained_ratio = max(inter / area_a, inter / area_b)
        return contained_ratio >= 0.7

    @classmethod
    def _group_detections(cls, detections: list[Detection]) -> list[list[Detection]]:
        groups: list[list[Detection]] = []
        used = [False] * len(detections)
        for idx, det in enumerate(detections):
            if used[idx]:
                continue
            used[idx] = True
            group = [det]
            queue = [idx]
            while queue:
                cur = queue.pop()
                for j, other in enumerate(detections):
                    if used[j]:
                        continue
                    if cls._is_connected(detections[cur], other):
                        used[j] = True
                        group.append(other)
                        queue.append(j)
            groups.append(group)
        return groups

    @staticmethod
    def _pick_primary_group(
        groups: list[list[Detection]], selected: Detection | None
    ) -> list[Detection] | None:
        if not groups:
            return None
        if selected is not None:
            for group in groups:
                if any(det is selected for det in group):
                    return group
                if any(det.bbox_xyxy == selected.bbox_xyxy for det in group):
                    return group
        # Fallback: choose group with highest max score.
        return max(groups, key=lambda g: max(det.score for det in g))
