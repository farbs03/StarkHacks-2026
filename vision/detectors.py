from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np


@dataclass(slots=True)
class Detection:
    label: str
    bbox_xyxy: tuple[int, int, int, int]
    score: float
    frame_id: int
    timestamp: float

    @property
    def center_xy(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.bbox_xyxy
        return max(0, x2 - x1) * max(0, y2 - y1)


class Detector(Protocol):
    def detect(self, frame: np.ndarray, frame_id: int, ts: float) -> list[Detection]:
        ...


class HaarFaceDetector:
    def __init__(
        self,
        *,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: tuple[int, int] = (40, 40),
    ) -> None:
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size
        self._cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        if self._cascade.empty():
            raise RuntimeError("Failed to load OpenCV haarcascade_frontalface_default.xml")

    def detect(self, frame: np.ndarray, frame_id: int, ts: float) -> list[Detection]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
        )

        detections: list[Detection] = []
        for x, y, w, h in faces:
            detections.append(
                Detection(
                    label="face",
                    bbox_xyxy=(int(x), int(y), int(x + w), int(y + h)),
                    score=0.85,
                    frame_id=frame_id,
                    timestamp=ts,
                )
            )
        return detections


class YoloOnnxDetector:
    """
    Lightweight OpenCV DNN ONNX detector.
    This is optional and only used when a valid model path is provided.
    """

    def __init__(
        self,
        model_path: str,
        class_names: list[str] | None = None,
        conf_threshold: float = 0.4,
        nms_threshold: float = 0.45,
        input_size: tuple[int, int] = (640, 640),
    ) -> None:
        self.class_names = class_names or []
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self._model = cv2.dnn_DetectionModel(model_path)
        self._model.setInputParams(
            size=input_size,
            scale=1 / 255.0,
            swapRB=True,
        )

    def detect(self, frame: np.ndarray, frame_id: int, ts: float) -> list[Detection]:
        class_ids, scores, boxes = self._model.detect(
            frame,
            confThreshold=self.conf_threshold,
            nmsThreshold=self.nms_threshold,
        )
        detections: list[Detection] = []
        if len(class_ids) == 0:
            return detections

        for class_id, score, box in zip(class_ids.flatten(), scores.flatten(), boxes):
            x, y, w, h = box
            label = (
                self.class_names[class_id]
                if 0 <= class_id < len(self.class_names)
                else f"class_{class_id}"
            )
            detections.append(
                Detection(
                    label=label,
                    bbox_xyxy=(int(x), int(y), int(x + w), int(y + h)),
                    score=float(score),
                    frame_id=frame_id,
                    timestamp=ts,
                )
            )
        return detections
