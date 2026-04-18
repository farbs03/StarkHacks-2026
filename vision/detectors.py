from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np
try:
    import onnxruntime as ort
except ImportError:
    ort = None


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


CONSTRUCTION_HAZARD_LABELS = [
    "Hardhat",
    "Mask",
    "NO-Hardhat",
    "NO-Mask",
    "NO-Safety Vest",
    "Person",
    "Safety Cone",
    "Safety Vest",
    "Machinery",
    "Utility Pole",
    "Vehicle",
]


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


class UltralyticsOnnxDetector:
    """
    ONNX detector for Ultralytics-style exports (e.g., YOLO11/YOLO26 family).
    Handles outputs shaped like (1, N, no) or (1, no, N).
    """

    def __init__(
        self,
        model_path: str,
        *,
        class_names: list[str] | None = None,
        conf_threshold: float = 0.35,
        nms_threshold: float = 0.45,
        input_size: int = 640,
        max_detections: int = 120,
        backend: str = "auto",
    ) -> None:
        self.model_path = model_path
        self.class_names = class_names or []
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        self.max_detections = max_detections
        self.backend = backend
        self._warned_fallback = False
        self._net = None
        self._ort_session = None
        self._ort_input_name = None

        if backend in {"auto", "opencv"}:
            self._net = cv2.dnn.readNetFromONNX(model_path)
        if backend in {"auto", "onnxruntime"}:
            self._init_onnxruntime()

        if backend == "onnxruntime" and self._ort_session is None:
            raise RuntimeError(
                "Detector backend set to onnxruntime, but onnxruntime is not installed. "
                "Install it with: pip install onnxruntime"
            )

    def _init_onnxruntime(self) -> None:
        if ort is None or self._ort_session is not None:
            return
        providers = ["CPUExecutionProvider"]
        self._ort_session = ort.InferenceSession(self.model_path, providers=providers)
        self._ort_input_name = self._ort_session.get_inputs()[0].name

    def _forward_opencv(self, frame: np.ndarray) -> np.ndarray:
        if self._net is None:
            raise RuntimeError("OpenCV backend not initialized.")
        blob = cv2.dnn.blobFromImage(
            frame,
            scalefactor=1.0 / 255.0,
            size=(self.input_size, self.input_size),
            swapRB=True,
            crop=False,
        )
        self._net.setInput(blob)
        return self._net.forward()

    def _forward_onnxruntime(self, frame: np.ndarray) -> np.ndarray:
        if self._ort_session is None or self._ort_input_name is None:
            raise RuntimeError("ONNX Runtime backend not initialized.")
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(img_rgb, (self.input_size, self.input_size))
        inp = resized.astype(np.float32) / 255.0
        inp = np.transpose(inp, (2, 0, 1))[None, ...]
        outputs = self._ort_session.run(None, {self._ort_input_name: inp})
        return outputs[0]

    def detect(self, frame: np.ndarray, frame_id: int, ts: float) -> list[Detection]:
        frame_h, frame_w = frame.shape[:2]
        try:
            if self.backend == "onnxruntime":
                raw = self._forward_onnxruntime(frame)
            else:
                raw = self._forward_opencv(frame)
        except cv2.error as exc:
            if self.backend in {"auto", "opencv"} and self._ort_session is not None:
                if not self._warned_fallback:
                    print(
                        "[WARN] OpenCV DNN failed for ONNX model; "
                        "falling back to ONNX Runtime backend."
                    )
                    self._warned_fallback = True
                raw = self._forward_onnxruntime(frame)
            else:
                raise RuntimeError(
                    "OpenCV DNN failed to run this ONNX model. "
                    "This usually means model ops are unsupported by OpenCV DNN. "
                    f"Model: {self.model_path}. "
                    "Install onnxruntime and run with backend=onnxruntime/auto."
                ) from exc
        pred = np.squeeze(raw)
        if pred.ndim != 2:
            return []

        expected_cols_obj = 5 + len(self.class_names) if self.class_names else None
        expected_cols_noobj = 4 + len(self.class_names) if self.class_names else None

        # Convert to (num_predictions, num_outputs) by checking expected output width.
        if expected_cols_obj is not None and pred.shape[1] not in {
            expected_cols_obj,
            expected_cols_noobj,
        }:
            if pred.shape[0] in {expected_cols_obj, expected_cols_noobj}:
                pred = pred.T
            elif pred.shape[0] < pred.shape[1]:
                pred = pred.T

        if pred.shape[1] < 5:
            return []

        use_objectness = False
        if self.class_names and pred.shape[1] == (len(self.class_names) + 5):
            use_objectness = True

        boxes_xywh: list[list[int]] = []
        scores: list[float] = []
        class_ids: list[int] = []

        scale_x = frame_w / float(self.input_size)
        scale_y = frame_h / float(self.input_size)

        for row in pred:
            cx, cy, w, h = row[:4]
            if use_objectness:
                objectness = float(row[4])
                class_scores = row[5:]
            else:
                objectness = 1.0
                class_scores = row[4:]

            if class_scores.size == 0:
                continue

            # Some exports output logits instead of probabilities.
            if np.max(class_scores) > 1.0 or np.min(class_scores) < 0.0:
                class_scores = 1.0 / (1.0 + np.exp(-class_scores))
            if objectness > 1.0 or objectness < 0.0:
                objectness = 1.0 / (1.0 + np.exp(-objectness))

            class_id = int(np.argmax(class_scores))
            class_score = float(class_scores[class_id])
            score = objectness * class_score
            if score < self.conf_threshold:
                continue

            x1 = (cx - (w / 2.0)) * scale_x
            y1 = (cy - (h / 2.0)) * scale_y
            bw = w * scale_x
            bh = h * scale_y

            x1_i = int(max(0, min(frame_w - 1, x1)))
            y1_i = int(max(0, min(frame_h - 1, y1)))
            bw_i = int(max(1, min(frame_w - x1_i, bw)))
            bh_i = int(max(1, min(frame_h - y1_i, bh)))

            boxes_xywh.append([x1_i, y1_i, bw_i, bh_i])
            scores.append(score)
            class_ids.append(class_id)

        if not boxes_xywh:
            return []

        kept = cv2.dnn.NMSBoxes(
            bboxes=boxes_xywh,
            scores=scores,
            score_threshold=self.conf_threshold,
            nms_threshold=self.nms_threshold,
        )
        if len(kept) == 0:
            return []

        detections: list[Detection] = []
        for idx in np.array(kept).reshape(-1):
            x, y, w, h = boxes_xywh[int(idx)]
            class_id = class_ids[int(idx)]
            label = (
                self.class_names[class_id]
                if 0 <= class_id < len(self.class_names)
                else f"class_{class_id}"
            )
            detections.append(
                Detection(
                    label=label,
                    bbox_xyxy=(x, y, x + w, y + h),
                    score=float(scores[int(idx)]),
                    frame_id=frame_id,
                    timestamp=ts,
                )
            )
        detections.sort(key=lambda det: det.score, reverse=True)
        return detections[: self.max_detections]


class ConstructionHazardOnnxDetector(UltralyticsOnnxDetector):
    def __init__(
        self,
        model_path: str,
        *,
        conf_threshold: float = 0.55,
        nms_threshold: float = 0.45,
        input_size: int = 640,
        max_detections: int = 80,
        backend: str = "auto",
    ) -> None:
        super().__init__(
            model_path=model_path,
            class_names=CONSTRUCTION_HAZARD_LABELS,
            conf_threshold=conf_threshold,
            nms_threshold=nms_threshold,
            input_size=input_size,
            max_detections=max_detections,
            backend=backend,
        )
