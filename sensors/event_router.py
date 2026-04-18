from __future__ import annotations

import time
from dataclasses import dataclass

from sensors.sensor_bridge import SensorEvent
from vision.detectors import Detection


@dataclass(slots=True)
class CaptureDecision:
    should_capture: bool
    reason: str | None
    risk_level: str
    trigger_score: float


class EventRouter:
    def __init__(
        self,
        *,
        capture_cooldown_sec: float = 3.0,
        visual_labels: set[str] | None = None,
        visual_score_threshold: float = 0.6,
    ) -> None:
        self.capture_cooldown_sec = capture_cooldown_sec
        self.visual_labels = visual_labels or {"face", "person", "smoke", "fire", "hazard"}
        self.visual_score_threshold = visual_score_threshold
        self._last_capture_ts = 0.0

    def decide(
        self,
        detection: Detection | None,
        sensor_events: list[SensorEvent],
    ) -> CaptureDecision:
        now = time.time()
        if (now - self._last_capture_ts) < self.capture_cooldown_sec:
            return CaptureDecision(
                should_capture=False,
                reason=None,
                risk_level="low",
                trigger_score=0.0,
            )

        high_sensor_events = [evt for evt in sensor_events if evt.severity == "high"]
        if high_sensor_events:
            top = high_sensor_events[0]
            self._last_capture_ts = now
            return CaptureDecision(
                should_capture=True,
                reason=f"sensor:{top.event_type}",
                risk_level="high",
                trigger_score=1.0,
            )

        if (
            detection is not None
            and detection.label in self.visual_labels
            and detection.score >= self.visual_score_threshold
        ):
            self._last_capture_ts = now
            return CaptureDecision(
                should_capture=True,
                reason=f"vision:{detection.label}",
                risk_level="medium",
                trigger_score=detection.score,
            )

        return CaptureDecision(
            should_capture=False,
            reason=None,
            risk_level="low",
            trigger_score=0.0,
        )
