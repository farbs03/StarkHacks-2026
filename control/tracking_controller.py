from __future__ import annotations

from dataclasses import dataclass

from control.scan_behavior import PanSweepScanner


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass(slots=True)
class ControlCommand:
    pan_delta: float
    lift_delta: float
    mode: str
    smoothed_target: tuple[float, float] | None


class TrackingController:
    def __init__(
        self,
        *,
        pan_gain: float,
        lift_gain: float,
        deadzone: float,
        max_step: float,
        max_joint: float,
        ema_alpha: float = 0.35,
        frames_before_scan: int = 10,
        scan_step_size: float = 0.7,
    ) -> None:
        self.pan_gain = pan_gain
        self.lift_gain = lift_gain
        self.deadzone = deadzone
        self.max_step = max_step
        self.max_joint = max_joint
        self.ema_alpha = ema_alpha
        self.frames_before_scan = frames_before_scan
        self.missed_frames = 0
        self.smoothed_target: tuple[float, float] | None = None
        self.scanner = PanSweepScanner(step_size=scan_step_size)

    def update(
        self,
        target_center_norm: tuple[float, float] | None,
        *,
        current_pan: float,
    ) -> ControlCommand:
        if target_center_norm is None:
            self.missed_frames += 1
            self.smoothed_target = None
            if self.missed_frames >= self.frames_before_scan:
                pan_step = self.scanner.next_pan_step(current_pan, self.max_joint)
                return ControlCommand(
                    pan_delta=pan_step,
                    lift_delta=0.0,
                    mode="scan",
                    smoothed_target=None,
                )
            return ControlCommand(
                pan_delta=0.0,
                lift_delta=0.0,
                mode="hold",
                smoothed_target=None,
            )

        self.missed_frames = 0
        if self.smoothed_target is None:
            self.smoothed_target = target_center_norm
        else:
            tx, ty = target_center_norm
            sx, sy = self.smoothed_target
            self.smoothed_target = (
                (self.ema_alpha * tx) + ((1.0 - self.ema_alpha) * sx),
                (self.ema_alpha * ty) + ((1.0 - self.ema_alpha) * sy),
            )

        smx, smy = self.smoothed_target
        err_x = smx - 0.5
        err_y = smy - 0.5

        if abs(err_x) < self.deadzone:
            err_x = 0.0
        if abs(err_y) < self.deadzone:
            err_y = 0.0

        pan_step = clamp(err_x * self.pan_gain, -self.max_step, self.max_step)
        lift_step = clamp(err_y * self.lift_gain, -self.max_step, self.max_step)
        return ControlCommand(
            pan_delta=pan_step,
            lift_delta=lift_step,
            mode="track",
            smoothed_target=self.smoothed_target,
        )
