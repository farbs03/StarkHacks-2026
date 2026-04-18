from __future__ import annotations


class PanSweepScanner:
    def __init__(self, *, step_size: float = 0.7, margin: float = 8.0) -> None:
        self.step_size = step_size
        self.margin = margin
        self._direction = 1.0

    def next_pan_step(self, current_pan: float, max_joint: float) -> float:
        if current_pan >= (max_joint - self.margin):
            self._direction = -1.0
        elif current_pan <= (-max_joint + self.margin):
            self._direction = 1.0
        return self.step_size * self._direction
