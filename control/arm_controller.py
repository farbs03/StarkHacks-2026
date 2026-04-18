from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass(slots=True)
class ArmControllerConfig:
    port: str
    send_hz: float = 30.0
    max_joint: float = 100.0
    robot_id: str = "follower"
    dry_run: bool = False


DEFAULT_STATE = {
    "shoulder_pan.pos": 0.0,
    "shoulder_lift.pos": 0.0,
    "elbow_flex.pos": 0.0,
    "wrist_flex.pos": 0.0,
    "wrist_roll.pos": 0.0,
    "gripper.pos": 0.0,
}


class SOArmController:
    def __init__(self, config: ArmControllerConfig) -> None:
        self.config = config
        self._state = dict(DEFAULT_STATE)
        self._state_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._robot: SO101Follower | None = None

    def connect(self) -> None:
        if self.config.dry_run:
            return
        self._robot = SO101Follower(
            SO101FollowerConfig(port=self.config.port, id=self.config.robot_id)
        )
        self._robot.connect()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._send_loop, daemon=True)
        self._thread.start()

    def _send_loop(self) -> None:
        period = 1.0 / self.config.send_hz
        while not self._stop_event.is_set():
            with self._state_lock:
                action = dict(self._state)
            if self._robot is not None:
                self._robot.send_action(action)
            time.sleep(period)

    def update_pan_lift(self, pan_delta: float, lift_delta: float) -> None:
        with self._state_lock:
            self._state["shoulder_pan.pos"] = clamp(
                self._state["shoulder_pan.pos"] + pan_delta,
                -self.config.max_joint,
                self.config.max_joint,
            )
            self._state["shoulder_lift.pos"] = clamp(
                self._state["shoulder_lift.pos"] + lift_delta,
                -self.config.max_joint,
                self.config.max_joint,
            )

    def set_neutral_pose(self, neutral_pose: dict[str, float]) -> None:
        with self._state_lock:
            for joint, value in neutral_pose.items():
                if joint in self._state:
                    self._state[joint] = clamp(
                        float(value), -self.config.max_joint, self.config.max_joint
                    )

    def get_state(self) -> dict[str, float]:
        with self._state_lock:
            return dict(self._state)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._robot is not None:
            self._robot.disconnect()
