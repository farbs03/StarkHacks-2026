from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from typing import TextIO

import serial


@dataclass(slots=True)
class SensorSnapshot:
    ldr: int
    ir: int
    sound: int
    gas: int
    dist: int
    risk: bool
    timestamp: float


@dataclass(slots=True)
class SensorEvent:
    event_type: str
    severity: str
    sensor_id: str
    value: float
    timestamp: float
    raw: SensorSnapshot


@dataclass(slots=True)
class ParserHealth:
    valid_lines: int = 0
    invalid_lines: int = 0
    last_valid_ts: float = 0.0


def parse_sensor_line(line: str, ts: float | None = None) -> SensorSnapshot | None:
    parts = [part.strip() for part in line.strip().split(",") if part.strip()]
    if len(parts) < 6:
        return None

    values: dict[str, str] = {}
    for part in parts:
        if ":" not in part:
            return None
        key, value = part.split(":", maxsplit=1)
        values[key.strip().upper()] = value.strip()

    required = ["LDR", "IR", "SOUND", "GAS", "DIST", "RISK"]
    if any(key not in values for key in required):
        return None

    now = time.time() if ts is None else ts
    try:
        return SensorSnapshot(
            ldr=int(values["LDR"]),
            ir=int(values["IR"]),
            sound=int(values["SOUND"]),
            gas=int(values["GAS"]),
            dist=int(values["DIST"]),
            risk=values["RISK"] in {"1", "true", "True"},
            timestamp=now,
        )
    except ValueError:
        return None


def parse_sensor_dict(data: dict[str, int], ts: float | None = None) -> SensorSnapshot | None:
    now = time.time() if ts is None else ts
    try:
        return SensorSnapshot(
            ldr=int(data["LDR"]),
            ir=int(data["IR"]),
            sound=int(data["SOUND"]),
            gas=int(data["GAS"]),
            dist=int(data["DIST"]),
            risk=int(data["RISK"]) == 1,
            timestamp=now,
        )
    except (KeyError, ValueError, TypeError):
        return None


class SensorBridge:
    def __init__(
        self,
        *,
        serial_port: str | None = None,
        baud_rate: int = 9600,
        replay_file: str | None = None,
        replay_sleep_sec: float = 0.1,
        event_cooldown_sec: float = 1.0,
        use_sensor_serial_module: bool = False,
    ) -> None:
        if use_sensor_serial_module and replay_file is not None:
            raise ValueError("replay_file is not supported with sensor_serial module mode")
        if not use_sensor_serial_module and serial_port is None and replay_file is None:
            raise ValueError("Provide either serial_port or replay_file for SensorBridge")

        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.replay_file = replay_file
        self.replay_sleep_sec = replay_sleep_sec
        self.event_cooldown_sec = event_cooldown_sec
        self.use_sensor_serial_module = use_sensor_serial_module

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._latest_snapshot: SensorSnapshot | None = None
        self._health = ParserHealth()
        self._event_queue: Queue[SensorEvent] = Queue()
        self._last_event_ts: dict[str, float] = {}

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        if self.use_sensor_serial_module:
            self._run_from_sensor_serial_module()
            return

        if self.serial_port is not None:
            with serial.Serial(self.serial_port, self.baud_rate, timeout=1) as ser:
                self._run_from_serial(ser)
            return

        replay_path = Path(self.replay_file or "")
        with replay_path.open("r", encoding="utf-8") as handle:
            self._run_from_replay(handle)

    def _run_from_sensor_serial_module(self) -> None:
        try:
            from Arduino import sensor_serial
        except Exception as exc:
            raise RuntimeError(
                "Could not import Arduino.sensor_serial. "
                "Ensure Arduino/sensor_serial.py exists and is importable."
            ) from exc

        reader_port = self.serial_port or sensor_serial.DEFAULT_PORT
        reader_baud = self.baud_rate or sensor_serial.DEFAULT_BAUD
        with sensor_serial.ArduinoSensorReader(
            port=reader_port,
            baudrate=reader_baud,
        ) as reader:
            while not self._stop_event.is_set():
                payload = reader.read_sensor_data()
                snapshot = parse_sensor_dict(payload) if payload is not None else None
                if snapshot is None:
                    self._health.invalid_lines += 1
                    continue
                self._latest_snapshot = snapshot
                self._health.valid_lines += 1
                self._health.last_valid_ts = snapshot.timestamp
                self._emit_events(snapshot)

    def _run_from_serial(self, ser: serial.Serial) -> None:
        while not self._stop_event.is_set():
            raw = ser.readline().decode(errors="ignore").strip()
            self._consume_line(raw)

    def _run_from_replay(self, handle: TextIO) -> None:
        while not self._stop_event.is_set():
            line = handle.readline()
            if not line:
                handle.seek(0)
                continue
            self._consume_line(line)
            time.sleep(self.replay_sleep_sec)

    def _consume_line(self, line: str) -> None:
        snapshot = parse_sensor_line(line)
        if snapshot is None:
            self._health.invalid_lines += 1
            return

        self._latest_snapshot = snapshot
        self._health.valid_lines += 1
        self._health.last_valid_ts = snapshot.timestamp
        self._emit_events(snapshot)

    def _emit_events(self, snap: SensorSnapshot) -> None:
        if snap.risk:
            self._emit_event(
                SensorEvent(
                    event_type="sensor_risk",
                    severity="high",
                    sensor_id="arduino",
                    value=1.0,
                    timestamp=snap.timestamp,
                    raw=snap,
                )
            )

        if snap.gas >= 400:
            self._emit_event(
                SensorEvent(
                    event_type="gas_spike",
                    severity="high",
                    sensor_id="gas",
                    value=float(snap.gas),
                    timestamp=snap.timestamp,
                    raw=snap,
                )
            )
        elif snap.sound >= 80:
            self._emit_event(
                SensorEvent(
                    event_type="sound_spike",
                    severity="medium",
                    sensor_id="sound",
                    value=float(snap.sound),
                    timestamp=snap.timestamp,
                    raw=snap,
                )
            )

    def _emit_event(self, event: SensorEvent) -> None:
        key = f"{event.event_type}:{event.sensor_id}"
        now = event.timestamp
        if (now - self._last_event_ts.get(key, 0.0)) < self.event_cooldown_sec:
            return
        self._last_event_ts[key] = now
        self._event_queue.put(event)

    def get_latest_snapshot(self) -> SensorSnapshot | None:
        return self._latest_snapshot

    def pop_events(self, max_events: int = 20) -> list[SensorEvent]:
        events: list[SensorEvent] = []
        for _ in range(max_events):
            try:
                events.append(self._event_queue.get_nowait())
            except Empty:
                break
        return events

    def get_health(self) -> ParserHealth:
        return ParserHealth(
            valid_lines=self._health.valid_lines,
            invalid_lines=self._health.invalid_lines,
            last_valid_ts=self._health.last_valid_ts,
        )

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
