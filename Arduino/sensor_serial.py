"""
Read and parse real-time sensor lines from an Arduino over serial.

Expected line format (one line per message):
    LDR:<int>,IR:<int>,SOUND:<int>,GAS:<int>,DIST:<int>,RISK:<0_or_1>
"""

from __future__ import annotations

import atexit
import sys
import time
from typing import Final

import serial

SENSOR_KEYS: Final[tuple[str, ...]] = (
    "LDR",
    "IR",
    "SOUND",
    "GAS",
    "DIST",
    "RISK",
)

DEFAULT_PORT: Final[str] = "COM4"
DEFAULT_BAUD: Final[int] = 9600

_default_serial: serial.Serial | None = None
_default_serial_announced: bool = False


def parse_sensor_line(line: str) -> dict[str, int] | None:
    """
    Parse a single sensor line into a dictionary of integers.

    Returns None if the line is empty, incomplete, or malformed.
    """
    text = line.strip()
    if not text:
        return None

    parts = text.split(",")
    if len(parts) != len(SENSOR_KEYS):
        return None

    result: dict[str, int] = {}
    for expected_key, segment in zip(SENSOR_KEYS, parts, strict=True):
        if ":" not in segment:
            return None
        key, _, value_str = segment.partition(":")
        key = key.strip()
        value_str = value_str.strip()
        if key != expected_key:
            return None
        try:
            result[expected_key] = int(value_str)
        except ValueError:
            return None

    return result


def _read_and_parse_line(ser: serial.Serial) -> dict[str, int] | None:
    """Read one line from an open port, parse, print on success; never raises."""
    if not ser.is_open:
        return None

    try:
        raw = ser.readline()
    except serial.SerialException:
        return None

    line = raw.decode("utf-8", errors="replace")

    parsed = parse_sensor_line(line)
    if parsed is not None:
        print(parsed, flush=True)
    return parsed


def _ensure_default_serial() -> serial.Serial:
    global _default_serial, _default_serial_announced
    if _default_serial is None or not _default_serial.is_open:
        _default_serial = serial.Serial(
            port=DEFAULT_PORT,
            baudrate=DEFAULT_BAUD,
            timeout=1.0,
        )
        time.sleep(2)
        if _default_serial.in_waiting:
            _default_serial.reset_input_buffer()
        if not _default_serial_announced:
            print(
                f"Serial open on {DEFAULT_PORT} @ {DEFAULT_BAUD} baud; "
                "waiting for lines like LDR:n,IR:n,...",
                flush=True,
            )
            _default_serial_announced = True
    return _default_serial


def shutdown_default_serial() -> None:
    """Close the lazily opened default COM port (safe to call multiple times)."""
    global _default_serial, _default_serial_announced
    if _default_serial is not None and _default_serial.is_open:
        _default_serial.close()
    _default_serial = None
    _default_serial_announced = False


atexit.register(shutdown_default_serial)


def read_sensor_data() -> dict[str, int] | None:
    """
    Read one line from the default serial port (COM4 @ 9600), parse, return dict.

    Returns None if reading or parsing fails. On success, prints the dict for
    debugging. Prefer :class:`ArduinoSensorReader` when you need explicit
    lifecycle control instead of the module-level default port.
    """
    ser = _ensure_default_serial()
    return _read_and_parse_line(ser)


class ArduinoSensorReader:
    """
    Blocking serial reader for Arduino sensor lines.

    Use as a context manager so the port is opened and closed cleanly.
    """

    def __init__(self, port: str = DEFAULT_PORT, baudrate: int = DEFAULT_BAUD) -> None:
        self._port = port
        self._baudrate = baudrate
        self._serial: serial.Serial | None = None

    def __enter__(self) -> ArduinoSensorReader:
        self._serial = serial.Serial(
            port=self._port,
            baudrate=self._baudrate,
            timeout=1.0,
        )
        time.sleep(2)
        if self._serial.in_waiting:
            self._serial.reset_input_buffer()
        print(
            f"Serial open on {self._port} @ {self._baudrate} baud; "
            "waiting for lines like LDR:n,IR:n,...",
            flush=True,
        )
        return self

    def __exit__(self, *args: object) -> None:
        if self._serial is not None and self._serial.is_open:
            self._serial.close()
        self._serial = None

    def read_sensor_data(self) -> dict[str, int] | None:
        """
        Read one line from this reader's serial port.

        Same contract as :func:`read_sensor_data` (including debug print).
        """
        if self._serial is None or not self._serial.is_open:
            return None
        return _read_and_parse_line(self._serial)


def stream_sensor_data(port: str = DEFAULT_PORT, baudrate: int = DEFAULT_BAUD) -> None:
    """Continuously read lines using an explicit context-managed serial port."""
    try:
        with ArduinoSensorReader(port=port, baudrate=baudrate) as reader:
            while True:
                reader.read_sensor_data()
    except (serial.SerialException, OSError) as exc:
        print(f"Serial error: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    try:
        while True:
            read_sensor_data()
    except (serial.SerialException, OSError) as exc:
        print(f"Serial error: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        shutdown_default_serial()
