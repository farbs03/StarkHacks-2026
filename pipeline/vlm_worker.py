from __future__ import annotations

import threading
from queue import Empty, Queue
from typing import Callable

from pipeline.capture_pipeline import CapturePipeline, CaptureRecord


def default_annotator(record: CaptureRecord) -> dict[str, str]:
    reason = record.metadata.get("trigger_reason", "unknown")
    risk_level = record.metadata.get("risk_level", "unknown")
    return {
        "description": f"Auto-generated event summary from {reason}.",
        "tags": [reason, f"risk:{risk_level}"],
        "risk_level": risk_level,
    }


class VLMWorker:
    def __init__(
        self,
        capture_pipeline: CapturePipeline,
        annotator: Callable[[CaptureRecord], dict] | None = None,
    ) -> None:
        self.capture_pipeline = capture_pipeline
        self.annotator = annotator or default_annotator
        self._queue: Queue[CaptureRecord] = Queue()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, record: CaptureRecord) -> None:
        self._queue.put(record)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                record = self._queue.get(timeout=0.2)
            except Empty:
                continue
            patch = self.annotator(record)
            self.capture_pipeline.update_metadata(record, patch)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
