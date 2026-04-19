from __future__ import annotations

import base64
import json
import os
import re
import threading
from pathlib import Path
from queue import Empty, Queue
from typing import Callable

import requests

from pipeline.capture_pipeline import CapturePipeline, CaptureRecord


def default_annotator(record: CaptureRecord) -> dict[str, str]:
    reason = record.metadata.get("trigger_reason", "unknown")
    risk_level = record.metadata.get("risk_level", "unknown")
    return {
        "description": f"Auto-generated event summary from {reason}.",
        "tags": [reason, f"risk:{risk_level}"],
        "risk_level": risk_level,
    }


def _extract_json_object(text: str) -> dict | None:
    text = text.strip()

    candidates = [text]
    fenced_blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    candidates.extend(block.strip() for block in fenced_blocks if block.strip())

    decoder = json.JSONDecoder()
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

        for idx, ch in enumerate(candidate):
            if ch != "{":
                continue
            try:
                obj, _ = decoder.raw_decode(candidate[idx:])
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                return obj
    return None


def _decode_json_string_literal(value: str) -> str:
    try:
        return str(json.loads(f'"{value}"'))
    except Exception:
        return value


def _extract_string_field(text: str, field: str) -> str:
    pattern = rf'"{re.escape(field)}"\s*:\s*"((?:\\.|[^"\\])*)"'
    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    return _decode_json_string_literal(match.group(1)).strip()


def _extract_string_array_field(text: str, field: str) -> list[str]:
    pattern = rf'"{re.escape(field)}"\s*:\s*\[(.*?)\]'
    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return []
    body = match.group(1)
    values = re.findall(r'"((?:\\.|[^"\\])*)"', body, flags=re.DOTALL)
    return [_decode_json_string_literal(v).strip() for v in values if v.strip()]


def _recover_annotation_from_text(
    text: str,
    *,
    default_risk_level: str,
    default_tag: str,
) -> dict | None:
    description = _extract_string_field(text, "description")
    risk_level = _extract_string_field(text, "risk_level").lower()
    tags = _extract_string_array_field(text, "tags")
    key_factors = _extract_string_array_field(text, "key_factors")

    if not description and not risk_level and not tags and not key_factors:
        return None

    return {
        "description": description or text.strip()[:800],
        "risk_level": risk_level or default_risk_level,
        "tags": tags or [default_tag],
        "key_factors": key_factors or ["partial_json_recovered"],
    }


def _annotation_patch_from_content(content: str, record: CaptureRecord) -> dict:
    parsed = _extract_json_object(content)
    if isinstance(parsed, dict):
        return {
            "description": str(parsed.get("description", "")),
            "risk_level": str(
                parsed.get("risk_level", record.metadata.get("risk_level", "medium"))
            ).lower(),
            "tags": parsed.get("tags", []),
            "key_factors": parsed.get("key_factors", []),
            "_vlm_raw_response": str(content),
        }

    recovered = _recover_annotation_from_text(
        content,
        default_risk_level=str(record.metadata.get("risk_level", "medium")).lower(),
        default_tag=str(record.metadata.get("trigger_reason", "vlm")),
    )
    if recovered is not None:
        recovered["_vlm_raw_response"] = str(content)
        recovered["vlm_parse_warning"] = "Recovered partial/non-strict JSON response."
        return recovered

    return {
        "description": str(content).strip()[:800],
        "risk_level": record.metadata.get("risk_level", "medium"),
        "tags": [record.metadata.get("trigger_reason", "vlm")],
        "key_factors": ["non_json_response"],
        "_vlm_raw_response": str(content),
    }


def _write_vlm_raw_response(record: CaptureRecord, raw_text: str) -> str:
    metadata_path = Path(record.metadata_path)
    base_name = metadata_path.stem
    raw_path = metadata_path.with_name(f"{base_name}.vlm_raw.txt")
    raw_path.write_text(raw_text, encoding="utf-8")
    return str(raw_path)


def build_openai_compatible_annotator(
    *,
    model: str,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout_sec: float = 45.0,
) -> Callable[[CaptureRecord], dict]:
    resolved_base = (base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip(
        "/"
    )
    resolved_key = api_key or os.getenv("OPENAI_API_KEY", "")

    local_base = (
        resolved_base.startswith("http://localhost")
        or resolved_base.startswith("http://127.0.0.1")
    )
    if not resolved_key and not local_base:
        raise ValueError(
            "Missing API key for VLM annotator. Set OPENAI_API_KEY or pass --vlm-api-key. "
            "For local endpoints (localhost/127.0.0.1), API key is optional."
        )

    endpoint = f"{resolved_base}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if resolved_key:
        headers["Authorization"] = f"Bearer {resolved_key}"

    def annotator(record: CaptureRecord) -> dict:
        image_path = record.metadata.get("photo_path", record.image_path)
        with open(image_path, "rb") as handle:
            image_b64 = base64.b64encode(handle.read()).decode("ascii")

        metadata_json = json.dumps(record.metadata, indent=2)
        prompt = (
            "You are a safety monitoring assistant.\n"
            "Given the image and structured metadata, produce a concise event summary.\n"
            "Consider both visual cues and sensor metadata.\n"
            "Return ONLY valid JSON with keys:\n"
            '  description: string,\n'
            '  risk_level: one of ["low","medium","high"],\n'
            "  tags: array of short strings,\n"
            "  key_factors: array of short strings.\n\n"
            "Event metadata:\n"
            f"{metadata_json}\n"
        )

        payload = {
            "model": model,
            "temperature": 0.2,
            "max_tokens": 260,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                        },
                    ],
                }
            ],
        }
        response = requests.post(
            endpoint,
            headers=headers,
            json=payload,
            timeout=timeout_sec,
        )
        response.raise_for_status()
        data = response.json()
        content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        return _annotation_patch_from_content(str(content), record)

    return annotator


def build_gemini_annotator(
    *,
    model: str,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout_sec: float = 45.0,
) -> Callable[[CaptureRecord], dict]:
    resolved_key = api_key or os.getenv("GEMINI_API_KEY", "")
    if not resolved_key:
        raise ValueError(
            "Missing Gemini API key. Set GEMINI_API_KEY or pass --vlm-api-key."
        )
    resolved_base = (base_url or "https://generativelanguage.googleapis.com/v1beta").rstrip(
        "/"
    )
    endpoint = f"{resolved_base}/models/{model}:generateContent"

    def annotator(record: CaptureRecord) -> dict:
        image_path = record.metadata.get("photo_path", record.image_path)
        with open(image_path, "rb") as handle:
            image_b64 = base64.b64encode(handle.read()).decode("ascii")

        metadata_json = json.dumps(record.metadata, indent=2)
        prompt = (
            "You are a safety monitoring assistant.\n"
            "Given the image and structured metadata, produce a concise event summary.\n"
            "Consider both visual cues and sensor metadata.\n"
            "Return ONLY valid JSON with keys:\n"
            '  description: string,\n'
            '  risk_level: one of ["low","medium","high"],\n'
            "  tags: array of short strings,\n"
            "  key_factors: array of short strings.\n\n"
            "Event metadata:\n"
            f"{metadata_json}\n"
        )

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": image_b64,
                            }
                        },
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 512,
                "response_mime_type": "application/json",
            },
        }
        response = requests.post(
            endpoint,
            headers={"Content-Type": "application/json"},
            json=payload,
            params={"key": resolved_key},
            timeout=timeout_sec,
        )
        if response.status_code >= 400:
            try:
                err_json = response.json()
                err_msg = (
                    err_json.get("error", {}).get("message")
                    or err_json.get("message")
                    or str(err_json)
                )
            except Exception:
                err_msg = response.text.strip()[:500]
            if response.status_code == 404:
                raise RuntimeError(
                    "Gemini API returned 404 (model not found / unavailable for this endpoint). "
                    f"Tried model '{model}'. "
                    "Try model 'gemini-2.0-flash' or 'gemini-1.5-flash-latest'. "
                    f"API message: {err_msg}"
                )
            raise RuntimeError(
                f"Gemini API error {response.status_code}: {err_msg}"
            )
        data = response.json()
        parts = (
            data.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [])
        )
        content = "\n".join(str(part.get("text", "")) for part in parts if isinstance(part, dict))
        return _annotation_patch_from_content(str(content), record)

    return annotator


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
            try:
                patch = self.annotator(record)
            except Exception as exc:
                patch = default_annotator(record)
                patch["vlm_error"] = str(exc)
                patch["_vlm_raw_response"] = str(exc)

            raw_text = str(patch.pop("_vlm_raw_response", "")).strip()
            if raw_text:
                try:
                    patch["vlm_raw_response_path"] = _write_vlm_raw_response(record, raw_text)
                except Exception as exc:
                    patch["vlm_raw_log_error"] = str(exc)
            self.capture_pipeline.update_metadata(record, patch)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
