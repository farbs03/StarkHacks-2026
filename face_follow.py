import argparse
import collections
import json
import signal
import sys
import time

import cv2

from control.arm_controller import ArmControllerConfig, SOArmController
from control.tracking_controller import ControlCommand, TrackingController
from pipeline.capture_pipeline import CapturePipeline
from pipeline.poi_exporter import POIExporter
from pipeline.vlm_worker import VLMWorker, default_annotator
from sensors.event_router import EventRouter
from sensors.sensor_bridge import SensorBridge
from vision.detectors import (
    ConstructionHazardOnnxDetector,
    Detection,
    HaarFaceDetector,
    UltralyticsOnnxDetector,
    YoloOnnxDetector,
)
from vision.target_selection import TargetSelector


def parse_priority_map(raw: str) -> dict[str, int]:
    out: dict[str, int] = {}
    if not raw.strip():
        return out
    for token in raw.split(","):
        if "=" not in token:
            continue
        key, value = token.split("=", maxsplit=1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        try:
            out[key] = int(value)
        except ValueError:
            continue
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Face/Hazard tracking pipeline with SO-101 + optional Arduino sensor fusion."
    )
    parser.add_argument("--port", default="COM3", help="Serial port for SO-101 follower")
    parser.add_argument("--camera-index", type=int, default=0, help="OpenCV camera index")
    parser.add_argument("--send-hz", type=float, default=30.0, help="Arm send loop Hz")
    parser.add_argument("--max-joint", type=float, default=100.0, help="Joint command clamp")
    parser.add_argument("--show-preview", action="store_true", help="Show OpenCV preview")
    parser.add_argument("--dry-run-arm", action="store_true", help="Skip hardware arm connection")

    parser.add_argument(
        "--detector",
        choices=["haar", "yolo", "construction_hazard"],
        default="haar",
    )
    parser.add_argument("--yolo-onnx", default="", help="Path to YOLO ONNX model")
    parser.add_argument(
        "--construction-hazard-onnx",
        default="",
        help="Path to Construction-Hazard-Detection ONNX model",
    )
    parser.add_argument(
        "--yolo-labels",
        default="",
        help="Comma-separated class labels for YOLO classes",
    )
    parser.add_argument("--detector-conf", type=float, default=0.35)
    parser.add_argument("--detector-nms", type=float, default=0.45)
    parser.add_argument("--detector-input-size", type=int, default=640)
    parser.add_argument(
        "--detector-backend",
        choices=["auto", "opencv", "onnxruntime"],
        default="auto",
        help="Inference backend for ultralytics-style ONNX models",
    )
    parser.add_argument("--max-detections", type=int, default=120)
    parser.add_argument(
        "--detector-interval-sec",
        type=float,
        default=0.0,
        help="Minimum seconds between detector queries (0 = every frame)",
    )
    parser.add_argument(
        "--debug-detections",
        action="store_true",
        help="Print per-second detection label counts for detector debugging",
    )
    parser.add_argument(
        "--priority-map",
        default="fire=100,smoke=95,gas=90,hazard=85,person=70,face=60",
        help="Target priority map (label=priority,...)",
    )

    parser.add_argument("--pan-gain", type=float, default=45.0)
    parser.add_argument("--lift-gain", type=float, default=35.0)
    parser.add_argument("--deadzone", type=float, default=0.07)
    parser.add_argument("--max-step", type=float, default=2.2)
    parser.add_argument("--ema-alpha", type=float, default=0.35)
    parser.add_argument("--frames-before-scan", type=int, default=10)
    parser.add_argument("--scan-step", type=float, default=0.7)

    parser.add_argument("--sensor-port", default="", help="Arduino serial port (optional)")
    parser.add_argument("--sensor-baud", type=int, default=9600)
    parser.add_argument(
        "--sensor-source",
        choices=["bridge", "sensor_serial"],
        default="bridge",
        help="Sensor reader source: direct bridge parser or Arduino/sensor_serial.py",
    )
    parser.add_argument(
        "--sensor-replay",
        default="",
        help="Path to replay file with SensorRead.ino style telemetry lines",
    )
    parser.add_argument("--sensor-cooldown", type=float, default=1.0)

    parser.add_argument("--enable-capture", action="store_true")
    parser.add_argument("--capture-dir", default="photos")
    parser.add_argument("--poi-jsonl", default="captures/poi.jsonl")
    parser.add_argument("--capture-cooldown", type=float, default=3.0)
    parser.add_argument("--visual-score-threshold", type=float, default=0.6)
    parser.add_argument("--enable-vlm-worker", action="store_true")
    parser.add_argument(
        "--default-depth-m",
        type=float,
        default=1.5,
        help="Fallback depth used when ultrasonic DIST is unavailable",
    )
    parser.add_argument(
        "--camera-hfov-deg",
        type=float,
        default=70.0,
        help="Approximate horizontal FOV for pseudo-3D projection",
    )
    parser.add_argument(
        "--camera-vfov-deg",
        type=float,
        default=43.0,
        help="Approximate vertical FOV for pseudo-3D projection",
    )
    return parser.parse_args()


def build_detector(args: argparse.Namespace):
    if args.detector == "haar":
        return HaarFaceDetector()
    if args.detector == "construction_hazard":
        model_path = args.construction_hazard_onnx or args.yolo_onnx
        if not model_path:
            raise ValueError(
                "--construction-hazard-onnx (or --yolo-onnx) is required when "
                "--detector construction_hazard"
            )
        return ConstructionHazardOnnxDetector(
            model_path=model_path,
            conf_threshold=args.detector_conf,
            nms_threshold=args.detector_nms,
            input_size=args.detector_input_size,
            max_detections=args.max_detections,
            backend=args.detector_backend,
        )
    if not args.yolo_onnx:
        raise ValueError("--yolo-onnx must be provided when --detector yolo")
    labels = [token.strip() for token in args.yolo_labels.split(",") if token.strip()]
    if labels:
        return UltralyticsOnnxDetector(
            model_path=args.yolo_onnx,
            class_names=labels,
            conf_threshold=args.detector_conf,
            nms_threshold=args.detector_nms,
            input_size=args.detector_input_size,
            max_detections=args.max_detections,
            backend=args.detector_backend,
        )
    return YoloOnnxDetector(
        model_path=args.yolo_onnx,
        class_names=labels,
        conf_threshold=args.detector_conf,
        nms_threshold=args.detector_nms,
        input_size=(args.detector_input_size, args.detector_input_size),
    )


def draw_preview(
    frame,
    detections: list[Detection],
    selected: Detection | None,
    mode: str,
    smoothed_target: tuple[float, float] | None,
) -> None:
    h, w = frame.shape[:2]
    frame_center = (int(w / 2), int(h / 2))
    cv2.circle(frame, frame_center, 4, (0, 180, 255), -1)
    for det in detections:
        x1, y1, x2, y2 = det.bbox_xyxy
        color = (50, 220, 50) if selected and det == selected else (255, 180, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"{det.label}:{det.score:.2f}",
            (x1, max(14, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    if smoothed_target is not None:
        tx = int(smoothed_target[0] * w)
        ty = int(smoothed_target[1] * h)
        cv2.circle(frame, (tx, ty), 4, (255, 50, 180), -1)
    cv2.putText(
        frame,
        f"mode={mode}",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (230, 230, 230),
        2,
        cv2.LINE_AA,
    )


def main() -> None:
    args = parse_args()

    arm = SOArmController(
        ArmControllerConfig(
            port=args.port,
            send_hz=args.send_hz,
            max_joint=args.max_joint,
            dry_run=args.dry_run_arm,
        )
    )
    detector = build_detector(args)
    target_selector = TargetSelector(class_priority=parse_priority_map(args.priority_map))
    tracker = TrackingController(
        pan_gain=args.pan_gain,
        lift_gain=args.lift_gain,
        deadzone=args.deadzone,
        max_step=args.max_step,
        max_joint=args.max_joint,
        ema_alpha=args.ema_alpha,
        frames_before_scan=args.frames_before_scan,
        scan_step_size=args.scan_step,
    )
    event_router = EventRouter(
        capture_cooldown_sec=args.capture_cooldown,
        visual_score_threshold=args.visual_score_threshold,
    )

    sensor_bridge = None
    if args.sensor_port or args.sensor_replay or args.sensor_source == "sensor_serial":
        sensor_bridge = SensorBridge(
            serial_port=args.sensor_port or None,
            baud_rate=args.sensor_baud,
            replay_file=args.sensor_replay or None,
            event_cooldown_sec=args.sensor_cooldown,
            use_sensor_serial_module=(args.sensor_source == "sensor_serial"),
        )
        sensor_bridge.start()

    capture_pipeline = None
    poi_exporter = None
    vlm_worker = None
    if args.enable_capture:
        capture_pipeline = CapturePipeline(output_dir=args.capture_dir, write_overlay=True)
        poi_exporter = POIExporter(output_jsonl=args.poi_jsonl)
        if args.enable_vlm_worker:
            vlm_worker = VLMWorker(capture_pipeline=capture_pipeline)
            vlm_worker.start()

    print(f"[INFO] Connecting to robot on {args.port}...")
    arm.connect()
    arm.start()
    print("[OK] Robot loop started")

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera index {args.camera_index}. Try another --camera-index."
        )

    should_stop = False

    def shutdown(*_args) -> None:
        nonlocal should_stop
        should_stop = True

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    frame_id = 0
    previous_target: tuple[float, float] | None = None
    last_telemetry_ts = time.time()
    frame_counter = 0
    label_counter: collections.Counter[str] = collections.Counter()
    last_detection_ts = -1e9
    cached_detections: list[Detection] = []

    print("[INFO] Tracking started. Press 'q' in preview window to stop.")
    try:
        while not should_stop:
            ok, frame = cap.read()
            if not ok:
                print("[WARN] Camera read failed; retrying...")
                time.sleep(0.05)
                continue

            frame_id += 1
            frame_counter += 1
            now = time.time()
            should_run_detector = (
                args.detector_interval_sec <= 0.0
                or (now - last_detection_ts) >= args.detector_interval_sec
                or not cached_detections
            )
            if should_run_detector:
                cached_detections = detector.detect(frame, frame_id=frame_id, ts=now)
                last_detection_ts = now
                if args.debug_detections:
                    for det in cached_detections:
                        label_counter[det.label] += 1

            detections = cached_detections
            selected_detection = None
            target_center = None
            arm_state = arm.get_state()

            if should_run_detector:
                selected_target = target_selector.select(
                    detections,
                    frame.shape,
                    previous_center_norm=previous_target,
                )
                selected_detection = (
                    selected_target.detection if selected_target is not None else None
                )
                target_center = (
                    selected_target.center_norm_xy if selected_target is not None else None
                )
                previous_target = target_center
                command = tracker.update(
                    target_center,
                    current_pan=arm_state["shoulder_pan.pos"],
                )
            else:
                # Hold the arm perfectly still between detector queries.
                command = ControlCommand(
                    pan_delta=0.0,
                    lift_delta=0.0,
                    mode="hold_between_queries",
                    smoothed_target=tracker.smoothed_target,
                )
            arm.update_pan_lift(command.pan_delta, command.lift_delta)

            sensor_snapshot = sensor_bridge.get_latest_snapshot() if sensor_bridge else None
            sensor_events = sensor_bridge.pop_events() if sensor_bridge else []
            # Trigger vision events only on fresh detector results.
            decision = event_router.decide(
                selected_detection if should_run_detector else None,
                sensor_events,
            )

            if decision.should_capture and capture_pipeline and poi_exporter:
                state_for_capture = arm.get_state()
                record = capture_pipeline.capture(
                    frame=frame,
                    detection=selected_detection,
                    detections=detections,
                    arm_state=state_for_capture,
                    sensor_snapshot=sensor_snapshot,
                    trigger_reason=decision.reason or "unknown",
                    risk_level=decision.risk_level,
                    trigger_score=decision.trigger_score,
                )
                if vlm_worker is not None:
                    vlm_worker.submit(record)
                else:
                    capture_pipeline.update_metadata(record, default_annotator(record))
                poi_exporter.export(
                    record=record,
                    arm_state=state_for_capture,
                    event_type=decision.reason or "unknown",
                    confidence=decision.trigger_score,
                    selected_detection=selected_detection,
                    frame_shape=frame.shape,
                    sensor_snapshot=sensor_snapshot,
                    default_depth_m=args.default_depth_m,
                    horizontal_fov_deg=args.camera_hfov_deg,
                    vertical_fov_deg=args.camera_vfov_deg,
                )

            if args.show_preview:
                draw_preview(
                    frame,
                    detections=detections,
                    selected=selected_detection if should_run_detector else None,
                    mode=command.mode,
                    smoothed_target=command.smoothed_target,
                )
                cv2.imshow("SO-101 Face/Hazard Follow", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    should_stop = True

            if (now - last_telemetry_ts) >= 1.0:
                fps = frame_counter / (now - last_telemetry_ts)
                state = arm.get_state()
                telemetry = {
                    "timestamp": now,
                    "mode": command.mode,
                    "fps": round(fps, 2),
                    "detections": len(detections),
                    "detector_ran_this_frame": should_run_detector,
                    "detector_interval_sec": args.detector_interval_sec,
                    "selected_label": selected_detection.label if selected_detection else None,
                    "selected_score": (
                        round(selected_detection.score, 3)
                        if selected_detection is not None
                        else None
                    ),
                    "joints": {
                        "shoulder_pan.pos": round(state["shoulder_pan.pos"], 3),
                        "shoulder_lift.pos": round(state["shoulder_lift.pos"], 3),
                    },
                }
                if sensor_bridge:
                    health = sensor_bridge.get_health()
                    telemetry["sensor_health"] = {
                        "valid_lines": health.valid_lines,
                        "invalid_lines": health.invalid_lines,
                        "last_valid_ts": health.last_valid_ts,
                    }
                if args.debug_detections:
                    telemetry["label_counts"] = dict(label_counter.most_common(8))
                    label_counter.clear()
                print(json.dumps(telemetry))
                last_telemetry_ts = now
                frame_counter = 0
    finally:
        cap.release()
        if args.show_preview:
            cv2.destroyAllWindows()
        if sensor_bridge is not None:
            sensor_bridge.stop()
        if vlm_worker is not None:
            vlm_worker.stop()
        arm.stop()
        print("[INFO] Shutdown complete.")
    sys.exit(0)


if __name__ == "__main__":
    main()
