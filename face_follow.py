import argparse
import signal
import sys
import threading
import time

import cv2
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig


# python face_follow.py --port COM3 --camera-index 1 --show-preview

DEFAULT_STATE = {
    "shoulder_pan.pos": 0.0,
    "shoulder_lift.pos": 0.0,
    "elbow_flex.pos": 0.0,
    "wrist_flex.pos": 0.0,
    "wrist_roll.pos": 0.0,
    "gripper.pos": 0.0,
}


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect a face and steer SO-101 camera toward it."
    )
    parser.add_argument("--port", default="COM3", help="Serial port for follower arm")
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="OpenCV camera index (0 = default webcam)",
    )
    parser.add_argument(
        "--send-hz", type=float, default=30.0, help="Robot send loop frequency"
    )
    parser.add_argument(
        "--pan-gain",
        type=float,
        default=45.0,
        help="Gain for horizontal correction (higher = more aggressive)",
    )
    parser.add_argument(
        "--lift-gain",
        type=float,
        default=35.0,
        help="Gain for vertical correction (higher = more aggressive)",
    )
    parser.add_argument(
        "--deadzone",
        type=float,
        default=0.07,
        help="Ignore tracking error magnitude below this normalized threshold",
    )
    parser.add_argument(
        "--max-step",
        type=float,
        default=2.2,
        help="Max joint change per frame in arm command units",
    )
    parser.add_argument(
        "--max-joint",
        type=float,
        default=100.0,
        help="Clamp for shoulder commands (matches teleop range)",
    )
    parser.add_argument(
        "--show-preview",
        action="store_true",
        help="Show camera preview window with detections",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state = dict(DEFAULT_STATE)
    state_lock = threading.Lock()
    stop_event = threading.Event()

    robot = SO101Follower(
        SO101FollowerConfig(
            port=args.port,
            id="follower",
        )
    )

    print(f"[INFO] Connecting to robot on {args.port}...")
    robot.connect()
    print("[OK] Robot connected")

    def shutdown(*_args) -> None:
        if stop_event.is_set():
            return
        print("\n[INFO] Shutting down...")
        stop_event.set()
        try:
            robot.disconnect()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    def send_loop() -> None:
        period = 1.0 / args.send_hz
        while not stop_event.is_set():
            with state_lock:
                robot.send_action(dict(state))
            time.sleep(period)

    sender = threading.Thread(target=send_loop, daemon=True)
    sender.start()

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        shutdown()
        raise RuntimeError(
            f"Could not open camera index {args.camera_index}. "
            "Try another --camera-index value."
        )

    face_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    if face_detector.empty():
        shutdown()
        raise RuntimeError("Failed to load OpenCV haarcascade_frontalface_default.xml")

    print("[INFO] Face tracking started. Press 'q' in preview window to quit.")
    while not stop_event.is_set():
        ok, frame = cap.read()
        if not ok:
            print("[WARN] Camera read failed; retrying...")
            time.sleep(0.05)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40),
        )

        frame_h, frame_w = frame.shape[:2]
        frame_cx = frame_w / 2.0
        frame_cy = frame_h / 2.0

        if len(faces) > 0:
            # Track the largest detected face to reduce jitter.
            x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
            face_cx = x + (w / 2.0)
            face_cy = y + (h / 2.0)

            err_x = (face_cx - frame_cx) / frame_w
            err_y = (face_cy - frame_cy) / frame_h

            if abs(err_x) < args.deadzone:
                err_x = 0.0
            if abs(err_y) < args.deadzone:
                err_y = 0.0

            step_pan = clamp(err_x * args.pan_gain, -args.max_step, args.max_step)
            # Positive vertical pixel error means face is lower in frame.
            step_lift = clamp(err_y * args.lift_gain, -args.max_step, args.max_step)

            with state_lock:
                state["shoulder_pan.pos"] = clamp(
                    state["shoulder_pan.pos"] + step_pan,
                    -args.max_joint,
                    args.max_joint,
                )
                state["shoulder_lift.pos"] = clamp(
                    state["shoulder_lift.pos"] + step_lift,
                    -args.max_joint,
                    args.max_joint,
                )

            if args.show_preview:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 220, 50), 2)
                cv2.circle(frame, (int(face_cx), int(face_cy)), 4, (50, 220, 50), -1)

        if args.show_preview:
            cv2.circle(frame, (int(frame_cx), int(frame_cy)), 5, (0, 180, 255), -1)
            cv2.imshow("SO-101 Face Follow", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                shutdown()
                break

    cap.release()
    shutdown()
    sys.exit(0)


if __name__ == "__main__":
    main()
