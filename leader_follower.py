import time
import traceback

from lerobot.teleoperators.so_leader import SO101LeaderConfig, SO101Leader
from lerobot.robots.so_follower import SO101FollowerConfig, SO101Follower


LEADER_PORT = "/dev/tty.usbmodem5AE60536821"
FOLLOWER_PORT = "/dev/tty.usbmodem5AE60844281"


def main():
    print("=== LeRobot SO-101 Teleop Debug Script ===")

    print(f"[INFO] Leader port: {LEADER_PORT}")
    print(f"[INFO] Follower port: {FOLLOWER_PORT}")

    # Configs
    robot_config = SO101FollowerConfig(
        port=FOLLOWER_PORT,
        id="follower_arm",
    )

    teleop_config = SO101LeaderConfig(
        port=LEADER_PORT,
        id="leader_arm",
    )

    # Initialize
    print("[INFO] Initializing devices...")
    robot = SO101Follower(robot_config)
    teleop_device = SO101Leader(teleop_config)

    # Connect
    print("[INFO] Connecting to follower...")
    robot.connect()
    print("[SUCCESS] Follower connected")

    print("[INFO] Connecting to leader...")
    teleop_device.connect()
    print("[SUCCESS] Leader connected")

    print("[INFO] Starting teleoperation loop...")
    print("[INFO] Move the leader arm to test\n")

    step = 0

    try:
        while True:
            step += 1

            action = teleop_device.get_action()

            if action is None:
                print(f"[WARN] Step {step}: No action received")
                time.sleep(0.05)
                continue

            # Print occasionally so logs aren’t insane
            if step % 20 == 0:
                print(f"[DEBUG] Step {step}: action = {action}")

            robot.send_action(action)

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user")

    except Exception as e:
        print("\n[ERROR] Exception occurred:")
        traceback.print_exc()

    finally:
        print("[INFO] Cleaning up (if supported)...")
        try:
            robot.disconnect()
        except Exception:
            pass

        try:
            teleop_device.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()