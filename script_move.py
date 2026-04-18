import tkinter as tk
import time
import threading
import signal
import sys

from lerobot.robots.so_follower import SO101FollowerConfig, SO101Follower


PORT = "/dev/tty.usbmodem5AE60844281"
#PORT = "COM3"


robot = SO101Follower(
    SO101FollowerConfig(
        port=PORT,
        id="follower",
    )
)

print("[INFO] Connecting...")
robot.connect()
print("[OK] Connected")


state = {
    "shoulder_pan.pos": 0.0,
    "shoulder_lift.pos": 0.0,
    "elbow_flex.pos": 0.0,
    "wrist_flex.pos": 0.0,
    "wrist_roll.pos": 0.0,
    "gripper.pos": 0.0,
}


def send_loop():
    while True:
        robot.send_action(state)


threading.Thread(target=send_loop, daemon=True).start()


root = tk.Tk()
root.title("SO-101 Teleop GUI")


def add_slider(name, row):
    def update(val):
        state[name] = float(val)

    tk.Label(root, text=name).grid(row=row, column=0)

    s = tk.Scale(
        root,
        from_=-150,
        to=150,
        resolution=1,
        orient=tk.HORIZONTAL,
        length=400,
        command=update,
    )
    s.set(0.0)
    s.grid(row=row, column=1)


joints = list(state.keys())

for i, j in enumerate(joints):
    add_slider(j, i)

def shutdown(*args):
    print("\n[INFO] SHUTDOWN: disabling torque")
    try:
        robot.disconnect()   # or robot.bus.disable_torque()
    except:
        pass
    sys.exit(0)

signal.signal(signal.SIGINT, shutdown)

root.mainloop()