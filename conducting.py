import time
import threading
import numpy as np
import librosa
import sounddevice as sd

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

PORT = "/dev/tty.usbmodem5AE60844281"
AUDIO_FILE = "songs/brandenburg_3_allegro.ogg"

# -----------------------
# ROBOT SETUP
# -----------------------
robot = SO101Follower(SO101FollowerConfig(port=PORT))
robot.connect()

# -----------------------
# TIME SIGNATURE DETECTION
# -----------------------
def estimate_time_signature(y, sr):
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)

    tempo = float(tempo)  # 🔥 FIX

    beat_times = librosa.frames_to_time(beats, sr=sr)

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    beat_strengths = onset_env[beats]
    beat_strengths /= (np.max(beat_strengths) + 1e-6)

    candidates = [2, 3, 4]
    scores = {}

    for meter in candidates:
        groups = len(beat_strengths) // meter
        if groups < 2:
            scores[meter] = 0
            continue

        pattern_scores = []

        for i in range(groups):
            group = beat_strengths[i*meter:(i+1)*meter]
            score = group[0] - np.mean(group[1:])
            pattern_scores.append(score)

        scores[meter] = float(np.mean(pattern_scores))

    print("[DEBUG] meter scores:", scores)

    best_meter = max(scores, key=scores.get)

    return best_meter, tempo, beats


# -----------------------
# LOAD AUDIO
# -----------------------
print("[INFO] Loading audio...")
y, sr = librosa.load(AUDIO_FILE)

TIME_SIG, tempo, beats = estimate_time_signature(y, sr)

print(f"[INFO] Detected: {TIME_SIG}/4 at {tempo:.2f} BPM")

beat_times = librosa.frames_to_time(beats, sr=sr)

# -----------------------
# AUDIO PLAYBACK
# -----------------------
def play_audio():
    print("[INFO] Playing audio...")
    sd.play(y, sr)
    sd.wait()

# -----------------------
# MOTION HELPERS
# -----------------------
SCALE = 30

def pose(x, y):
    return {
        "shoulder_pan.pos": x,
        "shoulder_lift.pos": y,
        "elbow_flex.pos": 20,
        "wrist_flex.pos": y * 0.3,
        "wrist_roll.pos": 0,
        "gripper.pos": 0,
    }

p_up = pose(0, SCALE)
p_down = pose(0, -SCALE)
p_left = pose(-SCALE, 0)
p_right = pose(SCALE, 0)

# -----------------------
# CONDUCTING LOOP (BEAT-SYNCED)
# -----------------------
def conduct():
    print("[INFO] Conducting (beat-synced)...")

    start_time = time.time()

    for i, bt in enumerate(beat_times):
        # wait until exact beat
        while time.time() - start_time < bt:
            time.sleep(0.001)

        beat_idx = i % TIME_SIG

        try:
            if TIME_SIG == 2:
                if beat_idx == 0:
                    robot.send_action(p_down)
                else:
                    robot.send_action(p_up)

            elif TIME_SIG == 3:
                if beat_idx == 0:
                    robot.send_action(p_down)
                elif beat_idx == 1:
                    robot.send_action(p_right)
                else:
                    robot.send_action(p_up)

            elif TIME_SIG == 4:
                if beat_idx == 0:
                    robot.send_action(p_down)
                elif beat_idx == 1:
                    robot.send_action(p_left)
                elif beat_idx == 2:
                    robot.send_action(p_right)
                else:
                    robot.send_action(p_up)

        except Exception as e:
            print("[ERROR]", e)
            break


# -----------------------
# RUN
# -----------------------
try:
    audio_thread = threading.Thread(target=play_audio)
    motion_thread = threading.Thread(target=conduct)

    audio_thread.start()
    motion_thread.start()

    audio_thread.join()
    motion_thread.join()

except KeyboardInterrupt:
    print("\n[INFO] Interrupted")

finally:
    print("[INFO] Releasing torque")
    try:
        robot.disconnect()
    except:
        pass