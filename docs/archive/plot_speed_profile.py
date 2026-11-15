from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from src.racing_env import RacingEnv
from src.track_geometry import TrackGeometry
from src.track_factory import make_wavy_track


project_dir = Path(__file__).resolve().parent


# =========================================================
# LOAD MODEL
# =========================================================

model = PPO.load(
    project_dir
    / "models"
    / "ppo_hardfivefamily_stage1"
)

print("Final model loaded!")


# =========================================================
# BUILD WAVY TRACK
# =========================================================

centerline = make_wavy_track(
    num_points=720
)

track = TrackGeometry(
    centerline_points=centerline,
    track_width=12.0,
    num_checkpoints=7
)

env = RacingEnv(
    track=track,
    random_start=False
)


# =========================================================
# RUN ONE EPISODE
# =========================================================

observation, info = env.reset(
    options={
        "start_index": 0
    }
)

terminated = False
truncated = False

previous_progress = info.get(
    "progress",
    info["start_progress"]
)

cumulative_progress = 0.0

progress_values = [0.0]
speed_values = [env.sim.car.speed]

while not (
    terminated
    or truncated
):
    action, _ = model.predict(
        observation,
        deterministic=True
    )

    (
        observation,
        reward,
        terminated,
        truncated,
        info
    ) = env.step(action)

    current_progress = info["progress"]

    progress_delta = (
        current_progress
        - previous_progress
    )

    if progress_delta < -50:
        progress_delta += 100
    elif progress_delta > 50:
        progress_delta -= 100

    cumulative_progress += progress_delta

    progress_values.append(
        cumulative_progress
    )

    speed_values.append(
        info["speed"]
    )

    previous_progress = current_progress

    if info["lap_completed"]:
        break


progress_values = np.asarray(
    progress_values
)

speed_values = np.asarray(
    speed_values
)


print(
    "Lap completed:",
    info["lap_completed"]
)

print(
    "Lap time:",
    f"{info['time']:.2f}s"
)


# =========================================================
# PLOT SPEED PROFILE
# =========================================================

plt.figure(figsize=(10, 6))

plt.plot(
    progress_values,
    speed_values,
    linewidth=2
)

plt.xlabel("Lap progress (%)")
plt.ylabel("Speed (m/s)")
plt.title("PPO Speed Profile on Wavy Track")
plt.grid(alpha=0.25)
plt.xlim(0, 100)
plt.tight_layout()

output_file = (
    project_dir
    / "results"
    / "final_speed_profile.png"
)

plt.savefig(
    output_file,
    dpi=300
)

plt.show()

print("\nSaved figure to:")
print(output_file)

env.close()