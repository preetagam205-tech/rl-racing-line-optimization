from pathlib import Path

from stable_baselines3 import PPO

from src.racing_env import RacingEnv
from src.track_geometry import TrackGeometry
from src.track_factory import make_wavy_track


project_dir = Path(__file__).resolve().parent


# -------------------------
# LOAD CURRENT MODEL
# -------------------------

model = PPO.load(
    project_dir
    / "models"
    / "ppo_longlookahead_stage1"
)


# -------------------------
# FIXED WAVY TRACK
# -------------------------

points = make_wavy_track(
    num_points=720
)

track = TrackGeometry(
    centerline_points=points,
    track_width=12.0,
    num_checkpoints=7
)

env = RacingEnv(
    track=track,
    random_start=False
)


# Start 160 previously completed a lap
observation, info = env.reset(
    options={
        "start_index": 160
    }
)

terminated = False
truncated = False


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


print("\n==============================")
print(" LAP-TIME REWARD CHECK")
print("==============================")

print(
    f"Lap completed: "
    f"{info['lap_completed']}"
)

print(
    f"Lap time: "
    f"{info['time']:.2f}s"
)

print(
    f"Theoretical minimum: "
    f"{info['theoretical_min_time']:.2f}s"
)

print(
    f"Lap efficiency: "
    f"{info['lap_efficiency']:.3f}"
)

print(
    f"Lap completion reward: "
    f"{info['reward_lap']:.2f}"
)

print(
    f"Lap-time reward: "
    f"{info['reward_lap_time']:.2f}"
)

print(
    f"Final-step total reward: "
    f"{info['reward_total']:.2f}"
)

env.close()