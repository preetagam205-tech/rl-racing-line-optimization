from pathlib import Path

from stable_baselines3 import PPO

from src.racing_env import RacingEnv
from src.track_geometry import TrackGeometry
from src.track_factory import make_wavy_track


# --------------------------------
# MODEL
# --------------------------------

project_dir = Path(__file__).resolve().parent

model = PPO.load(
    project_dir
    / "models"
    / "ppo_longlookahead_stage1"
)


# --------------------------------
# WAVY TRACK
# --------------------------------

track = TrackGeometry(
    centerline_points=make_wavy_track(
        num_points=720
    ),
    track_width=12.0,
    num_checkpoints=7
)

env = RacingEnv(
    track=track,
    random_start=False
)


# --------------------------------
# TEST START POSITIONS
# --------------------------------

start_indices = list(
    range(0, 720, 20)
)


print("\n")
print("========================================")
print(" WAVY TRACK FAILURE LOCATIONS")
print("========================================")


for start_index in start_indices:

    observation, reset_info = env.reset(
        options={
            "start_index": start_index
        }
    )

    terminated = False
    truncated = False

    steps = 0

    last_action = [0.0, 0.0]


    while not (
        terminated
        or truncated
    ):

        action, _ = model.predict(
            observation,
            deterministic=True
        )

        last_action = action

        (
            observation,
            reward,
            terminated,
            truncated,
            info
        ) = env.step(action)

        steps += 1


    print(
        f"Start {start_index:3d} | "
        f"StartProg {reset_info['start_progress']:6.2f}% | "
        f"EndProg {info['progress']:6.2f}% | "
        f"Speed {info['speed']:5.2f} | "
        f"Steer {float(last_action[0]):6.3f} | "
        f"Throttle {float(last_action[1]):6.3f} | "
        f"CP {env.sim.checkpoints_passed_count}/7 | "
        f"OnTrack {info['on_track']} | "
        f"Lap {info['lap_completed']}"
    )


env.close()