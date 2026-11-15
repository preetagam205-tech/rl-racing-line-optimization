from pathlib import Path

from stable_baselines3 import PPO

from src.racing_env import RacingEnv
from src.track_geometry import TrackGeometry
from src.track_factory import make_wavy_track


project_dir = Path(__file__).resolve().parent


model = PPO.load(
    project_dir
    / "models"
    / "ppo_laptime_stage1"
)


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


observation, info = env.reset(
    options={
        "start_index": 0
    }
)

terminated = False
truncated = False

step_counter = 0


print("\n")
print("=" * 75)
print(" FAST PROGRESS REWARD CHECK")
print("=" * 75)

print(
    "Step | Speed | ProgressDelta | "
    "ProgressReward | FastProgressBonus"
)


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

    step_counter += 1

    if step_counter % 20 == 0:

        print(
            f"{step_counter:4d} | "
            f"{info['speed']:5.2f} | "
            f"{info['progress_delta']:13.4f} | "
            f"{info['reward_progress']:14.4f} | "
            f"{info['reward_fast_progress']:17.4f}"
        )


print("\nFINAL")

print(
    f"Lap completed: "
    f"{info['lap_completed']}"
)

print(
    f"Time: "
    f"{info['time']:.2f}s"
)

env.close()