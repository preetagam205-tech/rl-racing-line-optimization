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
    / "ppo_mixedtracks_stage1"
)


# --------------------------------
# TRACK
# --------------------------------

track = TrackGeometry(
    centerline_points=make_wavy_track(
        num_points=720
    ),
    track_width=12.0,
    num_checkpoints=7
)


# --------------------------------
# FAIL VS SUCCESS
# --------------------------------

test_starts = [
    ("FAIL", 0),
    ("SUCCESS", 320),
]


for label, start_index in test_starts:

    env = RacingEnv(
        track=track,
        random_start=False
    )

    observation, info = env.reset(
        options={
            "start_index": start_index
        }
    )

    terminated = False
    truncated = False

    counter = 0

    print("\n")
    print("=" * 95)
    print(
        f"{label} APPROACH — "
        f"START INDEX {start_index}"
    )
    print("=" * 95)

    print(
        "Prog | Speed | LatErr | "
        "Offset10 | Offset20 | Offset30 | "
        "Steer | Throttle"
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
            next_observation,
            reward,
            terminated,
            truncated,
            info
        ) = env.step(action)

        progress = info["progress"]

        if 80.0 <= progress <= 89.0:

            counter += 1

            if counter % 4 == 0:

                offsets = (
                    track.get_lookahead_lateral_offsets(
                        env.sim.car.x,
                        env.sim.car.y
                    )
                )

                print(
                    f"{progress:5.2f} | "
                    f"{info['speed']:5.2f} | "
                    f"{observation[1]:6.3f} | "
                    f"{offsets[0]:8.3f} | "
                    f"{offsets[1]:8.3f} | "
                    f"{offsets[2]:8.3f} | "
                    f"{float(action[0]):6.3f} | "
                    f"{float(action[1]):8.3f}"
                )

        observation = (
            next_observation
        )

    print("\nFINAL")

    print(
        f"End progress: "
        f"{info['progress']:.2f}%"
    )

    print(
        f"On track: "
        f"{info['on_track']}"
    )

    print(
        f"Lap: "
        f"{info['lap_completed']}"
    )

    env.close()