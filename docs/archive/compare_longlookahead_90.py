from pathlib import Path

from stable_baselines3 import PPO

from src.racing_env import RacingEnv
from src.track_geometry import TrackGeometry
from src.track_factory import make_wavy_track


project_dir = Path(__file__).resolve().parent

model = PPO.load(
    project_dir
    / "models"
    / "ppo_longlookahead_stage1"
)


track = TrackGeometry(
    centerline_points=make_wavy_track(
        num_points=720
    ),
    track_width=12.0,
    num_checkpoints=7
)


test_starts = [
    ("FAIL", 0),
    ("SUCCESS", 160),
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
    print("=" * 120)
    print(
        f"{label} APPROACH — START INDEX {start_index}"
    )
    print("=" * 120)

    print(
        "Prog | Speed | LatErr | HeadErr | "
        "H10 | H20 | H30 | H40 | H50 | H60 | "
        "Authority | Steer | Throttle"
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

        if 78.0 <= progress <= 92.0:

            counter += 1

            if counter % 4 == 0:

                print(
                    f"{progress:5.2f} | "
                    f"{info['speed']:5.2f} | "
                    f"{observation[1]:6.3f} | "
                    f"{observation[2]:7.3f} | "
                    f"{observation[3]:5.3f} | "
                    f"{observation[4]:5.3f} | "
                    f"{observation[5]:5.3f} | "
                    f"{observation[6]:5.3f} | "
                    f"{observation[7]:5.3f} | "
                    f"{observation[8]:5.3f} | "
                    f"{observation[9]:9.3f} | "
                    f"{float(action[0]):6.3f} | "
                    f"{float(action[1]):8.3f}"
                )

        observation = next_observation


    print("\nFINAL")

    print(
        f"End progress: {info['progress']:.2f}%"
    )

    print(
        f"On track: {info['on_track']}"
    )

    print(
        f"Lap: {info['lap_completed']}"
    )

    env.close()