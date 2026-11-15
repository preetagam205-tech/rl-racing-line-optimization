from collections import Counter

from src.racing_env import RacingEnv


env = RacingEnv(
    random_start=True,
    mixed_tracks=True
)

family_counts = Counter()


for i in range(30):

    observation, info = env.reset(
        seed=1000 + i
    )

    family = info[
        "track_family"
    ]

    family_counts[
        family
    ] += 1

    print(
        f"Episode {i + 1:2d} | "
        f"Family: {family:8s} | "
        f"Start: "
        f"{info['start_index']:3d} | "
        f"Obs shape: "
        f"{observation.shape}"
    )


print("\n============================")
print(" FAMILY COUNTS")
print("============================")

for family, count in (
    family_counts.items()
):

    print(
        f"{family:8s}: "
        f"{count}"
    )


print(
    "\nObservation valid:",
    env.observation_space.contains(
        observation
    )
)

env.close()