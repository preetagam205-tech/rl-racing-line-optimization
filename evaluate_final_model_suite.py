from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from src.racing_env import RacingEnv
from src.track_geometry import TrackGeometry

from src.track_factory import (
    make_random_wavy_track,
    make_random_complex_track,
    make_random_spline_track,
    make_random_hard_spline_track,
)


project_dir = Path(__file__).resolve().parent


# =========================================================
# LOAD FINAL MODEL
# =========================================================

model = PPO.load(
    project_dir
    / "models"
    / "ppo_hardfivefamily_stage1"
)

print("Final model loaded!")


# =========================================================
# SETTINGS
# =========================================================

STARTS_PER_TRACK = 12
MAX_SPEED = 20.0


# =========================================================
# EVALUATE ONE TRACK
# =========================================================

def evaluate_track(
    track,
    starts_per_track=STARTS_PER_TRACK
):

    env = RacingEnv(
        track=track,
        random_start=False,
        curvature_observation=False
    )

    start_indices = np.linspace(
        0,
        track.num_points - 1,
        starts_per_track,
        dtype=int
    )

    successful_laps = 0

    lap_times = []
    efficiencies = []


    for start_index in start_indices:

        observation, info = env.reset(
            options={
                "start_index":
                    int(start_index)
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


        if info["lap_completed"]:

            successful_laps += 1

            lap_time = info["time"]

            lap_times.append(
                lap_time
            )

            theoretical_min_time = (
                track.total_track_length
                / MAX_SPEED
            )

            efficiency = (
                theoretical_min_time
                / lap_time
            )

            efficiencies.append(
                efficiency
            )


    env.close()


    return {
        "successes":
            successful_laps,

        "attempts":
            starts_per_track,

        "success_rate":
            100.0
            * successful_laps
            / starts_per_track,

        "lap_times":
            lap_times,

        "efficiencies":
            efficiencies,
    }


# =========================================================
# FAMILY EVALUATION
# =========================================================

def evaluate_family(
    family_name,
    seeds,
    track_generator
):

    total_successes = 0
    total_attempts = 0

    all_lap_times = []
    all_efficiencies = []

    track_success_rates = []


    print("\n")
    print("=" * 80)
    print(f"{family_name.upper()} FAMILY")
    print("=" * 80)


    for seed in seeds:

        rng = np.random.default_rng(
            seed
        )

        track = track_generator(
            rng
        )

        result = evaluate_track(
            track
        )

        total_successes += (
            result["successes"]
        )

        total_attempts += (
            result["attempts"]
        )

        all_lap_times.extend(
            result["lap_times"]
        )

        all_efficiencies.extend(
            result["efficiencies"]
        )

        track_success_rates.append(
            result["success_rate"]
        )


        print(
            f"Seed {seed} | "
            f"Length "
            f"{track.total_track_length:7.2f} m | "
            f"Success "
            f"{result['successes']:2d}/"
            f"{result['attempts']:2d} | "
            f"{result['success_rate']:6.2f}%"
        )


    overall_success = (
        100.0
        * total_successes
        / total_attempts
    )


    if all_lap_times:

        average_lap_time = (
            np.mean(
                all_lap_times
            )
        )

        average_efficiency = (
            100.0
            * np.mean(
                all_efficiencies
            )
        )

    else:

        average_lap_time = np.nan
        average_efficiency = np.nan


    return {
        "family":
            family_name,

        "successes":
            total_successes,

        "attempts":
            total_attempts,

        "success_rate":
            overall_success,

        "average_track_success":
            np.mean(
                track_success_rates
            ),

        "average_lap_time":
            average_lap_time,

        "average_efficiency":
            average_efficiency,
    }


# =========================================================
# TRACK GENERATORS
# =========================================================

def make_ellipse(rng):

    a = float(
        rng.uniform(
            35.0,
            60.0
        )
    )

    b = float(
        rng.uniform(
            15.0,
            25.0
        )
    )

    return TrackGeometry(
        a=a,
        b=b,
        track_width=12.0,
        num_points=720,
        num_checkpoints=7
    )


def make_wavy(rng):

    points = make_random_wavy_track(
        rng=rng,
        num_points=720
    )

    return TrackGeometry(
        centerline_points=points,
        track_width=12.0,
        num_checkpoints=7
    )


def make_complex(rng):

    points = make_random_complex_track(
        rng=rng,
        num_points=720
    )

    return TrackGeometry(
        centerline_points=points,
        track_width=12.0,
        num_checkpoints=7
    )


def make_spline(rng):

    points = make_random_spline_track(
        rng=rng,
        num_points=720
    )

    return TrackGeometry(
        centerline_points=points,
        track_width=12.0,
        num_checkpoints=7
    )


def make_hard_spline(rng):

    points = (
        make_random_hard_spline_track(
            rng=rng,
            num_points=720
        )
    )

    return TrackGeometry(
        centerline_points=points,
        track_width=12.0,
        num_checkpoints=7
    )


# =========================================================
# HELD-OUT SPLINE BENCHMARK
# =========================================================

def catmull_rom_closed(
    control_points,
    points_per_segment=60
):

    control_points = np.asarray(
        control_points,
        dtype=np.float64
    )

    generated = []

    n = len(control_points)


    for i in range(n):

        p0 = control_points[
            (i - 1) % n
        ]

        p1 = control_points[i]

        p2 = control_points[
            (i + 1) % n
        ]

        p3 = control_points[
            (i + 2) % n
        ]


        for j in range(
            points_per_segment
        ):

            t = (
                j
                / points_per_segment
            )

            t2 = t * t
            t3 = t2 * t


            point = 0.5 * (
                2.0 * p1

                + (
                    -p0
                    + p2
                ) * t

                + (
                    2.0 * p0
                    - 5.0 * p1
                    + 4.0 * p2
                    - p3
                ) * t2

                + (
                    -p0
                    + 3.0 * p1
                    - 3.0 * p2
                    + p3
                ) * t3
            )

            generated.append(
                point
            )


    return np.asarray(
        generated
    )


heldout_control_points = np.array(
    [
        [48.00,   0.00],
        [38.11,  17.60],
        [26.00,  36.03],
        [0.00,   32.00],
        [-23.50, 32.56],
        [-37.24, 17.20],
        [-50.00,  0.00],
        [-35.51, -16.40],
        [-24.50, -33.95],
        [0.00,   -36.00],
        [26.50,  -36.72],
        [36.37,  -16.80],
    ],
    dtype=np.float64
)


heldout_points = catmull_rom_closed(
    heldout_control_points,
    points_per_segment=60
)


heldout_track = TrackGeometry(
    centerline_points=heldout_points,
    track_width=12.0,
    num_checkpoints=7
)


# =========================================================
# RUN FAMILY TESTS
# =========================================================

results = []


results.append(
    evaluate_family(
        "Ellipse",
        [
            5101,
            5102,
            5103,
            5104,
            5105,
        ],
        make_ellipse
    )
)


results.append(
    evaluate_family(
        "Wavy",
        [
            5201,
            5202,
            5203,
            5204,
            5205,
        ],
        make_wavy
    )
)


results.append(
    evaluate_family(
        "Complex",
        [
            5301,
            5302,
            5303,
            5304,
            5305,
        ],
        make_complex
    )
)


results.append(
    evaluate_family(
        "Spline",
        [
            5401,
            5402,
            5403,
            5404,
            5405,
        ],
        make_spline
    )
)


results.append(
    evaluate_family(
        "Hard spline",
        [
            5501,
            5502,
            5503,
            5504,
            5505,
        ],
        make_hard_spline
    )
)


# =========================================================
# HELD-OUT BENCHMARK
# =========================================================

print("\n")
print("=" * 80)
print("HELD-OUT CONTROL-POINT SPLINE")
print("=" * 80)


heldout_result = evaluate_track(
    heldout_track,
    starts_per_track=36
)


heldout_success_rate = (
    heldout_result[
        "success_rate"
    ]
)


if heldout_result["lap_times"]:

    heldout_lap_time = np.mean(
        heldout_result[
            "lap_times"
        ]
    )

    heldout_efficiency = (
        100.0
        * np.mean(
            heldout_result[
                "efficiencies"
            ]
        )
    )

else:

    heldout_lap_time = np.nan
    heldout_efficiency = np.nan


print(
    f"Length: "
    f"{heldout_track.total_track_length:.2f} m"
)

print(
    f"Success: "
    f"{heldout_result['successes']}/"
    f"{heldout_result['attempts']} "
    f"({heldout_success_rate:.2f}%)"
)


results.append(
    {
        "family":
            "Held-out spline",

        "successes":
            heldout_result[
                "successes"
            ],

        "attempts":
            heldout_result[
                "attempts"
            ],

        "success_rate":
            heldout_success_rate,

        "average_track_success":
            heldout_success_rate,

        "average_lap_time":
            heldout_lap_time,

        "average_efficiency":
            heldout_efficiency,
    }
)


# =========================================================
# FINAL SUMMARY
# =========================================================

print("\n")
print("=" * 100)
print(" FINAL MODEL BENCHMARK")
print("=" * 100)

print(
    "Family           | Success   | Rate    | "
    "Avg Lap Time | Avg Efficiency"
)

print("-" * 100)


for result in results:

    if np.isnan(
        result[
            "average_lap_time"
        ]
    ):

        lap_text = "   ---   "
        efficiency_text = "   ---   "

    else:

        lap_text = (
            f"{result['average_lap_time']:7.2f}s"
        )

        efficiency_text = (
            f"{result['average_efficiency']:7.2f}%"
        )


    print(
        f"{result['family']:16s} | "
        f"{result['successes']:3d}/"
        f"{result['attempts']:<3d} | "
        f"{result['success_rate']:6.2f}% | "
        f"{lap_text:12s} | "
        f"{efficiency_text}"
    )