
"""Benchmark real PPO checkpoints on five track families."""

import argparse
import csv
import statistics
from collections import defaultdict
from functools import lru_cache
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

ROOT = Path(__file__).resolve().parent

FAMILIES = (
    "ellipse",
    "wavy",
    "complex",
    "spline",
    "hard_spline",
)

CHECKPOINTS = (
    0, 25000, 50000, 100000, 250000, 500000
)

FACTORIES = {
    "wavy": make_random_wavy_track,
    "complex": make_random_complex_track,
    "spline": make_random_spline_track,
    "hard_spline": make_random_hard_spline_track,
}


@lru_cache(maxsize=256)
def cached_points(family, trial):
    """Generate a reproducible track per test case."""

    seed = (
        20261115
        + 1009 * FAMILIES.index(family)
        + trial
    )

    for attempt in range(5):
        rng = np.random.default_rng(
            seed + attempt * 100003
        )

        try:
            return FACTORIES[family](
                rng=rng,
                num_points=720,
            )
        except RuntimeError:
            if attempt == 4:
                raise


def build_track(family, trial):
    if family == "ellipse":
        rng = np.random.default_rng(
            20261115 + trial
        )

        return TrackGeometry(
            a=float(rng.uniform(35.0, 60.0)),
            b=float(rng.uniform(15.0, 25.0)),
            track_width=12.0,
            num_points=720,
            num_checkpoints=7,
        )

    return TrackGeometry(
        centerline_points=cached_points(
            family, trial
        ).copy(),
        track_width=12.0,
        num_checkpoints=7,
    )


def evaluate(model, family, trial):
    dims = tuple(
        model.observation_space.shape
    )

    if dims not in ((10,), (16,)):
        raise ValueError(
            f"Unsupported observations: {dims}"
        )

    if tuple(model.action_space.shape) != (2,):
        raise ValueError(
            "Expected steering and throttle/brake"
        )

    env = RacingEnv(
        track=build_track(family, trial),
        curvature_observation=(dims == (16,)),
    )

    try:
        obs, _ = env.reset(
            seed=12345,
            options={"start_index": 0},
        )

        maximum_progress = 0.0
        steering_changes = []
        lateral_errors = []
        previous_steer = None
        final_info = {}

        steps = 0
        finished = False
        truncated = False

        while True:
            action, _ = model.predict(
                obs, deterministic=True
            )

            action = np.asarray(
                action, dtype=float
            )

            if previous_steer is not None:
                steering_changes.append(
                    abs(
                        float(action[0])
                        - previous_steer
                    )
                )

            previous_steer = float(action[0])

            obs, _, terminated, truncated, info = (
                env.step(action)
            )

            steps += 1

            maximum_progress = max(
                maximum_progress,
                float(info["progress"]),
            )

            lateral, _, _ = (
                env.sim.track.get_track_relative_state(
                    env.sim.car.x,
                    env.sim.car.y,
                    env.sim.car.heading,
                )
            )

            lateral_errors.append(
                abs(float(lateral))
            )

            final_info = info

            if terminated or truncated:
                finished = bool(
                    info.get(
                        "lap_completed", False
                    )
                )
                break

        return {
            "completed": int(finished),

            "progress_percent": round(
                100.0 if finished
                else min(
                    100.0, maximum_progress
                ),
                3,
            ),

            "lap_seconds": (
                round(
                    float(final_info["time"]),
                    3,
                )
                if finished else ""
            ),

            "elapsed_seconds": round(
                float(
                    final_info.get(
                        "time", steps * 0.05
                    )
                ),
                3,
            ),

            "off_track": int(
                not finished and not truncated
            ),

            "mean_steering_change": (
                round(
                    statistics.mean(
                        steering_changes
                    ),
                    5,
                )
                if steering_changes else 0.0
            ),

            "mean_abs_lateral_error": (
                round(
                    statistics.mean(
                        lateral_errors
                    ),
                    3,
                )
                if lateral_errors else 0.0
            ),
        }

    finally:
        env.close()


def write_csv(path, rows, columns):
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=columns,
        )

        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--trials",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--only-final",
        action="store_true",
    )

    args = parser.parse_args()

    if args.trials < 1:
        parser.error(
            "--trials must be at least 1"
        )

    steps_to_check = (
        (500000,)
        if args.only_final
        else CHECKPOINTS
    )

    models = {}

    for step in steps_to_check:
        path = (
            ROOT
            / "models"
            / "visual_training"
            / f"ppo_visual_{step:06d}.zip"
        )

        if not path.is_file():
            parser.error(
                f"Missing checkpoint: {path}"
            )

        models[step] = PPO.load(
            str(path),
            device="cpu",
        )

        print(
            f"Loaded {path.name}"
        )

    raw = []

    for family in FAMILIES:
        for trial in range(args.trials):
            for step, model in models.items():
                result = evaluate(
                    model,
                    family,
                    trial,
                )

                row = {
                    "family": family,
                    "trial": trial,
                    "ppo_steps": step,
                    **result,
                }

                raw.append(row)

                print(
                    f"{family:12} "
                    f"trial {trial + 1}/"
                    f"{args.trials}  "
                    f"PPO {step:>6}  "
                    f"progress "
                    f"{row['progress_percent']:6.1f}%  "
                    f"lap="
                    f"{'YES' if row['completed'] else 'NO'}",
                    flush=True,
                )

    raw_cols = [
        "family",
        "trial",
        "ppo_steps",
        "completed",
        "progress_percent",
        "lap_seconds",
        "elapsed_seconds",
        "off_track",
        "mean_steering_change",
        "mean_abs_lateral_error",
    ]

    output_dir = ROOT / "docs" / "benchmarks"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "ppo_benchmark_runs.csv"

    write_csv(
        raw_path,
        raw,
        raw_cols,
    )

    grouped = defaultdict(list)

    for row in raw:
        key = (
            row["family"],
            row["ppo_steps"],
        )

        grouped[key].append(row)

    summary = []

    for family in FAMILIES:
        for step in steps_to_check:
            group = grouped[
                (family, step)
            ]

            laps = [
                r["lap_seconds"]
                for r in group
                if r["completed"]
            ]

            summary.append({
                "family": family,
                "ppo_steps": step,
                "trials": len(group),

                "completion_rate_percent": round(
                    100 * sum(
                        r["completed"]
                        for r in group
                    ) / len(group),
                    1,
                ),

                "mean_progress_percent": round(
                    statistics.mean(
                        r["progress_percent"]
                        for r in group
                    ),
                    2,
                ),

                "mean_successful_lap_seconds": (
                    round(
                        statistics.mean(laps),
                        3,
                    )
                    if laps else ""
                ),

                "mean_steering_change": round(
                    statistics.mean(
                        r["mean_steering_change"]
                        for r in group
                    ),
                    5,
                ),

                "mean_abs_lateral_error": round(
                    statistics.mean(
                        r["mean_abs_lateral_error"]
                        for r in group
                    ),
                    3,
                ),
            })

    summary_path = (
        output_dir / "ppo_benchmark_summary.csv"
    )

    write_csv(
        summary_path,
        summary,
        list(summary[0]),
    )

    print(
        "\nSaved:",
        raw_path.name,
        "and",
        summary_path.name,
    )


if __name__ == "__main__":
    main()
