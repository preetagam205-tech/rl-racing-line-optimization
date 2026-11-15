
"""Compare PPO agents against a geometric centerline baseline.

Uses the existing RacingEnv and identical test circuits.
Does not retrain or modify PPO models.
"""

import argparse
import csv
import math
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

from benchmark_ppo import FAMILIES, build_track
from analyze_racing_lines import (
    centerline_for_case,
    track_boundaries,
)
from src.racing_env import RacingEnv

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "baseline_comparison"

MODEL_PATHS = {
    "Broad Domain PPO":
        ROOT / "models/ppo_broaddomain_stage1.zip",
    "Hard Five Family PPO":
        ROOT / "models/ppo_hardfivefamily_stage1.zip",
}

BASELINE_NAME = "Centerline Pure Pursuit"


def wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


class CenterlineFollower:
    """Pure-pursuit steering and curvature-aware speed control."""

    def __init__(self, centerline):
        self.points = np.asarray(centerline, dtype=float)

        self.segments = (
            np.roll(self.points, -1, axis=0) - self.points
        )

        self.lengths = np.linalg.norm(
            self.segments, axis=1
        )

        if np.min(self.lengths) <= 1e-8:
            raise ValueError("Degenerate track geometry")

        self.cumulative = np.r_[
            0, np.cumsum(self.lengths)
        ]

        self.total_length = float(
            self.cumulative[-1]
        )

        headings = np.arctan2(
            self.segments[:, 1],
            self.segments[:, 0],
        )

        heading_changes = np.array([
            wrap(value)
            for value in (
                np.roll(headings, -1) - headings
            )
        ])

        self.curvature = (
            np.abs(heading_changes)
            / np.maximum(
                (
                    self.lengths
                    + np.roll(self.lengths, -1)
                ) / 2,
                1e-6,
            )
        )

    def projection(self, position):
        displacement = (
            position[None, :] - self.points
        )

        fraction = np.clip(
            np.einsum(
                "ij,ij->i",
                displacement,
                self.segments,
            ) / (self.lengths ** 2),
            0.0,
            1.0,
        )

        feet = (
            self.points
            + fraction[:, None] * self.segments
        )

        index = int(np.argmin(
            np.sum(
                (feet - position) ** 2,
                axis=1,
            )
        ))

        return float(
            self.cumulative[index]
            + fraction[index] * self.lengths[index]
        )

    def point_at(self, distance):
        distance %= self.total_length

        index = min(
            len(self.points) - 1,
            int(
                np.searchsorted(
                    self.cumulative,
                    distance,
                    side="right",
                ) - 1
            ),
        )

        fraction = (
            distance - self.cumulative[index]
        ) / self.lengths[index]

        return (
            self.points[index]
            + fraction * self.segments[index]
        )

    def curvature_ahead(self, distance):
        positions = (
            distance + np.linspace(0, 25, 13)
        ) % self.total_length

        indices = np.searchsorted(
            self.cumulative,
            positions,
            side="right",
        ) - 1

        indices = np.clip(
            indices,
            0,
            len(self.curvature) - 1,
        )

        return float(
            np.max(self.curvature[indices])
        )

    def action(self, env):
        car = env.sim.car

        position = np.array(
            [float(car.x), float(car.y)]
        )

        speed = max(
            0.0, float(car.speed)
        )

        distance = self.projection(position)

        lookahead = 9.0 + 0.48 * speed

        target = self.point_at(
            distance + lookahead
        )

        difference = target - position

        target_distance = max(
            1e-6,
            float(np.linalg.norm(difference)),
        )

        heading_error = wrap(
            math.atan2(
                difference[1],
                difference[0],
            ) - float(car.heading)
        )

        wheelbase = float(
            getattr(car, "wheelbase", 3.2)
        )

        max_steering = max(
            1e-6,
            float(car.max_steering),
        )

        steering_angle = math.atan2(
            2.0 * wheelbase
            * math.sin(heading_error),
            target_distance,
        )

        steering = float(np.clip(
            steering_angle / max_steering,
            -1.0,
            1.0,
        ))

        curvature = max(
            self.curvature_ahead(distance),
            1e-5,
        )

        target_speed = float(np.clip(
            math.sqrt(7.0 / curvature),
            5.0,
            17.0,
        ))

        speed_error = target_speed - speed

        throttle_brake = float(np.clip(
            0.45 * speed_error,
            -1.0,
            1.0,
        ))

        return np.array(
            [steering, throttle_brake],
            dtype=np.float32,
        )


def rollout(name, model, family, trial, centerline):
    if name == BASELINE_NAME:
        observation_shape = (10,)
    else:
        observation_shape = tuple(
            model.observation_space.shape
        )

    if observation_shape not in ((10,), (16,)):
        raise ValueError(
            f"Incompatible observations: {observation_shape}"
        )

    if name != BASELINE_NAME:
        if tuple(model.action_space.shape) != (2,):
            raise ValueError("Incompatible action space")

    env = RacingEnv(
        track=build_track(family, trial),
        curvature_observation=(
            observation_shape == (16,)
        ),
    )

    controller = (
        CenterlineFollower(centerline)
        if name == BASELINE_NAME
        else None
    )

    try:
        observation, _ = env.reset(
            seed=12345,
            options={"start_index": 0},
        )

        car = env.sim.car

        path = [
            (float(car.x), float(car.y))
        ]

        telemetry = []
        previous_steering = None
        maximum_progress = 0.0

        info = {}
        truncated = False

        max_steps = int(
            np.ceil(
                env.max_episode_time / env.dt
            )
        ) + 2

        for step in range(max_steps):
            if controller is not None:
                action = controller.action(env)
            else:
                action, _ = model.predict(
                    observation,
                    deterministic=True,
                )

                action = np.asarray(
                    action,
                    dtype=np.float32,
                ).reshape(2)

            observation, _, terminated, truncated, info = (
                env.step(action)
            )

            car = env.sim.car

            lateral, _, _ = (
                env.sim.track.get_track_relative_state(
                    car.x,
                    car.y,
                    car.heading,
                )
            )

            path.append((
                float(car.x),
                float(car.y),
            ))

            maximum_progress = max(
                maximum_progress,
                float(info.get("progress", 0)),
            )

            steering = float(action[0])

            telemetry.append({
                "time": float(
                    info.get(
                        "time",
                        (step + 1) * env.dt,
                    )
                ),
                "speed": float(
                    info.get("speed", car.speed)
                ),
                "lateral": float(lateral),
                "steer": steering,
                "change": (
                    abs(
                        steering - previous_steering
                    )
                    if previous_steering is not None
                    else None
                ),
            })

            previous_steering = steering

            if terminated or truncated:
                break

        completed = bool(
            info.get("lap_completed", False)
        )

        positions = np.asarray(path)

        driven_distance = float(
            np.linalg.norm(
                np.diff(positions, axis=0),
                axis=1,
            ).sum()
        )

        lateral_errors = [
            abs(record["lateral"])
            for record in telemetry
        ]

        steering_changes = [
            record["change"]
            for record in telemetry
            if record["change"] is not None
        ]

        result = {
            "model": name,
            "family": family,
            "trial": trial,
            "completed": int(completed),
            "progress_percent": (
                100.0
                if completed
                else round(
                    min(100.0, maximum_progress),
                    3,
                )
            ),
            "lap_seconds": (
                round(float(info["time"]), 3)
                if completed else ""
            ),
            "elapsed_seconds": round(
                float(info.get(
                    "time",
                    len(telemetry) * env.dt,
                )),
                3,
            ),
            "distance_driven": round(
                driven_distance, 3
            ),
            "mean_speed": round(
                mean(t["speed"] for t in telemetry),
                3,
            ),
            "mean_abs_lateral_error": round(
                mean(lateral_errors), 3
            ),
            "mean_steering_change": (
                round(mean(steering_changes), 5)
                if steering_changes else 0.0
            ),
            "status": (
                "lap"
                if completed
                else "timeout"
                if truncated
                else "off_track"
            ),
        }

        return result, positions, telemetry

    finally:
        env.close()


def save_csv(path, records):
    if not records:
        return

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(records[0]),
        )
        writer.writeheader()
        writer.writerows(records)


def plot_comparison(family, trial, centerline, runs):
    fig, axes = plt.subplots(
        2, 2,
        figsize=(14, 9),
    )

    outer, inner = track_boundaries(
        centerline,
        half_width=6.0,
    )

    for boundary in (outer, inner):
        closed = np.vstack(
            [boundary, boundary[0]]
        )

        axes[0, 0].plot(
            closed[:, 0],
            closed[:, 1],
            color="0.62",
            linewidth=1.4,
        )

    closed_center = np.vstack(
        [centerline, centerline[0]]
    )

    axes[0, 0].plot(
        closed_center[:, 0],
        closed_center[:, 1],
        "--",
        color="0.6",
        label="Centerline",
    )

    axes[0, 0].plot(
        centerline[0, 0],
        centerline[0, 1],
        "ko",
        markersize=4,
    )

    for name, data in runs.items():
        result, path, telemetry = data

        status = (
            "LAP"
            if result["completed"]
            else f"{result['progress_percent']:.0f}%"
        )

        axes[0, 0].plot(
            path[:, 0],
            path[:, 1],
            linewidth=2,
            label=f"{name}: {status}",
        )

        axes[0, 0].plot(
            path[-1, 0],
            path[-1, 1],
            marker="x",
            markersize=7,
        )

        times = [
            row["time"]
            for row in telemetry
        ]

        for axis, field in (
            (axes[0, 1], "speed"),
            (axes[1, 0], "lateral"),
            (axes[1, 1], "steer"),
        ):
            axis.plot(
                times,
                [
                    row[field]
                    for row in telemetry
                ],
                label=name,
            )

    axes[0, 0].set_title(
        "Driven paths vs centerline"
    )

    axes[0, 0].set_aspect(
        "equal",
        adjustable="box",
    )

    axes[0, 0].set_xlabel("Track x")
    axes[0, 0].set_ylabel("Track y")

    axes[0, 1].set_title(
        "Speed vs simulation time"
    )

    axes[1, 0].set_title(
        "Signed lateral error"
    )

    axes[1, 1].set_title(
        "Steering input"
    )

    for axis in axes.flat:
        if axis is not axes[0, 0]:
            axis.set_xlabel("Time (s)")
            axis.grid(alpha=0.2)

        axis.legend(fontsize=8)

    fig.suptitle(
        f"Stage 17 | {family} | Trial {trial}"
    )

    fig.tight_layout()

    destination = (
        OUT / f"{family}_baseline_comparison.png"
    )

    fig.savefig(
        destination,
        dpi=150,
    )

    plt.close(fig)

    print(f"Saved {destination}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--trials",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--trial-start",
        type=int,
        default=10000,
    )

    args = parser.parse_args()

    if args.trials < 1:
        parser.error("--trials must be positive")

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    policies = {
        BASELINE_NAME: None
    }

    for name, path in MODEL_PATHS.items():
        if not path.is_file():
            parser.error(
                f"Missing PPO model: {path}"
            )

        policies[name] = PPO.load(
            str(path),
            device="cpu",
        )

        print(
            f"Loaded {name}: "
            f"{policies[name].observation_space.shape}"
        )

    all_results = []

    for family in FAMILIES:
        for trial in range(
            args.trial_start,
            args.trial_start + args.trials,
        ):
            centerline = centerline_for_case(
                family, trial
            )

            plots = {}

            for name, model in policies.items():
                result, path, telemetry = rollout(
                    name,
                    model,
                    family,
                    trial,
                    centerline,
                )

                all_results.append(result)

                print(
                    f"{family:12} "
                    f"{name:23} "
                    f"trial={trial} "
                    f"progress="
                    f"{result['progress_percent']:6.1f}% "
                    f"lap="
                    f"{'YES' if result['completed'] else 'NO'}",
                    flush=True,
                )

                if trial == args.trial_start:
                    plots[name] = (
                        result,
                        path,
                        telemetry,
                    )

            if plots:
                plot_comparison(
                    family,
                    trial,
                    centerline,
                    plots,
                )

    save_csv(
        OUT / "baseline_comparison_runs.csv",
        all_results,
    )

    summary = []

    for name in policies:
        for family in FAMILIES:
            group = [
                row
                for row in all_results
                if row["model"] == name
                and row["family"] == family
            ]

            successful_laps = [
                float(row["lap_seconds"])
                for row in group
                if row["completed"]
            ]

            summary.append({
                "model": name,
                "family": family,
                "runs": len(group),

                "completion_rate_percent": round(
                    100 * mean(
                        row["completed"]
                        for row in group
                    ),
                    1,
                ),

                "mean_progress_percent": round(
                    mean(
                        row["progress_percent"]
                        for row in group
                    ),
                    2,
                ),

                "mean_successful_lap_seconds": (
                    round(
                        mean(successful_laps),
                        3,
                    )
                    if successful_laps else ""
                ),

                "mean_abs_lateral_error": round(
                    mean(
                        row["mean_abs_lateral_error"]
                        for row in group
                    ),
                    3,
                ),

                "mean_steering_change": round(
                    mean(
                        row["mean_steering_change"]
                        for row in group
                    ),
                    5,
                ),
            })

    save_csv(
        OUT / "baseline_comparison_summary.csv",
        summary,
    )

    print(
        "\nResults saved to:",
        OUT,
    )


if __name__ == "__main__":
    main()
