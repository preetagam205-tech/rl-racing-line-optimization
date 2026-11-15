
"""Stage 17: analyze real PPO racing trajectories on matched circuits.

Uses benchmark_ppo.py track construction and original RacingEnv physics.

Run:
    python analyze_racing_lines.py --trials 1
    python analyze_racing_lines.py --trials 20
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

from benchmark_ppo import (
    FAMILIES,
    build_track,
    cached_points,
)
from src.racing_env import RacingEnv


ROOT = Path(__file__).resolve().parent

MODELS = {
    "Broad Domain":
        ROOT / "models/ppo_broaddomain_stage1.zip",
    "Hard Five Family":
        ROOT / "models/ppo_hardfivefamily_stage1.zip",
}

OUTPUT = ROOT / "racing_line_analysis"


def centerline_for_case(family, trial):
    """Reproduce the benchmark's track centerline."""

    if family != "ellipse":
        return np.asarray(
            cached_points(family, trial),
            dtype=float,
        ).copy()

    rng = np.random.default_rng(
        20261115 + trial
    )

    a = float(rng.uniform(35.0, 60.0))
    b = float(rng.uniform(15.0, 25.0))

    theta = np.linspace(
        0, 2 * np.pi, 720, endpoint=False
    )

    return np.column_stack((
        a * np.cos(theta),
        b * np.sin(theta),
    ))


def run_model(model, family, trial):
    shape = tuple(model.observation_space.shape)

    if shape not in ((10,), (16,)):
        raise ValueError(
            f"Unsupported observation shape: {shape}"
        )

    if tuple(model.action_space.shape) != (2,):
        raise ValueError(
            "Expected steering and throttle/brake"
        )

    env = RacingEnv(
        track=build_track(family, trial),
        curvature_observation=(shape == (16,)),
    )

    try:
        obs, _ = env.reset(
            seed=12345,
            options={"start_index": 0},
        )

        car = env.sim.car

        points = [
            (float(car.x), float(car.y))
        ]

        telemetry = []
        last_steer = None
        final_info = {}
        terminated = truncated = False

        max_steps = int(
            np.ceil(
                env.max_episode_time / env.dt
            )
        ) + 2

        for step in range(max_steps):
            action, _ = model.predict(
                obs,
                deterministic=True,
            )

            action = np.asarray(
                action, dtype=float
            ).reshape(-1)

            obs, _, terminated, truncated, info = (
                env.step(action)
            )

            car = env.sim.car

            lateral, heading_error, _ = (
                env.sim.track.get_track_relative_state(
                    car.x,
                    car.y,
                    car.heading,
                )
            )

            points.append((
                float(car.x),
                float(car.y),
            ))

            steering = float(action[0])

            telemetry.append({
                "time_s": float(
                    info.get(
                        "time",
                        (step + 1) * env.dt,
                    )
                ),
                "x": float(car.x),
                "y": float(car.y),
                "speed": float(
                    info.get("speed", car.speed)
                ),
                "progress_percent": float(
                    info.get("progress", 0.0)
                ),
                "lateral_error": float(lateral),
                "heading_error_rad": float(
                    heading_error
                ),
                "steering_action": steering,
                "throttle_brake_action": float(
                    action[1]
                ),
                "steering_change": (
                    0.0
                    if last_steer is None
                    else abs(steering - last_steer)
                ),
            })

            last_steer = steering
            final_info = info

            if terminated or truncated:
                break

        if not telemetry:
            raise RuntimeError(
                "No simulation steps recorded"
            )

        coords = np.asarray(points)

        path_distance = float(
            np.linalg.norm(
                np.diff(coords, axis=0),
                axis=1,
            ).sum()
        )

        completed = bool(
            final_info.get(
                "lap_completed", False
            )
        )

        max_progress = max(
            row["progress_percent"]
            for row in telemetry
        )

        lateral_errors = np.asarray([
            row["lateral_error"]
            for row in telemetry
        ])

        steering_changes = np.asarray([
            row["steering_change"]
            for row in telemetry[1:]
        ])

        metrics = {
            "completed": int(completed),

            "progress_percent": (
                100.0 if completed
                else round(
                    min(100.0, max_progress), 3
                )
            ),

            "lap_seconds": (
                round(
                    float(final_info["time"]), 3
                )
                if completed else ""
            ),

            "elapsed_seconds": round(
                float(
                    telemetry[-1]["time_s"]
                ), 3
            ),

            "driven_distance": round(
                path_distance, 3
            ),

            "mean_speed": round(
                float(np.mean([
                    row["speed"]
                    for row in telemetry
                ])), 3
            ),

            "mean_abs_lateral_error": round(
                float(
                    np.mean(
                        abs(lateral_errors)
                    )
                ), 3
            ),

            "max_abs_lateral_error": round(
                float(
                    np.max(
                        abs(lateral_errors)
                    )
                ), 3
            ),

            "mean_abs_steering_change": round(
                float(
                    np.mean(
                        steering_changes
                    )
                )
                if len(steering_changes)
                else 0.0,
                5,
            ),

            "finish_reason": (
                "lap"
                if completed
                else "time_limit"
                if truncated
                else "off_track"
            ),
        }

        return metrics, telemetry, coords

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


def track_boundaries(points, half_width=6.0):
    tangents = (
        np.roll(points, -1, axis=0)
        - np.roll(points, 1, axis=0)
    )

    tangents /= np.maximum(
        1e-9,
        np.linalg.norm(
            tangents,
            axis=1,
            keepdims=True,
        ),
    )

    normals = np.column_stack((
        -tangents[:, 1],
        tangents[:, 0],
    ))

    return (
        points + half_width * normals,
        points - half_width * normals,
    )


def plot_case(family, trial, points, runs):
    fig, axs = plt.subplots(
        2, 2,
        figsize=(14, 9),
    )

    ax = axs[0, 0]

    plus, minus = track_boundaries(
        points
    )

    for boundary in (plus, minus):
        closed = np.vstack((
            boundary,
            boundary[0],
        ))

        ax.plot(
            closed[:, 0],
            closed[:, 1],
            color="0.55",
            linewidth=1.3,
        )

    closed_centerline = np.vstack((
        points,
        points[0],
    ))

    ax.plot(
        closed_centerline[:, 0],
        closed_centerline[:, 1],
        color="0.75",
        linestyle="--",
        linewidth=1,
        label="Track centerline",
    )

    ax.plot(
        points[0, 0],
        points[0, 1],
        "ko",
        markersize=5,
        label="Start",
    )

    colors = [
        "tab:blue",
        "tab:orange",
    ]

    for (
        name,
        (metrics, telemetry, xy),
    ), color in zip(runs.items(), colors):

        if metrics["completed"]:
            status = "LAP COMPLETE"
        else:
            status = (
                f"{metrics['progress_percent']:.0f}%"
            )

        ax.plot(
            xy[:, 0],
            xy[:, 1],
            color=color,
            linewidth=2,
            label=f"{name}: {status}",
        )

        ax.plot(
            xy[-1, 0],
            xy[-1, 1],
            marker="x",
            color=color,
            markersize=8,
        )

        times = [
            row["time_s"]
            for row in telemetry
        ]

        axs[0, 1].plot(
            times,
            [
                row["speed"]
                for row in telemetry
            ],
            color=color,
            label=name,
        )

        axs[1, 0].plot(
            times,
            [
                row["lateral_error"]
                for row in telemetry
            ],
            color=color,
            label=name,
        )

        axs[1, 1].plot(
            times,
            [
                row["steering_action"]
                for row in telemetry
            ],
            color=color,
            label=name,
        )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.set_title(
        "Actual PPO racing trajectories"
    )

    ax.set_xlabel(
        "Track x (simulation units)"
    )

    ax.set_ylabel(
        "Track y (simulation units)"
    )

    ax.legend(fontsize=8)

    axs[0, 1].set(
        title="Speed vs time",
        xlabel="Simulation time (s)",
        ylabel="Speed (simulation units/s)",
    )

    axs[1, 0].set(
        title="Signed lateral error vs time",
        xlabel="Simulation time (s)",
        ylabel="Lateral error",
    )

    axs[1, 1].set(
        title="Steering input vs time",
        xlabel="Simulation time (s)",
        ylabel="Normalized steering action",
    )

    for axis in axs.flat[1:]:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)

    fig.suptitle(
        f"Stage 17 | "
        f"{family.replace('_', ' ').title()} "
        f"| Trial {trial}"
    )

    fig.tight_layout(
        rect=(0, 0, 1, 0.96)
    )

    destination = (
        OUTPUT
        / f"{family}_trajectory_comparison.png"
    )

    fig.savefig(
        destination,
        dpi=160,
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
        parser.error(
            "--trials must be at least 1"
        )

    OUTPUT.mkdir(
        exist_ok=True
    )

    models = {}

    for name, path in MODELS.items():
        if not path.is_file():
            parser.error(
                f"Model not found: {path}"
            )

        models[name] = PPO.load(
            str(path),
            device="cpu",
        )

        print(
            f"Loaded {name}: "
            f"{models[name].observation_space.shape}"
        )

    rows = []

    for family in FAMILIES:
        for trial in range(
            args.trial_start,
            args.trial_start + args.trials,
        ):

            points = (
                centerline_for_case(
                    family, trial
                )
                if trial == args.trial_start
                else None
            )

            plotted_runs = {}

            for name, model in models.items():
                metrics, telemetry, xy = (
                    run_model(
                        model, family, trial
                    )
                )

                rows.append({
                    "model": name,
                    "family": family,
                    "trial": trial,
                    **metrics,
                })

                print(
                    f"{name:<17} "
                    f"{family:<12} "
                    f"trial={trial} "
                    f"lap="
                    f"{'YES' if metrics['completed'] else 'NO'} "
                    f"progress="
                    f"{metrics['progress_percent']:5.1f}%"
                )

                if trial == args.trial_start:
                    plotted_runs[name] = (
                        metrics,
                        telemetry,
                        xy,
                    )

                    safe_name = (
                        name.lower().replace(
                            " ", "_"
                        )
                    )

                    save_csv(
                        OUTPUT / (
                            f"{family}_"
                            f"{safe_name}_telemetry.csv"
                        ),
                        telemetry,
                    )

            if plotted_runs:
                plot_case(
                    family,
                    trial,
                    points,
                    plotted_runs,
                )

    save_csv(
        OUTPUT / "trajectory_metrics.csv",
        rows,
    )

    summaries = []

    for name in MODELS:
        for family in FAMILIES:
            group = [
                row for row in rows
                if row["model"] == name
                and row["family"] == family
            ]

            successes = [
                float(row["lap_seconds"])
                for row in group
                if row["completed"]
            ]

            summaries.append({
                "model": name,
                "family": family,
                "runs": len(group),

                "completion_rate_percent": round(
                    100 * sum(
                        row["completed"]
                        for row in group
                    ) / len(group),
                    1,
                ),

                "mean_progress_percent": round(
                    float(np.mean([
                        row["progress_percent"]
                        for row in group
                    ])),
                    2,
                ),

                "mean_successful_lap_seconds": (
                    round(
                        float(np.mean(successes)),
                        3,
                    )
                    if successes else ""
                ),

                "mean_abs_lateral_error": round(
                    float(np.mean([
                        row["mean_abs_lateral_error"]
                        for row in group
                    ])),
                    3,
                ),

                "mean_abs_steering_change": round(
                    float(np.mean([
                        row["mean_abs_steering_change"]
                        for row in group
                    ])),
                    5,
                ),
            })

    save_csv(
        OUTPUT / "trajectory_summary.csv",
        summaries,
    )

    print(
        "\nCompleted! Results saved in:",
        OUTPUT,
    )


if __name__ == "__main__":
    main()
