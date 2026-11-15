
"""Evolve a geometric racing controller inside the SAME RacingEnv as PPO.

This is an evolutionary-tuned geometric controller, not a conversion of
population_evolution_multitrack.py and not a neural-network policy.

It sees the full centerline, whereas PPO uses local observations.
The comparison matches vehicle physics, tracks and starts, but NOT
observation information or training budget.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean

import numpy as np
from stable_baselines3 import PPO

from benchmark_ppo import FAMILIES, build_track
from src.racing_env import RacingEnv


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "evolution_shared_physics"

MODEL_FILES = {
    "Broad Domain PPO":
        ROOT / "models/ppo_broaddomain_stage1.zip",
    "Hard Five Family PPO":
        ROOT / "models/ppo_hardfivefamily_stage1.zip",
}


# Parameters that evolution can optimize
NAMES = [
    "lookahead_base",
    "lookahead_per_speed",
    "steer_gain",
    "corner_accel",
    "min_target_speed",
    "max_target_speed",
    "throttle_gain",
    "curvature_horizon",
    "outward_offset",
]

LOW = np.array([
    4.0, 0.10, 0.55, 3.0, 4.0,
    11.0, 0.15, 10.0, -3.0
])

HIGH = np.array([
    16.0, 0.90, 1.50, 12.0, 9.0,
    20.0, 1.0, 40.0, 3.0
])

DEFAULT = np.array([
    9.0, 0.48, 1.0, 7.0, 5.0,
    17.0, 0.45, 25.0, 0.0
])

SIGMA = np.array([
    2.5, 0.15, 0.20, 1.6, 1.0,
    2.0, 0.16, 6.0, 0.8
])


def wrap(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class TrackReference:
    """Precompute centerline geometry and curvature."""

    def __init__(self, points):
        self.points = np.asarray(points, dtype=float)

        self.delta = (
            np.roll(self.points, -1, axis=0)
            - self.points
        )

        self.lengths = np.linalg.norm(
            self.delta,
            axis=1
        )

        if np.any(self.lengths <= 1e-8):
            raise ValueError(
                "Track centerline has zero-length segments"
            )

        self.cumulative = np.r_[
            0.0,
            np.cumsum(self.lengths)
        ]

        self.total = float(self.cumulative[-1])

        heading = np.arctan2(
            self.delta[:, 1],
            self.delta[:, 0]
        )

        next_turn = np.array([
            wrap(x)
            for x in (
                np.roll(heading, -1) - heading
            )
        ])

        raw_curvature = next_turn / np.maximum(
            (
                self.lengths
                + np.roll(self.lengths, -1)
            ) / 2.0,
            1e-6
        )

        self.curvature = (
            np.roll(raw_curvature, 1)
            + 2 * raw_curvature
            + np.roll(raw_curvature, -1)
        ) / 4.0

        self.normals = np.column_stack([
            -np.sin(heading),
            np.cos(heading)
        ])

    def project(self, x, y):
        here = np.array([x, y])

        fraction = np.clip(
            np.einsum(
                "ij,ij->i",
                here - self.points,
                self.delta
            ) / self.lengths ** 2,
            0.0,
            1.0
        )

        feet = (
            self.points
            + fraction[:, None] * self.delta
        )

        index = int(np.argmin(
            np.sum(
                (feet - here) ** 2,
                axis=1
            )
        ))

        return float(
            self.cumulative[index]
            + fraction[index] * self.lengths[index]
        )

    def at(self, distance):
        distance = float(distance % self.total)

        index = int(
            np.searchsorted(
                self.cumulative,
                distance,
                side="right"
            ) - 1
        )

        index = min(
            len(self.points) - 1,
            index
        )

        fraction = (
            distance - self.cumulative[index]
        ) / self.lengths[index]

        point = (
            self.points[index]
            + fraction * self.delta[index]
        )

        return point, index

    def worst_curvature(self, distance, horizon):
        distances = (
            distance
            + np.linspace(0.0, horizon, 13)
        ) % self.total

        indices = np.searchsorted(
            self.cumulative,
            distances,
            side="right"
        ) - 1

        indices = np.clip(
            indices,
            0,
            len(self.points) - 1
        )

        return float(
            np.max(
                np.abs(self.curvature[indices])
            )
        )


class EvolvedController:
    """Controller whose parameters are optimized by evolution."""

    def __init__(self, genes, reference):
        genes = np.asarray(
            genes,
            dtype=float
        )

        if genes.shape != (len(NAMES),):
            raise ValueError(
                "Invalid controller genes"
            )

        self.g = np.clip(
            genes,
            LOW,
            HIGH
        )

        self.reference = reference

    def action(self, env):
        car = env.sim.car
        g = self.g

        x = float(car.x)
        y = float(car.y)
        speed = float(car.speed)

        along = self.reference.project(x, y)

        # Look farther ahead as vehicle speed increases
        lookahead = float(
            g[0] + g[1] * speed
        )

        target, index = self.reference.at(
            along + lookahead
        )

        # Evolved lateral racing-line offset
        curvature_here = float(
            self.reference.curvature[index]
        )

        offset = (
            -float(g[8])
            * math.tanh(curvature_here / 0.045)
        )

        target = (
            target
            + offset * self.reference.normals[index]
        )

        to_target = (
            target - np.array([x, y])
        )

        target_distance = max(
            1e-6,
            float(np.linalg.norm(to_target))
        )

        alpha = wrap(
            math.atan2(
                to_target[1],
                to_target[0]
            ) - float(car.heading)
        )

        # Pure-pursuit steering calculation
        steering_angle = math.atan2(
            2.0
            * float(car.wheelbase)
            * math.sin(alpha),
            target_distance
        )

        steer = float(np.clip(
            g[2]
            * steering_angle
            / float(car.max_steering),
            -1.0,
            1.0
        ))

        # Curvature-aware speed planning
        curve = max(
            self.reference.worst_curvature(
                along,
                g[7]
            ),
            1e-5
        )

        target_speed = float(np.clip(
            math.sqrt(g[3] / curve),
            g[4],
            g[5]
        ))

        throttle = float(np.clip(
            g[6] * (target_speed - speed),
            -1.0,
            1.0
        ))

        # Same two-dimensional action format as PPO
        return np.array(
            [steer, throttle],
            dtype=np.float32
        )


def episode(env, policy, reference=None):
    """Run one episode using the shared simulator."""

    obs, _ = env.reset(
        seed=12345,
        options={"start_index": 0}
    )

    controller = (
        EvolvedController(policy, reference)
        if reference is not None
        else None
    )

    maximum_progress = 0.0
    previous_steer = None

    distance = 0.0

    steering_total = 0.0
    steering_count = 0

    lateral_total = 0.0
    step_count = 0

    last_info = {}
    truncated = False

    while True:
        car = env.sim.car

        before = np.array([
            float(car.x),
            float(car.y)
        ])

        if controller is None:
            # PPO controls the vehicle
            action, _ = policy.predict(
                obs,
                deterministic=True
            )

            action = np.asarray(
                action,
                dtype=np.float32
            ).reshape(2)

        else:
            # Evolutionary controller controls the vehicle
            action = controller.action(env)

        obs, _, terminated, truncated, info = (
            env.step(action)
        )

        car = env.sim.car

        after = np.array([
            float(car.x),
            float(car.y)
        ])

        distance += float(
            np.linalg.norm(after - before)
        )

        lateral, _, _ = (
            env.sim.track.get_track_relative_state(
                car.x,
                car.y,
                car.heading
            )
        )

        lateral_total += abs(float(lateral))

        steering = float(action[0])

        if previous_steer is not None:
            steering_total += abs(
                steering - previous_steer
            )

            steering_count += 1

        previous_steer = steering

        step_count += 1

        maximum_progress = max(
            maximum_progress,
            float(info.get("progress", 0.0))
        )

        last_info = info

        if terminated or truncated:
            break

    completed = bool(
        last_info.get("lap_completed", False)
    )

    return {
        "completed": int(completed),

        "progress_percent": (
            100.0
            if completed
            else round(
                min(100.0, maximum_progress),
                3
            )
        ),

        "lap_seconds": (
            round(
                float(last_info["time"]),
                3
            )
            if completed
            else ""
        ),

        "elapsed_seconds": round(
            float(
                last_info.get(
                    "time",
                    step_count * env.dt
                )
            ),
            3
        ),

        "distance_driven": round(
            distance,
            3
        ),

        "mean_abs_lateral_error": round(
            lateral_total / max(1, step_count),
            3
        ),

        "mean_steering_change": round(
            steering_total / max(1, steering_count),
            5
        ),

        "status": (
            "lap"
            if completed
            else "timeout"
            if truncated
            else "off_track"
        )
    }


def fitness(result):
    """Reward successful, fast laps and partial progress."""

    if result["completed"]:
        return (
            300.0
            + 5.0 * max(
                0.0,
                60.0 - float(result["lap_seconds"])
            )
        )

    return (
        1.7 * float(result["progress_percent"])
        - 20.0
    )


def save_csv(path, rows):
    if not rows:
        return

    with path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0])
        )

        writer.writeheader()
        writer.writerows(rows)


def train(args):
    """Optimize controller genes through evolutionary search."""

    if (
        args.generations < 1
        or args.population < 4
        or args.train_trials < 1
    ):
        raise SystemExit(
            "Need >=1 generation, >=4 population, "
            ">=1 training trial"
        )

    rng = np.random.default_rng(args.seed)

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True
    )

    # Training tracks do not overlap the held-out test tracks
    train_ids = range(
        200,
        200 + args.train_trials
    )

    environments = []

    for family in FAMILIES:
        for trial in train_ids:
            env = RacingEnv(
                track=build_track(family, trial)
            )

            ref = TrackReference(
                env.sim.track.centerline
            )

            environments.append(
                (family, trial, env, ref)
            )

    champion = DEFAULT.copy()
    best_score = -float("inf")

    history = []

    # Initial population
    population = [champion.copy()]

    while len(population) < args.population:
        genes = np.clip(
            DEFAULT
            + rng.normal(size=len(NAMES)) * SIGMA,
            LOW,
            HIGH
        )

        population.append(genes)

    try:
        for generation in range(
            1,
            args.generations + 1
        ):
            scores = []

            # Evaluate every candidate controller
            for genes in population:
                results = [
                    episode(env, genes, ref)
                    for _, _, env, ref in environments
                ]

                score = float(
                    mean(
                        fitness(result)
                        for result in results
                    )
                )

                scores.append(score)

            # Rank candidates by fitness
            ranks = np.argsort(scores)[::-1]

            top_index = int(ranks[0])

            if scores[top_index] > best_score:
                best_score = scores[top_index]

                champion = population[
                    top_index
                ].copy()

                checkpoint = {
                    "controller":
                        "evolutionary_tuned_geometric_pure_pursuit",

                    "parameters": dict(
                        zip(
                            NAMES,
                            map(float, champion)
                        )
                    ),

                    "best_training_fitness":
                        best_score,

                    "seed": args.seed,

                    "training_trials_per_family":
                        args.train_trials,

                    "training_trial_ids":
                        list(train_ids),

                    "note":
                        "Geometry-aware policy; PPO observation inputs differ."
                }

                (
                    OUTPUT / "best_controller.json"
                ).write_text(
                    json.dumps(
                        checkpoint,
                        indent=2
                    ),
                    encoding="utf-8"
                )

            row = {
                "generation": generation,

                "generation_best_fitness": round(
                    scores[top_index],
                    4
                ),

                "best_fitness_ever": round(
                    best_score,
                    4
                )
            }

            history.append(row)

            save_csv(
                OUTPUT / "training_history.csv",
                history
            )

            print(
                f"Generation {generation:3}/"
                f"{args.generations} | "
                f"best={scores[top_index]:8.2f} | "
                f"champion={best_score:8.2f}",
                flush=True
            )

            # Select the strongest candidates
            elite_count = max(
                2,
                args.population // 4
            )

            elite = [
                population[int(i)].copy()
                for i in ranks[:elite_count]
            ]

            # Keep the champion and initial baseline
            next_population = [
                champion.copy(),
                DEFAULT.copy()
            ]

            sigma_scale = max(
                0.25,
                1.0 - 0.70
                * generation / args.generations
            )

            # Mutate elite controllers
            while len(next_population) < args.population:
                parent = elite[
                    int(rng.integers(len(elite)))
                ]

                child = np.clip(
                    parent
                    + rng.normal(
                        size=len(NAMES)
                    ) * SIGMA * sigma_scale,
                    LOW,
                    HIGH
                )

                next_population.append(child)

            population = next_population

    finally:
        for _, _, env, _ in environments:
            env.close()

    print(
        "\nSaved",
        OUTPUT / "best_controller.json"
    )

    print(
        "Training tracks use IDs 200+. "
        "Do NOT tune using held-out IDs 10000+."
    )


def evaluate(args):
    """Evaluate evolution and PPO on identical held-out tracks."""

    checkpoint = OUTPUT / "best_controller.json"

    if not checkpoint.is_file():
        raise SystemExit(
            "No evolved controller. Run 'train' first."
        )

    saved = json.loads(
        checkpoint.read_text(
            encoding="utf-8"
        )
    )

    genes = np.array([
        saved["parameters"][name]
        for name in NAMES
    ], dtype=float)

    models = {}

    # Load existing trained PPO agents
    for name, path in MODEL_FILES.items():
        if not path.is_file():
            raise SystemExit(
                f"Missing saved PPO model: {path}"
            )

        model = PPO.load(
            str(path),
            device="cpu"
        )

        if tuple(
            model.observation_space.shape
        ) not in ((10,), (16,)):
            raise SystemExit(
                f"Unsupported observation shape for {name}"
            )

        if tuple(
            model.action_space.shape
        ) != (2,):
            raise SystemExit(
                f"Unsupported action shape for {name}"
            )

        models[name] = model

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True
    )

    rows = []

    for family in FAMILIES:
        for trial in range(
            args.trial_start,
            args.trial_start + args.trials
        ):
            policies = [
                ("Evolved geometric", genes),
                *models.items()
            ]

            for name, policy in policies:
                dims = (
                    (10,)
                    if name == "Evolved geometric"
                    else tuple(
                        policy.observation_space.shape
                    )
                )

                env = RacingEnv(
                    track=build_track(family, trial),

                    curvature_observation=(
                        dims == (16,)
                    )
                )

                try:
                    reference = (
                        TrackReference(
                            env.sim.track.centerline
                        )
                        if name == "Evolved geometric"
                        else None
                    )

                    metrics = episode(
                        env,
                        policy,
                        reference
                    )

                finally:
                    env.close()

                rows.append({
                    "model": name,
                    "family": family,
                    "trial": trial,
                    **metrics
                })

                print(
                    f"{family:12} "
                    f"{trial:5} "
                    f"{name:24} "
                    f"lap="
                    f"{'YES' if metrics['completed'] else 'NO ':3} "
                    f"progress="
                    f"{metrics['progress_percent']:6.1f}%",
                    flush=True
                )

    save_csv(
        OUTPUT / "comparison_runs.csv",
        rows
    )

    summary = []

    for name in [
        "Evolved geometric",
        *models
    ]:
        for family in FAMILIES:
            group = [
                row
                for row in rows
                if (
                    row["model"] == name
                    and row["family"] == family
                )
            ]

            laps = [
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
                    1
                ),

                "mean_progress_percent": round(
                    mean(
                        row["progress_percent"]
                        for row in group
                    ),
                    2
                ),

                "mean_successful_lap_seconds": (
                    round(
                        mean(laps),
                        3
                    )
                    if laps else ""
                ),

                "mean_abs_lateral_error": round(
                    mean(
                        row["mean_abs_lateral_error"]
                        for row in group
                    ),
                    3
                ),

                "mean_steering_change": round(
                    mean(
                        row["mean_steering_change"]
                        for row in group
                    ),
                    5
                )
            })

    save_csv(
        OUTPUT / "comparison_summary.csv",
        summary
    )

    print(
        "\nSaved",
        OUTPUT / "comparison_summary.csv"
    )

    print(
        "Reminder: physics is shared, but the "
        "evolved controller sees the full centerline."
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True
    )

    train_cmd = subparsers.add_parser(
        "train",
        help="Train evolutionary controller"
    )

    train_cmd.add_argument(
        "--generations",
        type=int,
        default=20
    )

    train_cmd.add_argument(
        "--population",
        type=int,
        default=20
    )

    train_cmd.add_argument(
        "--train-trials",
        type=int,
        default=2
    )

    train_cmd.add_argument(
        "--seed",
        type=int,
        default=2026
    )

    evaluate_cmd = subparsers.add_parser(
        "evaluate",
        help="Compare on held-out tracks"
    )

    evaluate_cmd.add_argument(
        "--trials",
        type=int,
        default=20
    )

    evaluate_cmd.add_argument(
        "--trial-start",
        type=int,
        default=10000
    )

    args = parser.parse_args()

    if args.command == "train":
        train(args)

    else:
        if (
            args.trials < 1
            or args.trial_start < 10000
        ):
            raise SystemExit(
                "Evaluation requires --trials >= 1 "
                "and --trial-start >= 10000"
            )

        evaluate(args)


if __name__ == "__main__":
    main()
