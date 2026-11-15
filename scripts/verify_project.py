"""Fast, read-only checks for the published simulator, checkpoints and results.

Run from any directory with ``python /path/to/scripts/verify_project.py``.
This does not train agents, launch a window, or overwrite benchmark artifacts.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from benchmark_ppo import FAMILIES, build_track
from benchmark_centerline_tuned import CenterlineFollower
from evolve_shared_physics import EvolvedController, NAMES, TrackReference
from src.racing_env import RacingEnv


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def read_csv(relative_path):
    with (ROOT / relative_path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def verify_environment():
    for curvature in (False, True):
        env = RacingEnv(curvature_observation=curvature)
        try:
            check_env(env, warn=True)
            env.max_episode_time = 0.10
            env.reset(seed=42)
            _, _, terminated, truncated, _ = env.step(np.zeros(2, np.float32))
            require(not terminated and not truncated, "Episode ended too early")
            _, _, terminated, truncated, info = env.step(np.zeros(2, np.float32))
            require(truncated and not terminated, "Time limit must truncate")
            require(not info["lap_completed"], "Standing still cannot finish a lap")

            env.reset(seed=42)
            env.sim.car.x = 1e6
            env.sim.car.y = 1e6
            _, _, terminated, truncated, info = env.step(np.zeros(2, np.float32))
            require(terminated and not truncated, "Off-track must terminate")
            require(not info["run_valid"], "Off-track run must be invalid")
        finally:
            env.close()
    print("PASS: Gymnasium contract, timeout, and off-track termination (10/16 observations)")


def verify_tracks():
    for family in FAMILIES:
        first = build_track(family, 10000)
        second = build_track(family, 10000)
        np.testing.assert_array_equal(first.centerline, second.centerline)
        require(np.isfinite(first.centerline).all(), f"Nonfinite {family} track")
        require(np.all(first.segment_lengths > 0), f"Degenerate {family} segments")

        env = RacingEnv(track=first)
        try:
            action = np.array([0.1, 0.5], dtype=np.float32)
            initial, _ = env.reset(seed=42, options={"start_index": 37})
            observation, reward, terminated, truncated, info = env.step(action)
            repeated, _ = env.reset(seed=42, options={"start_index": 37})
            again = env.step(action)
            np.testing.assert_array_equal(initial, repeated)
            np.testing.assert_array_equal(observation, again[0])
            require((reward, terminated, truncated, info) == again[1:],
                    f"Nondeterministic reset/step for {family}")
        finally:
            env.close()
    print("PASS: All five track families and repeated seeded reset/step")


def verify_controllers():
    env = RacingEnv(track=build_track("ellipse", 10000))
    try:
        obs, _ = env.reset(seed=12345, options={"start_index": 0})
        baseline = CenterlineFollower(env.sim.track.centerline)
        checkpoint = json.loads(
            (ROOT / "evolution_shared_physics/best_controller.json").read_text()
        )
        genes = np.array([checkpoint["parameters"][name] for name in NAMES])
        evolved = EvolvedController(genes, TrackReference(env.sim.track.centerline))
        for name, controller in (("Pure Pursuit", baseline), ("Evolution", evolved)):
            action = controller.action(env)
            require(env.action_space.contains(action), f"Invalid {name} action")

        # A complete baseline lap exercises forward gate crossing and every checkpoint.
        for _ in range(math.ceil(env.max_episode_time / env.dt) + 1):
            obs, reward, terminated, truncated, info = env.step(baseline.action(env))
            require(np.isfinite(obs).all() and math.isfinite(reward), "Nonfinite rollout")
            if terminated or truncated:
                break
        require(info["lap_completed"] and info["run_valid"], "Baseline lap did not finish")
        require(env.sim.checkpoints_passed_count == env.sim.track.num_checkpoints,
                "Lap completed without passing all ordered checkpoints")
    finally:
        env.close()
    print("PASS: Saved evolutionary controller and complete Pure Pursuit lap")


def verify_models():
    paths = [
        ROOT / "models/ppo_broaddomain_stage1.zip",
        ROOT / "models/ppo_hardfivefamily_stage1.zip",
        *[ROOT / f"models/visual_training/ppo_visual_{steps:06d}.zip"
          for steps in (0, 25000, 50000, 100000, 250000, 500000)],
    ]
    for path in paths:
        require(path.is_file(), f"Missing published checkpoint: {path.relative_to(ROOT)}")
        model = PPO.load(path, device="cpu")
        require(model.observation_space.shape in ((10,), (16,)),
                f"Unsupported observation shape: {path.name}")
        env = RacingEnv(curvature_observation=model.observation_space.shape == (16,))
        try:
            observation, _ = env.reset(seed=42)
            action, _ = model.predict(observation, deterministic=True)
            require(env.action_space.contains(action), f"Invalid model action: {path.name}")
            observation, reward, _, _, _ = env.step(action)
            require(env.observation_space.contains(observation) and math.isfinite(reward),
                    f"Invalid model transition: {path.name}")
        finally:
            env.close()
    print(f"PASS: {len(paths)} published PPO checkpoints load and act in RacingEnv")


def verify_result_table(directory, basename, expected_models):
    rows = read_csv(f"{directory}/{basename}_runs.csv")
    summaries = read_csv(f"{directory}/{basename}_summary.csv")
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["family"])].append(row)
    expected_groups = {(model, family) for model in expected_models for family in FAMILIES}
    require(set(grouped) == expected_groups, f"Unexpected groups in {directory}")
    require(len(summaries) == len(expected_groups), f"Unexpected summary size in {directory}")
    require({(r["model"], r["family"]) for r in summaries} == expected_groups,
            f"Missing or duplicated summary groups in {directory}")
    for summary in summaries:
        key = summary["model"], summary["family"]
        group = grouped[key]
        require(len(group) == 20 and {int(r["trial"]) for r in group} == set(range(10000, 10020)),
                f"Expected held-out trials 10000–10019 for {key}")
        require(int(summary["runs"]) == len(group), f"Incorrect run count for {key}")
        successes = [r for r in group if int(r["completed"])]
        for row in group:
            require(bool(row["lap_seconds"]) == bool(int(row["completed"])),
                    f"Failed run has a successful lap time: {key}")
        require(math.isclose(float(summary["completion_rate_percent"]), 100 * len(successes) / len(group)),
                f"Incorrect completion rate for {key}")
        for column, output, digits in (
            ("progress_percent", "mean_progress_percent", 2),
            ("mean_abs_lateral_error", "mean_abs_lateral_error", 3),
            ("mean_steering_change", "mean_steering_change", 5),
        ):
            average = sum(float(r[column]) for r in group) / len(group)
            require(abs(float(summary[output]) - average) <= 0.51 * 10 ** -digits,
                    f"Incorrect {output} for {key}")
        if successes:
            average = sum(float(r["lap_seconds"]) for r in successes) / len(successes)
            require(abs(float(summary["mean_successful_lap_seconds"]) - average) <= 0.00051,
                    f"Incorrect successful-lap mean for {key}")
        else:
            require(not summary["mean_successful_lap_seconds"], f"Unexpected mean for {key}")
    return {(r["model"], r["family"], r["trial"]): r for r in rows}


def verify_results():
    ppo_models = ("Broad Domain PPO", "Hard Five Family PPO")
    baseline = verify_result_table("baseline_comparison_tuned", "baseline_comparison",
                                   ("Centerline Pure Pursuit", *ppo_models))
    evolution = verify_result_table("evolution_shared_physics", "comparison",
                                    ("Evolved geometric", *ppo_models))
    common_keys = baseline.keys() & evolution.keys()
    require(len(common_keys) == 200, "Expected the same 200 PPO episodes in both comparisons")
    for key in common_keys:
        for column in ("completed", "lap_seconds", "progress_percent"):
            require(baseline[key][column] == evolution[key][column],
                    f"Shared PPO runs disagree: {key}, {column}")
    print("PASS: 600 archived benchmark rows, held-out IDs, summaries, and shared PPO results")


def main():
    verify_environment()
    verify_tracks()
    verify_controllers()
    verify_models()
    verify_results()
    print("All verification checks passed. No training or benchmark outputs were written.")


if __name__ == "__main__":
    main()
