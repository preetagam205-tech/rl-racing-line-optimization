"""Regenerate the publication figures and GIFs from real experiment data.

Run ``python generate_assets.py`` to use the committed rollout cache, or add
``--refresh-rollouts`` to replay the committed policies in RacingEnv. No model
training occurs. The two animations distinguish PPO timesteps, evolutionary
generations, and simulated driving time.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "assets"
FAMILIES = ("ellipse", "wavy", "complex", "spline", "hard_spline")
MODELS = ("Broad Domain PPO", "Centerline Pure Pursuit", "Evolved geometric")
LABELS = ("PPO · Broad Domain", "Pure Pursuit · tuned", "Evolutionary · geometric")
COLORS = ("#49d7e8", "#ffbf69", "#bd9aff")
NOMINAL_STEPS = (0, 25000, 50000, 100000, 250000, 500000)
BG, PANEL, FG, MUTED = "#101925", "#172435", "#eff5fc", "#a6b9ce"
TRIAL = 10000


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def read_csv(path):
    with (ROOT / path).open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def configure():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "figure.facecolor": BG, "axes.facecolor": PANEL,
        "axes.edgecolor": "#3b4b5f", "text.color": FG,
        "axes.labelcolor": MUTED, "xtick.color": MUTED,
        "ytick.color": MUTED, "axes.titleweight": "bold",
        "axes.spines.top": False, "axes.spines.right": False,
        "grid.color": "#354457", "grid.alpha": .5,
        "savefig.facecolor": BG, "figure.dpi": 100,
    })


def collect_rollouts():
    # Keep plotting from the committed cache usable without loading Torch.
    import torch
    from stable_baselines3 import PPO
    from benchmark_ppo import build_track
    from benchmark_centerline_tuned import CenterlineFollower
    from evolve_shared_physics import EvolvedController, TrackReference, NAMES
    from ppo_learning_replay import track_points
    from src.racing_env import RacingEnv
    from src.track_geometry import TrackGeometry

    torch.set_num_threads(1)
    arrays, metadata = {}, {}

    def run(key, track, model=None, kind=None, genes=None, seed=12345):
        env = RacingEnv(track=track, curvature_observation=(
            model is not None and tuple(model.observation_space.shape) == (16,)
        ))
        obs, _ = env.reset(seed=seed, options={"start_index": 0})
        controller = None
        if kind == "baseline":
            controller = CenterlineFollower(track.centerline)
        elif kind == "evolved":
            controller = EvolvedController(genes, TrackReference(track.centerline))
        car = env.sim.car
        trace = [[0., car.x, car.y, car.heading, car.speed, 0.]]
        max_progress = 0.
        try:
            for _ in range(int(env.max_episode_time / env.dt) + 3):
                action = controller.action(env) if controller else model.predict(obs, deterministic=True)[0]
                obs, _, terminated, truncated, info = env.step(action)
                car = env.sim.car
                max_progress = max(max_progress, float(info["progress"]))
                trace.append([float(info["time"]), car.x, car.y, car.heading,
                              car.speed, max_progress])
                if terminated or truncated:
                    break
            else:
                raise RuntimeError(f"Episode did not terminate: {key}")
            completed = bool(info["lap_completed"])
            metadata[key] = {
                "status": "lap complete" if completed else "time limit" if truncated else "off track",
                "completed": completed,
                "progress_percent": 100. if completed else min(100., max_progress),
                "elapsed_seconds": float(info["time"]),
                "lap_seconds": float(info["time"]) if completed else None,
                "reset_seed": seed,
            }
            arrays[key] = np.asarray(trace, dtype=np.float64)
            for name in ("centerline", "outer_boundary", "inner_boundary"):
                arrays[f"{key}_{name}"] = getattr(track, name)
            print(f"{key}: {metadata[key]['status']}, {metadata[key]['progress_percent']:.1f}%", flush=True)
        finally:
            env.close()

    model = PPO.load(ROOT / "models/ppo_broaddomain_stage1.zip", device="cpu")
    parameters = json.loads((ROOT / "evolution_shared_physics/best_controller.json").read_text())["parameters"]
    genes = np.array([parameters[name] for name in NAMES])
    for family in FAMILIES:
        for index in range(3):
            run(f"{family}_{index}", build_track(family, TRIAL),
                model=model if index == 0 else None,
                kind=(None, "baseline", "evolved")[index], genes=genes)
    for nominal in NOMINAL_STEPS:
        checkpoint = ROOT / f"models/visual_training/ppo_visual_{nominal:06d}.zip"
        policy = PPO.load(checkpoint, device="cpu")
        key = f"learning_{nominal}"
        track = TrackGeometry(centerline_points=track_points("ellipse", 0),
                              track_width=12., num_checkpoints=7)
        run(key, track, model=policy, seed=20261115)
        metadata[key]["nominal_checkpoint_steps"] = nominal
        metadata[key]["actual_training_timesteps"] = int(policy.num_timesteps)

    sources = [
        "src/racing_env.py", "src/simulator.py", "src/vehicle.py",
        "src/track_geometry.py", "src/track_factory.py", "benchmark_ppo.py",
        "benchmark_centerline_tuned.py", "evolve_shared_physics.py",
        "ppo_learning_replay.py", "models/ppo_broaddomain_stage1.zip",
        "evolution_shared_physics/best_controller.json",
    ] + [f"models/visual_training/ppo_visual_{step:06d}.zip" for step in NOMINAL_STEPS]
    provenance = {
        "description": "Fresh deterministic RacingEnv rollouts; no training or fabricated trajectories",
        "benchmark_trial": TRIAL, "start_index": 0, "dt_seconds": .05,
        "columns": ["time_seconds", "x", "y", "heading_radians", "speed", "maximum_progress_percent"],
        "models": dict(enumerate(MODELS)),
        "input_sha256": {path: sha(path) for path in sources},
        "episodes": metadata,
    }
    np.savez_compressed(OUT / "rollouts.npz", **arrays)
    (OUT / "rollouts.json").write_text(json.dumps(provenance, indent=2) + "\n")


def load_rollouts():
    with np.load(OUT / "rollouts.npz", allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    metadata = json.loads((OUT / "rollouts.json").read_text())
    return arrays, metadata


def plot_track(ax, arrays, key):
    outer, inner = (arrays[f"{key}_{name}"] for name in ("outer_boundary", "inner_boundary"))
    ax.add_patch(Polygon(outer, closed=True, facecolor="#334154", edgecolor="#6c7d91", lw=.8))
    ax.add_patch(Polygon(inner, closed=True, facecolor=PANEL, edgecolor="#6c7d91", lw=.8))
    center = arrays[f"{key}_centerline"]
    closed = np.vstack([center, center[0]])
    ax.plot(*closed.T, color="#8293a7", lw=.7, ls=(0, (4, 5)), alpha=.5)
    ax.plot(*np.vstack([outer[0], inner[0]]).T, color=FG, lw=2, alpha=.7)
    extent = np.vstack([inner, outer])
    ax.set_xlim(extent[:, 0].min() - 6, extent[:, 0].max() + 6)
    ax.set_ylim(extent[:, 1].min() - 6, extent[:, 1].max() + 6)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def family_label(family):
    return family.replace("_", " ").title()


def legend_handles():
    return [Line2D([0], [0], color=color, lw=3, label=label)
            for color, label in zip(COLORS, LABELS)]


def title(fig, heading, subtitle):
    fig.text(.05, .95, heading, fontsize=23, weight="bold", va="top")
    fig.text(.05, .895, subtitle, fontsize=10.5, color=MUTED, va="top")


def trajectories(arrays, provenance):
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 8.5))
    fig.subplots_adjust(left=.04, right=.97, bottom=.1, top=.82, hspace=.35, wspace=.19)
    title(fig, "Different controllers. Different racing lines.",
          "Five reproducible circuits · trial 10000 · identical physics, track and starting pose")
    for family, ax in zip(FAMILIES, axes.flat):
        plot_track(ax, arrays, f"{family}_0")
        ax.set_title(family_label(family), loc="left", fontsize=13, pad=10)
        status_lines = []
        for index, color in enumerate(COLORS):
            key = f"{family}_{index}"
            trace = arrays[key]
            ax.plot(trace[:, 1], trace[:, 2], color=color, lw=1.9, alpha=.96)
            item = provenance["episodes"][key]
            if not item["completed"]:
                ax.scatter(*trace[-1, 1:3], marker="x", s=45, linewidths=2, color=color, zorder=5)
            status_lines.append(f"{item['lap_seconds']:.2f}s" if item["completed"]
                                else f"DNF {item['progress_percent']:.0f}%")
        for index, (label, color) in enumerate(zip(status_lines, COLORS)):
            ax.text(.02 + index * .33, -.1, label, color=color, transform=ax.transAxes,
                    fontsize=10, weight="bold")
    ax = axes.flat[-1]
    ax.axis("off")
    ax.legend(handles=legend_handles(), loc="upper left", frameon=False, fontsize=11,
              labelspacing=1.4, borderaxespad=0)
    ax.text(0, .60, "Solid lines = actual driven paths\n× = episode ended before a lap\nDashed line = track centerline",
            transform=ax.transAxes, color=MUTED, fontsize=11, linespacing=1.7, va="top")
    ax.text(0, .02, "One illustration per family.\nSee the 100-case benchmark for\nreliability and matched lap times.",
            transform=ax.transAxes, color=MUTED, fontsize=10, linespacing=1.6)
    fig.text(.05, .035, "PPO uses local observations; geometric controllers use the full centerline. These examples do not establish a general winner.",
             fontsize=10, color=MUTED)
    fig.savefig(OUT / "racing_trajectories.png", dpi=160)
    plt.close(fig)


def benchmark_data():
    baseline = read_csv("baseline_comparison_tuned/baseline_comparison_runs.csv")
    evolved = read_csv("evolution_shared_physics/comparison_runs.csv")
    lookup = {}
    for row in baseline + evolved:
        if row["model"] not in MODELS:
            continue
        key = (row["family"], row["trial"], row["model"])
        if key in lookup:
            for field in ("completed", "lap_seconds", "progress_percent"):
                if lookup[key][field] != row[field]:
                    raise ValueError(f"Conflicting archived benchmark result: {key} / {field}")
        lookup[key] = row
    completion, laps, matched_counts, denominators = [], [], [], []
    for family in FAMILIES:
        trial_sets = [{trial for f, trial, model in lookup if f == family and model == name}
                      for name in MODELS]
        if not all(trials == trial_sets[0] for trials in trial_sets):
            raise ValueError(f"Controller trial sets differ: {family}")
        trials = sorted(trial_sets[0], key=int)
        matched = [trial for trial in trials if all(
            int(lookup[family, trial, name]["completed"]) for name in MODELS)]
        denominators.append(len(trials))
        matched_counts.append(len(matched))
        completion.append([100 * np.mean([int(lookup[family, trial, name]["completed"])
                                          for trial in trials]) for name in MODELS])
        laps.append([np.mean([float(lookup[family, trial, name]["lap_seconds"])
                              for trial in matched]) if matched else np.nan for name in MODELS])
    return np.array(completion), np.array(laps), matched_counts, denominators


def performance():
    completion, laps, counts, denominators = benchmark_data()
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 9.3))
    fig.subplots_adjust(left=.08, right=.97, top=.79, bottom=.17, hspace=.49)
    title(fig, "Fast laps matter. Finishing matters too.",
          "Archived benchmark · five track families · 20 cases per family · same RacingEnv physics")
    fig.legend(handles=legend_handles(), loc="upper left", bbox_to_anchor=(.07, .855),
               frameon=False, ncol=3, fontsize=10.5)
    xs, width = np.arange(5), .23
    axes[0].set_title("Completion rate — all runs", loc="left", fontsize=13, pad=13)
    axes[1].set_title("Mean lap time — only cases completed by all three controllers", loc="left", fontsize=13, pad=13)
    for index, color in enumerate(COLORS):
        x = xs + (index - 1) * width
        axes[0].bar(x, completion[:, index], width, color=color, zorder=3)
        axes[1].bar(x, laps[:, index], width, color=color, zorder=3)
        for xpos, val in zip(x, completion[:, index]):
            axes[0].text(xpos, val + 2, f"{val:.0f}%", ha="center", color=color, fontsize=10, weight="bold")
        for xpos, val in zip(x, laps[:, index]):
            if np.isfinite(val):
                axes[1].text(xpos, val + .6, f"{val:.1f}", ha="center", color=color, fontsize=10, weight="bold")
    axes[0].set_ylim(0, 116)
    axes[0].set_yticks([0, 25, 50, 75, 100])
    axes[0].set_ylabel("Completed laps (%)")
    axes[1].set_ylim(0, 35)
    axes[1].set_ylabel("Lap time (seconds)")
    for ax in axes:
        ax.set_xticks(xs)
        ax.grid(axis="y")
        ax.set_axisbelow(True)
    axes[0].set_xticklabels([f"{family_label(f)}\nn = {n}" for f, n in zip(FAMILIES, denominators)])
    axes[1].set_xticklabels([f"{family_label(f)}\nmatched n = {n}" for f, n in zip(FAMILIES, counts)])
    fig.text(.08, .035, "Lap-time means exclude failures and describe a selected subset; read them with completion rates.\n"
             "Geometric policies use full-track information. Observation access and training budgets are not matched.",
             fontsize=10, color=MUTED, linespacing=1.7)
    fig.savefig(OUT / "performance_comparison.png", dpi=160)
    plt.close(fig)


def frame(fig):
    fig.canvas.draw()
    return Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())


def save_gif(path, frames, duration=100):
    # One palette avoids flicker and keeps the downloadable demonstrations small.
    palette = frames[0].quantize(colors=128)
    converted = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in frames]
    converted[0].save(path, save_all=True, append_images=converted[1:], duration=duration,
                      loop=0, optimize=False, disposal=2)


def training_demo(arrays, provenance):
    frames = []
    fig = plt.figure(figsize=(11, 6.6))
    title(fig, "Learning to drive · PPO checkpoint replay", "Scene 1 / 2 · actual saved policies on the same ellipse · deterministic evaluation")
    ax = fig.add_axes([.045, .21, .57, .58])
    plot_track(ax, arrays, "learning_0")
    trace_line, = ax.plot([], [], color=COLORS[0], lw=2.5)
    marker, = ax.plot([], [], "o", color=FG, ms=7, markeredgecolor=COLORS[0], markeredgewidth=2)
    step_text = fig.text(.66, .75, "", fontsize=22, weight="bold", color=COLORS[0])
    fig.text(.66, .70, "actual training timesteps", fontsize=10, color=MUTED)
    result_text = fig.text(.66, .56, "", fontsize=17, weight="bold")
    clock_text = fig.text(.66, .48, "", fontsize=11, color=MUTED)
    fig.text(.66, .29, "Saved checkpoints, replayed.\nNo retraining for this animation.\nLater checkpoints can regress\non unfamiliar track families.",
             fontsize=11, color=MUTED, linespacing=1.7)
    fig.text(.05, .09, "Each episode is compressed to the same playback length; the on-screen time is simulated time.", fontsize=10, color=MUTED)
    fig.text(.05, .05, "This ellipse is an illustration. Benchmark reliability is reported separately across five families.", fontsize=10, color=MUTED)
    for nominal in NOMINAL_STEPS:
        key = f"learning_{nominal}"
        trace, meta = arrays[key], provenance["episodes"][key]
        step_text.set_text(f"{meta['actual_training_timesteps']:,}")
        for proportion in np.r_[np.linspace(0, 1, 16), np.ones(5)]:
            end = min(len(trace), max(1, int(proportion * (len(trace) - 1)) + 1))
            trace_line.set_data(trace[:end, 1], trace[:end, 2])
            marker.set_data([trace[end - 1, 1]], [trace[end - 1, 2]])
            clock_text.set_text(f"Simulation time  {trace[end - 1, 0]:.2f}s")
            if end == len(trace):
                result_text.set_text(f"{meta['status'].upper()}\n{meta['progress_percent']:.1f}% progress")
            else:
                result_text.set_text(f"DRIVING\n{min(100., trace[end - 1, 5]):.1f}% progress")
            frames.append(frame(fig))
    plt.close(fig)

    history = read_csv("evolution_shared_physics/training_history.csv")
    generations = np.array([int(row["generation"]) for row in history])
    best = np.array([float(row["best_fitness_ever"]) for row in history])
    current = np.array([float(row["generation_best_fitness"]) for row in history])
    fig = plt.figure(figsize=(11, 6.6))
    title(fig, "Learning to drive · evolutionary search", "Scene 2 / 2 · genuine logged scores · geometric controller optimized inside RacingEnv")
    ax = fig.add_axes([.09, .23, .58, .52])
    ax.set_xlim(1, max(generations))
    ax.set_ylim(min(current) - 5, max(best) + 5)
    ax.set_xlabel("Evolutionary generation")
    ax.set_ylabel("Training fitness (higher is better)")
    ax.set_xticks([1, 5, 10, 15, 20] if max(generations) == 20 else np.unique(np.linspace(1, max(generations), 5, dtype=int)))
    ax.grid()
    current_line, = ax.plot([], [], color=MUTED, ls="--", lw=1.5, label="Best in generation")
    best_line, = ax.plot([], [], color=COLORS[2], marker="o", ms=4, lw=2.5, label="Best recorded so far")
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    generation_text = fig.text(.73, .69, "", fontsize=27, color=COLORS[2], weight="bold")
    fig.text(.73, .64, "generations evaluated", fontsize=10, color=MUTED)
    fitness_text = fig.text(.73, .51, "", fontsize=23, weight="bold")
    fig.text(.73, .46, "best recorded fitness", fontsize=10, color=MUTED)
    fig.text(.73, .25, "Training tracks: 200–201\nin each of five families.\nFitness is a training score,\nnot a held-out lap time.", fontsize=10, color=MUTED, linespacing=1.7)
    fig.text(.05, .1, "Only the final controller was preserved. The curve uses the training log; intermediate driving paths are unavailable.", fontsize=10, color=MUTED)
    fig.text(.05, .055, "Evolutionary generations and PPO timesteps are different training units and are not compared as equal budgets.", fontsize=10, color=MUTED)
    for count in range(1, len(history) + 1):
        current_line.set_data(generations[:count], current[:count])
        best_line.set_data(generations[:count], best[:count])
        generation_text.set_text(f"{generations[count - 1]} / {generations[-1]}")
        fitness_text.set_text(f"{best[count - 1]:.3f}")
        image = frame(fig)
        frames.extend([image] * (3 if count != len(history) else 16))
    fig.savefig(OUT / "evolution_learning_curve.png", dpi=160)
    plt.close(fig)
    save_gif(OUT / "training_demo.gif", frames)


def driving_demo(arrays, provenance):
    fig, axes = plt.subplots(2, 3, figsize=(11, 7.2))
    fig.subplots_adjust(left=.035, right=.97, bottom=.12, top=.81, hspace=.34, wspace=.18)
    title(fig, "Three controllers. Five circuits.", "Deterministic replay · benchmark trial 10000 · all vehicles share the same simulation clock")
    lines, markers, statuses = {}, {}, {}
    for family, ax in zip(FAMILIES, axes.flat):
        plot_track(ax, arrays, f"{family}_0")
        ax.set_title(family_label(family), loc="left", fontsize=12)
        for index, color in enumerate(COLORS):
            key = f"{family}_{index}"
            lines[key], = ax.plot([], [], color=color, lw=1.5)
            markers[key], = ax.plot([], [], "o", color=color, ms=5)
            statuses[key] = ax.text(.01 + .33 * index, -.1, "", fontsize=9, color=color, transform=ax.transAxes)
    axes.flat[-1].axis("off")
    axes.flat[-1].legend(handles=legend_handles(), loc="upper left", frameon=False, fontsize=10, labelspacing=1.3)
    sim_clock = axes.flat[-1].text(.02, .43, "", fontsize=21, weight="bold", transform=axes.flat[-1].transAxes)
    axes.flat[-1].text(.02, .05, "Vehicles stop at episode end.\n× marks a failed lap.\nPlayback is accelerated.", fontsize=10, color=MUTED,
                      linespacing=1.6, transform=axes.flat[-1].transAxes)
    fig.text(.05, .035, "Illustrations include failures. See performance_comparison.png for all 100 benchmark cases per controller.", fontsize=10, color=MUTED)
    end_time = max(provenance["episodes"][key]["elapsed_seconds"] for key in lines)
    frames = []
    for sim_time in np.r_[np.linspace(0, end_time, 100), np.repeat(end_time, 12)]:
        sim_clock.set_text(f"t = {sim_time:.1f}s")
        for key in lines:
            trace, meta = arrays[key], provenance["episodes"][key]
            end = max(1, int(np.searchsorted(trace[:, 0], sim_time, side="right")))
            lines[key].set_data(trace[:end, 1], trace[:end, 2])
            markers[key].set_data([trace[end - 1, 1]], [trace[end - 1, 2]])
            done = sim_time >= trace[-1, 0]
            markers[key].set_marker("x" if done and not meta["completed"] else "o")
            statuses[key].set_text((f"{meta['lap_seconds']:.1f}s" if meta["completed"] else "DNF")
                                    if done else f"{min(100., trace[end - 1, 5]):.0f}%")
        frames.append(frame(fig))
    plt.close(fig)
    save_gif(OUT / "driving_maps.gif", frames)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-rollouts", action="store_true", help="Replay all 21 episodes using the committed checkpoints")
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    configure()
    if args.refresh_rollouts or not (OUT / "rollouts.npz").exists():
        collect_rollouts()
    arrays, provenance = load_rollouts()
    trajectories(arrays, provenance)
    performance()
    training_demo(arrays, provenance)
    driving_demo(arrays, provenance)
    source_paths = ["baseline_comparison_tuned/baseline_comparison_runs.csv",
                    "evolution_shared_physics/comparison_runs.csv",
                    "evolution_shared_physics/training_history.csv",
                    "assets/rollouts.npz", "assets/rollouts.json", "generate_assets.py"]
    (OUT / "manifest.json").write_text(json.dumps({
        "generator": "python generate_assets.py --refresh-rollouts",
        "input_sha256": {path: sha(path) for path in source_paths},
        "note": "Static statistics are from archived CSVs. Driving visuals use fresh deterministic cached rollouts, not archived telemetry.",
        "matched_completed_cases_per_family": dict(zip(FAMILIES, benchmark_data()[2])),
    }, indent=2) + "\n")
    for path in sorted(OUT.glob("*")):
        if path.suffix in (".gif", ".png"):
            print(f"{path.relative_to(ROOT)}: {path.stat().st_size / 1_000_000:.2f} MB")


if __name__ == "__main__":
    main()
