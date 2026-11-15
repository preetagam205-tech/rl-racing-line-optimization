# Reinforcement Learning for Optimal Racing-Line Generation

**Autonomous racing on procedural tracks using PPO, evolutionary optimization, and classical control**

A Python research and visualization project exploring how reinforcement-learning agents learn to navigate racing tracks, trade speed for reliability, and generalize across unseen circuits. Built with Gymnasium, Stable-Baselines3, PyTorch, Pygame, NumPy, and Matplotlib.

> **Scope:** This project investigates *racing-line optimization*, but the results do not establish a globally optimal or provably minimum-time trajectory.

## What this project demonstrates

- **Custom racing simulator** with bicycle-style steering, speed-dependent tire-grip limits, off-track detection, checkpoints, and lap validation.
- **PPO agents** trained for multi-track driving, including models that use local curvature observations.
- **Procedurally generated tracks**: ellipse, wavy, complex, spline, and hard spline.
- **Classical and evolutionary baselines**: Pure Pursuit centerline tracking and an evolutionarily tuned geometry-aware controller.
- **Visual learning replay** across PPO checkpoints, multicar evolutionary visualization, and trajectory/telemetry plots.
- **Held-out benchmarking** on 20 generated circuits per track family, with completion rate, lap time, steering changes, and lateral error.

## Main findings

### Completion rates on 100 held-out circuits per controller

| Controller | Ellipse | Wavy | Complex | Spline | Hard spline | Total |
|---|---:|---:|---:|---:|---:|---:|
| Tuned Pure Pursuit | 100% | 100% | 100% | 100% | 100% | **100%** |
| Broad Domain PPO | 100% | 100% | 100% | 70% | 30% | **80%** |
| Evolved geometric | 35% | 60% | 90% | 100% | 100% | **77%** |
| Hard Five Family PPO | 100% | 85% | 50% | 55% | 40% | **66%** |

### Mean successful-lap times (seconds, lower is better)

| Controller | Ellipse | Wavy | Complex | Spline | Hard spline |
|---|---:|---:|---:|---:|---:|
| Tuned Pure Pursuit | 20.762 | 20.317 | 23.170 | 28.660 | 31.465 |
| Broad Domain PPO | **15.363** | 15.893 | **15.810** | **16.979** | **16.100** |
| Evolved geometric | 15.429 | **15.188** | 16.344 | 19.238 | 20.170 |
| Hard Five Family PPO | 16.905 | 17.015 | 16.675 | 18.568 | 17.475 |

**Interpretation:** Broad Domain PPO produces fast successful laps, whereas Pure Pursuit is the most reliable. The evolved controller completes all 40 of the tested spline and hard-spline layouts. Successful-lap averages are *conditional on finishing*, so they should not be interpreted as overall race performance or as a strict head-to-head comparison.

### Matched-lap result: Broad Domain PPO vs tuned Pure Pursuit

On circuits **both controllers successfully completed**, Broad Domain reduced lap time by a mean of **21.79–46.09%**, depending on the track family. Its path length was **1.09–2.08% shorter**. These are per-circuit paired percentage differences, not proof that the learned geometric line alone caused the time reduction.

Full tables and caveats: [Experiment results](docs/RESULTS.md).

## Technical stack

- Python, NumPy, Matplotlib
- Gymnasium (custom racing environment)
- Stable-Baselines3 PPO / PyTorch
- Pygame for interactive racing visualization

## Code layout

The following is the **existing local-project layout intended for publication**. Scripts and model files are not yet all uploaded to this repository.

```text
src/
  racing_env.py                 # Gymnasium agent interface
  simulator.py                  # Lap and checkpoint validation
  vehicle.py                    # Speed-dependent steering / grip physics
  track_geometry.py            # Centerline and track boundaries
  track_factory.py             # Procedural circuit generation
benchmark_ppo.py                # PPO evaluation
benchmark_multitrack_models.py  # Cross-model performance
analyze_racing_lines.py         # Trajectory and telemetry evaluation
benchmark_centerline.py         # Pure Pursuit baseline
benchmark_centerline_tuned.py   # Speed-limit experiment
compare_matched_laps.py         # Paired-circuit statistics
evolve_shared_physics.py        # Evolved controller + shared-physics test
racing_showcase.py              # Evolutionary population visualization
ppo_learning_replay.py          # Saved PPO training checkpoint replay
racing_results.py               # Evolutionary result viewer
docs/RESULTS.md                 # Experimental findings and limitations
```

The visual evolutionary population simulator and the **shared-physics** evolutionary evaluation are distinct implementations. Their raw lap times must not be compared.

## Run locally

Use the corresponding local Python source and trained policy ZIP files from this project. On macOS or Linux:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install numpy pygame-ce gymnasium stable-baselines3 torch matplotlib pandas
```

*The command above describes the Python dependencies used; exact package versions are not yet pinned or independently tested as a clean installation.*

After adding the source files and saved policies:

```bash
python racing_showcase.py
python ppo_learning_replay.py

# Evaluation on the same seeded held-out test circuits
python benchmark_centerline_tuned.py --trials 20 --trial-start 10000
python compare_matched_laps.py
python evolve_shared_physics.py train --generations 20 --population 20 --train-trials 2
python evolve_shared_physics.py evaluate --trials 20
```

PPO benchmarking and replay require trained model ZIP files in `models/`, which are **not yet distributed** with this repository.

## Evaluation caveats

1. **Shared physics is not a fully controlled algorithm comparison.** The evolved geometric controller reads the complete track centerline, whereas PPO receives local observations; training budgets and controller structures differ.
2. **Success conditioning.** Fast successful laps do not compensate for a low completion rate. For example, Broad Domain finishes only 6/20 hard-spline trials.
3. **Generalization is within the generator families.** Different held-out procedural seeds are not fundamentally new types of race circuit.
4. **Global optimality is not proven.** Neither Pure Pursuit nor the evolved policy is an independently verified minimum-time optimum.
5. **Reproducibility is still being packaged.** The local source tree, selected model checkpoints, raw result CSVs, and media should be uploaded before calling this a fully reproducible release.

## Project status

Main experiments and comparisons have been run locally. **GitHub publication, source syncing, and demo packaging are in progress.**
