# Checkpoints and reproducibility

The repository includes 29 compact Stable-Baselines3 PPO checkpoints (about 4.1 MiB total), the shared-physics evolutionary champion, and saved populations for the standalone showcase. A normal clone downloads them directly; Git LFS and separate model downloads are not required.

## Recommended checkpoints

| Artifact | Purpose | Observation size | Actual PPO steps |
| --- | --- | ---: | ---: |
| `models/ppo_broaddomain_stage1.zip` | Broad Domain PPO in the main comparison | 10 | 4,501,504 |
| `models/ppo_hardfivefamily_stage1.zip` | Hard Five Family PPO in the main comparison | 10 | 7,501,824 |
| `models/ppo_curvature_stage1.zip` | Curvature-observation experiment | 16 | See manifest |
| `models/visual_training/ppo_visual_*.zip` | Six snapshots of the single-ellipse demonstration | 10 | 0–507,904 |
| `evolution_shared_physics/best_controller.json` | Nine evolved geometric controller parameters | Full centerline | Not applicable |
| `racing_checkpoints/*.json` | Populations for the separate interactive showcase | Separate simulator | Not applicable |

Earlier PPO experiments are retained because historical analysis scripts reference them. Some use older observation sizes (for example, 8 values) and cannot be evaluated directly in the current 10/16-value environment. Use the recommended policies with the main benchmark scripts. Do not infer architectural compatibility from a checkpoint filename.

## Integrity and metadata

[`models/manifest.json`](../models/manifest.json) records every PPO file's SHA-256 digest, byte size, observation shape, saved timestep count, seed, and original runtime metadata. These fields were extracted from the ZIP archives. Verify file integrity from the repository root:

```bash
python - <<'PY'
import hashlib, json
from pathlib import Path
manifest = json.loads(Path('models/manifest.json').read_text())
for item in manifest['checkpoints']:
    path = Path(item['path'])
    assert path.stat().st_size == item['bytes'], path
    assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256'], path
print(f"Verified {len(manifest['checkpoints'])} PPO checkpoints")
PY
```

Only load model files from a trusted source: Stable-Baselines3 checkpoints include serialized Python objects. The numeric visualization cache can be read with `numpy.load(..., allow_pickle=False)`.

## Verified runtime

Local evaluation and smoke tests used Python **3.14.4** on macOS with CPU inference. [`requirements.txt`](../requirements.txt) pins the seven direct dependencies: NumPy 2.5.3, Gymnasium 1.3.0, Stable-Baselines3 2.9.0, Torch 2.14.0, Matplotlib 3.11.2, Pygame CE 2.5.8, and Pillow 12.3.0. Pygame CE supplies the `pygame` import.

Use Python 3.14 for the closest match. Exact numerical agreement on other operating systems or hardware is not guaranteed. These are direct dependency pins, not a lockfile of every transitive dependency. The archived model metadata captures the training runtime separately.

## What can be reproduced

- **Inference and published comparisons:** the saved weights, deterministic track IDs, start poses, controller parameters, and raw CSV rows are included. Run the commands in the [README](../README.md#reproduce-the-evaluation).
- **Visuals:** [`assets/manifest.json`](../assets/manifest.json) hashes the raw benchmark inputs and cached deterministic driving episodes. `python generate_assets.py` renders from this frozen cache; `--refresh-rollouts` evaluates the bundled checkpoints again.
- **New PPO training:** both trainers accept seeds, budgets, and separate output paths and record actual timestep counts and runtime metadata. Existing outputs require explicit `--overwrite` to replace.
- **Historical weights:** complete environment revisions and training manifests were not saved for every older experiment. The visual-training checkpoints were produced without a recorded seed (`seed: null`). Fresh training is therefore a new experiment, not a promise to reproduce identical archived weights.

The six visual checkpoint filenames denote requested milestones 0, 25,000, 50,000, 100,000, 250,000 and 500,000. Actual saved steps are 0, 26,624, 53,248, 104,448, 256,000 and 507,904 because PPO finishes full 2,048-step rollouts. The animation displays actual steps.

The evolved champion records seed 2026 and two training circuits per family (trial IDs 200–201), and its log records 20 generations. The original population size was not stored. Evaluation IDs 10000–10019 are separate from these evolutionary training IDs; no comparable guarantee of a completely untouched tuning set exists for all historical experiments.
