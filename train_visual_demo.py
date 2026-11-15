"""Train PPO on one ellipse and save a seeded checkpoint progression."""
import argparse
import json
import platform
from importlib.metadata import version
from pathlib import Path
from stable_baselines3 import PPO
from src.racing_env import RacingEnv
from src.track_geometry import TrackGeometry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--timesteps', type=int, default=500_000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output-dir', type=Path, default=Path('experiments/visual_training'))
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if args.timesteps <= 0:
        parser.error('--timesteps must be positive')
    milestones = [0] + [n for n in (25_000, 50_000, 100_000, 250_000, 500_000) if n < args.timesteps] + [args.timesteps]
    paths = [args.output_dir / f'ppo_visual_{n:06d}.zip' for n in milestones]
    metadata_path = args.output_dir / 'training_metadata.json'
    if not args.overwrite and any(p.exists() for p in paths + [metadata_path]):
        parser.error('Output exists. Choose --output-dir or explicitly use --overwrite.')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    env = RacingEnv(track=TrackGeometry(a=40.0, b=20.0, track_width=12.0, num_points=720, num_checkpoints=7))
    metadata = {
        'seed': args.seed, 'device': args.device, 'python': platform.python_version(),
        'packages': {name: version(name) for name in ('numpy', 'gymnasium', 'stable-baselines3', 'torch')},
        'track': {'family': 'ellipse', 'a': 40.0, 'b': 20.0, 'width': 12.0},
        'checkpoints': [],
        'note': 'Filenames are nominal milestones; actual steps include complete 2048-step PPO rollouts.',
    }
    try:
        model = PPO('MlpPolicy', env, verbose=1, seed=args.seed, device=args.device,
                    learning_rate=3e-4, n_steps=2048, batch_size=64, gamma=0.99,
                    gae_lambda=0.95, clip_range=0.2, ent_coef=0.01)
        previous_nominal = 0
        for target, path in zip(milestones, paths):
            if target:
                model.learn(total_timesteps=target - previous_nominal, reset_num_timesteps=False)
            model.save(path)
            metadata['checkpoints'].append({'path': path.name, 'nominal_timesteps': target,
                                            'actual_timesteps': model.num_timesteps})
            metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
            print(f'Saved {path}: nominal {target:,}, actual {model.num_timesteps:,} steps')
            previous_nominal = target
    finally:
        env.close()


if __name__ == '__main__':
    main()
