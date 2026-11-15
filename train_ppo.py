"""Train a seeded PPO policy without overwriting published checkpoints."""
import argparse
import json
import platform
from importlib.metadata import version
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from src.racing_env import RacingEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preset', choices=('broad-domain', 'hard-five-family', 'curvature'), default='curvature')
    parser.add_argument('--timesteps', type=int, help='Requested steps; PPO completes full rollouts.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=Path, default=Path('experiments/ppo_curvature'))
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    steps = args.timesteps if args.timesteps is not None else (4_500_000 if args.preset == 'broad-domain' else 7_500_000)
    if steps <= 0:
        parser.error('--timesteps must be positive')
    output = args.output if args.output.suffix == '.zip' else Path(str(args.output) + '.zip')
    metadata_path = output.with_suffix('.json')
    monitor_path = output.with_suffix('.monitor.csv')
    if not args.overwrite and any(p.exists() for p in (output, metadata_path, monitor_path)):
        parser.error('Output exists. Choose --output or explicitly use --overwrite.')
    output.parent.mkdir(parents=True, exist_ok=True)
    env = Monitor(RacingEnv(random_start=True, broad_tracks=args.preset == 'broad-domain',
                           hard_broad_tracks=args.preset != 'broad-domain',
                           curvature_observation=args.preset == 'curvature'), filename=str(monitor_path))
    try:
        model = PPO('MlpPolicy', env, seed=args.seed, device=args.device, verbose=1,
                    learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10,
                    gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0)
        model.learn(total_timesteps=steps)
        model.save(output)
        metadata = {
            'preset': args.preset, 'seed': args.seed, 'requested_timesteps': steps,
            'actual_timesteps': model.num_timesteps, 'device': args.device,
            'python': platform.python_version(),
            'packages': {name: version(name) for name in ('numpy', 'gymnasium', 'stable-baselines3', 'torch')},
            'observation_shape': list(env.observation_space.shape),
            'note': 'Seeded recipe; historical checkpoints may use earlier environment revisions.',
        }
        metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
        print(f'Saved {output} after {model.num_timesteps:,} steps')
    finally:
        env.close()


if __name__ == '__main__':
    main()
