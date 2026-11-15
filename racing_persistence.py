
"""Save and restore training for each racing map."""

import json
import math
import re
from pathlib import Path

import numpy as np
from population_evolution_multitrack import Car, Track

CHECKPOINTS = (
    Path(__file__).resolve().parent / "racing_checkpoints"
)


def checkpoint_path(model):
    family = model.stable_track.family
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", family)
    return CHECKPOINTS / f"{safe}.json"


def save_map(model):
    """Save the evolved population and learning history."""

    track = model.stable_track
    champion = model.global_champion

    data = {
        "version": 1,
        "algorithm": "Evolutionary controller (not PPO)",
        "family": track.family,
        "seed": int(model.seed),
        "population": int(model.population),
        "generation": int(model.generation),

        "genomes": [
            car.genes.tolist()
            for car in model.cars
        ],

        "track": {
            "name": track.name,
            "source": track.source,
            "family": track.family,
            "width": float(track.width),
            "points": track.points.tolist(),
        },

        "history": [
            float(value)
            for value in model.history
        ],

        "history_families": list(
            getattr(model, "history_families", [])
        ),

        "track_bests": {
            k: float(v)
            for k, v in model.track_bests.items()
        },

        "track_laps": {
            k: float(v)
            for k, v in model.track_laps.items()
        },

        "alltime_best": float(getattr(model, "alltime_best", 0.0)),

        "global_champion": (
            champion.tolist()
            if champion is not None
            else None
        ),

        "best_across_training": (
            float(model.best_across_training)
            if math.isfinite(model.best_across_training)
            else None
        ),

        "rng_state": model.rng.bit_generator.state,

        "reset_on_leader_crash": bool(
            model.reset_on_leader_crash
        ),
    }

    path = checkpoint_path(model)
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(data, separators=(",", ":")),
        encoding="utf-8",
    )

    temporary.replace(path)
    return path


def load_map(model):
    """Restore a previously trained population."""

    path = checkpoint_path(model)

    if not path.exists():
        return False

    try:
        data = json.loads(
            path.read_text(encoding="utf-8")
        )

        family = model.stable_track.family

        if (
            data.get("version") != 1
            or data.get("family") != family
        ):
            raise ValueError("Incompatible checkpoint")

        if data.get("seed") != model.seed:
            raise ValueError("Training seed has changed")

        if data.get("population") != model.population:
            raise ValueError(
                "Population changed. Use the previous --cars value."
            )

        genes = np.asarray(
            data["genomes"], dtype=float
        )

        if (
            genes.shape != (model.population, 6)
            or not np.isfinite(genes).all()
        ):
            raise ValueError("Invalid saved controllers")

        saved_track = data["track"]

        track = Track(
            name=saved_track["name"],
            points=np.asarray(
                saved_track["points"], dtype=float
            ),
            width=float(saved_track["width"]),
            source=saved_track["source"],
            family=family,
        )

        champion = data.get("global_champion")

        champion_genes = (
            None
            if champion is None
            else np.asarray(champion, dtype=float)
        )

        if (
            champion_genes is not None
            and champion_genes.shape != (6,)
        ):
            raise ValueError("Invalid saved champion")

        model.stable_track = track
        model.track = track

        model.generation = max(
            1, int(data["generation"])
        )

        model.steps = 0

        model.cars = [
            Car(g.copy(), i)
            for i, g in enumerate(genes)
        ]

        for car in model.cars:
            car.reset(track)

        model.history = [
            float(v) for v in data.get("history", [])
        ]

        model.history_families = list(
            data.get("history_families", [])
        )

        model.track_bests.update({
            k: float(v)
            for k, v in data.get(
                "track_bests", {}
            ).items()
        })

        model.track_laps.update({
            k: float(v)
            for k, v in data.get(
                "track_laps", {}
            ).items()
        })

        model.alltime_best = float(
            data.get("alltime_best", 0)
        )

        model.global_champion = champion_genes

        old_best = data.get(
            "best_across_training"
        )

        model.best_across_training = (
            float(old_best)
            if old_best is not None
            else -float("inf")
        )

        model.rng.bit_generator.state = (
            data["rng_state"]
        )

        model.reset_on_leader_crash = bool(
            data.get("reset_on_leader_crash", False)
        )

        model.pause_frames = 0
        model.last_cars = None
        model.last_gen = 0

        model.last_outcome = (
            f"Resumed generation {model.generation}"
        )

        return True

    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(
            f"Cannot restore {model.stable_track.family}: {exc}"
        )
        return False


def save_all(states):
    """Save all maps separately."""

    return [
        save_map(model)
        for model in states.values()
    ]
