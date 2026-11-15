
"""Multi-track evolutionary racing simulator.

30 AI cars learn by evolutionary selection.
Supports 5 track families and optional PPO project track imports.

This is a separate evolutionary algorithm, not PPO training.
"""

import argparse
import importlib
import inspect
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

TAU = 2 * math.pi
WIDTH = 12.0
DT = 0.045
WHEELBASE = 3.2
MAX_STEER = 0.52
EPISODE_SECONDS = 24.0

FAMILIES = (
    "ellipse",
    "wavy",
    "complex",
    "spline",
    "hard_spline",
)

LABELS = {
    "ellipse": "ELLIPSE",
    "wavy": "WAVY / HARMONICS",
    "complex": "COMPLEX HARMONICS",
    "spline": "PROCEDURAL SPLINE",
    "hard_spline": "HARD SPLINE",
    "ood_harmonic": "UNSEEN HARMONIC",
    "ood_spline": "UNSEEN SPLINE",
}

BOUNDS = np.array([
    [-1.2, 4.5],
    [-1.0, 2.5],
    [0.035, 0.50],
    [8.0, 32.0],
    [0.0, 0.8],
    [-0.35, 0.35],
])

MUTATION_SIGMA = np.array([
    0.40, 0.30, 0.046, 2.2, 0.12, 0.08
])


def wrap(angle):
    return (angle + math.pi) % TAU - math.pi


@dataclass
class Track:
    name: str
    points: np.ndarray
    width: float = WIDTH
    source: str = "Built-in demo"
    family: str = "custom"

    def __post_init__(self):
        points = np.asarray(self.points, dtype=float)

        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("Track must be an Nx2 array")

        if len(points) < 20:
            raise ValueError("Track needs at least 20 points")

        if np.linalg.norm(points[-1] - points[0]) < 1e-7:
            points = points[:-1]

        seg = np.roll(points, -1, axis=0) - points
        lengths = np.linalg.norm(seg, axis=1)

        if np.min(lengths) < 1e-6:
            raise ValueError("Track has duplicate consecutive points")

        self.points = points
        self.deltas = seg
        self.segment_lengths = lengths
        self.cumdist = np.r_[0.0, np.cumsum(lengths)]
        self.length = float(self.cumdist[-1])
        self.tangents = seg / lengths[:, None]

        before = np.roll(self.tangents, 1, axis=0)
        tangents = before + self.tangents

        tangents /= np.maximum(
            1e-8, np.linalg.norm(tangents, axis=1)
        )[:, None]

        area = np.sum(
            points[:, 0] * np.roll(points[:, 1], -1)
            - points[:, 1] * np.roll(points[:, 0], -1)
        )

        outside = 1.0 if area >= 0 else -1.0

        self.normals = outside * np.stack(
            [tangents[:, 1], -tangents[:, 0]],
            axis=1,
        )

        angles = np.unwrap(
            np.arctan2(
                self.tangents[:, 1],
                self.tangents[:, 0],
            )
        )

        da = (
            np.roll(angles, -1)
            - np.roll(angles, 1)
            + math.pi
        ) % TAU - math.pi

        ds = np.roll(lengths, 1) + lengths

        self.curvatures = np.clip(
            np.abs(da / np.maximum(ds, 1e-6)) / 0.16,
            0,
            1,
        )

    def at(self, s):
        s = float(s % self.length)

        i = min(
            len(self.points) - 1,
            int(np.searchsorted(
                self.cumdist, s, side="right"
            ) - 1),
        )

        t = (s - self.cumdist[i]) / self.segment_lengths[i]

        return self.points[i] + t * self.deltas[i]

    def curvature(self, s):
        i = min(
            len(self.points) - 1,
            int(np.searchsorted(
                self.cumdist,
                s % self.length,
                side="right",
            ) - 1),
        )

        return float(self.curvatures[i])

    def project(self, x, y, hint=None):
        n = len(self.points)

        if hint is None:
            ids = np.arange(n)
        else:
            ids = (int(hint) + np.arange(-28, 29)) % n

        p = np.array([x, y])
        v = p - self.points[ids]

        t = np.clip(
            np.einsum("ij,ij->i", v, self.deltas[ids])
            / (self.segment_lengths[ids] ** 2),
            0.0,
            1.0,
        )

        feet = (
            self.points[ids]
            + t[:, None] * self.deltas[ids]
        )

        distances = np.einsum(
            "ij,ij->i", p - feet, p - feet
        )

        j = int(np.argmin(distances))
        idx = int(ids[j])

        dif = p - feet[j]
        tangent = self.tangents[idx]

        lateral = float(
            tangent[0] * dif[1]
            - tangent[1] * dif[0]
        )

        s = float(
            self.cumdist[idx]
            + t[j] * self.segment_lengths[idx]
        )

        return s, lateral, idx

    def start(self):
        p = self.points[0]
        t = self.tangents[0]

        return (
            float(p[0]),
            float(p[1]),
            float(math.atan2(t[1], t[0])),
        )

    def bounds(self):
        return (
            *np.min(self.points, axis=0),
            *np.max(self.points, axis=0),
        )


def catmull_rom_loop(control, count=720):
    control = np.asarray(control, dtype=float)

    m = len(control)
    q = np.linspace(0, m, count, endpoint=False)
    k = np.floor(q).astype(int)
    t = (q - k)[:, None]

    p0, p1, p2, p3 = [
        control[(k + shift) % m]
        for shift in (-1, 0, 1, 2)
    ]

    return 0.5 * (
        2 * p1
        + (-p0 + p2) * t
        + (2*p0 - 5*p1 + 4*p2 - p3) * t**2
        + (-p0 + 3*p1 - 3*p2 + p3) * t**3
    )


def builtin_track(family, seed):
    rng = np.random.default_rng(seed)
    theta = np.linspace(0, TAU, 720, endpoint=False)

    if family == "ellipse":
        pts = np.column_stack([
            40 * np.cos(theta),
            20 * np.sin(theta),
        ])

    elif family in ("wavy", "complex", "ood_harmonic"):
        if family == "wavy":
            coeff = [
                (2, rng.uniform(.05, .085)),
                (3, rng.uniform(.05, .11)),
            ]

        elif family == "complex":
            coeff = [
                (k, rng.uniform(.025, .075))
                for k in range(2, 6)
            ]

        else:
            coeff = [
                (k, rng.uniform(.035, .09))
                for k in (3, 5, 7)
            ]

        shape = np.ones_like(theta)

        for k, amp in coeff:
            phase = rng.uniform(-math.pi, math.pi)
            shape += amp * np.cos(k * theta + phase)

        pts = np.column_stack([
            42 * shape * np.cos(theta),
            26 * shape * np.sin(theta),
        ])

    elif family in ("spline", "hard_spline", "ood_spline"):
        if family == "spline":
            n, jitter = 10, .13
        elif family == "hard_spline":
            n, jitter = 13, .23
        else:
            n, jitter = 12, .19

        angles = TAU * np.arange(n) / n
        r = 1 + rng.uniform(-jitter, jitter, n)

        control = np.column_stack([
            42 * r * np.cos(angles),
            27 * r * np.sin(angles),
        ])

        pts = catmull_rom_loop(control)

    else:
        raise ValueError(f"Unknown family: {family}")

    start = int(np.argmax(pts[:, 0]))
    pts = np.roll(pts, -start, axis=0)

    return Track(
        LABELS.get(family, family.upper()),
        pts,
        WIDTH,
        "Built-in demo",
        family,
    )


def get_centerline(result):
    if isinstance(result, (tuple, list)):
        for obj in result:
            points = get_centerline(obj)
            if points is not None:
                return points
        return None

    if isinstance(result, np.ndarray):
        points = np.asarray(result, dtype=float)

        if points.ndim == 2 and points.shape[1] == 2:
            if len(points) >= 20:
                return points

        return None

    for attr in (
        "centerline_points",
        "centerline",
        "track_points",
        "points",
    ):
        if hasattr(result, attr):
            try:
                points = np.asarray(
                    getattr(result, attr),
                    dtype=float,
                )

                if (
                    points.ndim == 2
                    and points.shape[1] == 2
                    and len(points) >= 20
                ):
                    return points

            except (ValueError, TypeError):
                pass

    return None


class TrackBank:
    def __init__(self, source="auto", seed=12,
                 tracks_dir=None):
        self.source = source
        self.seed = seed
        self.warning = ""
        self.project_factory = None
        self.exported = {}

        if tracks_dir:
            root = Path(tracks_dir)

            if not root.is_dir():
                raise FileNotFoundError(root)

            for file in sorted(root.glob("*.json")):
                raw = json.loads(file.read_text())

                points = raw.get(
                    "centerline_points",
                    raw.get("centerline", raw.get("points")),
                )

                if points is None:
                    continue

                self.exported[file.stem] = Track(
                    str(raw.get("name", file.stem)),
                    np.asarray(points, dtype=float),
                    float(raw.get("width", WIDTH)),
                    f"Imported {file.name}",
                    file.stem,
                )

            for file in sorted(root.glob("*.npz")):
                with np.load(file, allow_pickle=False) as data:
                    if "centerline_points" not in data:
                        continue

                    width = (
                        float(data["width"])
                        if "width" in data
                        else WIDTH
                    )

                    self.exported[file.stem] = Track(
                        file.stem.upper(),
                        data["centerline_points"],
                        width,
                        f"Imported {file.name}",
                        file.stem,
                    )

            if not self.exported:
                raise ValueError("No valid tracks found")

        if source in ("auto", "project") and not self.exported:
            roots = (
                Path.cwd(),
                Path(__file__).resolve().parent,
            )

            for root in roots:
                factory = root / "src" / "track_factory.py"

                if factory.is_file():
                    if str(root) not in sys.path:
                        sys.path.insert(0, str(root))

                    try:
                        self.project_factory = (
                            importlib.import_module(
                                "src.track_factory"
                            )
                        )
                        break
                    except Exception as exc:
                        self.warning = str(exc)

            if source == "project" and self.project_factory is None:
                raise RuntimeError(
                    "Cannot load src/track_factory.py"
                )

            if self.project_factory is None:
                self.warning = (
                    self.warning
                    or "Using built-in demonstration tracks"
                )

        self.families = (
            tuple(self.exported)
            if self.exported
            else FAMILIES
        )

    def from_factory(self, family, seed):
        if self.project_factory is None:
            return None

        candidates = {
            "wavy": [
                "make_random_wavy_track",
                "make_wavy_track",
            ],
            "complex": [
                "make_random_complex_track",
            ],
            "spline": [
                "make_random_spline_track",
            ],
            "hard_spline": [
                "make_random_hard_spline_track",
            ],
        }.get(family, [])

        for name in candidates:
            fn = getattr(self.project_factory, name, None)

            if fn is None:
                continue

            try:
                kwargs = {}

                for pname, param in inspect.signature(
                    fn
                ).parameters.items():
                    low = pname.lower()

                    if low in (
                        "rng",
                        "generator",
                        "random_state",
                    ):
                        kwargs[pname] = (
                            np.random.default_rng(seed)
                        )

                    elif "seed" in low:
                        kwargs[pname] = seed

                    elif low in (
                        "width",
                        "track_width",
                        "road_width",
                    ):
                        kwargs[pname] = WIDTH

                    elif low in (
                        "num_points",
                        "n_points",
                        "n_samples",
                        "num_samples",
                    ):
                        kwargs[pname] = 720

                    elif low == "a":
                        kwargs[pname] = 40.0

                    elif low == "b":
                        kwargs[pname] = 20.0

                    elif param.default is inspect.Parameter.empty:
                        raise TypeError(
                            f"Unknown required parameter {pname}"
                        )

                result = fn(**kwargs)
                points = get_centerline(result)

                if points is None:
                    raise ValueError(
                        "No compatible centerline returned"
                    )

                width = float(
                    getattr(
                        result,
                        "width",
                        getattr(result, "track_width", WIDTH),
                    )
                )

                return Track(
                    LABELS.get(family, family.upper()),
                    points,
                    width,
                    f"PPO factory: {name}",
                    family,
                )

            except Exception as exc:
                self.warning = (
                    f"{name} could not be adapted: {exc}"
                )

        return None

    def make(self, family, seed):
        if family in self.exported:
            return self.exported[family]

        if self.project_factory is not None:
            track = self.from_factory(family, seed)

            if track is not None:
                return track

            if self.source == "project" and family != "ellipse":
                raise RuntimeError(
                    f"Could not load project track: {family}"
                )

        return builtin_track(family, seed)

    def heldout(self):
        return [
            builtin_track("ood_harmonic", 90505),
            builtin_track("ood_spline", 70907),
        ]


@dataclass
class Car:
    genes: np.ndarray
    number: int

    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0
    speed: float = 0.0

    alive: bool = True
    completed: bool = False

    progress: float = 0.0
    driven: float = 0.0
    last_s: float = 0.0
    idx: int = 0
    steps: int = 0

    finish_time: float | None = None
    trail: list = field(default_factory=list)

    def reset(self, track):
        self.x, self.y, self.heading = track.start()

        self.speed = 0.0
        self.progress = 0.0
        self.driven = 0.0

        self.alive = True
        self.completed = False

        self.last_s = 0.0
        self.idx = 0
        self.steps = 0
        self.finish_time = None

        self.trail = [(self.x, self.y)]

    def step(self, track):
        if not self.alive:
            return

        g = self.genes

        s, lateral, idx = track.project(
            self.x, self.y, self.idx
        )
        self.idx = idx

        lookahead = float(
            g[2] * track.length / TAU
        )

        target = track.at(s + lookahead)

        desired = math.atan2(
            target[1] - self.y,
            target[0] - self.x,
        )

        error = wrap(desired - self.heading)

        steer = float(np.clip(
            g[0] * error
            - g[1] * lateral / (track.width / 2)
            + g[5],
            -MAX_STEER,
            MAX_STEER,
        ))

        self.speed = float(
            g[3] * (
                1 - .60 * g[4] * track.curvature(s)
            )
        )

        self.heading = wrap(
            self.heading
            + self.speed / WHEELBASE
            * math.tan(steer) * DT
        )

        self.x += self.speed * math.cos(
            self.heading
        ) * DT

        self.y += self.speed * math.sin(
            self.heading
        ) * DT

        new_s, new_lat, self.idx = track.project(
            self.x, self.y, self.idx
        )

        delta = (
            new_s - self.last_s + track.length / 2
        ) % track.length - track.length / 2

        self.last_s = new_s
        self.driven += delta

        self.progress = max(
            self.progress,
            max(0.0, min(
                1.0, self.driven / track.length
            )),
        )

        self.steps += 1

        if self.steps % 3 == 0:
            self.trail.append((self.x, self.y))

            if len(self.trail) > 220:
                self.trail.pop(0)

        if abs(new_lat) >= track.width / 2:
            self.alive = False

        elif self.driven >= track.length:
            self.alive = False
            self.completed = True
            self.progress = 1.0
            self.finish_time = self.steps * DT

    @property
    def fitness(self):
        if self.completed and self.finish_time is not None:
            return (
                1200
                + 600 / max(1.0, self.finish_time)
            )

        return self.progress * 1000


def evaluate_genome(genes, track):
    car = Car(genes.copy(), -1)
    car.reset(track)

    for _ in range(int(EPISODE_SECONDS / DT)):
        if not car.alive:
            break

        car.step(track)

    return car.progress, car.finish_time


class Evolution:
    def __init__(
        self,
        bank,
        population=30,
        seed=12,
        mode="cycle",
        selected=None,
    ):
        if population < 6:
            raise ValueError("Minimum population is 6")

        self.bank = bank
        self.population = population
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.mode = mode
        self.selected = selected or bank.families[0]

        self.generation = 0
        self.history = []

        self.track_bests = {
            family: 0.0
            for family in bank.families
        }

        self.track_laps = {}

        self.global_champion = None
        self.best_across_training = -float("inf")

        self.last_outcome = ""
        self.reset_on_leader_crash = True
        self.steps = 0
        self.validation = {}

        self.track = bank.make(self.selected, seed)
        self.cars = []

        self.spawn(self.initial_genes())

    def initial_genes(self):
        genomes = []

        for _ in range(self.population):
            genes = self.rng.uniform(
                BOUNDS[:, 0], BOUNDS[:, 1]
            )

            genes[0] = self.rng.uniform(-1.0, .25)
            genes[1] = self.rng.uniform(-1.0, .35)
            genes[3] = self.rng.uniform(15, 30)

            genomes.append(genes)

        return genomes

    def choose_track(self):
        families = self.bank.families

        if self.mode == "cycle":
            family = families[
                (self.generation - 1) % len(families)
            ]

        elif self.mode == "random":
            family = families[
                int(self.rng.integers(len(families)))
            ]

        else:
            family = self.selected

        seed = (
            self.seed * 1009
            + self.generation * 97
            + list(families).index(family) * 421
        )

        self.track = self.bank.make(family, seed)

    def spawn(self, genomes):
        self.generation += 1
        self.steps = 0

        self.choose_track()

        self.cars = [
            Car(genes.copy(), index)
            for index, genes in enumerate(genomes)
        ]

        for car in self.cars:
            car.reset(self.track)

    @property
    def alive_count(self):
        return sum(car.alive for car in self.cars)

    @property
    def leader(self):
        running = [
            car for car in self.cars if car.alive
        ]

        return max(
            running or self.cars,
            key=lambda car: (
                car.progress,
                car.fitness,
            ),
        )

    def update(self):
        self.steps += 1

        for car in self.cars:
            car.step(self.track)

        timed_out = (
            self.steps * DT >= EPISODE_SECONDS
        )

        if self.reset_on_leader_crash and self.steps > 12:
            best = max(
                self.cars,
                key=lambda car: car.progress,
            )

            if not best.alive:
                self.end_generation(
                    "Leader finished or crashed"
                )
                return True

        if self.alive_count == 0 or timed_out:
            reason = (
                "Time limit"
                if timed_out
                else "All cars stopped"
            )

            self.end_generation(reason)
            return True

        return False

    def end_generation(self, reason="Next generation"):
        ranked = sorted(
            self.cars,
            key=lambda car: car.fitness,
            reverse=True,
        )

        best = ranked[0]

        self.history.append(best.progress)

        family = self.track.family

        self.track_bests[family] = max(
            self.track_bests.get(family, 0.0),
            best.progress,
        )

        if best.finish_time is not None:
            self.track_laps[family] = min(
                self.track_laps.get(
                    family, float("inf")
                ),
                best.finish_time,
            )

        self.last_outcome = (
            f"Generation {self.generation} | "
            f"{family} | {reason} | "
            f"Best {best.progress * 100:.1f}%"
        )

        elite_count = max(
            3, self.population // 6
        )

        elites = [
            car.genes.copy()
            for car in ranked[:elite_count]
        ]

        if best.fitness > self.best_across_training:
            self.best_across_training = best.fitness
            self.global_champion = best.genes.copy()

        children = [elites[0].copy()]

        if self.global_champion is not None:
            children.append(
                self.global_champion.copy()
            )

        while len(children) < self.population:
            if self.rng.random() < .08:
                child = self.rng.uniform(
                    BOUNDS[:, 0],
                    BOUNDS[:, 1],
                )

            else:
                parent = elites[
                    int(self.rng.integers(len(elites)))
                ]

                if self.rng.random() < .25:
                    parent2 = elites[
                        int(self.rng.integers(
                            len(elites)
                        ))
                    ]

                    mask = self.rng.integers(
                        0, 2, size=len(parent)
                    ).astype(bool)

                    parent = np.where(
                        mask, parent, parent2
                    )

                sigma = (
                    .35
                    if self.rng.random() < .55
                    else 1.0
                )

                child = np.clip(
                    parent + self.rng.normal(
                        0, MUTATION_SIGMA * sigma
                    ),
                    BOUNDS[:, 0],
                    BOUNDS[:, 1],
                )

            children.append(child)

        self.spawn(children)

    def set_track(self, family):
        if family not in self.bank.families:
            raise ValueError(family)

        self.mode = "fixed"
        self.selected = family

        self.spawn([
            car.genes for car in self.cars
        ])

    def cycle_mode(self):
        options = {
            "cycle": "random",
            "random": "fixed",
            "fixed": "cycle",
        }

        self.mode = options[self.mode]

        self.spawn([
            car.genes for car in self.cars
        ])

    def validate(self):
        champion = self.global_champion

        if champion is None:
            champion = max(
                self.cars,
                key=lambda car: car.fitness,
            ).genes

        self.validation = {
            track.family: evaluate_genome(
                champion, track
            )
            for track in self.bank.heldout()
        }

        return self.validation

    def save_champion(self):
        champion = self.global_champion

        if champion is None:
            champion = max(
                self.cars,
                key=lambda car: car.fitness,
            ).genes

        data = {
            "algorithm": "Evolution, not PPO",
            "generation": self.generation,
            "training_mode": self.mode,
            "genes": champion.tolist(),
            "best_progress_by_track": self.track_bests,
            "best_lap_seconds": self.track_laps,
        }

        path = "racing_multitrack_champion.json"

        Path(path).write_text(
            json.dumps(data, indent=2)
        )

        return path


def headless(bank, population, seed,
             generations, mode, selected, validate):
    evolution = Evolution(
        bank, population, seed, mode, selected
    )

    for _ in range(generations):
        while not evolution.update():
            pass

        print(evolution.last_outcome)

    print("\nBest results:")

    for family, progress in evolution.track_bests.items():
        print(
            f"{family}: {progress * 100:.1f}%"
        )

    if validate:
        print("\nUnseen track evaluation:")

        for family, (progress, lap) in (
            evolution.validate().items()
        ):
            print(
                f"{family}: {progress * 100:.1f}%"
                f" | Lap time: {lap}"
            )


def visual(bank, population, seed, mode, selected):
    try:
        import pygame
    except ImportError:
        raise SystemExit(
            "Install dependencies: "
            "python -m pip install pygame-ce numpy"
        )

    pygame.init()

    screen = pygame.display.set_mode(
        (1280, 760),
        pygame.RESIZABLE,
    )

    pygame.display.set_caption(
        "Racing Line | Multi-Track AI Evolution"
    )

    clock = pygame.time.Clock()

    fonts = {
        "title": pygame.font.SysFont(
            "arial", 25, bold=True
        ),
        "sub": pygame.font.SysFont(
            "arial", 17, bold=True
        ),
        "regular": pygame.font.SysFont(
            "arial", 15
        ),
        "small": pygame.font.SysFont(
            "arial", 12
        ),
        "big": pygame.font.SysFont(
            "arial", 45, bold=True
        ),
    }

    evo = Evolution(
        bank, population, seed, mode, selected
    )

    BG = (10, 16, 27)
    ROAD = (44, 57, 76)
    WALL = (234, 93, 86)
    CENTER = (125, 153, 174)
    WHITE = (235, 242, 253)
    MUTE = (154, 174, 193)
    GREEN = (72, 239, 173)
    GOLD = (255, 198, 87)

    palette = [
        (101, 178, 255),
        (191, 126, 250),
        (249, 181, 91),
        (247, 122, 171),
        (107, 222, 220),
        (195, 233, 108),
    ]

    paused = False
    fast = False
    banner = 0

    def txt(message, x, y,
            face="regular", color=WHITE):
        image = fonts[face].render(
            str(message), True, color
        )

        screen.blit(
            image, (int(x), int(y))
        )

    def project(points, origin, scale):
        return [
            (
                int(origin[0] + x * scale),
                int(origin[1] - y * scale),
            )
            for x, y in points
        ]

    def draw_track(track, origin, scale):
        outer = project(
            track.points
            + track.normals * (track.width / 2),
            origin,
            scale,
        )

        inner = project(
            track.points
            - track.normals * (track.width / 2),
            origin,
            scale,
        )

        pygame.draw.polygon(
            screen, ROAD, outer
        )

        pygame.draw.polygon(
            screen, BG, inner
        )

        pygame.draw.lines(
            screen, WALL, True, outer, 3
        )

        pygame.draw.lines(
            screen, WALL, True, inner, 3
        )

        center = project(
            track.points, origin, scale
        )

        jump = max(
            4, len(center) // 90
        )

        for i in range(0, len(center), jump * 2):
            pygame.draw.line(
                screen,
                CENTER,
                center[i],
                center[(i + jump) % len(center)],
                1,
            )

        p = track.points[0]
        n = track.normals[0]

        start = project(
            [
                p - n * track.width / 2,
                p + n * track.width / 2,
            ],
            origin,
            scale,
        )

        pygame.draw.line(
            screen,
            WHITE,
            start[0],
            start[1],
            4,
        )

        txt(
            "START / FINISH",
            start[0][0] - 34,
            start[0][1] - 26,
            "small",
        )

    def draw_car(
        car, origin, scale, highlight, offset
    ):
        if highlight and len(car.trail) > 2:
            pygame.draw.lines(
                screen,
                (65, 214, 158),
                False,
                project(car.trail, origin, scale),
                3,
            )

        dx = math.cos(car.heading)
        dy = math.sin(car.heading)

        nx, ny = -dy, dx

        x = car.x + nx * offset
        y = car.y + ny * offset

        vertices = [
            (x + dx * 1.7, y + dy * 1.7),
            (
                x - dx * 1.1 + nx * .85,
                y - dy * 1.1 + ny * .85,
            ),
            (
                x - dx * 1.1 - nx * .85,
                y - dy * 1.1 - ny * .85,
            ),
        ]

        points = project(
            vertices, origin, scale
        )

        if highlight:
            color = GREEN
        elif not car.alive:
            color = (111, 62, 75)
        else:
            color = palette[
                car.number % len(palette)
            ]

        pygame.draw.polygon(
            screen, color, points
        )

        if highlight:
            center = project(
                [(x, y)], origin, scale
            )[0]

            pygame.draw.circle(
                screen, GREEN, center, 12, 2
            )

    def chart(x, y, w, h):
        pygame.draw.rect(
            screen,
            (18, 28, 43),
            (x, y, w, h),
            border_radius=8,
        )

        txt(
            "BEST PROGRESS / GENERATION",
            x + 12,
            y + 10,
            "small",
        )

        pygame.draw.line(
            screen,
            (58, 76, 97),
            (x + 12, y + h - 24),
            (x + w - 12, y + h - 24),
            1,
        )

        values = evo.history[-50:]

        if len(values) >= 2:
            points = [
                (
                    int(
                        x + 18
                        + i * (w - 36)
                        / max(1, len(values) - 1)
                    ),
                    int(
                        y + h - 25
                        - value * (h - 62)
                    ),
                )
                for i, value in enumerate(values)
            ]

            pygame.draw.lines(
                screen, GREEN, False, points, 2
            )

        txt(
            "100%", x + w - 45,
            y + 30, "small", MUTE
        )

        txt(
            "0%", x + 18,
            y + h - 20, "small", MUTE
        )

    running = True

    while running:
        clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_SPACE:
                    paused = not paused

                elif event.key == pygame.K_f:
                    fast = not fast

                elif event.key == pygame.K_l:
                    evo.reset_on_leader_crash = (
                        not evo.reset_on_leader_crash
                    )

                elif event.key == pygame.K_n:
                    evo.end_generation("Skipped")
                    banner = 100

                elif event.key == pygame.K_m:
                    evo.cycle_mode()
                    banner = 100

                elif event.key == pygame.K_t:
                    index = bank.families.index(
                        evo.track.family
                    )

                    next_track = bank.families[
                        (index + 1) % len(bank.families)
                    ]

                    evo.set_track(next_track)
                    banner = 100

                elif pygame.K_1 <= event.key <= pygame.K_5:
                    index = event.key - pygame.K_1

                    if index < len(bank.families):
                        evo.set_track(
                            bank.families[index]
                        )
                        banner = 100

                elif event.key == pygame.K_v:
                    evo.validate()
                    banner = 120

                elif event.key == pygame.K_b:
                    print(
                        "Saved:",
                        evo.save_champion(),
                    )
                    banner = 100

                elif event.key == pygame.K_r:
                    evo = Evolution(
                        bank,
                        population,
                        seed,
                        mode,
                        selected,
                    )

                    banner = 0

        if not paused:
            steps = 8 if fast else 2

            for _ in range(steps):
                if evo.update():
                    banner = 110
                    break

        if banner:
            banner -= 1

        W, H = screen.get_size()

        right_width = 320 if W >= 1050 else 245
        left_width = W - right_width - 30

        track = evo.track

        min_x, min_y, max_x, max_y = (
            track.bounds()
        )

        margin = track.width / 2 + 7

        scale = max(
            1.0,
            min(
                (left_width - 60)
                / (max_x - min_x + 2 * margin),
                (H - 164)
                / (max_y - min_y + 2 * margin),
            ),
        )

        origin = (
            left_width / 2
            - (min_x + max_x) / 2 * scale
            + 7,

            H / 2 + 20
            + (min_y + max_y) / 2 * scale,
        )

        side = W - right_width + 15

        screen.fill(BG)

        txt(
            "RACING LINE / MULTI-TRACK EVOLUTION",
            24, 20, "title",
        )

        txt(
            "30-car evolutionary learning | "
            "Separate from PPO",
            25, 54, "small", MUTE,
        )

        draw_track(
            track, origin, scale
        )

        leader = evo.leader

        for car in evo.cars:
            if car is not leader:
                offset = (
                    (car.number % 7) - 3
                ) * .12

                draw_car(
                    car,
                    origin,
                    scale,
                    False,
                    offset,
                )

        draw_car(
            leader, origin, scale, True, 0
        )

        txt(
            "ACTIVE TRACK",
            side, 20, "small", MUTE,
        )

        txt(
            track.name,
            side, 37, "sub",
        )

        source_color = (
            GOLD
            if "demo" in track.source.lower()
            else GREEN
        )

        txt(
            track.source,
            side, 59, "small", source_color,
        )

        txt(
            f"MODE: {evo.mode.upper()}",
            side, 80, "small", MUTE,
        )

        txt(
            "GENERATION",
            side, 112, "small", MUTE,
        )

        txt(
            str(evo.generation),
            side, 126, "big", GREEN,
        )

        txt(
            "CARS ALIVE",
            side, 190, "small", MUTE,
        )

        txt(
            f"{evo.alive_count} / {evo.population}",
            side, 207, "title",
        )

        txt(
            "LEADER PROGRESS",
            side, 244, "small", MUTE,
        )

        txt(
            f"{leader.progress * 100:.1f}%",
            side, 263, "sub", GREEN,
        )

        txt(
            "BEST ON THIS TRACK",
            side, 297, "small", MUTE,
        )

        best_progress = evo.track_bests.get(
            track.family, 0.0
        )

        txt(
            f"{best_progress * 100:.1f}%",
            side, 315, "sub", GOLD,
        )

        chart(
            side - 8,
            352,
            right_width - 26,
            140,
        )

        if evo.validation:
            txt(
                "UNSEEN TRACK TEST",
                side, 505, "small", GOLD,
            )

            for index, (name, result) in enumerate(
                evo.validation.items()
            ):
                progress, lap = result

                detail = (
                    f"{name}: {progress * 100:.0f}%"
                )

                if lap is not None:
                    detail += f" | {lap:.1f}s"

                txt(
                    detail,
                    side,
                    522 + index * 17,
                    "small",
                )

        controls_y = H - 141

        txt(
            "1-5 map  T next  M mode  V test",
            side, controls_y, "small",
        )

        txt(
            "SPACE pause  F fast  N next",
            side, controls_y + 19, "small",
        )

        txt(
            "L reset rule  B save  R restart",
            side, controls_y + 38, "small",
        )

        reset_mode = (
            "RESET: LEADER"
            if evo.reset_on_leader_crash
            else "RESET: LAST CAR"
        )

        txt(
            reset_mode,
            side, controls_y + 58,
            "small", MUTE,
        )

        status = (
            "PAUSED"
            if paused
            else "FAST MODE"
            if fast
            else "TRAINING"
        )

        txt(
            status,
            side, controls_y + 79,
            "sub",
            GOLD if paused else GREEN,
        )

        txt(
            "Green: leader | Red: crashed | "
            "Green line: trajectory",
            24, H - 27, "small", MUTE,
        )

        if banner:
            txt(
                evo.last_outcome
                or "Updated track / mode",
                24, H - 51, "small", GOLD,
            )

        pygame.display.flip()

    pygame.quit()


def main():
    parser = argparse.ArgumentParser(
        description="Multi-track AI racing evolution"
    )

    parser.add_argument(
        "--population", type=int, default=30
    )

    parser.add_argument(
        "--seed", type=int, default=12
    )

    parser.add_argument(
        "--mode",
        choices=["cycle", "random", "fixed"],
        default="cycle",
    )

    parser.add_argument(
        "--track", default="ellipse"
    )

    parser.add_argument(
        "--source",
        choices=["auto", "project", "builtin"],
        default="auto",
    )

    parser.add_argument(
        "--tracks-dir", default=None
    )

    parser.add_argument(
        "--headless", action="store_true"
    )

    parser.add_argument(
        "--generations", type=int, default=20
    )

    parser.add_argument(
        "--validate", action="store_true"
    )

    args = parser.parse_args()

    bank = TrackBank(
        args.source,
        args.seed,
        args.tracks_dir,
    )

    if args.track not in bank.families:
        if args.track == "ellipse" and bank.exported:
            args.track = bank.families[0]
        else:
            parser.error(
                f"Unknown track: {args.track}"
            )

    if bank.warning:
        print("Track source:", bank.warning)

    if args.headless:
        headless(
            bank,
            args.population,
            args.seed,
            args.generations,
            args.mode,
            args.track,
            args.validate,
        )

    else:
        visual(
            bank,
            args.population,
            args.seed,
            args.mode,
            args.track,
        )


if __name__ == "__main__":
    main()
