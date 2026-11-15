
"""Multi-car racing showcase with manual map selection.

Requires population_evolution_multitrack.py in the same folder.

N = next map, P = previous map.
Every map trains separately and keeps the same track layout.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from population_evolution_multitrack import (
    Car,
    Evolution,
    TrackBank,
)

from racing_persistence import save_map, load_map, save_all


class MapTrainer(Evolution):
    """One independent population per fixed racing circuit."""

    def __init__(self, bank, family, cars, seed):
        self.stable_track = bank.make(family, seed + 101)
        self.pause_frames = 0
        self.last_cars = None
        self.last_gen = 0

        super().__init__(
            bank,
            population=cars,
            seed=seed,
            mode="fixed",
            selected=family,
        )

        self.reset_on_leader_crash = False

    def choose_track(self):
        # Keep exactly the same circuit between generations.
        self.track = self.stable_track

    def end_generation(self, reason="Next generation"):
        self.last_cars = self.cars
        self.last_gen = self.generation

        super().end_generation(reason)

        self.pause_frames = 20

        # Automatically save after every generation
        save_map(self)

    def tick(self, quick=False):
        if self.pause_frames > 0:
            self.pause_frames -= 6 if quick else 1
            return

        for _ in range(5 if quick else 1):
            if self.update():
                break


def write_results(states, path="racing_final_results.json"):
    result = {
        "algorithm": "Evolutionary controllers, not PPO",
        "maps": {},
    }

    for name, model in states.items():
        champ = model.global_champion

        if champ is None:
            champ = max(
                model.cars,
                key=lambda car: car.fitness,
            ).genes

        result["maps"][name] = {
            "generation": model.generation,
            "best_progress_percent": round(
                model.track_bests[name] * 100, 3
            ),
            "fastest_lap_seconds": model.track_laps.get(name),
            "genes": champ.tolist(),
            "source": model.track.source,
            "seed": model.seed,
        }

    Path(path).write_text(json.dumps(result, indent=2))
    return path


def run_check(states, generations):
    for name, model in states.items():
        original_track = model.track.points.copy()

        for _ in range(generations):
            current = model.generation

            while model.generation == current:
                model.update()

        assert np.array_equal(
            model.track.points,
            original_track,
        ), "Map changed unexpectedly"

        print(
            f"{name}: generation {model.generation}, "
            f"best {model.track_bests[name]*100:.1f}%"
        )

    print("All fixed-map checks passed.")


def show(states, names):
    try:
        import pygame
    except ImportError as exc:
        raise SystemExit(
            "Install pygame-ce: "
            "python -m pip install pygame-ce"
        ) from exc

    pygame.init()

    screen = pygame.display.set_mode(
        (1400, 850),
        pygame.RESIZABLE,
    )

    pygame.display.set_caption(
        "AI Racing | Multi-Car Learning & Final Results"
    )

    clock = pygame.time.Clock()

    font = {
        "s": pygame.font.SysFont("arial", 15),
        "m": pygame.font.SysFont("arial", 19, bold=True),
        "l": pygame.font.SysFont("arial", 29, bold=True),
        "xl": pygame.font.SysFont("arial", 43, bold=True),
    }

    INK = (236, 244, 246)
    MUTED = (181, 195, 207)
    ROAD = (53, 72, 88)
    GROUND = (128, 150, 166)
    GREEN = (95, 246, 164)
    RED = (244, 92, 78)
    GOLD = (255, 210, 108)

    index = 0
    camera_mode = "FOLLOW"

    paused = False
    fast = False
    replay = False
    results_page = False

    replay_car = None
    replay_hold = 0

    save_note = ""
    save_note_frames = 0

    cam = None
    zoom = 11.0

    def text(value, x, y, kind="s", color=INK):
        image = font[kind].render(
            str(value), True, color
        )

        screen.blit(
            image, (int(x), int(y))
        )

    def panel(x, y, w, h):
        surface = pygame.Surface(
            (w, h),
            pygame.SRCALPHA,
        )

        surface.fill((23, 40, 56, 225))

        pygame.draw.rect(
            surface,
            (192, 211, 224, 60),
            surface.get_rect(),
            1,
            border_radius=12,
        )

        screen.blit(surface, (x, y))

    def pixels(points, origin, scale):
        return [
            (
                int(origin[0] + x * scale),
                int(origin[1] - y * scale),
            )
            for x, y in points
        ]

    def draw_map(track, origin, scale):
        outer = pixels(
            track.points
            + track.normals * track.width / 2,
            origin,
            scale,
        )

        inner = pixels(
            track.points
            - track.normals * track.width / 2,
            origin,
            scale,
        )

        pygame.draw.polygon(
            screen, ROAD, outer
        )

        pygame.draw.polygon(
            screen, GROUND, inner
        )

        pygame.draw.lines(
            screen,
            (242, 249, 251),
            True,
            outer,
            3,
        )

        pygame.draw.lines(
            screen,
            (242, 249, 251),
            True,
            inner,
            3,
        )

        center = pixels(
            track.points, origin, scale
        )

        for i in range(0, len(center), 18):
            pygame.draw.line(
                screen,
                (121, 148, 170),
                center[i],
                center[(i + 7) % len(center)],
                2,
            )

        start = pixels(
            [
                track.points[0]
                + track.normals[0] * track.width / 2,

                track.points[0]
                - track.normals[0] * track.width / 2,
            ],
            origin,
            scale,
        )

        pygame.draw.line(
            screen,
            GOLD,
            start[0],
            start[1],
            5,
        )

    
    def car_shape(car, origin, scale, color, halo=False):
        """Draw a top-down Formula 1-style car."""

        dx = math.cos(car.heading)
        dy = math.sin(car.heading)

        nx, ny = -dy, dx

        # Spread cars visually near the starting line.
        amount = max(0.0, 1.0 - car.steps / 32.0)

        lateral = ((car.number % 12) - 5.5) * 0.55 * amount
        behind = (car.number // 12) * 0.9 * amount

        x = car.x + nx * lateral - dx * behind
        y = car.y + ny * lateral - dy * behind

        def point(forward, sideways):
            px = x + dx * forward + nx * sideways
            py = y + dy * forward + ny * sideways

            return (
                int(origin[0] + px * scale),
                int(origin[1] - py * scale),
            )

        def polygon(coords, fill):
            pygame.draw.polygon(
                screen,
                fill,
                [point(f, s) for f, s in coords],
            )

        BLACK = (22, 27, 34)
        GLASS = (10, 23, 34)
        METAL = (140, 150, 159)
        HIGHLIGHT = (238, 240, 244)

        # Rear wing
        polygon([
            (-1.85, -1.20),
            (-1.52, -1.20),
            (-1.52, 1.20),
            (-1.85, 1.20),
        ], BLACK)

        # Four black tires
        for side in (-1, 1):
            polygon([
                (-1.38, side * 0.66),
                (-0.62, side * 0.66),
                (-0.62, side * 1.15),
                (-1.38, side * 1.15),
            ], BLACK)

            polygon([
                (0.58, side * 0.69),
                (1.36, side * 0.69),
                (1.36, side * 1.15),
                (0.58, side * 1.15),
            ], BLACK)

        # Front wing
        polygon([
            (1.43, -1.33),
            (1.76, -1.33),
            (1.76, 1.33),
            (1.43, 1.33),
        ], BLACK)

        # Main chassis
        polygon([
            (-1.60, -0.40),
            (-0.90, -0.62),
            (-0.10, -0.57),
            (0.45, -0.33),
            (1.65, -0.14),
            (1.78, 0),
            (1.65, 0.14),
            (0.45, 0.33),
            (-0.10, 0.57),
            (-0.90, 0.62),
            (-1.60, 0.40),
        ], color)

        # Engine cover
        polygon([
            (-1.42, -0.30),
            (-0.45, -0.37),
            (-0.45, 0.37),
            (-1.42, 0.30),
        ], color)

        # Cockpit
        polygon([
            (-0.46, -0.32),
            (0.24, -0.31),
            (0.43, 0),
            (0.24, 0.31),
            (-0.46, 0.32),
        ], GLASS)

        # Driver helmet
        pygame.draw.circle(
            screen,
            HIGHLIGHT,
            point(-0.06, 0),
            max(2, int(scale * 0.19)),
        )

        # Nose stripe
        pygame.draw.line(
            screen,
            METAL,
            point(0.52, 0),
            point(1.60, 0),
            max(1, int(scale * 0.09)),
        )

        # Highlight the best car
        if halo:
            pygame.draw.circle(
                screen,
                GREEN,
                point(0, 0),
                max(12, int(scale * 2.1)),
                2,
            )



    def graph(history, x, y, w, h):
        panel(x, y, w, h)

        text(
            "BEST LAP PROGRESS / GENERATIONS",
            x + 12,
            y + 10,
            "s",
            MUTED,
        )

        pygame.draw.line(
            screen,
            (107, 133, 151),
            (x + 12, y + h - 17),
            (x + w - 12, y + h - 17),
            1,
        )

        values = history[-60:]

        if len(values) > 1:
            points = [
                (
                    x + 13 + int(
                        i * (w - 28)
                        / (len(values) - 1)
                    ),
                    y + h - 19 - int(
                        value * (h - 52)
                    ),
                )
                for i, value in enumerate(values)
            ]

            pygame.draw.lines(
                screen,
                GREEN,
                False,
                points,
                2,
            )

        text(
            "100%",
            x + w - 48,
            y + 37,
            "s",
            MUTED,
        )

    def reset_replay(model):
        champion = model.global_champion

        if champion is None:
            champion = max(
                model.cars,
                key=lambda car: car.fitness,
            ).genes

        car = Car(champion.copy(), 0)
        car.reset(model.track)

        return car

    running = True

    while running:
        clock.tick(60)

        W, H = screen.get_size()
        model = states[names[index]]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                key = event.key

                if key == pygame.K_ESCAPE:
                    running = False

                elif key == pygame.K_n:
                    # NEXT MAP, not next generation.
                    index = (
                        index + 1
                    ) % len(names)

                    replay_car = None
                    replay_hold = 0
                    cam = None

                elif key == pygame.K_p:
                    # PREVIOUS MAP.
                    index = (
                        index - 1
                    ) % len(names)

                    replay_car = None
                    replay_hold = 0
                    cam = None

                elif pygame.K_1 <= key <= pygame.K_5:
                    destination = (
                        key - pygame.K_1
                    )

                    if destination < len(names):
                        index = destination
                        replay_car = None
                        replay_hold = 0
                        cam = None

                elif key == pygame.K_SPACE:
                    paused = not paused

                elif key == pygame.K_f:
                    fast = not fast

                elif key == pygame.K_c:
                    camera_mode = (
                        "FIT"
                        if camera_mode == "FOLLOW"
                        else "FOLLOW"
                    )

                    cam = None

                elif key == pygame.K_LEFTBRACKET:
                    zoom = max(4, zoom - 1)

                elif key == pygame.K_RIGHTBRACKET:
                    zoom = min(24, zoom + 1)

                elif key == pygame.K_l:
                    model.reset_on_leader_crash = (
                        not model.reset_on_leader_crash
                    )

                elif key == pygame.K_TAB:
                    replay = not replay
                    replay_car = None
                    replay_hold = 0

                elif key == pygame.K_r:
                    results_page = not results_page

                elif key == pygame.K_s:
                    save_all(states)
                    save_note = (
                        "Saved checkpoints + "
                        + write_results(states)
                    )
                    save_note_frames = 120

        model = states[names[index]]

        # Train ONLY the currently selected map.
        if not paused and not results_page:
            if replay:
                if replay_car is None:
                    replay_car = reset_replay(model)

                if replay_car.alive:
                    for _ in range(5 if fast else 1):
                        if replay_car.alive:
                            replay_car.step(
                                model.track
                            )
                else:
                    replay_hold += 1

                    if replay_hold > 45:
                        replay_car = reset_replay(model)
                        replay_hold = 0

            else:
                model.tick(quick=fast)

        # Keep crashes visible briefly at generation end.
        visible = (
            model.last_cars
            if (
                model.pause_frames > 0
                and model.last_cars is not None
                and not replay
            )
            else model.cars
        )

        leader = (
            replay_car
            if replay and replay_car is not None
            else max(
                visible,
                key=lambda car: car.progress,
            )
        )

        track = model.track

        # CAMERA
        if camera_mode == "FIT":
            xmin, ymin, xmax, ymax = (
                track.bounds()
            )

            scale = min(
                (W - 130)
                / (xmax - xmin + track.width + 12),

                (H - 160)
                / (ymax - ymin + track.width + 12),
            )

            origin = (
                W / 2 - (xmin + xmax) / 2 * scale,
                H / 2 + (ymin + ymax) / 2 * scale,
            )

        else:
            target = np.array(
                [leader.x, leader.y],
                dtype=float,
            )

            if cam is None:
                cam = target
            else:
                cam = (
                    0.89 * cam
                    + 0.11 * target
                )

            scale = zoom

            origin = (
                W * 0.51 - cam[0] * scale,
                H * 0.54 + cam[1] * scale,
            )

        # BACKGROUND
        screen.fill(GROUND)

        for xx in range(0, W, 78):
            pygame.draw.line(
                screen,
                (132, 153, 169),
                (xx, 0),
                (xx, H),
                1,
            )

        for yy in range(0, H, 78):
            pygame.draw.line(
                screen,
                (132, 153, 169),
                (0, yy),
                (W, yy),
                1,
            )

        draw_map(track, origin, scale)

        # CARS
        if replay:
            if replay_car is not None:
                if len(replay_car.trail) > 1:
                    pygame.draw.lines(
                        screen,
                        GREEN,
                        False,
                        pixels(
                            replay_car.trail,
                            origin,
                            scale,
                        ),
                        3,
                    )

                car_shape(
                    replay_car,
                    origin,
                    scale,
                    GREEN,
                    True,
                )

        else:
            for car in visible:
                if car is leader:
                    continue

                color = (
                    RED
                    if not car.alive
                    else (77, 204, 230)
                )

                car_shape(
                    car,
                    origin,
                    scale,
                    color,
                )

            if len(leader.trail) > 1:
                pygame.draw.lines(
                    screen,
                    GREEN,
                    False,
                    pixels(
                        leader.trail,
                        origin,
                        scale,
                    ),
                    3,
                )

            car_shape(
                leader,
                origin,
                scale,
                GREEN,
                True,
            )

        # LEFT STATISTICS PANEL
        panel(18, 72, 270, 342)

        text("TRACK", 36, 90, "s", MUTED)

        text(
            f"{index+1:02}  {track.name[:20]}",
            36, 112, "m",
        )

        text(
            "ALGORITHM",
            36, 150, "s", MUTED,
        )

        text(
            "Evolutionary learning (not PPO)",
            36, 173, "s",
        )

        pygame.draw.line(
            screen,
            (85, 112, 133),
            (35, 204),
            (272, 204),
            1,
        )

        shown_gen = (
            model.last_gen
            if model.pause_frames > 0 and not replay
            else model.generation
        )

        alive = sum(
            car.alive for car in visible
        )

        avg_speed = (
            sum(car.speed for car in visible)
            / len(visible)
            if visible else 0
        )

        text(
            "Generation",
            36, 219, "s", MUTED,
        )

        text(
            shown_gen,
            232, 219, "m",
        )

        text(
            "Cars alive",
            36, 248, "s", MUTED,
        )

        text(
            f"{alive}/{model.population}",
            208, 248, "m",
        )

        text(
            "Best progress",
            36, 277, "s", MUTED,
        )

        text(
            f"{model.track_bests[names[index]]*100:.1f}%",
            198, 277, "m",
        )

        text(
            "Best lap",
            36, 306, "s", MUTED,
        )

        lap = model.track_laps.get(
            names[index]
        )

        text(
            f"{lap:.2f}s" if lap is not None else "--",
            203, 306, "m",
        )

        text(
            "Avg speed",
            36, 335, "s", MUTED,
        )

        text(
            f"{avg_speed:.1f}",
            226, 335, "m",
        )

        text(
            "Green: leader     Red: crashed",
            36, 379, "s", GOLD,
        )

        # RIGHT MAP PANEL
        panel(W - 299, 72, 280, 364)

        x = W - 280

        text(
            "MAP SELECTOR",
            x, 91, "m",
        )

        text(
            "N next map   P previous",
            x, 123, "s", GOLD,
        )

        text(
            "Each map has its OWN generation",
            x, 146, "s", MUTED,
        )

        for i, family in enumerate(names):
            current = states[family]
            selected = i == index

            label = (
                ("> " if selected else "  ")
                + f"{i+1}. {family[:15]}"
            )

            text(
                label,
                x,
                181 + i * 29,
                "m" if selected else "s",
                GREEN if selected else INK,
            )

            text(
                f"G{current.generation}",
                W - 82,
                185 + i * 29,
                "s",
                GOLD if selected else MUTED,
            )

        text(
            "TAB best-car replay",
            x, 342, "s",
        )

        text(
            "R results table     S save",
            x, 365, "s",
        )

        text(
            "C camera     [ ] zoom",
            x, 388, "s",
        )

        # LEARNING GRAPH
        graph(
            model.history,
            W - 299,
            448,
            280,
            136,
        )

        # BOTTOM STATUS BAR
        panel(
            18, H - 92,
            W - 36, 72,
        )

        current_mode = (
            "FINAL REPLAY"
            if replay
            else "LIVE GENERATIONS"
        )

        running_label = (
            "PAUSED"
            if paused
            else "FAST"
            if fast
            else "PLAYING"
        )

        reset_label = (
            "RESET ON LEADER"
            if model.reset_on_leader_crash
            else "RESET ON LAST CAR"
        )

        text(
            f"{current_mode}  |  "
            f"{running_label}  |  "
            f"{reset_label}",
            36,
            H - 77,
            "m",
            GOLD,
        )

        text(
            "N/P change MAP (never automatically)"
            "   |   SPACE pause"
            "   |   F fast"
            "   |   L reset rule"
            "   |   TAB replay"
            "   |   R results"
            "   |   S save",
            36,
            H - 44,
            "s",
        )

        if model.pause_frames > 0 and not replay:
            text(
                model.last_outcome,
                38, 420, "s", GOLD,
            )

        if save_note_frames > 0:
            save_note_frames -= 1

            text(
                save_note,
                30,
                H - 120,
                "m",
                GOLD,
            )

        # FINAL RESULTS OVERLAY
        if results_page:
            wx = W // 2 - 360
            wy = H // 2 - 240
            ww = 720
            wh = 470

            panel(wx, wy, ww, wh)

            text(
                "FINAL RESULTS - ALL MAPS",
                wx + 24,
                wy + 20,
                "l",
                GREEN,
            )

            text(
                "Map",
                wx + 27, wy + 75,
                "m", MUTED,
            )

            text(
                "Generation",
                wx + 260, wy + 75,
                "m", MUTED,
            )

            text(
                "Best %",
                wx + 435, wy + 75,
                "m", MUTED,
            )

            text(
                "Lap time",
                wx + 558, wy + 75,
                "m", MUTED,
            )

            for i, name in enumerate(names):
                state = states[name]
                yy = wy + 117 + i * 58

                pygame.draw.line(
                    screen,
                    (71, 94, 112),
                    (wx + 24, yy + 35),
                    (wx + ww - 24, yy + 35),
                    1,
                )

                text(
                    name,
                    wx + 27,
                    yy,
                    "m",
                    GREEN if i == index else INK,
                )

                text(
                    str(state.generation),
                    wx + 279,
                    yy,
                    "m",
                )

                text(
                    f"{state.track_bests[name]*100:.1f}%",
                    wx + 438,
                    yy,
                    "m",
                )

                best_lap = state.track_laps.get(name)

                text(
                    (
                        f"{best_lap:.2f}s"
                        if best_lap is not None
                        else "--"
                    ),
                    wx + 558,
                    yy,
                    "m",
                )

            text(
                "Press R to close. S saves results as JSON.",
                wx + 24,
                wy + wh - 41,
                "s",
                MUTED,
            )

        pygame.display.flip()

    save_all(states)
    write_results(states)
    pygame.quit()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cars",
        type=int,
        default=120,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--source",
        choices=["auto", "project", "builtin"],
        default="auto",
    )

    parser.add_argument(
        "--tracks-dir",
        default=None,
    )

    parser.add_argument(
        "--check",
        type=int,
        default=0,
        metavar="GENERATIONS",
    )

    args = parser.parse_args()

    if args.cars < 6:
        parser.error("At least 6 cars are required")

    bank = TrackBank(
        args.source,
        args.seed,
        args.tracks_dir,
    )

    if bank.warning:
        print(
            "Track source note:",
            bank.warning,
        )

    names = bank.families

    states = {
        family: MapTrainer(
            bank,
            family,
            args.cars,
            args.seed + i * 1000,
        )
        for i, family in enumerate(names)
    }

    for family, trainer in states.items():
        if load_map(trainer):
            print(
                f"Restored {family}: "
                f"Generation {trainer.generation}"
            )

    if args.check:
        run_check(states, args.check)
    else:
        show(states, names)


if __name__ == "__main__":
    main()
