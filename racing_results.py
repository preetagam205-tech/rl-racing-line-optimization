
"""Final-results viewer for the evolutionary racing simulator.

Reads checkpoints saved by src/racing_persistence.py.
This program replays trained controllers; it does not train them.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from src.population_evolution_multitrack import Track, Car, DT


FAMILY_ORDER = [
    "ellipse",
    "wavy",
    "complex",
    "spline",
    "hard_spline",
]

BG = (10, 17, 27)
PANEL = (20, 31, 46)
ROAD = (54, 70, 87)
EDGE = (159, 180, 193)
WHITE = (240, 247, 250)
MUTED = (158, 179, 194)
GREEN = (92, 242, 163)
GOLD = (255, 200, 101)
RED = (247, 113, 99)


def load_results(directory):
    entries = {}

    for path in sorted(Path(directory).glob("*.json")):
        try:
            data = json.loads(
                path.read_text(encoding="utf-8")
            )

            meta = data["track"]
            name = data["family"]

            track = Track(
                str(meta["name"]),
                np.asarray(meta["points"], dtype=float),
                float(meta["width"]),
                str(meta.get("source", "Saved checkpoint")),
                str(meta.get("family", name)),
            )

            genes = data.get("global_champion")

            if genes is None:
                genes = data["genomes"][0]

            genes = np.asarray(genes, dtype=float)

            if (
                genes.shape != (6,)
                or not np.all(np.isfinite(genes))
            ):
                raise ValueError("Invalid controller")

            history = [
                float(v)
                for v in data.get("history", [])
            ]

            progress = float(
                data.get("track_bests", {}).get(name, 0.0)
            )

            lap_time = data.get(
                "track_laps", {}
            ).get(name)

            entries[name] = {
                "family": name,
                "track": track,
                "genes": genes,
                "history": history,
                "completed_generations": len(history),
                "best_progress": max(
                    0.0, min(1.0, progress)
                ),
                "lap": (
                    float(lap_time)
                    if lap_time is not None
                    else None
                ),
                "generations_with_finisher": sum(
                    v >= 0.99999 for v in history
                ),
            }

        except (
            KeyError,
            ValueError,
            TypeError,
            OSError,
        ) as exc:
            print(
                f"Skipping {path.name}: {exc}"
            )

    ordered = [
        name
        for name in FAMILY_ORDER
        if name in entries
    ]

    ordered.extend(
        name
        for name in entries
        if name not in ordered
    )

    return [entries[name] for name in ordered]


def viewer(entries):
    import pygame

    pygame.init()

    screen = pygame.display.set_mode(
        (1280, 770),
        pygame.RESIZABLE,
    )

    pygame.display.set_caption(
        "Racing Line | Final Results"
    )

    clock = pygame.time.Clock()

    fonts = {
        "sm": pygame.font.SysFont("arial", 14),
        "md": pygame.font.SysFont(
            "arial", 17, bold=True
        ),
        "lg": pygame.font.SysFont(
            "arial", 25, bold=True
        ),
        "xl": pygame.font.SysFont(
            "arial", 34, bold=True
        ),
    }

    index = 0
    paused = False
    fast = False

    def reset_car():
        entry = entries[index]

        car = Car(
            entry["genes"].copy(),
            0,
        )

        car.reset(entry["track"])
        car.full_path = [(car.x, car.y)]

        return car

    car = reset_car()

    def text(
        message, x, y,
        style="sm", color=WHITE
    ):
        image = fonts[style].render(
            str(message), True, color
        )

        screen.blit(
            image, (int(x), int(y))
        )

    def panel(rect):
        pygame.draw.rect(
            screen,
            PANEL,
            rect,
            border_radius=12,
        )

        pygame.draw.rect(
            screen,
            (48, 67, 83),
            rect,
            1,
            border_radius=12,
        )

    def project(points, origin, scale):
        return [
            (
                int(origin[0] + x * scale),
                int(origin[1] - y * scale),
            )
            for x, y in points
        ]

    def track_geometry(track, box):
        x, y, w, h = box

        lo = (
            track.points.min(axis=0)
            - track.width / 2 - 5
        )

        hi = (
            track.points.max(axis=0)
            + track.width / 2 + 5
        )

        scale = min(
            (w - 34) / max(hi[0] - lo[0], 1),
            (h - 34) / max(hi[1] - lo[1], 1),
        )

        center = (lo + hi) / 2

        origin = (
            x + w / 2 - center[0] * scale,
            y + h / 2 + center[1] * scale,
        )

        return origin, scale

    def draw_track(track, box):
        origin, scale = track_geometry(
            track, box
        )

        outer = project(
            track.points
            + track.normals * track.width / 2,
            origin,
            scale,
        )

        inner = project(
            track.points
            - track.normals * track.width / 2,
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
            screen, EDGE, True, outer, 2
        )

        pygame.draw.lines(
            screen, EDGE, True, inner, 2
        )

        centerline = project(
            track.points, origin, scale
        )

        for k in range(0, len(centerline), 19):
            pygame.draw.line(
                screen,
                (102, 122, 141),
                centerline[k],
                centerline[
                    (k + 9) % len(centerline)
                ],
                1,
            )

        start = project(
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
            4,
        )

        return origin, scale

    def draw_f1_car(car, origin, scale):
        dx = math.cos(car.heading)
        dy = math.sin(car.heading)

        nx, ny = -dy, dx

        def point(forward, side):
            return (
                int(
                    origin[0]
                    + (
                        car.x
                        + dx * forward
                        + nx * side
                    ) * scale
                ),
                int(
                    origin[1]
                    - (
                        car.y
                        + dy * forward
                        + ny * side
                    ) * scale
                ),
            )

        def polygon(coords, color):
            pygame.draw.polygon(
                screen,
                color,
                [
                    point(f, s)
                    for f, s in coords
                ],
            )

        tire = (12, 17, 23)

        # Four tires
        for side in (-1, 1):
            polygon([
                (-1.3, side * .65),
                (-.63, side * .65),
                (-.63, side * 1.1),
                (-1.3, side * 1.1),
            ], tire)

            polygon([
                (.54, side * .65),
                (1.25, side * .65),
                (1.25, side * 1.1),
                (.54, side * 1.1),
            ], tire)

        # Rear wing
        polygon([
            (-1.78, -1.18),
            (-1.46, -1.18),
            (-1.46, 1.18),
            (-1.78, 1.18),
        ], tire)

        # Front wing
        polygon([
            (1.42, -1.29),
            (1.72, -1.29),
            (1.72, 1.29),
            (1.42, 1.29),
        ], tire)

        # Main body
        polygon([
            (-1.55, -.36),
            (-.9, -.56),
            (-.2, -.5),
            (.5, -.3),
            (1.66, -.12),
            (1.76, 0),
            (1.66, .12),
            (.5, .3),
            (-.2, .5),
            (-.9, .56),
            (-1.55, .36),
        ], GREEN)

        # Cockpit
        polygon([
            (-.48, -.28),
            (.28, -.27),
            (.42, 0),
            (.28, .27),
            (-.48, .28),
        ], (11, 29, 39))

        pygame.draw.circle(
            screen,
            WHITE,
            point(-.08, 0),
            max(2, int(scale * .17)),
        )

    def draw_chart(history, rect):
        x, y, w, h = rect

        panel(rect)

        text(
            "BEST PROGRESS BY COMPLETED GENERATION",
            x + 13,
            y + 9,
            "sm",
            MUTED,
        )

        pygame.draw.line(
            screen,
            (78, 98, 112),
            (x + 15, y + h - 22),
            (x + w - 15, y + h - 22),
            1,
        )

        values = history[-120:]

        if len(values) > 1:
            points = [
                (
                    int(
                        x + 16
                        + i * (w - 32)
                        / (len(values) - 1)
                    ),
                    int(
                        y + h - 23
                        - max(0, min(1, v))
                        * (h - 58)
                    ),
                )
                for i, v in enumerate(values)
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
            x + w - 50,
            y + 33,
            "sm",
            MUTED,
        )

        text(
            "0%",
            x + 16,
            y + h - 19,
            "sm",
            MUTED,
        )

    running = True

    while running:
        clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                key = event.key

                if key == pygame.K_ESCAPE:
                    running = False

                elif key == pygame.K_n:
                    index = (
                        index + 1
                    ) % len(entries)

                    car = reset_car()

                elif key == pygame.K_p:
                    index = (
                        index - 1
                    ) % len(entries)

                    car = reset_car()

                elif pygame.K_1 <= key <= pygame.K_9:
                    target = key - pygame.K_1

                    if target < len(entries):
                        index = target
                        car = reset_car()

                elif key == pygame.K_SPACE:
                    paused = not paused

                elif key == pygame.K_f:
                    fast = not fast

                elif key == pygame.K_r:
                    car = reset_car()

        entry = entries[index]
        track = entry["track"]

        if not paused and car.alive:
            for _ in range(7 if fast else 2):
                if car.alive:
                    car.step(track)

                    if car.steps % 3 == 0:
                        car.full_path.append(
                            (car.x, car.y)
                        )

        W, H = screen.get_size()

        screen.fill(BG)

        text(
            "RACING LINE / FINAL RESULTS",
            26, 19, "lg",
        )

        text(
            "Saved evolutionary controllers | Replay only",
            28, 52, "sm", MUTED,
        )

        chart_h = 140
        bottom_h = 98

        main_box = (
            22,
            84,
            max(320, W - 356),
            max(
                210,
                H - 84 - chart_h - bottom_h - 25
            ),
        )

        panel(main_box)

        origin, scale = draw_track(
            track,
            (
                main_box[0] + 12,
                main_box[1] + 14,
                main_box[2] - 24,
                main_box[3] - 28,
            ),
        )

        # Draw complete driven trajectory
        if len(car.full_path) > 1:
            pygame.draw.lines(
                screen,
                GREEN,
                False,
                project(
                    car.full_path,
                    origin,
                    scale,
                ),
                3,
            )

        draw_f1_car(
            car, origin, scale
        )

        text(
            track.name,
            main_box[0] + 16,
            main_box[1] + 12,
            "md",
        )

        text(
            "GREEN = best-car replay trajectory",
            main_box[0] + 16,
            main_box[1] + 38,
            "sm",
            MUTED,
        )

        # Statistics panel
        side_x = W - 310
        side_y = 84
        side_w = 288
        side_h = main_box[3]

        panel((
            side_x, side_y,
            side_w, side_h
        ))

        x = side_x + 17

        text(
            f"MAP {index+1}/{len(entries)}",
            x, side_y + 18,
            "md", GOLD,
        )

        text(
            entry["family"].replace(
                "_", " "
            ).upper(),
            x, side_y + 52, "lg",
        )

        text(
            "Completed generations",
            x, side_y + 98,
            "sm", MUTED,
        )

        text(
            entry["completed_generations"],
            x, side_y + 119, "lg",
        )

        text(
            "Best recorded progress",
            x, side_y + 162,
            "sm", MUTED,
        )

        text(
            f"{entry['best_progress']*100:.1f}%",
            x, side_y + 183,
            "lg", GREEN,
        )

        text(
            "Best recorded lap",
            x, side_y + 226,
            "sm", MUTED,
        )

        lap = entry["lap"]

        text(
            (
                f"{lap:.2f} s"
                if lap is not None
                else "No completed lap"
            ),
            x, side_y + 247,
            "md",
            GOLD if lap is not None else RED,
        )

        text(
            "Generations with a finisher",
            x, side_y + 285,
            "sm", MUTED,
        )

        text(
            entry["generations_with_finisher"],
            x, side_y + 307,
            "md",
        )

        if side_h >= 390:
            text(
                "CURRENT REPLAY",
                x, side_y + 344,
                "sm", MUTED,
            )

            state = (
                "COMPLETED LAP"
                if car.completed
                else "CRASHED"
                if not car.alive
                else "PAUSED"
                if paused
                else "RUNNING"
            )

            text(
                state,
                x, side_y + 365,
                "md",
                GREEN
                if car.completed
                else RED
                if not car.alive
                else WHITE,
            )

            text(
                f"Progress {car.progress*100:.1f}%"
                f" | {car.steps*DT:.1f}s",
                x, side_y + 396,
                "sm",
            )

        # Learning graph
        graph_y = (
            main_box[1]
            + main_box[3]
            + 12
        )

        draw_chart(
            entry["history"],
            (
                22,
                graph_y,
                max(320, W - 44),
                chart_h,
            ),
        )

        # Bottom controls
        footer_y = (
            graph_y + chart_h + 10
        )

        if footer_y + 50 < H:
            panel((
                22,
                footer_y,
                W - 44,
                H - footer_y - 14,
            ))

            text(
                "N NEXT MAP   P PREVIOUS   "
                "1-5 DIRECT   SPACE PAUSE   "
                "F FAST   R RESTART   ESC EXIT",
                34,
                footer_y + 14,
                "sm",
                MUTED,
            )

            count = len(entries)

            for i, item in enumerate(entries):
                xx = (
                    36
                    + i * max(
                        140,
                        (W - 65) // count
                    )
                )

                label = (
                    ("● " if i == index else "○ ")
                    + item["family"].replace(
                        "_", " "
                    ).upper()
                )

                if xx < W - 130:
                    text(
                        label,
                        xx,
                        footer_y + 41,
                        "sm",
                        GREEN
                        if i == index
                        else WHITE,
                    )

        pygame.display.flip()

    pygame.quit()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoints",
        default=str(Path(__file__).resolve().parent / "racing_checkpoints"),
    )

    args = parser.parse_args()

    entries = load_results(
        args.checkpoints
    )

    if not entries:
        raise SystemExit(
            "No saved checkpoints found. "
            "Train using racing_showcase.py first."
        )

    print(
        f"Loaded {len(entries)} maps: "
        + ", ".join(
            e["family"] for e in entries
        )
    )

    viewer(entries)


if __name__ == "__main__":
    main()
