from pathlib import Path
import time

import numpy as np
import pygame
from stable_baselines3 import PPO

from src.racing_env import RacingEnv
from src.track_geometry import TrackGeometry
from src.track_factory import (
    make_random_spline_track,
    make_random_hard_spline_track,
)


# =========================================================
# SETTINGS
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parent

SCREEN_WIDTH = 1250
SCREEN_HEIGHT = 800

MAP_WIDTH = 930
HUD_WIDTH = SCREEN_WIDTH - MAP_WIDTH

FPS = 30

TRACK_WIDTH = 12.0

AUTO_ADVANCE_SECONDS = 2.0


# =========================================================
# COLORS
# =========================================================

BACKGROUND = (18, 20, 24)
HUD_BACKGROUND = (27, 30, 36)

TRACK_COLOR = (60, 64, 70)
TRACK_EDGE = (225, 225, 225)
CENTERLINE_COLOR = (125, 130, 140)

TRAIL_COLOR = (255, 90, 70)

CAR_COLOR = (255, 210, 60)
CAR_OUTLINE = (15, 15, 15)

TEXT_MAIN = (240, 240, 240)
TEXT_SECONDARY = (165, 170, 180)

SUCCESS = (80, 220, 130)
FAILURE = (235, 90, 90)
ACCENT = (90, 170, 255)


# =========================================================
# LOAD FINAL FROZEN MODEL
# =========================================================

model = PPO.load(
    PROJECT_DIR
    / "models"
    / "ppo_hardfivefamily_stage1"
)

print("Final frozen PPO model loaded!")


# =========================================================
# REPRESENTATIVE TRACK GENERATORS
# =========================================================

def make_demo_ellipse():

    theta = np.linspace(
        0,
        2 * np.pi,
        720,
        endpoint=False
    )

    a = 40.0
    b = 21.0

    return np.column_stack(
        (
            a * np.cos(theta),
            b * np.sin(theta)
        )
    )


def make_demo_wavy(seed=5201):

    rng = np.random.default_rng(seed)

    theta = np.linspace(
        0,
        2 * np.pi,
        720,
        endpoint=False
    )

    base_radius = rng.uniform(
        34.0,
        44.0
    )

    amplitude2 = rng.uniform(
        1.0,
        4.0
    )

    amplitude3 = rng.uniform(
        2.0,
        7.0
    )

    phase2 = rng.uniform(
        0,
        2 * np.pi
    )

    phase3 = rng.uniform(
        0,
        2 * np.pi
    )

    y_scale = rng.uniform(
        0.65,
        0.90
    )

    radius = (
        base_radius
        + amplitude2
        * np.sin(
            2 * theta
            + phase2
        )
        + amplitude3
        * np.sin(
            3 * theta
            + phase3
        )
    )

    x = (
        radius
        * np.cos(theta)
    )

    y = (
        y_scale
        * radius
        * np.sin(theta)
    )

    return np.column_stack(
        (
            x,
            y
        )
    )


def make_demo_complex(seed=5301):

    rng = np.random.default_rng(seed)

    theta = np.linspace(
        0,
        2 * np.pi,
        720,
        endpoint=False
    )

    base_radius = rng.uniform(
        36.0,
        44.0
    )

    amplitudes = {
        2: rng.uniform(1.0, 3.0),
        3: rng.uniform(1.0, 4.0),
        4: rng.uniform(1.0, 4.0),
        5: rng.uniform(0.5, 3.0),
    }

    phases = {
        harmonic:
            rng.uniform(
                0,
                2 * np.pi
            )
        for harmonic
        in amplitudes
    }

    radius = np.full_like(
        theta,
        base_radius
    )

    for harmonic in (
        2,
        3,
        4,
        5
    ):

        radius += (
            amplitudes[harmonic]
            * np.sin(
                harmonic * theta
                + phases[harmonic]
            )
        )

    y_scale = rng.uniform(
        0.70,
        0.90
    )

    x = (
        radius
        * np.cos(theta)
    )

    y = (
        y_scale
        * radius
        * np.sin(theta)
    )

    return np.column_stack(
        (
            x,
            y
        )
    )


# =========================================================
# TRACK LIST
# =========================================================

def spline_track():

    rng = np.random.default_rng(
        5402
    )

    return make_random_spline_track(
        rng,
        num_points=720
    )


def hard_spline_track():

    rng = np.random.default_rng(
        5505
    )

    return make_random_hard_spline_track(
        rng,
        num_points=720
    )


TRACKS = [
    {
        "name": "Ellipse",
        "description": "Simple smooth geometry",
        "generator": make_demo_ellipse,
    },

    {
        "name": "Wavy",
        "description": "2nd + 3rd harmonic geometry",
        "generator": make_demo_wavy,
    },

    {
        "name": "Complex",
        "description": "2nd–5th harmonic geometry",
        "generator": make_demo_complex,
    },

    {
        "name": "Spline",
        "description": "Procedural Catmull-Rom spline",
        "generator": spline_track,
    },

    {
        "name": "Hard Spline",
        "description": "High-curvature targeted geometry",
        "generator": hard_spline_track,
    },
]


# =========================================================
# PYGAME
# =========================================================

pygame.init()

screen = pygame.display.set_mode(
    (
        SCREEN_WIDTH,
        SCREEN_HEIGHT
    )
)

pygame.display.set_caption(
    "RL Racing-Line Generalization"
)

clock = pygame.time.Clock()


font_title = pygame.font.SysFont(
    "Arial",
    27,
    bold=True
)

font_large = pygame.font.SysFont(
    "Arial",
    28,
    bold=True
)

font_medium = pygame.font.SysFont(
    "Arial",
    20,
    bold=True
)

font_normal = pygame.font.SysFont(
    "Arial",
    17
)

font_small = pygame.font.SysFont(
    "Arial",
    14
)


# =========================================================
# GLOBAL STATE
# =========================================================

track_index = 0

track = None
env = None

centerline = None
left_boundary = None
right_boundary = None

world_center = None
SCALE = None

observation = None
info = None

trajectory = []

episode_finished = False
finish_timestamp = None

paused = False
auto_mode = False

results = {}


# =========================================================
# TEXT
# =========================================================

def draw_text(
    text,
    font,
    color,
    x,
    y
):

    rendered = font.render(
        text,
        True,
        color
    )

    screen.blit(
        rendered,
        (
            x,
            y
        )
    )


# =========================================================
# WORLD TO SCREEN
# =========================================================

def world_to_screen(point):

    x = (
        MAP_WIDTH / 2
        + (
            point[0]
            - world_center[0]
        )
        * SCALE
    )

    y = (
        SCREEN_HEIGHT / 2
        - (
            point[1]
            - world_center[1]
        )
        * SCALE
    )

    return (
        int(x),
        int(y)
    )


# =========================================================
# LOAD TRACK
# =========================================================

def load_track(index):

    global track
    global env
    global centerline
    global left_boundary
    global right_boundary
    global world_center
    global SCALE
    global observation
    global info
    global trajectory
    global episode_finished
    global finish_timestamp


    # Close previous environment
    if env is not None:
        env.close()


    config = TRACKS[index]

    centerline = (
        config["generator"]()
    )


    track = TrackGeometry(
        centerline_points=centerline,
        track_width=TRACK_WIDTH,
        num_checkpoints=7
    )


    # ---------------------------------------------
    # BOUNDARIES
    # ---------------------------------------------

    previous_points = np.roll(
        centerline,
        1,
        axis=0
    )

    next_points = np.roll(
        centerline,
        -1,
        axis=0
    )

    tangent = (
        next_points
        - previous_points
    )

    tangent_length = np.linalg.norm(
        tangent,
        axis=1,
        keepdims=True
    )

    tangent = tangent / np.maximum(
        tangent_length,
        1e-12
    )

    normal = np.column_stack(
        (
            -tangent[:, 1],
            tangent[:, 0]
        )
    )

    half_width = (
        TRACK_WIDTH / 2
    )

    left_boundary = (
        centerline
        + normal * half_width
    )

    right_boundary = (
        centerline
        - normal * half_width
    )


    # ---------------------------------------------
    # CAMERA
    # ---------------------------------------------

    all_points = np.vstack(
        (
            left_boundary,
            right_boundary
        )
    )

    world_min = np.min(
        all_points,
        axis=0
    )

    world_max = np.max(
        all_points,
        axis=0
    )

    world_center = (
        world_min
        + world_max
    ) / 2

    world_size = (
        world_max
        - world_min
    )


    padding = 60


    scale_x = (
        MAP_WIDTH
        - 2 * padding
    ) / world_size[0]


    scale_y = (
        SCREEN_HEIGHT
        - 2 * padding
    ) / world_size[1]


    SCALE = min(
        scale_x,
        scale_y
    )


    # ---------------------------------------------
    # ENVIRONMENT
    # ---------------------------------------------

    env = RacingEnv(
        track=track,
        random_start=False,
        curvature_observation=False
    )


    observation, info = env.reset(
        options={
            "start_index": 0
        }
    )


    trajectory = [
        (
            env.sim.car.x,
            env.sim.car.y
        )
    ]


    episode_finished = False
    finish_timestamp = None


    print("\n")
    print("=" * 65)

    print(
        f"Track "
        f"{index + 1}"
        f"/"
        f"{len(TRACKS)}"
    )

    print(
        config["name"]
    )

    print(
        config["description"]
    )

    print("=" * 65)


# =========================================================
# DRAW TRACK
# =========================================================

def draw_track():

    left = [
        world_to_screen(point)
        for point in left_boundary
    ]

    right = [
        world_to_screen(point)
        for point in right_boundary
    ]

    center = [
        world_to_screen(point)
        for point in centerline
    ]


    polygon = (
        left
        + right[::-1]
    )


    pygame.draw.polygon(
        screen,
        TRACK_COLOR,
        polygon
    )


    pygame.draw.lines(
        screen,
        TRACK_EDGE,
        True,
        left,
        3
    )


    pygame.draw.lines(
        screen,
        TRACK_EDGE,
        True,
        right,
        3
    )


    for i in range(
        0,
        len(center),
        12
    ):

        j = min(
            i + 5,
            len(center) - 1
        )

        pygame.draw.line(
            screen,
            CENTERLINE_COLOR,
            center[i],
            center[j],
            2
        )


# =========================================================
# DRAW TRAJECTORY
# =========================================================

def draw_trajectory():

    if len(trajectory) < 2:
        return


    points = [
        world_to_screen(point)
        for point in trajectory
    ]


    pygame.draw.lines(
        screen,
        TRAIL_COLOR,
        False,
        points,
        4
    )


# =========================================================
# DRAW CAR
# =========================================================

def draw_car():

    car = env.sim.car


    position = np.array(
        [
            car.x,
            car.y
        ]
    )


    heading = car.heading


    shape = np.array(
        [
            [2.3, 0],
            [-1.3, 1.0],
            [-0.7, 0],
            [-1.3, -1.0]
        ]
    )


    rotation = np.array(
        [
            [
                np.cos(heading),
                -np.sin(heading)
            ],

            [
                np.sin(heading),
                np.cos(heading)
            ]
        ]
    )


    world_points = (
        shape
        @ rotation.T
        + position
    )


    screen_points = [
        world_to_screen(point)
        for point in world_points
    ]


    pygame.draw.polygon(
        screen,
        CAR_COLOR,
        screen_points
    )


    pygame.draw.polygon(
        screen,
        CAR_OUTLINE,
        screen_points,
        2
    )


# =========================================================
# FAMILY PROGRESS INDICATOR
# =========================================================

def draw_family_indicator():

    x = 35
    y = SCREEN_HEIGHT - 42

    width = (
        MAP_WIDTH - 70
    )


    pygame.draw.line(
        screen,
        (80, 84, 92),
        (
            x,
            y
        ),
        (
            x + width,
            y
        ),
        3
    )


    for i, config in enumerate(
        TRACKS
    ):

        marker_x = (
            x
            + int(
                i
                / (
                    len(TRACKS) - 1
                )
                * width
            )
        )


        if i < track_index:

            color = SUCCESS

        elif i == track_index:

            color = ACCENT

        else:

            color = TEXT_SECONDARY


        pygame.draw.circle(
            screen,
            color,
            (
                marker_x,
                y
            ),
            7
        )


# =========================================================
# HUD
# =========================================================

def draw_hud():

    pygame.draw.rect(
        screen,
        HUD_BACKGROUND,
        (
            MAP_WIDTH,
            0,
            HUD_WIDTH,
            SCREEN_HEIGHT
        )
    )


    x = MAP_WIDTH + 27


    config = TRACKS[
        track_index
    ]


    draw_text(
        "GENERALIZATION",
        font_title,
        TEXT_MAIN,
        x,
        30
    )


    draw_text(
        "Frozen final PPO model",
        font_small,
        TEXT_SECONDARY,
        x,
        69
    )


    pygame.draw.line(
        screen,
        TEXT_SECONDARY,
        (
            x,
            100
        ),
        (
            SCREEN_WIDTH - 25,
            100
        ),
        1
    )


    # ---------------------------------------------
    # TRACK FAMILY
    # ---------------------------------------------

    draw_text(
        "TRACK FAMILY",
        font_small,
        TEXT_SECONDARY,
        x,
        130
    )


    draw_text(
        config["name"],
        font_large,
        ACCENT,
        x,
        151
    )


    draw_text(
        config["description"],
        font_small,
        TEXT_SECONDARY,
        x,
        190
    )


    draw_text(
        f"Track "
        f"{track_index + 1}"
        f" / "
        f"{len(TRACKS)}",
        font_small,
        TEXT_SECONDARY,
        x,
        218
    )


    # ---------------------------------------------
    # LIVE TELEMETRY
    # ---------------------------------------------

    draw_text(
        "SPEED",
        font_small,
        TEXT_SECONDARY,
        x,
        270
    )


    draw_text(
        f"{env.sim.car.speed:.1f} m/s",
        font_large,
        TEXT_MAIN,
        x,
        290
    )


    draw_text(
        "LAP TIME",
        font_small,
        TEXT_SECONDARY,
        x,
        350
    )


    draw_text(
        f"{info.get('time', 0):.2f} s",
        font_large,
        TEXT_MAIN,
        x,
        370
    )


    draw_text(
        "PROGRESS",
        font_small,
        TEXT_SECONDARY,
        x,
        430
    )


    display_progress = info.get(
        "progress",
        0.0
    )


    if (
        episode_finished
        and info.get(
            "lap_completed",
            False
        )
    ):

        display_progress = 100.0


    draw_text(
        f"{display_progress:.1f} %",
        font_large,
        TEXT_MAIN,
        x,
        450
    )


    # ---------------------------------------------
    # RESULT
    # ---------------------------------------------

    draw_text(
        "RESULT",
        font_small,
        TEXT_SECONDARY,
        x,
        515
    )


    if episode_finished:

        if info.get(
            "lap_completed",
            False
        ):

            draw_text(
                "LAP COMPLETE",
                font_medium,
                SUCCESS,
                x,
                540
            )

        else:

            draw_text(
                "FAILED",
                font_medium,
                FAILURE,
                x,
                540
            )

    else:

        draw_text(
            "DRIVING...",
            font_medium,
            TEXT_MAIN,
            x,
            540
        )


    # ---------------------------------------------
    # PREVIOUS RESULTS
    # ---------------------------------------------

    draw_text(
        "COMPLETED DEMOS",
        font_small,
        TEXT_SECONDARY,
        x,
        595
    )


    row_y = 618


    for name, result in results.items():

        symbol = (
            "✓"
            if result["success"]
            else "✕"
        )

        color = (
            SUCCESS
            if result["success"]
            else FAILURE
        )


        text = (
            f"{symbol} "
            f"{name:<11} "
        )


        if result["success"]:

            text += (
                f"{result['time']:.2f}s"
            )

        else:

            text += (
                f"{result['progress']:.0f}%"
            )


        draw_text(
            text,
            font_small,
            color,
            x,
            row_y
        )


        row_y += 21


    # ---------------------------------------------
    # CONTROLS
    # ---------------------------------------------

    pygame.draw.line(
        screen,
        TEXT_SECONDARY,
        (
            x,
            705
        ),
        (
            SCREEN_WIDTH - 25,
            705
        ),
        1
    )


    draw_text(
        "N   Next track",
        font_small,
        TEXT_SECONDARY,
        x,
        720
    )


    draw_text(
        "R   Restart current track",
        font_small,
        TEXT_SECONDARY,
        x,
        740
    )


    auto_text = (
        "ON"
        if auto_mode
        else "OFF"
    )


    draw_text(
        f"A   Auto-cycle: {auto_text}",
        font_small,
        TEXT_SECONDARY,
        x,
        760
    )


    draw_text(
        "ESC Exit",
        font_small,
        TEXT_SECONDARY,
        x,
        780
    )


# =========================================================
# SAVE RESULT
# =========================================================

def finish_episode():

    global episode_finished
    global finish_timestamp


    if episode_finished:
        return


    episode_finished = True
    finish_timestamp = time.time()


    success = bool(
        info.get(
            "lap_completed",
            False
        )
    )


    results[
        TRACKS[
            track_index
        ]["name"]
    ] = {

        "success":
            success,

        "time":
            float(
                info.get(
                    "time",
                    0.0
                )
            ),

        "progress":
            float(
                info.get(
                    "progress",
                    0.0
                )
            )
    }


    print(
        "Lap completed:",
        success
    )


    print(
        "Time:",
        f"{info.get('time', 0):.2f}s"
    )


    print(
        "Progress:",
        f"{info.get('progress', 0):.2f}%"
    )


# =========================================================
# NEXT TRACK
# =========================================================

def next_track():

    global track_index


    if track_index < (
        len(TRACKS) - 1
    ):

        track_index += 1

    else:

        track_index = 0


    load_track(
        track_index
    )


# =========================================================
# START
# =========================================================

load_track(
    track_index
)


running = True


# =========================================================
# MAIN LOOP
# =========================================================

while running:

    # ---------------------------------------------
    # EVENTS
    # ---------------------------------------------

    for event in pygame.event.get():

        if event.type == pygame.QUIT:

            running = False


        elif event.type == pygame.KEYDOWN:

            if event.key == pygame.K_ESCAPE:

                running = False


            elif event.key == pygame.K_SPACE:

                paused = not paused


            elif event.key == pygame.K_n:

                next_track()


            elif event.key == pygame.K_r:

                load_track(
                    track_index
                )


            elif event.key == pygame.K_a:

                auto_mode = (
                    not auto_mode
                )


    # ---------------------------------------------
    # PPO
    # ---------------------------------------------

    if (
        not paused
        and not episode_finished
    ):

        action, _ = model.predict(
            observation,
            deterministic=True
        )


        (
            observation,
            reward,
            terminated,
            truncated,
            info
        ) = env.step(
            action
        )


        trajectory.append(
            (
                env.sim.car.x,
                env.sim.car.y
            )
        )


        if (
            terminated
            or truncated
            or info.get(
                "lap_completed",
                False
            )
        ):

            finish_episode()


    # ---------------------------------------------
    # AUTO ADVANCE
    # ---------------------------------------------

    if (
        auto_mode
        and episode_finished
        and finish_timestamp is not None
    ):

        if (
            time.time()
            - finish_timestamp
            >= AUTO_ADVANCE_SECONDS
        ):

            next_track()


    # ---------------------------------------------
    # DRAW
    # ---------------------------------------------

    screen.fill(
        BACKGROUND
    )


    draw_track()

    draw_trajectory()

    draw_car()

    draw_family_indicator()

    draw_hud()


    draw_text(
        "RL Racing-Line Generalization",
        font_title,
        TEXT_MAIN,
        28,
        20
    )


    draw_text(
        "One frozen PPO policy • multiple unseen geometries",
        font_small,
        TEXT_SECONDARY,
        28,
        57
    )


    pygame.display.flip()

    clock.tick(
        FPS
    )


# =========================================================
# FINAL SUMMARY
# =========================================================

print("\n")
print("=" * 65)
print(" GENERALIZATION VISUAL DEMO SUMMARY")
print("=" * 65)


for name, result in results.items():

    if result["success"]:

        print(
            f"{name:<15} "
            f"COMPLETE | "
            f"{result['time']:.2f}s"
        )

    else:

        print(
            f"{name:<15} "
            f"FAILED   | "
            f"{result['progress']:.2f}%"
        )


if env is not None:
    env.close()


pygame.quit()