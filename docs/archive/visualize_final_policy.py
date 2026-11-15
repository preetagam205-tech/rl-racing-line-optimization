from pathlib import Path

import numpy as np
import pygame

from stable_baselines3 import PPO

from src.racing_env import RacingEnv
from src.track_geometry import TrackGeometry
from src.track_factory import make_wavy_track


# =========================================================
# SETTINGS
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parent

SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800

MAP_WIDTH = 900
HUD_WIDTH = SCREEN_WIDTH - MAP_WIDTH

FPS = 30

TRACK_WIDTH = 12.0


# =========================================================
# COLORS
# =========================================================

BACKGROUND = (18, 20, 24)

TRACK_COLOR = (60, 64, 70)
TRACK_EDGE = (220, 220, 220)

CENTERLINE_COLOR = (125, 130, 140)

PPO_TRAIL_COLOR = (255, 90, 70)

CAR_COLOR = (255, 210, 60)
CAR_OUTLINE = (20, 20, 20)

HUD_BACKGROUND = (27, 30, 36)

TEXT_MAIN = (240, 240, 240)
TEXT_SECONDARY = (170, 175, 185)

SUCCESS_COLOR = (80, 220, 130)

BAR_BACKGROUND = (55, 58, 65)
BAR_POSITIVE = (80, 200, 120)
BAR_NEGATIVE = (220, 90, 90)


# =========================================================
# LOAD FINAL MODEL
# =========================================================

model_path = (
    PROJECT_DIR
    / "models"
    / "ppo_hardfivefamily_stage1"
)

model = PPO.load(model_path)

print("Final PPO model loaded!")


# =========================================================
# CREATE TRACK
# =========================================================

centerline = make_wavy_track(
    num_points=720
)

track = TrackGeometry(
    centerline_points=centerline,
    track_width=TRACK_WIDTH,
    num_checkpoints=7
)


# =========================================================
# COMPUTE TRACK BOUNDARIES
# =========================================================

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

half_width = TRACK_WIDTH / 2.0

left_boundary = (
    centerline
    + normal * half_width
)

right_boundary = (
    centerline
    - normal * half_width
)


# =========================================================
# CAMERA / WORLD-TO-SCREEN
# =========================================================

all_track_points = np.vstack(
    (
        left_boundary,
        right_boundary
    )
)

world_min = np.min(
    all_track_points,
    axis=0
)

world_max = np.max(
    all_track_points,
    axis=0
)

world_center = (
    world_min
    + world_max
) / 2.0

world_size = (
    world_max
    - world_min
)

padding = 55

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


def world_to_screen(point):

    x = (
        MAP_WIDTH / 2
        + (point[0] - world_center[0]) * SCALE
    )

    y = (
        SCREEN_HEIGHT / 2
        - (point[1] - world_center[1]) * SCALE
    )

    return (
        int(x),
        int(y)
    )


# =========================================================
# PYGAME INITIALIZATION
# =========================================================

pygame.init()

screen = pygame.display.set_mode(
    (
        SCREEN_WIDTH,
        SCREEN_HEIGHT
    )
)

pygame.display.set_caption(
    "RL Racing-Line Generation"
)

clock = pygame.time.Clock()


font_large = pygame.font.SysFont(
    "Arial",
    30,
    bold=True
)

font_medium = pygame.font.SysFont(
    "Arial",
    21,
    bold=True
)

font_normal = pygame.font.SysFont(
    "Arial",
    18
)

font_small = pygame.font.SysFont(
    "Arial",
    15
)


# =========================================================
# ENVIRONMENT
# =========================================================

env = RacingEnv(
    track=track,
    random_start=False,
    curvature_observation=False
)


def reset_episode():

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

    return (
        observation,
        info,
        trajectory
    )


observation, info, trajectory = (
    reset_episode()
)


terminated = False
truncated = False

paused = False
lap_finished = False

last_action = np.array(
    [0.0, 0.0],
    dtype=np.float32
)


# =========================================================
# DRAW HELPERS
# =========================================================

def draw_text(
    surface,
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

    surface.blit(
        rendered,
        (
            x,
            y
        )
    )


def draw_track():

    left_screen = [
        world_to_screen(point)
        for point in left_boundary
    ]

    right_screen = [
        world_to_screen(point)
        for point in right_boundary
    ]

    center_screen = [
        world_to_screen(point)
        for point in centerline
    ]


    # ---------------------------------------------
    # TRACK SURFACE
    # ---------------------------------------------

    track_polygon = (
        left_screen
        + right_screen[::-1]
    )

    pygame.draw.polygon(
        screen,
        TRACK_COLOR,
        track_polygon
    )


    # ---------------------------------------------
    # TRACK BOUNDARIES
    # ---------------------------------------------

    pygame.draw.lines(
        screen,
        TRACK_EDGE,
        True,
        left_screen,
        3
    )

    pygame.draw.lines(
        screen,
        TRACK_EDGE,
        True,
        right_screen,
        3
    )


    # ---------------------------------------------
    # DASHED CENTERLINE
    # ---------------------------------------------

    dash_step = 12

    for i in range(
        0,
        len(center_screen),
        dash_step
    ):

        start = center_screen[i]

        end_index = min(
            i + 5,
            len(center_screen) - 1
        )

        end = center_screen[end_index]

        pygame.draw.line(
            screen,
            CENTERLINE_COLOR,
            start,
            end,
            2
        )


def draw_trajectory():

    if len(trajectory) < 2:
        return

    points = [
        world_to_screen(point)
        for point in trajectory
    ]

    pygame.draw.lines(
        screen,
        PPO_TRAIL_COLOR,
        False,
        points,
        4
    )


def draw_car():

    car = env.sim.car

    position = np.array(
        [
            car.x,
            car.y
        ]
    )

    heading = car.heading


    # Top-down triangular car shape in meters
    local_points = np.array(
        [
            [2.2, 0.0],
            [-1.4, 1.0],
            [-0.8, 0.0],
            [-1.4, -1.0]
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
        local_points
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


def draw_bar(
    x,
    y,
    width,
    height,
    value,
    minimum=-1.0,
    maximum=1.0
):

    pygame.draw.rect(
        screen,
        BAR_BACKGROUND,
        (
            x,
            y,
            width,
            height
        ),
        border_radius=4
    )


    normalized = (
        value - minimum
    ) / (
        maximum - minimum
    )

    normalized = np.clip(
        normalized,
        0.0,
        1.0
    )


    center = 0.5

    if value >= 0:

        start_x = (
            x
            + int(
                center * width
            )
        )

        bar_width = int(
            (
                normalized
                - center
            )
            * width
        )

        color = BAR_POSITIVE

    else:

        start_x = (
            x
            + int(
                normalized
                * width
            )
        )

        bar_width = int(
            (
                center
                - normalized
            )
            * width
        )

        color = BAR_NEGATIVE


    if bar_width > 0:

        pygame.draw.rect(
            screen,
            color,
            (
                start_x,
                y,
                bar_width,
                height
            ),
            border_radius=4
        )


    center_x = (
        x
        + width // 2
    )

    pygame.draw.line(
        screen,
        TEXT_SECONDARY,
        (
            center_x,
            y
        ),
        (
            center_x,
            y + height
        ),
        1
    )


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


    hud_x = MAP_WIDTH + 30


    draw_text(
        screen,
        "FINAL PPO POLICY",
        font_large,
        TEXT_MAIN,
        hud_x,
        35
    )


    draw_text(
        screen,
        "Hard five-family model",
        font_small,
        TEXT_SECONDARY,
        hud_x,
        76
    )


    pygame.draw.line(
        screen,
        TEXT_SECONDARY,
        (
            hud_x,
            108
        ),
        (
            SCREEN_WIDTH - 30,
            108
        ),
        1
    )


    car = env.sim.car


    # ---------------------------------------------
    # SPEED
    # ---------------------------------------------

    draw_text(
        screen,
        "SPEED",
        font_small,
        TEXT_SECONDARY,
        hud_x,
        140
    )

    draw_text(
        screen,
        f"{car.speed:5.1f} m/s",
        font_large,
        TEXT_MAIN,
        hud_x,
        160
    )


    # ---------------------------------------------
    # LAP TIME
    # ---------------------------------------------

    draw_text(
        screen,
        "LAP TIME",
        font_small,
        TEXT_SECONDARY,
        hud_x,
        220
    )

    draw_text(
        screen,
        f"{info.get('time', 0.0):5.2f} s",
        font_large,
        TEXT_MAIN,
        hud_x,
        240
    )


    # ---------------------------------------------
    # PROGRESS
    # ---------------------------------------------

    draw_text(
        screen,
        "PROGRESS",
        font_small,
        TEXT_SECONDARY,
        hud_x,
        300
    )

    progress = info.get(
        "progress",
        0.0
    )

    display_progress = progress

    if lap_finished:
        display_progress = 100.0

    draw_text(
        screen,
        f"{display_progress:5.1f} %",
        font_large,
        TEXT_MAIN,
        hud_x,
        320
    )


    # ---------------------------------------------
    # STEERING
    # ---------------------------------------------

    steering = float(
        last_action[0]
    )

    draw_text(
        screen,
        "STEERING",
        font_small,
        TEXT_SECONDARY,
        hud_x,
        385
    )

    draw_text(
        screen,
        f"{steering:+.2f}",
        font_medium,
        TEXT_MAIN,
        hud_x,
        407
    )

    draw_bar(
        hud_x,
        440,
        230,
        14,
        steering
    )


    # ---------------------------------------------
    # THROTTLE / BRAKE
    # ---------------------------------------------

    throttle = float(
        last_action[1]
    )

    draw_text(
        screen,
        "THROTTLE / BRAKE",
        font_small,
        TEXT_SECONDARY,
        hud_x,
        485
    )

    draw_text(
        screen,
        f"{throttle:+.2f}",
        font_medium,
        TEXT_MAIN,
        hud_x,
        507
    )

    draw_bar(
        hud_x,
        540,
        230,
        14,
        throttle
    )


    # ---------------------------------------------
    # STATUS
    # ---------------------------------------------

    draw_text(
        screen,
        "STATUS",
        font_small,
        TEXT_SECONDARY,
        hud_x,
        590
    )


    if lap_finished:

        draw_text(
            screen,
            "LAP COMPLETE",
            font_medium,
            SUCCESS_COLOR,
            hud_x,
            614
        )

    elif paused:

        draw_text(
            screen,
            "PAUSED",
            font_medium,
            TEXT_MAIN,
            hud_x,
            614
        )

    else:

        draw_text(
            screen,
            "RACING",
            font_medium,
            TEXT_MAIN,
            hud_x,
            614
        )


    # ---------------------------------------------
    # CONTROLS
    # ---------------------------------------------

    pygame.draw.line(
        screen,
        TEXT_SECONDARY,
        (
            hud_x,
            675
        ),
        (
            SCREEN_WIDTH - 30,
            675
        ),
        1
    )


    draw_text(
        screen,
        "SPACE   Pause / resume",
        font_small,
        TEXT_SECONDARY,
        hud_x,
        700
    )

    draw_text(
        screen,
        "R       Restart lap",
        font_small,
        TEXT_SECONDARY,
        hud_x,
        725
    )

    draw_text(
        screen,
        "ESC     Exit",
        font_small,
        TEXT_SECONDARY,
        hud_x,
        750
    )


# =========================================================
# MAIN LOOP
# =========================================================

running = True


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


            elif event.key == pygame.K_r:

                (
                    observation,
                    info,
                    trajectory
                ) = reset_episode()

                terminated = False
                truncated = False

                lap_finished = False
                paused = False

                last_action = np.array(
                    [0.0, 0.0],
                    dtype=np.float32
                )


    # ---------------------------------------------
    # PPO STEP
    # ---------------------------------------------

    if (
        not paused
        and not lap_finished
    ):

        action, _ = model.predict(
            observation,
            deterministic=True
        )

        last_action = np.asarray(
            action,
            dtype=np.float32
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
            info.get(
                "lap_completed",
                False
            )
            or terminated
            or truncated
        ):

            lap_finished = True

            print("\n")
            print("=" * 60)
            print(" FINAL PPO VISUAL DEMO")
            print("=" * 60)

            print(
                "Lap completed:",
                info.get(
                    "lap_completed",
                    False
                )
            )

            print(
                "Lap time:",
                f"{info.get('time', 0.0):.2f}s"
            )


    # ---------------------------------------------
    # DRAW
    # ---------------------------------------------

    screen.fill(
        BACKGROUND
    )


    draw_track()

    draw_trajectory()

    draw_car()

    draw_hud()


    # ---------------------------------------------
    # MAP TITLE
    # ---------------------------------------------

    draw_text(
        screen,
        "RL Racing-Line Generation",
        font_medium,
        TEXT_MAIN,
        25,
        20
    )

    draw_text(
        screen,
        "Wavy Track • Learned PPO Racing Line",
        font_small,
        TEXT_SECONDARY,
        25,
        49
    )


    pygame.display.flip()

    clock.tick(
        FPS
    )


# =========================================================
# CLEANUP
# =========================================================

env.close()

pygame.quit()