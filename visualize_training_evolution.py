from pathlib import Path
import time

import numpy as np
import pygame
from stable_baselines3 import PPO

from src.racing_env import RacingEnv
from src.track_geometry import TrackGeometry


# =========================================================
# SETTINGS
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parent

MODEL_DIR = (
    PROJECT_DIR
    / "models"
    / "visual_training"
)

SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800

MAP_WIDTH = 900
HUD_WIDTH = SCREEN_WIDTH - MAP_WIDTH

FPS = 30
TRACK_WIDTH = 12.0

# How long to keep the result visible before next checkpoint
RESULT_HOLD_SECONDS = 2.0

# Safety limit in case an early model barely moves
MAX_VISUAL_STEPS = 2500


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
CAR_OUTLINE = (20, 20, 20)

TEXT_MAIN = (240, 240, 240)
TEXT_SECONDARY = (165, 170, 180)

SUCCESS = (80, 220, 130)
FAILURE = (235, 90, 90)

ACCENT = (90, 170, 255)


# =========================================================
# TRAINING CHECKPOINTS
# =========================================================

CHECKPOINTS = [
    {
        "steps": 0,
        "label": "Untrained policy",
        "file": "ppo_visual_000000"
    },
    {
        "steps": 25_000,
        "label": "Early training",
        "file": "ppo_visual_025000"
    },
    {
        "steps": 50_000,
        "label": "Early improvement",
        "file": "ppo_visual_050000"
    },
    {
        "steps": 100_000,
        "label": "Developing policy",
        "file": "ppo_visual_100000"
    },
    {
        "steps": 250_000,
        "label": "Intermediate policy",
        "file": "ppo_visual_250000"
    },
    {
        "steps": 500_000,
        "label": "Final visual-training policy",
        "file": "ppo_visual_500000"
    },
]


# =========================================================
# CHECK FILES
# =========================================================

for checkpoint in CHECKPOINTS:

    path = (
        MODEL_DIR
        / checkpoint["file"]
    )

    zip_path = Path(
        str(path) + ".zip"
    )

    if not zip_path.exists():
        raise FileNotFoundError(
            f"\nMissing checkpoint:\n{zip_path}\n"
        )


# =========================================================
# BUILD TRAINING TRACK
# =========================================================

num_points = 720

theta = np.linspace(
    0.0,
    2.0 * np.pi,
    num_points,
    endpoint=False
)

a = 40.0
b = 20.0

centerline = np.column_stack(
    (
        a * np.cos(theta),
        b * np.sin(theta)
    )
)

track = TrackGeometry(
    centerline_points=centerline,
    track_width=TRACK_WIDTH,
    num_checkpoints=7
)


# =========================================================
# TRACK BOUNDARIES
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
# CAMERA
# =========================================================

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
) / 2.0

world_size = (
    world_max
    - world_min
)

padding = 70

scale_x = (
    MAP_WIDTH - 2 * padding
) / world_size[0]

scale_y = (
    SCREEN_HEIGHT - 2 * padding
) / world_size[1]

SCALE = min(
    scale_x,
    scale_y
)


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
    "PPO Training Evolution"
)

clock = pygame.time.Clock()

font_title = pygame.font.SysFont(
    "Arial",
    28,
    bold=True
)

font_large = pygame.font.SysFont(
    "Arial",
    27,
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
# ENVIRONMENT
# =========================================================

env = RacingEnv(
    track=track,
    random_start=False,
    curvature_observation=False
)


# =========================================================
# STATE
# =========================================================

checkpoint_index = 0

model = None
observation = None
info = None

trajectory = []

terminated = False
truncated = False

episode_finished = False
finish_timestamp = None

visual_steps = 0

last_action = np.array(
    [0.0, 0.0],
    dtype=np.float32
)

results = []


# =========================================================
# LOAD CHECKPOINT
# =========================================================

def load_checkpoint(index):

    global model
    global observation
    global info
    global trajectory
    global terminated
    global truncated
    global episode_finished
    global finish_timestamp
    global visual_steps
    global last_action

    checkpoint = CHECKPOINTS[index]

    model_path = (
        MODEL_DIR
        / checkpoint["file"]
    )

    model = PPO.load(
        model_path
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

    terminated = False
    truncated = False

    episode_finished = False
    finish_timestamp = None

    visual_steps = 0

    last_action = np.array(
        [0.0, 0.0],
        dtype=np.float32
    )

    print("\n")
    print("=" * 65)

    print(
        f"Checkpoint: "
        f"{checkpoint['steps']:,} steps"
    )

    print(
        checkpoint["label"]
    )

    print("=" * 65)


# =========================================================
# DRAW HELPERS
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

    dash_step = 14

    for i in range(
        0,
        len(center),
        dash_step
    ):

        j = min(
            i + 6,
            len(center) - 1
        )

        pygame.draw.line(
            screen,
            CENTERLINE_COLOR,
            center[i],
            center[j],
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
        TRAIL_COLOR,
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

    car_shape = np.array(
        [
            [2.2, 0.0],
            [-1.3, 1.0],
            [-0.7, 0.0],
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
        car_shape
        @ rotation.T
        + position
    )

    points = [
        world_to_screen(point)
        for point in world_points
    ]

    pygame.draw.polygon(
        screen,
        CAR_COLOR,
        points
    )

    pygame.draw.polygon(
        screen,
        CAR_OUTLINE,
        points,
        2
    )


# =========================================================
# PROGRESS BAR
# =========================================================

def draw_training_progress():

    x = 35
    y = SCREEN_HEIGHT - 55

    width = MAP_WIDTH - 70

    pygame.draw.rect(
        screen,
        (45, 48, 54),
        (
            x,
            y,
            width,
            9
        ),
        border_radius=5
    )

    fraction = (
        checkpoint_index
        / (
            len(CHECKPOINTS) - 1
        )
    )

    pygame.draw.rect(
        screen,
        ACCENT,
        (
            x,
            y,
            int(width * fraction),
            9
        ),
        border_radius=5
    )

    for i in range(
        len(CHECKPOINTS)
    ):

        marker_x = (
            x
            + int(
                width
                * i
                / (
                    len(CHECKPOINTS) - 1
                )
            )
        )

        pygame.draw.circle(
            screen,
            TEXT_MAIN,
            (
                marker_x,
                y + 4
            ),
            5
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

    x = MAP_WIDTH + 28

    checkpoint = CHECKPOINTS[
        checkpoint_index
    ]

    draw_text(
        "TRAINING EVOLUTION",
        font_title,
        TEXT_MAIN,
        x,
        35
    )

    draw_text(
        f"Stage "
        f"{checkpoint_index + 1}"
        f" / "
        f"{len(CHECKPOINTS)}",
        font_small,
        TEXT_SECONDARY,
        x,
        78
    )

    pygame.draw.line(
        screen,
        TEXT_SECONDARY,
        (
            x,
            110
        ),
        (
            SCREEN_WIDTH - 28,
            110
        ),
        1
    )


    # ---------------------------------------------
    # TRAINING STEPS
    # ---------------------------------------------

    draw_text(
        "TRAINING STEPS",
        font_small,
        TEXT_SECONDARY,
        x,
        140
    )

    draw_text(
        f"{checkpoint['steps']:,}",
        font_large,
        TEXT_MAIN,
        x,
        162
    )

    draw_text(
        checkpoint["label"],
        font_small,
        ACCENT,
        x,
        200
    )


    # ---------------------------------------------
    # LIVE VALUES
    # ---------------------------------------------

    draw_text(
        "SPEED",
        font_small,
        TEXT_SECONDARY,
        x,
        250
    )

    draw_text(
        f"{env.sim.car.speed:.1f} m/s",
        font_large,
        TEXT_MAIN,
        x,
        270
    )


    draw_text(
        "PROGRESS",
        font_small,
        TEXT_SECONDARY,
        x,
        330
    )

    progress = info.get(
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
        progress = 100.0

    draw_text(
        f"{progress:.1f} %",
        font_large,
        TEXT_MAIN,
        x,
        350
    )


    draw_text(
        "TIME",
        font_small,
        TEXT_SECONDARY,
        x,
        410
    )

    draw_text(
        f"{info.get('time', 0):.2f} s",
        font_large,
        TEXT_MAIN,
        x,
        430
    )


    # ---------------------------------------------
    # RESULT
    # ---------------------------------------------

    draw_text(
        "RESULT",
        font_small,
        TEXT_SECONDARY,
        x,
        505
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
                530
            )

        else:

            draw_text(
                "FAILED",
                font_medium,
                FAILURE,
                x,
                530
            )

    else:

        draw_text(
            "DRIVING...",
            font_medium,
            TEXT_MAIN,
            x,
            530
        )


    # ---------------------------------------------
    # ACTION
    # ---------------------------------------------

    draw_text(
        "STEERING",
        font_small,
        TEXT_SECONDARY,
        x,
        590
    )

    draw_text(
        f"{float(last_action[0]):+.2f}",
        font_normal,
        TEXT_MAIN,
        x,
        610
    )

    draw_text(
        "THROTTLE / BRAKE",
        font_small,
        TEXT_SECONDARY,
        x,
        650
    )

    draw_text(
        f"{float(last_action[1]):+.2f}",
        font_normal,
        TEXT_MAIN,
        x,
        670
    )


    # ---------------------------------------------
    # CONTROLS
    # ---------------------------------------------

    pygame.draw.line(
        screen,
        TEXT_SECONDARY,
        (
            x,
            710
        ),
        (
            SCREEN_WIDTH - 28,
            710
        ),
        1
    )

    draw_text(
        "N   Next checkpoint",
        font_small,
        TEXT_SECONDARY,
        x,
        730
    )

    draw_text(
        "R   Restart sequence",
        font_small,
        TEXT_SECONDARY,
        x,
        750
    )

    draw_text(
        "ESC Exit",
        font_small,
        TEXT_SECONDARY,
        x,
        770
    )


# =========================================================
# SAVE RESULT
# =========================================================

def finish_current_episode():

    global episode_finished
    global finish_timestamp

    if episode_finished:
        return

    episode_finished = True

    finish_timestamp = time.time()

    checkpoint = CHECKPOINTS[
        checkpoint_index
    ]

    lap_completed = bool(
        info.get(
            "lap_completed",
            False
        )
    )

    result = {
        "steps":
            checkpoint["steps"],

        "lap_completed":
            lap_completed,

        "progress":
            float(
                info.get(
                    "progress",
                    0.0
                )
            ),

        "time":
            float(
                info.get(
                    "time",
                    0.0
                )
            )
    }

    results.append(
        result
    )

    print(
        "Lap completed:",
        lap_completed
    )

    print(
        "Progress:",
        f"{result['progress']:.2f}%"
    )

    print(
        "Time:",
        f"{result['time']:.2f}s"
    )


# =========================================================
# NEXT CHECKPOINT
# =========================================================

def next_checkpoint():

    global checkpoint_index

    if (
        checkpoint_index
        < len(CHECKPOINTS) - 1
    ):

        checkpoint_index += 1

        load_checkpoint(
            checkpoint_index
        )


# =========================================================
# START
# =========================================================

load_checkpoint(
    checkpoint_index
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


            elif event.key == pygame.K_n:

                next_checkpoint()


            elif event.key == pygame.K_r:

                checkpoint_index = 0

                results.clear()

                load_checkpoint(
                    checkpoint_index
                )


    # ---------------------------------------------
    # POLICY STEP
    # ---------------------------------------------

    if not episode_finished:

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

        visual_steps += 1

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
            or visual_steps
            >= MAX_VISUAL_STEPS
        ):

            finish_current_episode()


    # ---------------------------------------------
    # AUTO ADVANCE
    # ---------------------------------------------

    elif (
        finish_timestamp is not None
        and checkpoint_index
        < len(CHECKPOINTS) - 1
    ):

        if (
            time.time()
            - finish_timestamp
            >= RESULT_HOLD_SECONDS
        ):

            next_checkpoint()


    # ---------------------------------------------
    # DRAW
    # ---------------------------------------------

    screen.fill(
        BACKGROUND
    )

    draw_track()

    draw_trajectory()

    draw_car()

    draw_training_progress()

    draw_hud()


    draw_text(
        "How PPO Learns to Drive",
        font_title,
        TEXT_MAIN,
        30,
        25
    )

    draw_text(
        "Same track • same starting position • progressively trained policies",
        font_small,
        TEXT_SECONDARY,
        30,
        62
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
print(" TRAINING EVOLUTION SUMMARY")
print("=" * 65)

for result in results:

    print(
        f"{result['steps']:>8,} steps | "
        f"Lap: {result['lap_completed']} | "
        f"Progress: {result['progress']:.2f}% | "
        f"Time: {result['time']:.2f}s"
    )


env.close()

pygame.quit()