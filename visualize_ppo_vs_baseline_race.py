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

SCREEN_WIDTH = 1300
SCREEN_HEIGHT = 800

MAP_WIDTH = 930
HUD_WIDTH = SCREEN_WIDTH - MAP_WIDTH

FPS = 30

TRACK_WIDTH = 12.0


# =========================================================
# COLORS
# =========================================================

BACKGROUND = (18, 20, 24)
HUD_BACKGROUND = (27, 30, 36)

TRACK_COLOR = (60, 64, 70)
TRACK_EDGE = (225, 225, 225)
CENTERLINE_COLOR = (125, 130, 140)

PPO_COLOR = (255, 90, 70)
BASELINE_COLOR = (80, 170, 255)

PPO_CAR = (255, 210, 60)
BASELINE_CAR = (80, 190, 255)

TEXT_MAIN = (240, 240, 240)
TEXT_SECONDARY = (165, 170, 180)

SUCCESS = (80, 220, 130)


# =========================================================
# MODEL
# =========================================================

model = PPO.load(
    PROJECT_DIR
    / "models"
    / "ppo_hardfivefamily_stage1"
)

print("Final PPO model loaded!")


# =========================================================
# TRACK
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
# BASELINE CONTROLLER HELPERS
# =========================================================

def wrap_angle(angle):

    return (
        angle + np.pi
    ) % (
        2.0 * np.pi
    ) - np.pi


def nearest_centerline_index(x, y):

    differences = (
        centerline
        - np.array([x, y])
    )

    distances_squared = np.sum(
        differences ** 2,
        axis=1
    )

    return int(
        np.argmin(
            distances_squared
        )
    )


def lookahead_target(
    start_index,
    distance_ahead=12.0
):

    total_distance = 0.0

    index = start_index

    while total_distance < distance_ahead:

        next_index = (
            index + 1
        ) % len(centerline)

        segment = (
            centerline[next_index]
            - centerline[index]
        )

        total_distance += np.linalg.norm(
            segment
        )

        index = next_index

    return centerline[index]


def baseline_action(env):

    car = env.sim.car

    nearest_index = (
        nearest_centerline_index(
            car.x,
            car.y
        )
    )

    target = lookahead_target(
        nearest_index,
        distance_ahead=12.0
    )

    desired_heading = np.arctan2(
        target[1] - car.y,
        target[0] - car.x
    )

    heading_error = wrap_angle(
        desired_heading
        - car.heading
    )


    # ---------------------------------------------
    # STEERING
    # ---------------------------------------------

    steering = np.clip(
        1.8
        * heading_error
        / car.max_steering,
        -1.0,
        1.0
    )


    # ---------------------------------------------
    # SPEED PLANNING
    # ---------------------------------------------

    heading_changes = (
        track.get_lookahead_heading_changes(
            car.x,
            car.y,
            distances=(
                20.0,
                40.0
            )
        )
    )

    corner_severity = max(
        abs(heading_changes[0]),
        abs(heading_changes[1])
    )


    if corner_severity > 1.0:
        target_speed = 10.0

    elif corner_severity > 0.65:
        target_speed = 13.0

    elif corner_severity > 0.35:
        target_speed = 16.0

    else:
        target_speed = 20.0


    speed_error = (
        target_speed
        - car.speed
    )


    if speed_error >= 0.0:

        throttle = np.clip(
            speed_error / 5.0,
            0.0,
            1.0
        )

    else:

        throttle = np.clip(
            speed_error / 4.0,
            -1.0,
            0.0
        )


    return np.array(
        [
            steering,
            throttle
        ],
        dtype=np.float32
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
    "PPO vs Centerline Baseline"
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
# ENVIRONMENTS
# =========================================================

ppo_env = RacingEnv(
    track=track,
    random_start=False,
    curvature_observation=False
)

baseline_env = RacingEnv(
    track=track,
    random_start=False,
    curvature_observation=False
)


# =========================================================
# RESET
# =========================================================

def reset_race():

    ppo_observation, ppo_info = (
        ppo_env.reset(
            options={
                "start_index": 0
            }
        )
    )

    baseline_observation, baseline_info = (
        baseline_env.reset(
            options={
                "start_index": 0
            }
        )
    )


    ppo_trail = [
        (
            ppo_env.sim.car.x,
            ppo_env.sim.car.y
        )
    ]

    baseline_trail = [
        (
            baseline_env.sim.car.x,
            baseline_env.sim.car.y
        )
    ]


    return (
        ppo_observation,
        ppo_info,
        baseline_observation,
        baseline_info,
        ppo_trail,
        baseline_trail
    )


(
    ppo_observation,
    ppo_info,
    baseline_observation,
    baseline_info,
    ppo_trail,
    baseline_trail
) = reset_race()


ppo_finished = False
baseline_finished = False

ppo_time = None
baseline_time = None

paused = False


# =========================================================
# DRAW TEXT
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
# DRAW TRAILS
# =========================================================

def draw_trail(
    trail,
    color
):

    if len(trail) < 2:
        return


    points = [
        world_to_screen(point)
        for point in trail
    ]


    pygame.draw.lines(
        screen,
        color,
        False,
        points,
        4
    )


# =========================================================
# DRAW CAR
# =========================================================

def draw_car(
    env,
    color
):

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
            [2.3, 0.0],
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
        shape
        @ rotation.T
        + position
    )


    points = [
        world_to_screen(point)
        for point in world_points
    ]


    pygame.draw.polygon(
        screen,
        color,
        points
    )


    pygame.draw.polygon(
        screen,
        (15, 15, 15),
        points,
        2
    )


# =========================================================
# HUD
# =========================================================

def draw_driver_panel(
    y,
    title,
    color,
    env,
    info,
    finished,
    lap_time
):

    x = MAP_WIDTH + 28


    draw_text(
        title,
        font_medium,
        color,
        x,
        y
    )


    draw_text(
        "Speed",
        font_small,
        TEXT_SECONDARY,
        x,
        y + 42
    )


    draw_text(
        f"{env.sim.car.speed:.1f} m/s",
        font_large,
        TEXT_MAIN,
        x,
        y + 60
    )


    draw_text(
        "Progress",
        font_small,
        TEXT_SECONDARY,
        x + 180,
        y + 42
    )


    progress = info.get(
        "progress",
        0.0
    )

    if finished:
        progress = 100.0


    draw_text(
        f"{progress:.1f}%",
        font_large,
        TEXT_MAIN,
        x + 180,
        y + 60
    )


    draw_text(
        "Lap time",
        font_small,
        TEXT_SECONDARY,
        x,
        y + 110
    )


    if lap_time is not None:

        display_time = lap_time

    else:

        display_time = info.get(
            "time",
            0.0
        )


    draw_text(
        f"{display_time:.2f} s",
        font_large,
        TEXT_MAIN,
        x,
        y + 128
    )


    if finished:

        draw_text(
            "FINISHED",
            font_medium,
            SUCCESS,
            x + 180,
            y + 132
        )


# =========================================================
# COMPLETE HUD
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


    draw_text(
        "HEAD-TO-HEAD",
        font_title,
        TEXT_MAIN,
        x,
        30
    )


    draw_text(
        "Same track • same start",
        font_small,
        TEXT_SECONDARY,
        x,
        70
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


    # PPO
    draw_driver_panel(
        125,
        "PPO POLICY",
        PPO_COLOR,
        ppo_env,
        ppo_info,
        ppo_finished,
        ppo_time
    )


    pygame.draw.line(
        screen,
        TEXT_SECONDARY,
        (
            x,
            315
        ),
        (
            SCREEN_WIDTH - 25,
            315
        ),
        1
    )


    # BASELINE
    draw_driver_panel(
        345,
        "CENTERLINE BASELINE",
        BASELINE_COLOR,
        baseline_env,
        baseline_info,
        baseline_finished,
        baseline_time
    )


    pygame.draw.line(
        screen,
        TEXT_SECONDARY,
        (
            x,
            540
        ),
        (
            SCREEN_WIDTH - 25,
            540
        ),
        1
    )


    # ---------------------------------------------
    # RESULT
    # ---------------------------------------------

    draw_text(
        "RACE RESULT",
        font_small,
        TEXT_SECONDARY,
        x,
        570
    )


    if (
        ppo_finished
        and baseline_finished
    ):

        difference = (
            baseline_time
            - ppo_time
        )

        improvement = (
            100.0
            * difference
            / baseline_time
        )


        draw_text(
            "PPO WINS",
            font_large,
            SUCCESS,
            x,
            595
        )


        draw_text(
            f"Time saved: {difference:.2f} s",
            font_normal,
            TEXT_MAIN,
            x,
            640
        )


        draw_text(
            f"Lap-time reduction: "
            f"{improvement:.2f}%",
            font_normal,
            TEXT_MAIN,
            x,
            668
        )

    else:

        draw_text(
            "RACING...",
            font_medium,
            TEXT_MAIN,
            x,
            595
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
            SCREEN_WIDTH - 25,
            710
        ),
        1
    )


    draw_text(
        "SPACE   Pause / resume",
        font_small,
        TEXT_SECONDARY,
        x,
        730
    )

    draw_text(
        "R       Restart race",
        font_small,
        TEXT_SECONDARY,
        x,
        752
    )

    draw_text(
        "ESC     Exit",
        font_small,
        TEXT_SECONDARY,
        x,
        774
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
                    ppo_observation,
                    ppo_info,
                    baseline_observation,
                    baseline_info,
                    ppo_trail,
                    baseline_trail
                ) = reset_race()


                ppo_finished = False
                baseline_finished = False

                ppo_time = None
                baseline_time = None

                paused = False


    # =====================================================
    # SIMULATION
    # =====================================================

    if not paused:

        # ---------------------------------------------
        # PPO
        # ---------------------------------------------

        if not ppo_finished:

            ppo_action, _ = model.predict(
                ppo_observation,
                deterministic=True
            )


            (
                ppo_observation,
                reward,
                terminated,
                truncated,
                ppo_info
            ) = ppo_env.step(
                ppo_action
            )


            ppo_trail.append(
                (
                    ppo_env.sim.car.x,
                    ppo_env.sim.car.y
                )
            )


            if (
                ppo_info.get(
                    "lap_completed",
                    False
                )
                or terminated
                or truncated
            ):

                ppo_finished = True

                ppo_time = float(
                    ppo_info.get(
                        "time",
                        0.0
                    )
                )


        # ---------------------------------------------
        # BASELINE
        # ---------------------------------------------

        if not baseline_finished:

            action = baseline_action(
                baseline_env
            )


            (
                baseline_observation,
                reward,
                terminated,
                truncated,
                baseline_info
            ) = baseline_env.step(
                action
            )


            baseline_trail.append(
                (
                    baseline_env.sim.car.x,
                    baseline_env.sim.car.y
                )
            )


            if (
                baseline_info.get(
                    "lap_completed",
                    False
                )
                or terminated
                or truncated
            ):

                baseline_finished = True

                baseline_time = float(
                    baseline_info.get(
                        "time",
                        0.0
                    )
                )


    # =====================================================
    # DRAW
    # =====================================================

    screen.fill(
        BACKGROUND
    )


    draw_track()


    draw_trail(
        baseline_trail,
        BASELINE_COLOR
    )


    draw_trail(
        ppo_trail,
        PPO_COLOR
    )


    draw_car(
        baseline_env,
        BASELINE_CAR
    )


    draw_car(
        ppo_env,
        PPO_CAR
    )


    draw_hud()


    draw_text(
        "PPO vs Centerline Baseline",
        font_title,
        TEXT_MAIN,
        28,
        20
    )


    draw_text(
        "Red = learned PPO trajectory   •   Blue = non-RL baseline",
        font_small,
        TEXT_SECONDARY,
        28,
        58
    )


    pygame.display.flip()


    clock.tick(
        FPS
    )


# =========================================================
# TERMINAL SUMMARY
# =========================================================

if (
    ppo_time is not None
    and baseline_time is not None
):

    difference = (
        baseline_time
        - ppo_time
    )

    improvement = (
        100.0
        * difference
        / baseline_time
    )


    print("\n")
    print("=" * 60)
    print(" ANIMATED RACE RESULT")
    print("=" * 60)

    print(
        f"PPO:      {ppo_time:.2f}s"
    )

    print(
        f"Baseline: {baseline_time:.2f}s"
    )

    print(
        f"Saved:    {difference:.2f}s"
    )

    print(
        f"Reduction:{improvement:.2f}%"
    )


ppo_env.close()
baseline_env.close()

pygame.quit()