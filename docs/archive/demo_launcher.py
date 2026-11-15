from pathlib import Path
import subprocess
import sys

import pygame


# =========================================================
# PATHS
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parent


DEMOS = [
    {
        "key": "1",
        "title": "Final PPO Lap",
        "subtitle": "Watch the frozen final policy drive a learned racing line",
        "script": "visualize_final_policy.py",
    },
    {
        "key": "2",
        "title": "Training Evolution",
        "subtitle": "Watch PPO policies improve across training checkpoints",
        "script": "visualize_training_evolution.py",
    },
    {
        "key": "3",
        "title": "PPO vs Baseline",
        "subtitle": "Head-to-head race against the centerline controller",
        "script": "visualize_ppo_vs_baseline_race.py",
    },
    {
        "key": "4",
        "title": "Generalization",
        "subtitle": "Same frozen PPO model across multiple track geometries",
        "script": "visualize_generalization.py",
    },
]


# =========================================================
# DISPLAY
# =========================================================

WIDTH = 1000
HEIGHT = 720

BACKGROUND = (18, 20, 24)
CARD = (30, 33, 39)
CARD_HOVER = (42, 46, 54)

TEXT_MAIN = (242, 242, 242)
TEXT_SECONDARY = (165, 172, 184)

ACCENT = (255, 90, 70)
ACCENT_SECONDARY = (90, 170, 255)

SUCCESS = (80, 220, 130)

pygame.init()

screen = pygame.display.set_mode(
    (WIDTH, HEIGHT)
)

pygame.display.set_caption(
    "RL Racing-Line Generation"
)

clock = pygame.time.Clock()


font_title = pygame.font.SysFont(
    "Arial",
    42,
    bold=True
)

font_subtitle = pygame.font.SysFont(
    "Arial",
    19
)

font_card = pygame.font.SysFont(
    "Arial",
    24,
    bold=True
)

font_card_subtitle = pygame.font.SysFont(
    "Arial",
    16
)

font_number = pygame.font.SysFont(
    "Arial",
    25,
    bold=True
)

font_small = pygame.font.SysFont(
    "Arial",
    14
)


# =========================================================
# HELPERS
# =========================================================

def draw_text(
    text,
    font,
    color,
    x,
    y,
):

    surface = font.render(
        text,
        True,
        color
    )

    screen.blit(
        surface,
        (x, y)
    )


def launch_demo(script_name):

    script_path = (
        PROJECT_DIR
        / script_name
    )

    if not script_path.exists():

        print(
            f"Missing demo script: "
            f"{script_path}"
        )

        return


    print("\n")
    print("=" * 65)

    print(
        f"Launching: "
        f"{script_name}"
    )

    print("=" * 65)


    # sys.executable ensures the same .venv Python is used
    subprocess.run(
        [
            sys.executable,
            str(script_path)
        ],
        cwd=PROJECT_DIR
    )


# =========================================================
# CARD LAYOUT
# =========================================================

CARD_X = 120
CARD_WIDTH = 760
CARD_HEIGHT = 92
CARD_GAP = 18
CARD_START_Y = 225


def get_card_rect(index):

    y = (
        CARD_START_Y
        + index
        * (
            CARD_HEIGHT
            + CARD_GAP
        )
    )

    return pygame.Rect(
        CARD_X,
        y,
        CARD_WIDTH,
        CARD_HEIGHT
    )


# =========================================================
# MAIN LOOP
# =========================================================

running = True


while running:

    mouse_position = pygame.mouse.get_pos()


    # ---------------------------------------------
    # EVENTS
    # ---------------------------------------------

    for event in pygame.event.get():

        if event.type == pygame.QUIT:

            running = False


        elif event.type == pygame.KEYDOWN:

            if event.key == pygame.K_ESCAPE:

                running = False


            elif event.key == pygame.K_1:

                launch_demo(
                    DEMOS[0]["script"]
                )


            elif event.key == pygame.K_2:

                launch_demo(
                    DEMOS[1]["script"]
                )


            elif event.key == pygame.K_3:

                launch_demo(
                    DEMOS[2]["script"]
                )


            elif event.key == pygame.K_4:

                launch_demo(
                    DEMOS[3]["script"]
                )


        elif (
            event.type
            == pygame.MOUSEBUTTONDOWN
            and event.button == 1
        ):

            for index, demo in enumerate(
                DEMOS
            ):

                rect = get_card_rect(
                    index
                )

                if rect.collidepoint(
                    event.pos
                ):

                    launch_demo(
                        demo["script"]
                    )


    # ---------------------------------------------
    # DRAW
    # ---------------------------------------------

    screen.fill(
        BACKGROUND
    )


    # =====================================================
    # HEADER
    # =====================================================

    draw_text(
        "RL Racing-Line Generation",
        font_title,
        TEXT_MAIN,
        120,
        55
    )


    draw_text(
        "Reinforcement learning for fast racing-line and speed-control behaviour",
        font_subtitle,
        TEXT_SECONDARY,
        122,
        112
    )


    # ---------------------------------------------
    # PROJECT RESULT
    # ---------------------------------------------

    pygame.draw.line(
        screen,
        (65, 69, 78),
        (120, 160),
        (880, 160),
        1
    )


    draw_text(
        "FINAL PPO",
        font_small,
        TEXT_SECONDARY,
        120,
        177
    )

    draw_text(
        "16.45 s",
        font_card,
        ACCENT,
        205,
        168
    )


    draw_text(
        "vs",
        font_small,
        TEXT_SECONDARY,
        315,
        177
    )


    draw_text(
        "20.55 s baseline",
        font_card,
        ACCENT_SECONDARY,
        350,
        168
    )


    draw_text(
        "19.95% lap-time reduction",
        font_card_subtitle,
        SUCCESS,
        655,
        175
    )


    # =====================================================
    # DEMO CARDS
    # =====================================================

    for index, demo in enumerate(
        DEMOS
    ):

        rect = get_card_rect(
            index
        )

        hovered = rect.collidepoint(
            mouse_position
        )


        card_color = (
            CARD_HOVER
            if hovered
            else CARD
        )


        pygame.draw.rect(
            screen,
            card_color,
            rect,
            border_radius=12
        )


        # Number
        number_center = (
            rect.x + 42,
            rect.centery
        )


        pygame.draw.circle(
            screen,
            ACCENT,
            number_center,
            23
        )


        number_surface = (
            font_number.render(
                demo["key"],
                True,
                TEXT_MAIN
            )
        )


        number_rect = (
            number_surface.get_rect(
                center=number_center
            )
        )


        screen.blit(
            number_surface,
            number_rect
        )


        # Title
        draw_text(
            demo["title"],
            font_card,
            TEXT_MAIN,
            rect.x + 85,
            rect.y + 18
        )


        # Description
        draw_text(
            demo["subtitle"],
            font_card_subtitle,
            TEXT_SECONDARY,
            rect.x + 85,
            rect.y + 53
        )


        # Arrow
        draw_text(
            "→",
            font_card,
            (
                ACCENT
                if hovered
                else TEXT_SECONDARY
            ),
            rect.right - 48,
            rect.y + 30
        )


    # =====================================================
    # FOOTER
    # =====================================================

    draw_text(
        "Press 1–4 or click a demo",
        font_small,
        TEXT_SECONDARY,
        120,
        675
    )


    draw_text(
        "ESC  Exit",
        font_small,
        TEXT_SECONDARY,
        820,
        675
    )


    pygame.display.flip()

    clock.tick(
        60
    )


pygame.quit()