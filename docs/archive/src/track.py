import pygame
import numpy as np
import json
import csv
from pathlib import Path
from datetime import datetime

from vehicle import Vehicle
from track_geometry import TrackGeometry

pygame.init()

# Window settings
WIDTH, HEIGHT = 800, 600

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("RL Racing Simulator")

clock = pygame.time.Clock()
# Create shared track geometry
track = TrackGeometry()

print("Shared track loaded successfully!")
# Font for vehicle information
font = pygame.font.Font(None, 28)

# Colors
GRASS = (34, 115, 40)
ROAD = (55, 55, 55)
WHITE = (245, 245, 245)
RED = (215, 35, 35)
YELLOW = (245, 205, 45)
BLUE = (30, 150, 255)

# -------------------------
# SHARED TRACK GEOMETRY
# -------------------------

# Screen scaling
scale = 6

# Track parameters
a = track.a
b = track.b

track_width = track.track_width
num_points = track.num_points
num_checkpoints = track.num_checkpoints

# Centerline and boundaries
centerline = track.centerline

outer_boundary = track.outer_boundary
inner_boundary = track.inner_boundary

# Track segments
segment_starts = track.segment_starts
segment_ends = track.segment_ends
segment_vectors = track.segment_vectors
segment_lengths = track.segment_lengths

# Distance calculations
cumulative_distances = track.cumulative_distances
total_track_length = track.total_track_length

# Checkpoints
checkpoint_gates = track.checkpoint_gates

print(
    f"Shared track length: {total_track_length:.2f} units"
)

print(
    f"Shared checkpoints: {len(checkpoint_gates)}"
)


# Convert world coordinates to Pygame coordinates
def to_screen(points):

    result = []

    for x, y in points:

        sx = int(WIDTH / 2 + x * scale)
        sy = int(HEIGHT / 2 - y * scale)

        result.append((sx, sy))

    return result


center = to_screen(centerline)
outer = to_screen(outer_boundary)
inner = to_screen(inner_boundary)

# -------------------------
# TRACK BOUNDARY DETECTION
# -------------------------

def is_on_track(car_x, car_y):

    return track.is_on_track(
        car_x,
        car_y
    )

# -------------------------
# TRACK PROGRESS DETECTION
# -------------------------

def calculate_progress(car_x, car_y):

    return track.calculate_progress(
        car_x,
        car_y
    )
# -------------------------
# FINISH LINE DETECTION
# -------------------------

def crossed_finish_forward(previous_pos, current_pos):

    # Finish line endpoints
    gate_start = inner_boundary[0]
    gate_end = outer_boundary[0]

    # Car movement vector
    movement = current_pos - previous_pos

    # Finish line vector
    gate = gate_end - gate_start

    # 2D cross product
    def cross2d(a, b):
        return a[0] * b[1] - a[1] * b[0]

    denominator = cross2d(movement, gate)

    # No crossing if movement is parallel to gate
    if abs(denominator) < 1e-9:
        return False

    relative = gate_start - previous_pos

    t = cross2d(relative, gate) / denominator
    u = cross2d(relative, movement) / denominator

    # Check whether the movement segment
    # intersects the actual finish line
    intersects = (
        0 < t <= 1
        and 0 <= u <= 1
    )

    # Direction of the track at the finish
    track_direction = (
        segment_vectors[0] / segment_lengths[0]
    )

    # Check that the car moves forward
    moving_forward = (
        np.dot(movement, track_direction) > 0
    )

    return intersects and moving_forward

# -------------------------
# CREATE VEHICLE
# -------------------------

car_position = centerline[0]
next_position = centerline[1]

direction = next_position - car_position

car_heading = np.arctan2(
    direction[1],
    direction[0]
)

car = Vehicle(
    x=car_position[0],
    y=car_position[1],
    heading=car_heading
)

# -------------------------
# CHECKPOINT TRACKING
# -------------------------

# The next checkpoint the car must cross
next_checkpoint = 1

# Remember the car's previous track progress
previous_progress = calculate_progress(
    car.x,
    car.y
)

# Remember whether the car was on the road
previous_on_track = is_on_track(
    car.x,
    car.y
)

# Whether this attempt is still valid
run_valid = True
# Number of completed laps
lap_count = 0

# -------------------------
# LAP TIMER VARIABLES
# -------------------------

# Time elapsed in the current lap
lap_elapsed = 0.0

lap_started = False

# Time of the most recently completed lap
last_lap_time = None

# Fastest completed lap
best_lap_time = None

# -------------------------
# RACING DATA STORAGE
# -------------------------

# Data for the current lap
trajectory = []

# Data for completed laps
completed_laps = []

# -------------------------
# CSV STORAGE SETUP
# -------------------------

# Find the main project directory
project_dir = Path(__file__).resolve().parent.parent

# Create results folder
results_dir = project_dir / "results"

results_dir.mkdir(
    parents=True,
    exist_ok=True
)

# -------------------------
# LOAD PERSONAL BEST
# -------------------------

best_file = results_dir / "personal_best.json"

# Load previous best if the file exists
if best_file.exists():

    try:
        with open(best_file, "r") as file:
            data = json.load(file)

        best_lap_time = float(data["best_lap_time"])

        print(
            f"Loaded personal best: {best_lap_time:.2f} seconds"
        )

    except (OSError, ValueError, KeyError, TypeError):
        best_lap_time = None
        print("Could not load personal best.")

else:

    best_lap_time = None
    print("No previous personal best found.")

# Unique identifier for this session
session_id = datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)

print("Results folder:", results_dir)


# -------------------------
# DRAW TRACK
# -------------------------

def draw_track():

    screen.fill(GRASS)

    pygame.draw.polygon(screen, ROAD, outer)
    pygame.draw.polygon(screen, GRASS, inner)

    # White boundaries
    pygame.draw.aalines(screen, WHITE, True, outer)
    pygame.draw.aalines(screen, WHITE, True, inner)

    # Red and white kerbs
    for i in range(0, num_points, 10):

        j = (i + 10) % num_points

        color = RED if (i // 10) % 2 == 0 else WHITE

        pygame.draw.line(
            screen, color,
            outer[i], outer[j], 5
        )

        pygame.draw.line(
            screen, color,
            inner[i], inner[j], 5
        )

    # Dashed centerline
    for i in range(0, num_points, 24):

        j = (i + 12) % num_points

        pygame.draw.line(
            screen,
            YELLOW,
            center[i],
            center[j],
            2
        )

    # -------------------------
    # DRAW CHECKPOINTS
    # -------------------------

    for number, gate in enumerate(
        checkpoint_gates,
        start=1
    ):

        inner_point, outer_point = gate

        # Convert checkpoint endpoints to pixels
        gate_points = to_screen([
            inner_point,
            outer_point
        ])

        # Draw cyan checkpoint line
        pygame.draw.line(
            screen,
            (0, 220, 255),
            gate_points[0],
            gate_points[1],
            3
        )

        # Calculate checkpoint label position
        midpoint = (
            np.array(gate_points[0])
            + np.array(gate_points[1])
        ) / 2

        # Draw checkpoint number
        label = font.render(
            str(number),
            True,
            WHITE
        )

        label_rect = label.get_rect(
            center=(int(midpoint[0]), int(midpoint[1]))
        )

        screen.blit(label, label_rect)

    # Start / finish line
    pygame.draw.line(
        screen,
        WHITE,
        inner[0],
        outer[0],
        8
    )


# -------------------------
# DRAW VEHICLE
# -------------------------

def draw_car():

    car_screen = np.array(
        to_screen([(car.x, car.y)])[0]
    )

    forward = np.array([
        np.cos(car.heading),
        -np.sin(car.heading)
    ])

    side = np.array([
        -forward[1],
        forward[0]
    ])

    front_left = car_screen + forward * 12 + side * 6
    front_right = car_screen + forward * 12 - side * 6

    rear_left = car_screen - forward * 12 + side * 6
    rear_right = car_screen - forward * 12 - side * 6

    # Car body
    pygame.draw.polygon(
        screen,
        BLUE,
        [
            front_left,
            front_right,
            rear_right,
            rear_left
        ]
    )

    # Windshield
    pygame.draw.line(
        screen,
        (180, 230, 255),
        car_screen + forward * 5 + side * 5,
        car_screen + forward * 5 - side * 5,
        3
    )

# -------------------------
# VEHICLE DASHBOARD
# -------------------------

def draw_dashboard():

    # Dashboard background
    pygame.draw.rect(
        screen,
        (25, 25, 25),
        (10, 10, 260, 120),
        border_radius=8
    )

    # Vehicle information
    speed_text = font.render(
        f"Speed: {car.speed:.1f} units/s",
        True,
        WHITE
    )

    position_text = font.render(
        f"Position: ({car.x:.1f}, {car.y:.1f})",
        True,
        WHITE
    )

    heading_text = font.render(
        f"Heading: {np.degrees(car.heading) % 360:.1f} deg",
        True,
        WHITE
    )

    # Calculate track progress
    progress = calculate_progress(car.x, car.y)

    progress_text = font.render(
        f"Progress: {progress:.1f}%",
        True,
        YELLOW
    )

    # Display text
    screen.blit(speed_text, (20, 17))
    screen.blit(position_text, (20, 44))
    screen.blit(heading_text, (20, 71))
    screen.blit(progress_text, (20, 98))

# -------------------------
# TRACK STATUS DISPLAY
# -------------------------

def draw_track_status():

    on_track = is_on_track(car.x, car.y)

    if on_track:
        status = "ON TRACK"
        color = (0, 200, 0)

    else:
        status = "OFF TRACK"
        color = (255, 50, 50)

    # Draw status background
    pygame.draw.rect(
        screen,
        (25, 25, 25),
        (WIDTH - 190, 10, 180, 40),
        border_radius=8
    )

    # Create text
    status_text = font.render(
        status,
        True,
        color
    )

    # Display text
    screen.blit(
        status_text,
        (WIDTH - 175, 20)
    )

# -------------------------
# CHECKPOINT DASHBOARD
# -------------------------

def draw_checkpoint_status():

    if not run_valid:

        text = "INVALID RUN - PRESS R"
        color = RED

    elif next_checkpoint > num_checkpoints:

        text = "ALL CHECKPOINTS PASSED"
        color = (0, 255, 0)

    else:

        text = (
            f"Next: CP{next_checkpoint}"
            f" / {num_checkpoints}"
        )

        color = YELLOW

    # Background
    pygame.draw.rect(
        screen,
        (25, 25, 25),
        (WIDTH - 270, 60, 260, 40),
        border_radius=8
    )

    # Text
    status_text = font.render(
        text,
        True,
        color
    )

    screen.blit(
        status_text,
        (WIDTH - 255, 70)
    )

        # Lap counter background
    pygame.draw.rect(
        screen,
        (25, 25, 25),
        (WIDTH - 270, 110, 260, 40),
        border_radius=8
    )

    # Display completed laps
    lap_text = font.render(
        f"Laps: {lap_count}",
        True,
        WHITE
    )

    screen.blit(
        lap_text,
        (WIDTH - 255, 120)
    )

# -------------------------
# FORMAT LAP TIME
# -------------------------

def format_lap_time(seconds):

    if seconds is None:
        return "--:--.--"

    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60

    return f"{minutes:02d}:{remaining_seconds:05.2f}"


# -------------------------
# DRAW LAP TIMER
# -------------------------

def draw_lap_timer():

    # Dashboard background
    pygame.draw.rect(
        screen,
        (25, 25, 25),
        (10, HEIGHT - 120, 270, 110),
        border_radius=8
    )

    # Format timer values
    current_text = format_lap_time(lap_elapsed)
    last_text = format_lap_time(last_lap_time)
    best_text = format_lap_time(best_lap_time)

    # Render text
    current_surface = font.render(
        f"Current: {current_text}",
        True,
        YELLOW
    )

    last_surface = font.render(
        f"Last: {last_text}",
        True,
        WHITE
    )

    best_surface = font.render(
        f"Best: {best_text}",
        True,
        (0, 255, 0)
    )

    # Display text
    screen.blit(
        current_surface,
        (20, HEIGHT - 110)
    )

    screen.blit(
        last_surface,
        (20, HEIGHT - 78)
    )

    screen.blit(
        best_surface,
        (20, HEIGHT - 46)
    )



# -------------------------
# MAIN GAME LOOP
# -------------------------

running = True

while running:

    # Time since previous frame
    dt = min(clock.tick(60) / 1000.0, 0.05)

    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            running = False

        # Reset car when R is pressed
        elif event.type == pygame.KEYDOWN:

            if event.key == pygame.K_r:
                # Clear data from the interrupted lap
                trajectory = []

                # Reset vehicle
                car.reset(
                    car_position[0],
                    car_position[1],
                    car_heading
                )

                # Reset checkpoint tracking
                next_checkpoint = 1
                lap_count = 0
                # Reset lap timing
                lap_elapsed = 0.0
                last_lap_time = None
                lap_started = False

                previous_progress = calculate_progress(
                    car.x,
                    car.y
                )

                previous_on_track = True

                run_valid = True

                print("Car and checkpoints reset!")
    # Read keyboard
    keys = pygame.key.get_pressed()

    acceleration = 0
    steering = 0

    # Accelerate
    if keys[pygame.K_w]:
        acceleration = 6

    # Brake
    if keys[pygame.K_s]:
        acceleration = -10

    # Steer left
    if keys[pygame.K_a]:
        steering = car.max_steering

    # Steer right
    if keys[pygame.K_d]:
        steering = -car.max_steering

        # Remember position before moving
    previous_car_pos = np.array([
        car.x,
        car.y
    ])

    # Update vehicle physics
    car.update(
        acceleration,
        steering,
        dt
    )

    # -------------------------
    # CHECKPOINT DETECTION
    # -------------------------

    current_progress = calculate_progress(
        car.x,
        car.y
    )

    current_on_track = is_on_track(
        car.x,
        car.y
    )

    # Leaving the track invalidates this attempt
    if not current_on_track:
        run_valid = False

    # -------------------------
    # UPDATE LAP TIMER
    # -------------------------

    # Start timing when the car first moves
    if run_valid and not lap_started and car.speed > 0:
        lap_started = True

    # Update timer after the lap has started
    if run_valid and lap_started:
        lap_elapsed += dt

    # -------------------------
    # RECORD VEHICLE DATA
    # -------------------------

    if run_valid and lap_started:

        data = {
            "time": lap_elapsed,
            "x": float(car.x),
            "y": float(car.y),
            "speed": float(car.speed),
            "heading": float(car.heading),
            "steering": float(steering),
            "acceleration": float(acceleration),
            "progress": float(current_progress)
        }

        trajectory.append(data)
    # Detect checkpoints in the correct order
    if (
        run_valid
        and previous_on_track
        and current_on_track
        and next_checkpoint <= num_checkpoints
    ):

        # Progress percentage of the next checkpoint
        checkpoint_progress = (
            next_checkpoint / (num_checkpoints + 1)
        ) * 100

        # How much progress changed this frame
        progress_change = (
            current_progress - previous_progress
        )

        # Detect a forward crossing
        crossed = (
            0 < progress_change < 5
            and previous_progress < checkpoint_progress
            <= current_progress
        )

        if crossed:

            print(
                f"Checkpoint {next_checkpoint} passed!"
            )

            next_checkpoint += 1

    # -------------------------
    # LAP COMPLETION
    # -------------------------

    current_car_pos = np.array([
        car.x,
        car.y
    ])

    # Check whether a valid lap was completed
    if (
        run_valid
        and previous_on_track
        and current_on_track
        and next_checkpoint > num_checkpoints
        and crossed_finish_forward(
            previous_car_pos,
            current_car_pos
        )
    ):

        # One full lap completed
        lap_count += 1

        # Store the trajectory in memory
        completed_laps.append(trajectory.copy())

        # Create a unique CSV filename
        filename = (
            f"lap_{session_id}_{lap_count:03d}.csv"
        )

        filepath = results_dir / filename

        # Column names for our dataset
        fieldnames = [
            "time",
            "x",
            "y",
            "speed",
            "heading",
            "steering",
            "acceleration",
            "progress"
        ]

        # Write the trajectory to CSV
        with open(
            filepath,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames
            )

            writer.writeheader()

            writer.writerows(trajectory)

        print(
            f"Recorded {len(trajectory)} data points"
        )

        print(
            f"Lap data saved to: {filepath}"
        )

        # Clear trajectory for the next lap
        trajectory = []
        # Record completed lap time
        last_lap_time = lap_elapsed

        # Update personal best
        if (
            best_lap_time is None
            or last_lap_time < best_lap_time
        ):

            best_lap_time = last_lap_time

            # Save the new record
            data = {
                "best_lap_time": best_lap_time
            }

            with open(best_file, "w") as file:
                json.dump(data, file, indent=4)

            print("NEW PERSONAL BEST!")
            print("Personal best saved!")

        print(
            f"Lap {lap_count} completed in "
            f"{last_lap_time:.2f} seconds!"
        )

        # Reset timer for the next lap
        lap_elapsed = 0.0

        # Restart checkpoint tracking
        next_checkpoint = 1
    
    # Save current values for the next frame
    previous_progress = current_progress
    previous_on_track = current_on_track

    # Draw everything
    # Draw everything
    draw_track()
    draw_car()
    draw_dashboard()
    draw_track_status()
    draw_checkpoint_status()
    draw_lap_timer()

    pygame.display.flip()
pygame.quit()