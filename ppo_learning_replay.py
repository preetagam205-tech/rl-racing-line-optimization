
"""Replay real Stable-Baselines3 PPO checkpoints on the original RacingEnv.

Uses existing src/racing_env.py, src/track_geometry.py and src/track_factory.py.
No PPO training or model modification occurs. Tracks change ONLY on N/P.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pygame
from stable_baselines3 import PPO

from src.racing_env import RacingEnv
from src.track_geometry import TrackGeometry
from src.track_factory import (
    make_random_wavy_track,
    make_random_complex_track,
    make_random_spline_track,
    make_random_hard_spline_track,
)

ROOT = Path(__file__).resolve().parent
CHECKPOINTS = [0, 25_000, 50_000, 100_000, 250_000, 500_000]
FAMILIES = ["ellipse", "wavy", "complex", "spline", "hard_spline"]
COLORS = [
    (153, 169, 185), (86, 182, 245), (160, 122, 241),
    (246, 184, 87), (255, 121, 116), (88, 239, 163),
]
DARK, WHITE, MUTED = (12, 20, 31), (235, 243, 251), (151, 172, 193)


def track_points(family: str, index: int) -> np.ndarray:
    """One reproducible track per family; never regenerated during replay."""
    rng = np.random.default_rng(20261115 + 1471 * index)
    if family == "ellipse":
        theta = np.linspace(0, 2 * np.pi, 720, endpoint=False)
        points = np.column_stack((40 * np.cos(theta), 20 * np.sin(theta)))
    else:
        factory = {
            "wavy": make_random_wavy_track,
            "complex": make_random_complex_track,
            "spline": make_random_spline_track,
            "hard_spline": make_random_hard_spline_track,
        }[family]
        points = factory(rng=rng, num_points=720)
    return np.asarray(points, dtype=float)


class PPOReplay:
    def __init__(self, model, checkpoint: int, color):
        self.model = model
        self.checkpoint = checkpoint
        self.color = color
        dims = tuple(model.observation_space.shape)
        if dims not in [(10,), (16,)]:
            raise ValueError(
                f"Checkpoint {checkpoint} expects {dims} observations, "
                "but RacingEnv supports 10 or 16."
            )
        if tuple(model.action_space.shape) != (2,):
            raise ValueError(f"Checkpoint {checkpoint} does not have 2 actions")
        self.curvature = dims == (16,)
        self.reset_data()

    def reset_data(self):
        self.env = None
        self.obs = None
        self.trace = []
        self.done = False
        self.progress = 0.0
        self.steps = 0
        self.lap_time = None
        self.status = "READY"
        self.speed = 0.0

    def start(self, points):
        if self.env is not None:
            self.env.close()
        self.reset_data()
        geom = TrackGeometry(
            centerline_points=points.copy(), track_width=12.0, num_checkpoints=7
        )
        self.env = RacingEnv(track=geom, curvature_observation=self.curvature)
        self.obs, _ = self.env.reset(seed=20261115, options={"start_index": 0})
        car = self.env.sim.car
        self.trace = [(float(car.x), float(car.y))]
        self.status = "DRIVING"

    def step(self):
        if self.done:
            return
        action, _ = self.model.predict(self.obs, deterministic=True)
        self.obs, _, terminated, truncated, info = self.env.step(action)
        car = self.env.sim.car
        self.trace.append((float(car.x), float(car.y)))
        self.steps += 1
        self.progress = max(self.progress, float(info.get("progress", 0.0)))
        self.speed = float(info.get("speed", 0.0))
        self.done = bool(terminated or truncated)
        if self.done:
            if info.get("lap_completed", False):
                self.progress = 100.0
                self.status = "LAP COMPLETE"
                self.lap_time = float(info.get("time", self.steps * 0.05))
            elif truncated:
                self.status = "TIME LIMIT"
            else:
                self.status = "OFF TRACK"

    def pose(self):
        car = self.env.sim.car
        return float(car.x), float(car.y), float(car.heading)


def save_results(all_replays: dict):
    data = {
        "description": "Deterministic PPO checkpoint replay; one run per map/checkpoint",
        "track_seed": 20261115,
        "results": {},
    }
    for family, replays in all_replays.items():
        data["results"][family] = [
            {
                "checkpoint_steps": r.checkpoint,
                "observation_size": 16 if r.curvature else 10,
                "progress_percent": round(r.progress, 3),
                "lap_completed": r.status == "LAP COMPLETE",
                "lap_time_seconds": r.lap_time,
                "status": r.status,
                "simulated_seconds": round(r.steps * 0.05, 2),
            }
            for r in replays
        ]
    path = ROOT / "assets" / "ppo_learning_replay_results.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Saved {path}")
    return path


def run():
    print("Loading real PPO checkpoints (no training performed)...")
    base_replays = []
    for j, step in enumerate(CHECKPOINTS):
        filename = ROOT / "models" / "visual_training" / f"ppo_visual_{step:06d}.zip"
        if not filename.exists():
            raise FileNotFoundError(f"Missing PPO checkpoint: {filename}")
        model = PPO.load(str(filename), device="cpu")
        base_replays.append((model, step, COLORS[j]))
        print(f"  {filename.name}: observation shape {model.observation_space.shape}")

    # Cached map geometry and replay state. Switching maps never changes tracks.
    points_by_family = {}
    all_replays = {}

    def get_replays(index):
        family = FAMILIES[index]
        if family not in points_by_family:
            points_by_family[family] = track_points(family, index)
        if family not in all_replays:
            racers = [PPOReplay(m, step, col) for m, step, col in base_replays]
            for racer in racers:
                racer.start(points_by_family[family])
            all_replays[family] = racers
        return all_replays[family]

    pygame.init()
    screen = pygame.display.set_mode((1240, 740), pygame.RESIZABLE)
    pygame.display.set_caption("Real PPO Learning | Checkpoint Replay")
    clock = pygame.time.Clock()
    fonts = {
        "sm": pygame.font.SysFont("arial", 14),
        "md": pygame.font.SysFont("arial", 18, bold=True),
        "lg": pygame.font.SysFont("arial", 26, bold=True),
    }
    index, selected = 0, 5
    paused, fast, show_all = False, False, True

    def label(text, x, y, color=WHITE, size="sm"):
        screen.blit(fonts[size].render(str(text), True, color), (int(x), int(y)))

    def poly(points, origin, scale):
        return [(int(origin[0] + a * scale), int(origin[1] - b * scale))
                for a, b in points]

    def draw_f1(racer, origin, scale):
        x, y, angle = racer.pose()
        dx, dy = math.cos(angle), math.sin(angle)
        nx, ny = -dy, dx

        def local(f, s):
            return (x + dx * f + nx * s, y + dy * f + ny * s)

        def polygon(coords, color):
            pygame.draw.polygon(screen, color,
                                poly([local(f, s) for f, s in coords], origin, scale))

        black = (12, 16, 21)
        polygon([(-1.8,-1.13), (-1.5,-1.13),(-1.5,1.13),(-1.8,1.13)], black)
        polygon([(1.35,-1.25),(1.65,-1.25),(1.65,1.25),(1.35,1.25)], black)
        for side in (-1, 1):
            polygon([(-1.3,side*.66),(-.65,side*.66),(-.65,side*1.13),(-1.3,side*1.13)], black)
            polygon([(.55,side*.66),(1.2,side*.66),(1.2,side*1.13),(.55,side*1.13)], black)
        polygon([(-1.5,-.38),(-.9,-.56),(-.2,-.5),(.5,-.28),(1.7,-.12),
                 (1.8,0),(1.7,.12),(.5,.28),(-.2,.5),(-.9,.56),(-1.5,.38)], racer.color)
        polygon([(-.43,-.27),(.25,-.27),(.39,0),(.25,.27),(-.43,.27)], black)
        pygame.draw.circle(screen, WHITE, poly([local(-.05,0)],origin,scale)[0], max(2,int(scale*.14)))

    def draw_track(points, W, H):
        area = (25, 88, max(370, W-375), H-155)
        x0, y0, ww, hh = area
        lo = points.min(axis=0) - 13
        hi = points.max(axis=0) + 13
        scale = min((ww-35) / (hi[0]-lo[0]), (hh-35)/(hi[1]-lo[1]))
        center = (lo + hi) / 2
        origin = (x0+ww/2-center[0]*scale, y0+hh/2+center[1]*scale)
        directions = np.roll(points,-1,axis=0)-np.roll(points,1,axis=0)
        directions /= np.maximum(1e-9,np.linalg.norm(directions,axis=1))[:,None]
        # All five generators produce counterclockwise circuits.
        normals = np.column_stack((directions[:,1],-directions[:,0]))
        pygame.draw.polygon(screen,(54,68,82),poly(points+normals*6,origin,scale))
        pygame.draw.polygon(screen,DARK,poly(points-normals*6,origin,scale))
        pygame.draw.lines(screen,(214,228,236),True,poly(points+normals*6,origin,scale),2)
        pygame.draw.lines(screen,(214,228,236),True,poly(points-normals*6,origin,scale),2)
        for k in range(0,len(points),18):
            pygame.draw.line(screen,(104,122,141),
                             poly([points[k],points[(k+8)%len(points)]],origin,scale)[0],
                             poly([points[k],points[(k+8)%len(points)]],origin,scale)[1],1)
        start = poly([points[0]+normals[0]*6,points[0]-normals[0]*6],origin,scale)
        pygame.draw.line(screen,(253,214,107),start[0],start[1],3)
        return origin,scale

    running = True
    try:
        while running:
            clock.tick(60)
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                elif e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_ESCAPE:
                        running = False
                    elif e.key == pygame.K_n:
                        index = (index + 1) % len(FAMILIES)
                    elif e.key == pygame.K_p:
                        index = (index - 1) % len(FAMILIES)
                    elif pygame.K_1 <= e.key <= pygame.K_5:
                        index = e.key - pygame.K_1
                    elif e.key == pygame.K_SPACE:
                        paused = not paused
                    elif e.key == pygame.K_f:
                        fast = not fast
                    elif e.key == pygame.K_TAB:
                        show_all = not show_all
                    elif e.key == pygame.K_UP:
                        selected = (selected - 1) % len(CHECKPOINTS)
                    elif e.key == pygame.K_DOWN:
                        selected = (selected + 1) % len(CHECKPOINTS)
                    elif e.key == pygame.K_r:
                        family = FAMILIES[index]
                        for racer in get_replays(index):
                            racer.start(points_by_family[family])
                    elif e.key == pygame.K_s:
                        save_results(all_replays)

            family = FAMILIES[index]
            racers = get_replays(index)
            if not paused:
                for _ in range(6 if fast else 2):
                    for racer in racers:
                        racer.step()

            W,H = screen.get_size()
            screen.fill(DARK)
            label("RACING LINE  /  REAL PPO TRAINING CHECKPOINTS",25,19,size="lg")
            label("Saved policy snapshots - not new PPO training | N/P manual map switching",25,52,MUTED)
            points = points_by_family[family]
            origin,scale = draw_track(points,W,H)
            visible = racers if show_all else [racers[selected]]
            for racer in visible:
                if len(racer.trace) > 1:
                    pygame.draw.lines(screen,racer.color,False,
                                      poly(racer.trace,origin,scale),2)
                draw_f1(racer,origin,scale)
            sx = W - 340
            pygame.draw.rect(screen,(23,34,49),(sx-14,86,328,H-157),border_radius=12)
            label(f"MAP {index+1}/5  {family.upper()}",sx,105,(253,211,102),"md")
            label("N NEXT   P PREVIOUS   1-5 DIRECT",sx,139,MUTED)
            label("Each car = one saved PPO checkpoint",sx,168,MUTED)
            label("STEPS        PROGRESS        RESULT",sx,201,MUTED)
            for j,racer in enumerate(racers):
                yy = 240 + j*57
                label(f"{'>' if selected == j else ' '} {racer.checkpoint:>6,}",sx,yy,racer.color,"md")
                label(f"{racer.progress:5.1f}%",sx+130,yy,WHITE,"md")
                label(racer.status[:13],sx+212,yy,racer.color)
                if racer.lap_time is not None:
                    label(f"Lap: {racer.lap_time:.2f}s",sx+130,yy+24,MUTED)
                else:
                    label(f"Run: {racer.steps*.05:.1f}s",sx+130,yy+24,MUTED)
            label("UP/DOWN select checkpoint",sx,H-182,MUTED)
            label("TAB toggle all/single  R replay",sx,H-155,MUTED)
            label("SPACE pause  F fast  S save JSON",sx,H-128,MUTED)
            label("FAST" if fast else "NORMAL SPEED",sx,H-101,(253,211,102))
            pygame.draw.rect(screen,(23,34,49),(20,H-57,W-40,40),border_radius=9)
            label("Checkpoint colors show training snapshots; each policy drives separately with the ORIGINAL PPO physics.",35,H-44,MUTED)
            pygame.display.flip()
    finally:
        save_results(all_replays)
        for replays in all_replays.values():
            for racer in replays:
                if racer.env is not None:
                    racer.env.close()
        pygame.quit()


if __name__ == "__main__":
    run()
