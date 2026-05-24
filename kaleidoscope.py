"""
kaleidoscope.py
════════════════════════════════════════════════════════════════════════════════
A visually stunning, interactive kaleidoscope rendered with Pygame + NumPy.

Controls
────────
  SPACE       — next pattern (manual)
  A           — toggle auto-cycle mode (cycles every AUTO_CYCLE_SECS seconds)
  ↑ / ↓       — increase / decrease rotation speed
  → / ←       — add / remove mirror slices
  F           — toggle fade-blend between patterns
  S           — save screenshot (PNG)
  ESC / Q     — quit

Patterns
────────
  0  Plasma        — psychedelic sine-noise field
  1  Rainbow       — smooth rolling gradient
  2  Splatter      — neon paint-blob burst
  3  Swirl         — concentric radial ripples
  4  Stripes       — shifting neon lattice
  5  Mandala       — polar-rose / petal geometry
  6  Lava          — molten colour blobs
  7  Starfield     — zooming star tunnel
  8  Crystals      — angular Voronoi-ish shards
  9  Aurora        — northern-lights curtain bands
════════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import math
import os
import random
import sys
import time
from typing import Callable, List

import numpy as np
import pygame

# ─────────────────────────────────────────────
# CONFIG  (feel free to tweak)
# ─────────────────────────────────────────────
WIDTH, HEIGHT       = 800, 800
FPS                 = 60
SLICES              = 12          # mirror segments
ROT_SPEED           = 0.5         # degrees per frame
T_STEP              = 0.03        # time increment per frame
AUTO_CYCLE_SECS     = 6.0         # seconds per pattern in auto-cycle mode
FADE_FRAMES         = 45          # blend frames between patterns
SCREENSHOT_DIR      = "screenshots"

# ─────────────────────────────────────────────
# PATTERN GENERATORS  (all return uint8 HxWx3)
# ─────────────────────────────────────────────

def pattern_plasma(t: float) -> np.ndarray:
    """Psychedelic plasma noise."""
    x = np.linspace(0, 6, WIDTH)
    y = np.linspace(0, 6, HEIGHT)
    X, Y = np.meshgrid(x, y)
    Z = (np.sin(X + t) + np.sin(Y * 1.3 + t * 1.2)
         + np.sin((X + Y) * 0.7 + t * 0.5))
    Z = (Z - Z.min()) / (Z.max() - Z.min() + 1e-9)
    r = np.uint8(255 * np.abs(np.sin(np.pi * Z + t * 0.3)))
    g = np.uint8(255 * np.abs(np.sin(np.pi * Z + t * 0.3 + 2.09)))
    b = np.uint8(255 * np.abs(np.sin(np.pi * Z + t * 0.3 + 4.19)))
    return np.stack([r, g, b], axis=2)


def pattern_rainbow(t: float) -> np.ndarray:
    """Smooth rolling hue gradient."""
    x = np.linspace(0, 1, WIDTH)
    y = np.linspace(0, 1, HEIGHT)
    X, Y = np.meshgrid(x, y)
    R = ((np.sin(8 * X + t) + 1) * 127).astype(np.uint8)
    G = ((np.sin(8 * Y + t * 1.5) + 1) * 127).astype(np.uint8)
    B = ((np.sin(8 * (X + Y) + t * 0.7) + 1) * 127).astype(np.uint8)
    return np.stack([R, G, B], axis=2)


def pattern_splatter(t: float) -> np.ndarray:
    """Neon paint-blob burst — seeded by time for smooth drift."""
    rng = np.random.default_rng(int(t * 20) & 0xFFFF)
    img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    n = 120
    cx = rng.integers(0, WIDTH,  n)
    cy = rng.integers(0, HEIGHT, n)
    radii = rng.integers(15, 70, n)
    colors = rng.integers(80, 255, (n, 3), dtype=np.uint8)
    # neon bias: at least one channel is bright
    colors[np.arange(n), rng.integers(0, 3, n)] = 255
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH]
    for i in range(n):
        mask = (xx - cx[i])**2 + (yy - cy[i])**2 <= radii[i]**2
        img[mask] = colors[i]
    return img


def pattern_swirl(t: float) -> np.ndarray:
    """Concentric radial ripples."""
    x = np.linspace(-3, 3, WIDTH)
    y = np.linspace(-3, 3, HEIGHT)
    X, Y = np.meshgrid(x, y)
    r = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)
    Z = np.sin(10 * r - t * 2 + 4 * theta)
    Z = (Z + 1) / 2
    r_ch = np.uint8(255 * np.abs(np.sin(np.pi * Z + 0.0)))
    g_ch = np.uint8(255 * np.abs(np.sin(np.pi * Z + 2.09)))
    b_ch = np.uint8(255 * np.abs(np.sin(np.pi * Z + 4.19)))
    return np.stack([r_ch, g_ch, b_ch], axis=2)


def pattern_stripes(t: float) -> np.ndarray:
    """Shifting neon lattice."""
    x = np.linspace(0, 20, WIDTH)
    y = np.linspace(0, 20, HEIGHT)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(X * 2.5 + t) * np.cos(Y * 1.8 + t * 0.8)
    Z = (Z - Z.min()) / (Z.max() - Z.min() + 1e-9)
    r = np.uint8(255 * Z)
    g = np.uint8(255 * np.roll(Z, WIDTH // 4, axis=1))
    b = np.uint8(255 * np.roll(Z, HEIGHT // 3, axis=0))
    return np.stack([r, g, b], axis=2)


def pattern_mandala(t: float) -> np.ndarray:
    """Polar rose / petal geometry."""
    x = np.linspace(-1, 1, WIDTH)
    y = np.linspace(-1, 1, HEIGHT)
    X, Y = np.meshgrid(x, y)
    r = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)
    petals = 7
    Z = np.abs(np.cos(petals * theta + t)) * np.exp(-3 * r)
    Z2 = np.abs(np.sin(petals * 2 * theta - t * 1.3)) * np.exp(-2 * r)
    R = np.uint8(np.clip(Z  * 600, 0, 255))
    G = np.uint8(np.clip(Z2 * 500, 0, 255))
    B = np.uint8(np.clip((Z + Z2) * 300, 0, 255))
    return np.stack([R, G, B], axis=2)


def pattern_lava(t: float) -> np.ndarray:
    """Molten colour blobs — slow boiling lava lamp feel."""
    x = np.linspace(0, 4, WIDTH)
    y = np.linspace(0, 4, HEIGHT)
    X, Y = np.meshgrid(x, y)
    blob = (np.sin(X * 1.7 + np.cos(t * 0.4) * 3)
            + np.cos(Y * 2.1 + np.sin(t * 0.6) * 2)
            + np.sin((X + Y) * 1.1 + t * 0.3))
    blob = (blob - blob.min()) / (blob.max() - blob.min() + 1e-9)
    # fire palette: black → red → orange → yellow → white
    R = np.uint8(np.clip(blob * 3.0 * 255,        0, 255))
    G = np.uint8(np.clip((blob - 0.4) * 2.0 * 255, 0, 255))
    B = np.uint8(np.clip((blob - 0.8) * 5.0 * 255, 0, 255))
    return np.stack([R, G, B], axis=2)


def pattern_starfield(t: float) -> np.ndarray:
    """Zooming star-tunnel — forward warp through space."""
    img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    rng = np.random.default_rng(42)
    N = 600
    # stars in 3-D: x,y in [-1,1], z in (0,1]
    sx = rng.uniform(-1, 1, N)
    sy = rng.uniform(-1, 1, N)
    sz = ((rng.uniform(0, 1, N) + t * 0.4) % 1.0) + 0.001
    # project
    px = (sx / sz * 0.5 + 0.5) * WIDTH
    py = (sy / sz * 0.5 + 0.5) * HEIGHT
    brightness = np.clip((1 - sz) * 255, 20, 255).astype(np.uint8)
    size = np.clip((1 - sz) * 3, 1, 4).astype(int)
    mask_x = (px >= 0) & (px < WIDTH)
    mask_y = (py >= 0) & (py < HEIGHT)
    valid = mask_x & mask_y
    for i in np.where(valid)[0]:
        ix, iy = int(px[i]), int(py[i])
        s = size[i]
        c = brightness[i]
        # colour by depth band
        hue = (sz[i] * 3) % 1.0
        cr = int(c * abs(math.sin(hue * math.pi)))
        cg = int(c * abs(math.sin(hue * math.pi + 2.09)))
        cb = int(c * abs(math.sin(hue * math.pi + 4.19)))
        y0, y1 = max(0, iy - s), min(HEIGHT, iy + s + 1)
        x0, x1 = max(0, ix - s), min(WIDTH,  ix + s + 1)
        img[y0:y1, x0:x1] = [cr, cg, cb]
    return img


def pattern_crystals(t: float) -> np.ndarray:
    """Angular Voronoi-like crystal shards."""
    rng = np.random.default_rng(7)
    n_pts = 18
    pts_x = rng.uniform(0, WIDTH,  n_pts)
    pts_y = rng.uniform(0, HEIGHT, n_pts)
    # slowly drift seed points
    pts_x = (pts_x + np.sin(np.arange(n_pts) * 1.3 + t * 0.5) * 40) % WIDTH
    pts_y = (pts_y + np.cos(np.arange(n_pts) * 0.9 + t * 0.7) * 40) % HEIGHT
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float32)
    # distance to nearest seed
    dists = np.full((HEIGHT, WIDTH), np.inf)
    labels = np.zeros((HEIGHT, WIDTH), dtype=int)
    for i, (px, py) in enumerate(zip(pts_x, pts_y)):
        d = (xx - px)**2 + (yy - py)**2
        closer = d < dists
        dists[closer] = d[closer]
        labels[closer] = i
    # colour each region
    hues = np.linspace(0, 2 * math.pi, n_pts) + t * 0.4
    R = np.uint8(np.clip(200 * np.abs(np.sin(hues[labels])),          0, 255))
    G = np.uint8(np.clip(200 * np.abs(np.sin(hues[labels] + 2.09)),   0, 255))
    B = np.uint8(np.clip(255 * np.abs(np.sin(hues[labels] + 4.19)),   0, 255))
    # thin edge lines
    edge = (np.roll(labels, 1, 0) != labels) | (np.roll(labels, 1, 1) != labels)
    R[edge] = 255; G[edge] = 255; B[edge] = 255
    return np.stack([R, G, B], axis=2)


def pattern_aurora(t: float) -> np.ndarray:
    """Northern-lights curtain bands."""
    x = np.linspace(0, 2 * math.pi, WIDTH)
    y = np.linspace(0, 1, HEIGHT)
    X, Y = np.meshgrid(x, y)
    # undulating curtains
    wave = (np.sin(X + t * 0.6) * 0.25
            + np.sin(X * 2.1 - t * 0.4) * 0.15
            + np.sin(X * 0.7 + t * 0.9) * 0.10)
    band_y = 0.45 + wave          # vertical centre of curtain
    spread = 0.18 + 0.06 * np.sin(X * 1.5 + t)
    mask = np.exp(-((Y - band_y) / spread) ** 2)
    # colour: teal → green → violet sweep
    R = np.uint8(np.clip(mask * 120 * (0.5 + 0.5 * np.sin(X + t)),       0, 255))
    G = np.uint8(np.clip(mask * 255 * (0.7 + 0.3 * np.cos(X * 0.5 + t)), 0, 255))
    B = np.uint8(np.clip(mask * 200 * (0.5 + 0.5 * np.sin(X * 2 - t)),   0, 255))
    # dark sky above
    sky = np.clip(1 - Y * 1.2, 0, 1)
    R = np.uint8(R * sky + np.clip(5 * sky * 255, 0, 15))
    G = np.uint8(G * sky)
    B = np.uint8(np.clip(B * sky + sky * 18, 0, 255))
    return np.stack([R, G, B], axis=2)


# Ordered list — index = mode number shown in HUD
PATTERNS: List[Callable[[float], np.ndarray]] = [
    pattern_plasma,
    pattern_rainbow,
    pattern_splatter,
    pattern_swirl,
    pattern_stripes,
    pattern_mandala,
    pattern_lava,
    pattern_starfield,
    pattern_crystals,
    pattern_aurora,
]

PATTERN_NAMES = [
    "Plasma", "Rainbow", "Splatter", "Swirl", "Stripes",
    "Mandala", "Lava", "Starfield", "Crystals", "Aurora",
]


# ─────────────────────────────────────────────
# HUD OVERLAY
# ─────────────────────────────────────────────

def draw_hud(
    surface: pygame.Surface,
    font: pygame.font.Font,
    mode: int,
    slices: int,
    speed: float,
    auto: bool,
    fade_enabled: bool,
    fade_progress: float,          # 0..1 while fading, else -1
) -> None:
    lines = [
        f"Pattern : {PATTERN_NAMES[mode]}  [{mode+1}/{len(PATTERNS)}]",
        f"Slices  : {slices}",
        f"Speed   : {speed:.1f}",
        f"Auto    : {'ON' for auto}{'OFF' if not auto else ''}",
        f"Fade    : {'ON' for fade_enabled}{'OFF' if not fade_enabled else ''}",
    ]
    if 0 <= fade_progress < 1:
        lines.append(f"Fading  : {int(fade_progress*100):3d}%")
    lines += ["", "SPACE next  A auto  F fade", "↑↓ speed  ←→ slices  S save"]

    pad, line_h = 8, 16
    panel_w = 220
    panel_h = len(lines) * line_h + pad * 2
    panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
    panel.fill((0, 0, 0, 140))
    surface.blit(panel, (8, 8))

    for i, line in enumerate(lines):
        col = (200, 255, 180) if i < 5 else (140, 160, 140)
        txt = font.render(line, True, col)
        surface.blit(txt, (8 + pad, 8 + pad + i * line_h))


# ─────────────────────────────────────────────
# KALEIDOSCOPE BLIT
# ─────────────────────────────────────────────

def draw_kaleidoscope(
    screen: pygame.Surface,
    surf: pygame.Surface,
    angle: float,
    slices: int,
) -> None:
    slice_angle = 360.0 / slices
    cx, cy = WIDTH // 2, HEIGHT // 2
    for i in range(slices):
        rotated = pygame.transform.rotate(surf, angle + i * slice_angle)
        rect = rotated.get_rect(center=(cx, cy))
        screen.blit(rotated, rect)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Kaleidoscope")
    clock = pygame.time.Clock()

    try:
        font = pygame.font.SysFont("monospace", 13, bold=False)
    except Exception:
        font = pygame.font.Font(None, 14)

    # ── state ──
    pattern_mode  = 0
    slices        = SLICES
    rot_speed     = ROT_SPEED
    angle         = 0.0
    t             = 0.0

    auto_cycle    = False       # press A to toggle
    auto_timer    = 0.0         # seconds since last switch
    fade_enabled  = True        # press F to toggle
    fade_frame    = -1          # -1 = not fading; 0..FADE_FRAMES = active fade
    next_mode     = 0

    # Pre-render a surface so we can fade between two
    arr          = PATTERNS[pattern_mode](t)
    current_surf = pygame.surfarray.make_surface(arr)
    next_surf    = current_surf.copy()

    os.makedirs(SCREENSHOT_DIR, exist_ok=True)
    screenshot_idx = 0

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0   # seconds

        # ── events ──
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False

                elif event.key == pygame.K_SPACE:
                    # manual advance
                    next_mode   = (pattern_mode + 1) % len(PATTERNS)
                    fade_frame  = 0 if fade_enabled else -1
                    if not fade_enabled:
                        pattern_mode = next_mode
                    auto_timer = 0.0

                elif event.key == pygame.K_a:
                    auto_cycle = not auto_cycle
                    auto_timer = 0.0

                elif event.key == pygame.K_f:
                    fade_enabled = not fade_enabled

                elif event.key == pygame.K_UP:
                    rot_speed = round(min(rot_speed + 0.2, 10.0), 1)
                elif event.key == pygame.K_DOWN:
                    rot_speed = round(max(rot_speed - 0.2, 0.1), 1)

                elif event.key == pygame.K_RIGHT:
                    slices = min(slices + 1, 36)
                elif event.key == pygame.K_LEFT:
                    slices = max(slices - 1, 2)

                elif event.key == pygame.K_s:
                    path = os.path.join(SCREENSHOT_DIR, f"kaleid_{screenshot_idx:04d}.png")
                    pygame.image.save(screen, path)
                    screenshot_idx += 1
                    print(f"Screenshot saved: {path}")

        # ── auto-cycle ──
        if auto_cycle:
            auto_timer += dt
            if auto_timer >= AUTO_CYCLE_SECS:
                auto_timer = 0.0
                next_mode  = (pattern_mode + 1) % len(PATTERNS)
                fade_frame = 0 if fade_enabled else -1
                if not fade_enabled:
                    pattern_mode = next_mode

        # ── update time / generate frame ──
        t     += T_STEP
        angle += rot_speed

        arr          = PATTERNS[pattern_mode](t)
        current_surf = pygame.surfarray.make_surface(arr)

        if fade_frame >= 0:
            arr2     = PATTERNS[next_mode](t)
            next_surf = pygame.surfarray.make_surface(arr2)

        # ── draw ──
        screen.fill((0, 0, 0))

        if fade_frame >= 0:
            # blend two kaleidoscopes by alpha-overlaying the next onto current
            alpha = fade_frame / FADE_FRAMES
            draw_kaleidoscope(screen, current_surf, angle, slices)

            overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            draw_kaleidoscope(overlay, next_surf, angle, slices)
            overlay.set_alpha(int(alpha * 255))
            screen.blit(overlay, (0, 0))

            fade_frame += 1
            if fade_frame >= FADE_FRAMES:
                pattern_mode = next_mode
                fade_frame   = -1
        else:
            draw_kaleidoscope(screen, current_surf, angle, slices)

        # ── HUD ──
        fp = fade_frame / FADE_FRAMES if fade_frame >= 0 else -1.0
        draw_hud(screen, font, pattern_mode, slices, rot_speed,
                 auto_cycle, fade_enabled, fp)

        pygame.display.flip()

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
