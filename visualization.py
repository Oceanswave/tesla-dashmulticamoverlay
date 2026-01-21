"""
Visualization components for Tesla dashcam overlay.

Provides dashboard and composite frame rendering using Pillow.
Map rendering is handled by the map_renderer module.
"""

import logging
import numpy as np
import math
import threading
from typing import Tuple, Optional, Set
from PIL import Image, ImageDraw, ImageFont
import dashcam_pb2

logger = logging.getLogger(__name__)

from constants import (
    COLORS,
    OUTPUT_WIDTH, OUTPUT_HEIGHT,
    DASHBOARD_WIDTH, DASHBOARD_HEIGHT,
    SPEEDOMETER_CENTER, SPEEDOMETER_RADIUS, SPEEDOMETER_MAX_VALUE,
    STEERING_CENTER, STEERING_RADIUS, STEERING_THICKNESS,
    PEDAL_CENTER, PEDAL_BAR_WIDTH, PEDAL_BAR_HEIGHT,
    GBALL_CENTER, GBALL_RADIUS, GBALL_MAX_G, GBALL_DOT_RADIUS, MPS2_TO_G,
    BLINKER_Y, BLINKER_LEFT_X, BLINKER_RIGHT_X, BLINKER_SIZE,
    OVERLAY_OPACITY, CANVAS_OPACITY,
    TOP_SECTION_HEIGHT, FRONT_CAMERA_WIDTH, FRONT_CAMERA_X,
    PILLAR_WIDTH, PILLAR_HEIGHT, PILLAR_Y,
    BOTTOM_SECTION_Y, BOTTOM_SECTION_HEIGHT, BOTTOM_CAMERA_WIDTH,
    SPLIT_HEIGHT,
    REPEATER_LAYOUT_CENTER_WIDTH, REPEATER_LAYOUT_SIDE_WIDTH,
    GRID_2X2_HALF_WIDTH, GRID_2X2_HALF_HEIGHT,
    SIDE_LAYOUT_FRONT_WIDTH, SIDE_LAYOUT_SIDE_WIDTH,
    # PiP layout constants (bottom-anchored with separate row sizes)
    PIP_TOP_THUMB_WIDTH, PIP_TOP_THUMB_HEIGHT,
    PIP_BOTTOM_THUMB_WIDTH, PIP_BOTTOM_THUMB_HEIGHT,
    PIP_TOP_LEFT_X, PIP_TOP_RIGHT_X,
    PIP_BOTTOM_LEFT_X, PIP_BOTTOM_CENTER_X, PIP_BOTTOM_RIGHT_X,
    PIP_TOP_ROW_Y, PIP_BOTTOM_ROW_Y,
    PIP_REAR_CROP_PERCENT,
    MPS_TO_MPH,
    SPEED_ZONE_ECO, SPEED_ZONE_CITY, SPEED_ZONE_SUBURBAN,
    SPEED_ZONE_HIGHWAY, SPEED_ZONE_FAST, SPEED_ZONE_VERY_FAST,
    SPEED_ZONE_LOW, SPEED_ZONE_MID, SPEED_ZONE_HIGH,
    GBALL_ZONE_SAFE, GBALL_ZONE_SPORT,
    EMPHASIS_MAX_SCALE_BOOST,
    EMPHASIS_REAR_SCALE_BOOST,
    EMPHASIS_VISIBILITY_THRESHOLD,
)

# Import emphasis types (optional - graceful fallback if not available)
try:
    from emphasis import EmphasisState, CameraEmphasis
    EMPHASIS_AVAILABLE = True
except ImportError:
    EMPHASIS_AVAILABLE = False
    EmphasisState = None
    CameraEmphasis = None

# Re-export MapRenderer for backward compatibility
from map_renderer import MapRenderer


_warned_overlay_bounds = False  # Only warn once about overlay bounds
_warned_overlay_bounds_lock = threading.Lock()  # Thread-safe warning flag access


def apply_overlay(canvas: np.ndarray, overlay: np.ndarray, x: int, y: int,
                  overlay_weight: float = OVERLAY_OPACITY,
                  canvas_weight: float = CANVAS_OPACITY) -> None:
    """
    Apply an overlay image onto a canvas with alpha blending.

    Args:
        canvas: The destination canvas (modified in-place), RGB format
        overlay: The overlay image to blend, RGB format
        x: X position on canvas
        y: Y position on canvas
        overlay_weight: Weight of overlay in blend (default: 0.8)
        canvas_weight: Weight of canvas in blend (default: 0.2)
    """
    global _warned_overlay_bounds
    oh, ow = overlay.shape[:2]
    ch, cw = canvas.shape[:2]

    # Bounds checking
    if y < 0 or x < 0 or y + oh > ch or x + ow > cw:
        with _warned_overlay_bounds_lock:
            if not _warned_overlay_bounds:
                logger.warning(
                    f"Overlay dropped: position ({x},{y}) + size ({ow}x{oh}) exceeds canvas ({cw}x{ch}). "
                    "Try reducing --overlay-scale."
                )
                _warned_overlay_bounds = True
        return

    roi = canvas[y:y + oh, x:x + ow]
    # Alpha blending using numpy
    blended = (roi.astype(np.float32) * canvas_weight +
               overlay.astype(np.float32) * overlay_weight).astype(np.uint8)
    canvas[y:y + oh, x:x + ow] = blended


# Font cache to avoid repeated disk lookups
_font_cache: dict = {}
_font_cache_lock = threading.Lock()  # Thread-safe font cache access
_font_path: Optional[str] = None  # Cached successful font path


def _get_font(size: float = 12) -> ImageFont.FreeTypeFont:
    """Get a cached font instance.

    Caches font instances by size to avoid repeated disk lookups.
    On first call, finds a working font path and reuses it.
    Thread-safe for parallel rendering.
    """
    global _font_path

    int_size = int(size)

    # Return cached font if available (check with lock)
    with _font_cache_lock:
        if int_size in _font_cache:
            return _font_cache[int_size]

    # Find a working font path (only needed once)
    # This is idempotent so doesn't need strict locking
    if _font_path is None:
        for font_name in ["DejaVuSans.ttf", "Arial.ttf", "Helvetica.ttf",
                          "/System/Library/Fonts/Helvetica.ttc"]:
            try:
                ImageFont.truetype(font_name, 12)  # Test load
                _font_path = font_name
                break
            except (OSError, IOError):
                continue

    # Load and cache the font
    try:
        if _font_path:
            font = ImageFont.truetype(_font_path, int_size)
        else:
            font = ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    with _font_cache_lock:
        _font_cache[int_size] = font
    return font


def _draw_text_shadow(draw: ImageDraw.Draw, pos: Tuple[int, int], text: str,
                      font: ImageFont.FreeTypeFont, color: Tuple[int, int, int],
                      shadow_color: Tuple[int, int, int], offset: int = 2) -> None:
    """Draw text with drop shadow effect."""
    draw.text((pos[0] + offset, pos[1] + offset), text, fill=shadow_color, font=font)
    draw.text(pos, text, fill=color, font=font)


def _get_speed_zone_color(speed_mph: float) -> Tuple[int, int, int]:
    """Return color based on speed zone thresholds.

    Uses expanded 7-zone gradient for smoother color transitions:
    - 0-25 mph: Bright green (eco cruising)
    - 25-40 mph: Yellow-green (city driving)
    - 40-55 mph: Yellow (suburban)
    - 55-70 mph: Amber (highway)
    - 70-85 mph: Orange (fast highway)
    - 85-100 mph: Red-orange (very fast)
    - 100+ mph: Red (danger)
    """
    if speed_mph < SPEED_ZONE_ECO:
        return COLORS.SPEED_ECO
    elif speed_mph < SPEED_ZONE_CITY:
        return COLORS.SPEED_CITY
    elif speed_mph < SPEED_ZONE_SUBURBAN:
        return COLORS.SPEED_SUBURBAN
    elif speed_mph < SPEED_ZONE_HIGHWAY:
        return COLORS.SPEED_HIGHWAY
    elif speed_mph < SPEED_ZONE_FAST:
        return COLORS.SPEED_FAST
    elif speed_mph < SPEED_ZONE_VERY_FAST:
        return COLORS.SPEED_VERY_FAST
    else:
        return COLORS.SPEED_DANGER


def _get_gball_zone_color(g_magnitude: float) -> Tuple[int, int, int]:
    """Return color based on G-force magnitude zone thresholds."""
    if g_magnitude < GBALL_ZONE_SAFE:
        return COLORS.GBALL_SAFE
    elif g_magnitude < GBALL_ZONE_SPORT:
        return COLORS.GBALL_SPORT
    else:
        return COLORS.GBALL_LIMIT


class DashboardRenderer:
    """Renders vehicle telemetry dashboard overlay.

    Pre-renders static elements (arcs, circles, outlines) once in __init__
    to avoid redundant drawing on every frame. Only dynamic elements
    (speed value, steering spokes, pedal fills, status text) are drawn per-frame.

    Args:
        scale: Scaling factor for overlay size (default 1.0)
        bg_color: Background color in RGB
    """

    def __init__(self, scale: float = 1.0,
                 bg_color: Tuple[int, int, int] = COLORS.VOID_BLACK):
        self.scale = scale
        self.width = int(DASHBOARD_WIDTH * scale)
        self.height = int(DASHBOARD_HEIGHT * scale)
        self.bg_color = bg_color

        # Scaled fonts (increased for better readability)
        self._large_font = _get_font(34 * scale)
        self._medium_font = _get_font(17 * scale)
        self._small_font = _get_font(13 * scale)
        self._status_font = _get_font(16 * scale)

        # Scaled component positions and dimensions
        self._speedometer_center = (int(SPEEDOMETER_CENTER[0] * scale), int(SPEEDOMETER_CENTER[1] * scale))
        self._speedometer_radius = int(SPEEDOMETER_RADIUS * scale)
        self._steering_center = (int(STEERING_CENTER[0] * scale), int(STEERING_CENTER[1] * scale))
        self._steering_radius = int(STEERING_RADIUS * scale)
        self._steering_thickness = max(1, int(STEERING_THICKNESS * scale))

        # Load Cybertruck yoke image
        self._yoke_image = self._load_yoke_image()

        # Frame counter for blinker animation (toggles every 15 frames at 30fps = 0.5s blink)
        self._frame_count = 0
        self._blink_interval = 15  # frames per blink state

        self._pedal_center = (int(PEDAL_CENTER[0] * scale), int(PEDAL_CENTER[1] * scale))
        self._pedal_bar_width = int(PEDAL_BAR_WIDTH * scale)
        self._pedal_bar_height = int(PEDAL_BAR_HEIGHT * scale)

        # G-Ball (acceleration indicator) - scaled
        self._gball_center = (int(GBALL_CENTER[0] * scale), int(GBALL_CENTER[1] * scale))
        self._gball_radius = int(GBALL_RADIUS * scale)
        self._gball_dot_radius = max(2, int(GBALL_DOT_RADIUS * scale))

        # Blinker indicators - scaled
        self._blinker_y = int(BLINKER_Y * scale)
        self._blinker_left_x = int(BLINKER_LEFT_X * scale)
        self._blinker_right_x = int(BLINKER_RIGHT_X * scale)
        self._blinker_size = int(BLINKER_SIZE * scale)

        # Cache pedal coordinates (scaled) - must be before _create_base_image
        self._x_accel = self._pedal_center[0] + int(10 * scale)
        self._x_brake = self._pedal_center[0] - int(30 * scale)
        self._y_bot = self._pedal_center[1] + self._pedal_bar_height // 2

        # Cache gauge parameters (scaled)
        self._gauge_start_angle = 135
        self._gauge_end_angle = 405
        self._gauge_bbox = [
            self._speedometer_center[0] - self._speedometer_radius,
            self._speedometer_center[1] - self._speedometer_radius,
            self._speedometer_center[0] + self._speedometer_radius,
            self._speedometer_center[1] + self._speedometer_radius
        ]

        # Pre-render static elements once (must be after all cached values are set)
        self._base_image = self._create_base_image()

    def _create_base_image(self) -> Image.Image:
        """Create base image with all static elements pre-rendered."""
        img = Image.new('RGB', (self.width, self.height), self.bg_color)
        draw = ImageDraw.Draw(img)

        # 0. Panel border (GUNMETAL, 2px)
        border_width = max(1, int(2 * self.scale))
        draw.rectangle(
            [0, 0, self.width - 1, self.height - 1],
            outline=COLORS.GUNMETAL,
            width=border_width
        )

        # 1. Speedometer zone arcs (static, pre-rendered)
        # Gauge spans 135° to 405° (270° total for 0-120 MPH)
        bbox = [self._speedometer_center[0] - self._speedometer_radius,
                self._speedometer_center[1] - self._speedometer_radius,
                self._speedometer_center[0] + self._speedometer_radius,
                self._speedometer_center[1] + self._speedometer_radius]

        # Draw 7 zone arcs (each zone is a portion of the 270° arc)
        zone_arc_width = max(1, int(2 * self.scale))
        deg_per_mph = 270.0 / SPEEDOMETER_MAX_VALUE

        # Zone boundaries in degrees from start (135°) - expanded 7-zone gradient
        zones = [
            (0, SPEED_ZONE_ECO, COLORS.SPEED_ECO),                    # 0-25 bright green
            (SPEED_ZONE_ECO, SPEED_ZONE_CITY, COLORS.SPEED_CITY),     # 25-40 yellow-green
            (SPEED_ZONE_CITY, SPEED_ZONE_SUBURBAN, COLORS.SPEED_SUBURBAN),  # 40-55 yellow
            (SPEED_ZONE_SUBURBAN, SPEED_ZONE_HIGHWAY, COLORS.SPEED_HIGHWAY),  # 55-70 amber
            (SPEED_ZONE_HIGHWAY, SPEED_ZONE_FAST, COLORS.SPEED_FAST),   # 70-85 orange
            (SPEED_ZONE_FAST, SPEED_ZONE_VERY_FAST, COLORS.SPEED_VERY_FAST),  # 85-100 red-orange
            (SPEED_ZONE_VERY_FAST, SPEEDOMETER_MAX_VALUE, COLORS.SPEED_DANGER),  # 100-120 red
        ]

        for zone_start, zone_end, zone_color in zones:
            start_angle = 135 + zone_start * deg_per_mph
            end_angle = 135 + zone_end * deg_per_mph
            # Draw with dimmed color (33% brightness for static zone indicators)
            dim_color = tuple(c // 3 for c in zone_color)
            draw.arc(bbox, start_angle, end_angle, fill=dim_color, width=zone_arc_width)

        # 2. Tick marks (7 major ticks at 0, 20, 40, 60, 80, 100, 120)
        tick_length = int(8 * self.scale)
        tick_inner_radius = self._speedometer_radius - tick_length
        for mph_val in range(0, SPEEDOMETER_MAX_VALUE + 1, 20):
            angle_deg = 135 + mph_val * deg_per_mph
            angle_rad = math.radians(angle_deg)
            # Outer point (on arc)
            outer_x = self._speedometer_center[0] + self._speedometer_radius * math.cos(angle_rad)
            outer_y = self._speedometer_center[1] + self._speedometer_radius * math.sin(angle_rad)
            # Inner point
            inner_x = self._speedometer_center[0] + tick_inner_radius * math.cos(angle_rad)
            inner_y = self._speedometer_center[1] + tick_inner_radius * math.sin(angle_rad)
            draw.line([(inner_x, inner_y), (outer_x, outer_y)],
                      fill=COLORS.STEEL_MID, width=max(1, int(1 * self.scale)))

        # MPH label (static, STEEL_DARK)
        label_bbox = draw.textbbox((0, 0), "MPH", font=self._small_font)
        label_w = label_bbox[2] - label_bbox[0]
        draw.text((self._speedometer_center[0] - label_w // 2, self._speedometer_center[1] + int(15 * self.scale)),
                  "MPH", fill=COLORS.STEEL_DARK, font=self._small_font)

        # 2. Cybertruck Yoke (static outline) - rectangular with cutouts
        # The yoke shape is drawn dynamically since it rotates, but we draw
        # a reference frame here (will be overdrawn by dynamic yoke)
        # No static drawing needed - yoke is fully dynamic

        # 3. Pedal box outlines (static, STEEL_DARK) - uses pre-computed scaled values
        draw.rectangle([self._x_accel, self._y_bot - self._pedal_bar_height,
                       self._x_accel + self._pedal_bar_width, self._y_bot], outline=COLORS.STEEL_DARK, width=1)
        draw.rectangle([self._x_brake, self._y_bot - self._pedal_bar_height,
                       self._x_brake + self._pedal_bar_width, self._y_bot], outline=COLORS.STEEL_DARK, width=1)

        # Pedal labels (static, STEEL_DARK)
        draw.text((self._x_accel + int(4 * self.scale), self._y_bot + int(2 * self.scale)), "P", fill=COLORS.STEEL_DARK, font=self._small_font)
        draw.text((self._x_brake + int(4 * self.scale), self._y_bot + int(2 * self.scale)), "B", fill=COLORS.STEEL_DARK, font=self._small_font)

        # 4. G-Ball indicator with zone rings
        # Outer ring (1.0g limit) - GBALL_LIMIT dimmed
        gball_bbox = [
            self._gball_center[0] - self._gball_radius,
            self._gball_center[1] - self._gball_radius,
            self._gball_center[0] + self._gball_radius,
            self._gball_center[1] + self._gball_radius
        ]
        dim_limit = tuple(c // 3 for c in COLORS.GBALL_LIMIT)
        draw.ellipse(gball_bbox, outline=dim_limit, width=max(1, int(2 * self.scale)))

        # Sport zone ring (0.6g) - GBALL_SPORT dimmed
        sport_radius = int(self._gball_radius * GBALL_ZONE_SPORT / GBALL_MAX_G)
        sport_bbox = [
            self._gball_center[0] - sport_radius,
            self._gball_center[1] - sport_radius,
            self._gball_center[0] + sport_radius,
            self._gball_center[1] + sport_radius
        ]
        dim_sport = tuple(c // 3 for c in COLORS.GBALL_SPORT)
        draw.ellipse(sport_bbox, outline=dim_sport, width=1)

        # Safe zone ring (0.3g) - GBALL_SAFE dimmed
        safe_radius = int(self._gball_radius * GBALL_ZONE_SAFE / GBALL_MAX_G)
        safe_bbox = [
            self._gball_center[0] - safe_radius,
            self._gball_center[1] - safe_radius,
            self._gball_center[0] + safe_radius,
            self._gball_center[1] + safe_radius
        ]
        dim_safe = tuple(c // 3 for c in COLORS.GBALL_SAFE)
        draw.ellipse(safe_bbox, outline=dim_safe, width=1)

        # Crosshairs (STEEL_DARK for subtlety)
        draw.line([
            (self._gball_center[0] - self._gball_radius, self._gball_center[1]),
            (self._gball_center[0] + self._gball_radius, self._gball_center[1])
        ], fill=COLORS.STEEL_DARK, width=1)
        draw.line([
            (self._gball_center[0], self._gball_center[1] - self._gball_radius),
            (self._gball_center[0], self._gball_center[1] + self._gball_radius)
        ], fill=COLORS.STEEL_DARK, width=1)

        # 5. Blinker indicators (static dimmed arrows - GUNMETAL)
        self._draw_arrow(draw, self._blinker_left_x, self._blinker_y, self._blinker_size, "left", COLORS.BLINKER_INACTIVE)
        self._draw_arrow(draw, self._blinker_right_x, self._blinker_y, self._blinker_size, "right", COLORS.BLINKER_INACTIVE)

        return img

    def _draw_arrow(self, draw: ImageDraw.Draw, cx: int, cy: int, size: int,
                    direction: str, color: Tuple[int, int, int]) -> None:
        """Draw an arrow indicator (blinker style).

        Args:
            draw: ImageDraw context
            cx, cy: Center position
            size: Arrow size
            direction: "left" or "right"
            color: Fill color
        """
        half = size // 2
        if direction == "left":
            # Left-pointing arrow: tip on left, base on right
            points = [
                (cx - half, cy),          # Left tip
                (cx + half, cy - half),   # Top right
                (cx + half, cy + half)    # Bottom right
            ]
        else:
            # Right-pointing arrow: tip on right, base on left
            points = [
                (cx + half, cy),          # Right tip
                (cx - half, cy - half),   # Top left
                (cx - half, cy + half)    # Bottom left
            ]
        draw.polygon(points, fill=color)

    def _draw_gball(self, draw: ImageDraw.Draw, meta) -> None:
        """Draw G-ball indicator dot based on acceleration.

        Maps lateral (X) and longitudinal (Y) acceleration to dot position.
        Dot color changes based on G-magnitude (green→orange→red).
        Displays g-force value below the indicator.
        - Positive X (turning right) moves dot left
        - Positive Y (braking) moves dot up (forward in vehicle frame)
        """
        # Convert m/s² to G-force
        g_x = meta.linear_acceleration_mps2_x * MPS2_TO_G
        g_y = meta.linear_acceleration_mps2_y * MPS2_TO_G

        # Calculate magnitude for color coding and display
        g_magnitude = math.sqrt(g_x * g_x + g_y * g_y)

        # Clamp to max G for full deflection
        g_x_clamped = max(-GBALL_MAX_G, min(GBALL_MAX_G, g_x))
        g_y_clamped = max(-GBALL_MAX_G, min(GBALL_MAX_G, g_y))

        # Map G-force to pixel offset (inverted X: positive G = left on screen)
        # Y is inverted for screen coords (positive G = up = negative screen Y)
        offset_x = int(-g_x_clamped / GBALL_MAX_G * self._gball_radius)
        offset_y = int(-g_y_clamped / GBALL_MAX_G * self._gball_radius)

        dot_x = self._gball_center[0] + offset_x
        dot_y = self._gball_center[1] + offset_y

        # Get zone-based color for dot
        dot_color = _get_gball_zone_color(g_magnitude)

        # Draw filled dot with zone color
        dot_bbox = [
            dot_x - self._gball_dot_radius,
            dot_y - self._gball_dot_radius,
            dot_x + self._gball_dot_radius,
            dot_y + self._gball_dot_radius
        ]
        draw.ellipse(dot_bbox, fill=dot_color)

        # Draw g-force value below the indicator
        g_text = f"{g_magnitude:.2f}g"
        g_bbox = draw.textbbox((0, 0), g_text, font=self._small_font)
        g_w = g_bbox[2] - g_bbox[0]
        g_x_pos = self._gball_center[0] - g_w // 2
        g_y_pos = self._gball_center[1] + self._gball_radius + int(5 * self.scale)
        draw.text((g_x_pos, g_y_pos), g_text, fill=dot_color, font=self._small_font)

    def _draw_blinkers(self, draw: ImageDraw.Draw, meta) -> None:
        """Draw active blinker indicators with blinking animation.

        Blinkers flash on/off based on frame counter (~1Hz at 30fps).
        When "on" phase, draws Cybertruck orange arrows with glow effect.
        """
        # Determine if we're in the "on" phase of the blink cycle
        blink_on = (self._frame_count // self._blink_interval) % 2 == 0

        if meta.blinker_on_left and blink_on:
            # Draw glow (larger, semi-transparent orange)
            glow_size = int(self._blinker_size * 1.3)
            self._draw_arrow(draw, self._blinker_left_x, self._blinker_y,
                           glow_size, "left", COLORS.CT_ORANGE_DIM)
            # Draw main arrow
            self._draw_arrow(draw, self._blinker_left_x, self._blinker_y,
                           self._blinker_size, "left", COLORS.BLINKER_ACTIVE)

        if meta.blinker_on_right and blink_on:
            # Draw glow (larger, semi-transparent orange)
            glow_size = int(self._blinker_size * 1.3)
            self._draw_arrow(draw, self._blinker_right_x, self._blinker_y,
                           glow_size, "right", COLORS.CT_ORANGE_DIM)
            # Draw main arrow
            self._draw_arrow(draw, self._blinker_right_x, self._blinker_y,
                           self._blinker_size, "right", COLORS.BLINKER_ACTIVE)

    def render(self, meta: dashcam_pb2.SeiMetadata) -> np.ndarray:
        """Render dashboard overlay from SEI metadata.

        Copies pre-rendered base image and draws only dynamic elements.
        Includes frame-based blinker animation.
        """
        # Increment frame counter for blinker animation
        self._frame_count += 1

        # Copy base image (much faster than recreating)
        img = self._base_image.copy()
        draw = ImageDraw.Draw(img)

        # 1. Speedometer value arc (dynamic, zone-colored)
        speed_mph = meta.vehicle_speed_mps * MPS_TO_MPH
        val_angle = self._gauge_start_angle + (min(speed_mph, SPEEDOMETER_MAX_VALUE) / SPEEDOMETER_MAX_VALUE) * (self._gauge_end_angle - self._gauge_start_angle)
        if val_angle > self._gauge_start_angle:
            arc_color = _get_speed_zone_color(speed_mph)
            draw.arc(self._gauge_bbox, self._gauge_start_angle, val_angle, fill=arc_color, width=max(1, int(4 * self.scale)))

        # Speed text with shadow (dynamic)
        text = f"{int(speed_mph)}"
        text_bbox = draw.textbbox((0, 0), text, font=self._large_font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        text_pos = (self._speedometer_center[0] - text_w // 2, self._speedometer_center[1] - text_h // 2)
        shadow_offset = max(1, int(2 * self.scale))
        _draw_text_shadow(draw, text_pos, text, self._large_font,
                          COLORS.STEEL_BRIGHT, COLORS.GUNMETAL, shadow_offset)

        # 2. Cybertruck yoke (dynamic - rotates with steering angle)
        self._draw_steering_yoke(img, meta.steering_wheel_angle)

        # 3. Pedal fills (dynamic)
        self._draw_pedal_fills(draw, meta.accelerator_pedal_position, meta.brake_applied)

        # 4. Status Text (dynamic)
        self._draw_status(draw, meta)

        # 5. G-Ball indicator (dynamic dot)
        self._draw_gball(draw, meta)

        # 6. Blinker indicators (dynamic with animation)
        self._draw_blinkers(draw, meta)

        return np.array(img)

    def _load_yoke_image(self) -> Optional[Image.Image]:
        """Load and scale the Cybertruck yoke image.

        Returns scaled RGBA image or None if not found.
        """
        import os

        # Try to find the yoke image relative to this file or the project root
        possible_paths = [
            os.path.join(os.path.dirname(__file__), "images", "CT_Squirkel.png"),
            os.path.join(os.path.dirname(__file__), "..", "images", "CT_Squirkel.png"),
            "images/CT_Squirkel.png",
        ]

        for path in possible_paths:
            if os.path.exists(path):
                try:
                    img = Image.open(path).convert('RGBA')
                    # Scale to fit steering area (width = 2 * steering_radius)
                    target_width = int(self._steering_radius * 2.2)
                    aspect = img.height / img.width
                    target_height = int(target_width * aspect)
                    img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
                    return img
                except Exception as e:
                    logger.warning(f"Failed to load yoke image: {e}")
                    return None

        logger.debug("Yoke image not found, will use programmatic drawing")
        return None

    def _draw_steering_yoke(self, img: Image.Image, angle: float):
        """Draw Cybertruck-style yoke steering wheel using image or fallback.

        Uses CT_Squirkel.png image if available, rotated by steering angle.
        Falls back to programmatic drawing if image not found.
        Displays steering angle in degrees below the yoke.
        """
        cx, cy = self._steering_center
        draw = ImageDraw.Draw(img)

        if self._yoke_image is not None:
            # Rotate the yoke image by steering angle with expand=True to prevent clipping
            rotated = self._yoke_image.rotate(-angle, resample=Image.Resampling.BILINEAR,
                                               expand=True)

            # Calculate paste position (center the expanded image on steering center)
            paste_x = cx - rotated.width // 2
            paste_y = cy - rotated.height // 2

            # Paste with alpha mask for transparency
            img.paste(rotated, (paste_x, paste_y), rotated)

            # Draw orange dot marker at top of yoke (rotates with steering)
            rad = math.radians(angle)
            marker_distance = int(self._steering_radius * 0.75)  # Distance from center to top
            marker_x = cx + int(marker_distance * math.sin(rad))
            marker_y = cy - int(marker_distance * math.cos(rad))
            marker_r = max(2, int(3 * self.scale))
            draw.ellipse([marker_x - marker_r, marker_y - marker_r,
                          marker_x + marker_r, marker_y + marker_r],
                         fill=COLORS.CT_ORANGE)
        else:
            # Fallback: draw simple yoke shape
            rad = math.radians(angle)
            half_w = int(self._steering_radius * 1.1)
            half_h = int(self._steering_radius * 0.8)

            def rotate_point(px, py):
                rx = px * math.cos(rad) - py * math.sin(rad)
                ry = px * math.sin(rad) + py * math.cos(rad)
                return (int(cx + rx), int(cy + ry))

            # Simple rounded rectangle outline
            pts = [
                (-half_w, -half_h), (half_w, -half_h),
                (half_w, half_h), (-half_w, half_h)
            ]
            rotated_pts = [rotate_point(px, py) for px, py in pts]
            draw.polygon(rotated_pts, fill=COLORS.GUNMETAL, outline=COLORS.STEEL_MID)

        # Draw steering angle in degrees below the yoke
        angle_text = f"{angle:+.0f}°"
        angle_bbox = draw.textbbox((0, 0), angle_text, font=self._small_font)
        angle_w = angle_bbox[2] - angle_bbox[0]
        angle_x = cx - angle_w // 2
        angle_y = cy + self._steering_radius + int(8 * self.scale)
        draw.text((angle_x, angle_y), angle_text, fill=COLORS.STEEL_BRIGHT, font=self._small_font)

    def _draw_pedal_fills(self, draw: ImageDraw.Draw, accel: float, brake: bool):
        """Draw only the pedal fill bars (dynamic part of pedals).

        Uses Cybertruck colors: CT_GREEN for accelerator, CT_RED for brake.
        Shows percentage label above bar when pressed.
        """
        # Accelerator fill (uses scaled bar dimensions)
        # Handle both 0-1 and 0-100 range (Tesla may use 0-1)
        # Clamp to valid range to prevent drawing outside pedal bounds
        accel_pct = accel if accel > 1.0 else accel * 100.0
        accel_pct = max(0.0, min(100.0, accel_pct))  # Clamp to 0-100%
        h_accel = int((accel_pct / 100.0) * self._pedal_bar_height)
        if h_accel > 0:
            draw.rectangle([self._x_accel, self._y_bot - h_accel,
                           self._x_accel + self._pedal_bar_width, self._y_bot], fill=COLORS.CT_GREEN)
            # Show percentage above bar when pressed
            if accel_pct >= 5:  # Only show if noticeable
                pct_text = f"{int(accel_pct)}%"
                pct_bbox = draw.textbbox((0, 0), pct_text, font=self._small_font)
                pct_w = pct_bbox[2] - pct_bbox[0]
                pct_x = self._x_accel + (self._pedal_bar_width - pct_w) // 2
                pct_y = self._y_bot - h_accel - int(12 * self.scale)
                draw.text((pct_x, pct_y), pct_text, fill=COLORS.CT_GREEN, font=self._small_font)

        # Brake fill
        if brake:
            draw.rectangle([self._x_brake, self._y_bot - self._pedal_bar_height,
                           self._x_brake + self._pedal_bar_width, self._y_bot], fill=COLORS.CT_RED)
            # Show "ON" label above brake bar when pressed
            brake_text = "ON"
            brake_bbox = draw.textbbox((0, 0), brake_text, font=self._small_font)
            brake_w = brake_bbox[2] - brake_bbox[0]
            brake_x = self._x_brake + (self._pedal_bar_width - brake_w) // 2
            brake_y = self._y_bot - self._pedal_bar_height - int(12 * self.scale)
            draw.text((brake_x, brake_y), brake_text, fill=COLORS.CT_RED, font=self._small_font)

    def _draw_status(self, draw: ImageDraw.Draw, meta):
        """Draw gear and autopilot status with badge background and mode coloring.

        Mode colors:
        - Manual: STEEL_MID (neutral)
        - TACC: CT_GREEN (active assist)
        - Autosteer: CT_ORANGE (active steering)
        - FSD: CT_ORANGE_GLOW (full autonomy)
        """
        gear_map = {0: "P", 1: "D", 2: "R", 3: "N"}
        gear = gear_map.get(meta.gear_state, "?")

        ap_map = {0: "Manual", 1: "FSD", 2: "Autosteer", 3: "TACC"}
        ap = ap_map.get(meta.autopilot_state, "Unknown")

        # Mode-specific colors
        mode_colors = {
            0: COLORS.STEEL_MID,      # Manual
            1: COLORS.CT_ORANGE_GLOW, # FSD
            2: COLORS.CT_ORANGE,      # Autosteer
            3: COLORS.CT_GREEN,       # TACC
        }
        mode_color = mode_colors.get(meta.autopilot_state, COLORS.STEEL_MID)

        # Build text parts
        gear_text = f"GEAR: {gear}"
        mode_text = f"MODE: {ap}"
        separator = "  |  "
        full_text = gear_text + separator + mode_text

        # Calculate dimensions
        full_bbox = draw.textbbox((0, 0), full_text, font=self._status_font)
        text_w = full_bbox[2] - full_bbox[0]
        text_h = full_bbox[3] - full_bbox[1]

        # Badge background (GUNMETAL rounded rectangle)
        padding_x = int(12 * self.scale)
        padding_y = int(4 * self.scale)
        badge_x = (self.width - text_w) // 2 - padding_x
        badge_y = self.height - int(22 * self.scale) - padding_y
        badge_w = text_w + padding_x * 2
        badge_h = text_h + padding_y * 2
        corner_radius = int(4 * self.scale)

        draw.rounded_rectangle(
            [badge_x, badge_y, badge_x + badge_w, badge_y + badge_h],
            radius=corner_radius,
            fill=COLORS.GUNMETAL
        )

        # Draw gear text (STEEL_BRIGHT)
        text_x = (self.width - text_w) // 2
        text_y = self.height - int(20 * self.scale)
        draw.text((text_x, text_y), gear_text, fill=COLORS.STEEL_BRIGHT, font=self._status_font)

        # Draw separator (STEEL_DARK)
        gear_bbox = draw.textbbox((0, 0), gear_text, font=self._status_font)
        sep_x = text_x + gear_bbox[2] - gear_bbox[0]
        draw.text((sep_x, text_y), separator, fill=COLORS.STEEL_DARK, font=self._status_font)

        # Draw mode text (colored based on autopilot state)
        sep_bbox = draw.textbbox((0, 0), separator, font=self._status_font)
        mode_x = sep_x + sep_bbox[2] - sep_bbox[0]
        draw.text((mode_x, text_y), mode_text, fill=mode_color, font=self._status_font)


def _resize_frame(frame: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize a frame using OpenCV with INTER_LINEAR resampling.

    OpenCV resize is 3-5x faster than Pillow's resize due to:
    - Native C++ implementation with SIMD optimizations
    - Direct numpy array operations (no PIL conversion overhead)
    - Multi-threaded by default on modern OpenCV builds

    INTER_LINEAR (bilinear) provides good quality at 30fps playback
    where each frame is only visible for ~33ms.

    Optimization: Skip resize if frame is already the target size.
    """
    import cv2
    # Skip resize if already correct size (saves ~1ms per frame)
    h, w = frame.shape[:2]
    target_w, target_h = size
    if w == target_w and h == target_h:
        return frame
    # OpenCV uses (width, height) order for resize
    return cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR)


def _flip_horizontal(frame: np.ndarray) -> np.ndarray:
    """Flip a frame horizontally using numpy.

    np.fliplr() operates directly on the array without PIL conversion,
    making it significantly faster than Image.transpose().
    """
    return np.fliplr(frame)


def _crop_center(frame: np.ndarray, crop_percent: float) -> np.ndarray:
    """Crop to center portion of frame, removing edges.

    Used for "punch-in" effect to remove wide-angle lens vignetting
    (e.g., rear camera with fisheye distortion at edges).

    Args:
        frame: Input frame (H, W, C)
        crop_percent: Percentage to remove from each edge (e.g., 0.05 = 5%)

    Returns:
        Cropped frame (smaller dimensions)
    """
    h, w = frame.shape[:2]
    crop_h = int(h * crop_percent)
    crop_w = int(w * crop_percent)
    return frame[crop_h:h-crop_h, crop_w:w-crop_w]


def _draw_text(canvas: np.ndarray, text: str, position: Tuple[int, int],
               color: Tuple[int, int, int] = COLORS.WHITE, size: int = 16) -> None:
    """Draw text on canvas using a small overlay buffer.

    Instead of converting the entire canvas (1920x1080) to PIL Image for
    each text label, this creates a small text overlay and blends it in.
    This is much faster when drawing multiple labels per frame.
    """
    font = _get_font(size)

    # Get text dimensions
    # Create a temporary draw context for measuring
    temp_img = Image.new('RGB', (1, 1))
    temp_draw = ImageDraw.Draw(temp_img)
    bbox = temp_draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0] + 4  # Small padding
    text_h = bbox[3] - bbox[1] + 4

    # Create small overlay with black background (matches canvas bg)
    overlay = Image.new('RGB', (text_w, text_h), (0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.text((2, 2), text, fill=color, font=font)

    # Convert to numpy and place on canvas
    overlay_arr = np.array(overlay)
    x, y = position
    ch, cw = canvas.shape[:2]

    # Bounds check
    if x < 0 or y < 0 or x + text_w > cw or y + text_h > ch:
        return

    canvas[y:y + text_h, x:x + text_w] = overlay_arr


def _draw_rectangle(canvas: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                    color: Tuple[int, int, int], fill: bool = False) -> None:
    """Draw rectangle on canvas."""
    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)
    if fill:
        draw.rectangle([x1, y1, x2, y2], fill=color)
    else:
        draw.rectangle([x1, y1, x2, y2], outline=color)
    canvas[:] = np.array(img)


def _draw_emphasis_border(canvas: np.ndarray, x: int, y: int, width: int, height: int,
                          color: Tuple[int, int, int], thickness: int = 4) -> None:
    """Draw an inset border inside the camera frame bounds.

    The border is drawn INSIDE the frame, never extending beyond allocated space.
    This prevents overlap with adjacent cameras regardless of emphasis state.

    Args:
        canvas: The canvas to draw on (modified in-place)
        x: Left edge of camera frame
        y: Top edge of camera frame
        width: Width of camera frame
        height: Height of camera frame
        color: RGB border color
        thickness: Border thickness in pixels (drawn inward)
    """
    # Clamp to canvas bounds
    ch, cw = canvas.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(cw, x + width)
    y2 = min(ch, y + height)

    # Ensure thickness doesn't exceed half the frame dimension
    t = min(thickness, (x2 - x1) // 4, (y2 - y1) // 4)
    if t < 1:
        return

    # Draw four border rectangles (inset)
    color_arr = np.array(color, dtype=np.uint8)

    # Top border
    canvas[y1:y1+t, x1:x2] = color_arr
    # Bottom border
    canvas[y2-t:y2, x1:x2] = color_arr
    # Left border
    canvas[y1:y2, x1:x1+t] = color_arr
    # Right border
    canvas[y1:y2, x2-t:x2] = color_arr


def _maybe_draw_emphasis_border(canvas: np.ndarray, x: int, y: int, width: int, height: int,
                                 emph: Optional['CameraEmphasis']) -> None:
    """Draw emphasis border if emphasis is active and visible.

    Consolidates the repeated pattern of checking emphasis state before drawing.
    Does nothing if emph is None, has no border_color, or weight is below threshold.

    Args:
        canvas: The canvas to draw on (modified in-place)
        x: Left edge of camera frame
        y: Top edge of camera frame
        width: Width of camera frame
        height: Height of camera frame
        emph: CameraEmphasis object or None
    """
    if emph and emph.border_color and emph.weight > EMPHASIS_VISIBILITY_THRESHOLD:
        _draw_emphasis_border(canvas, x, y, width, height,
                             emph.border_color, emph.border_width)


def _apply_emphasis_scale(frame: np.ndarray, base_size: Tuple[int, int],
                          weight: float, grow_direction: str = "center"
                          ) -> Tuple[np.ndarray, int, int]:
    """Scale a frame based on emphasis weight with directional growth.

    Args:
        frame: Input frame to scale
        base_size: Target (width, height) without emphasis
        weight: Emphasis weight 0.0-1.0
        grow_direction: How the frame grows:
            - "center": Grows toward center (both directions equally)
            - "right": Grows rightward (for left-side cameras)
            - "left": Grows leftward (for right-side cameras)
            - "up": Grows upward (for bottom cameras)

    Returns:
        Tuple of (scaled_frame, x_offset, y_offset)
        Offsets indicate how much to adjust placement position
    """
    base_w, base_h = base_size

    # Fast path: if weight is negligible, skip scale calculation entirely
    # This saves the scale_boost multiplication and extra resize overhead
    if weight < 0.02:  # 2% threshold - imperceptible difference
        resized = _resize_frame(frame, base_size)
        return resized, 0, 0

    scale_boost = weight * EMPHASIS_MAX_SCALE_BOOST

    new_w = int(base_w * (1 + scale_boost))
    new_h = int(base_h * (1 + scale_boost))

    # Calculate growth amount
    grow_w = new_w - base_w
    grow_h = new_h - base_h

    # Resize frame to new dimensions
    resized = _resize_frame(frame, (new_w, new_h))

    # Calculate offsets based on growth direction
    if grow_direction == "center":
        x_offset = -grow_w // 2
        y_offset = -grow_h // 2
    elif grow_direction == "right":
        # Left cameras grow right - no x offset needed (anchored left)
        x_offset = 0
        y_offset = -grow_h // 2
    elif grow_direction == "left":
        # Right cameras grow left - offset left by full growth
        x_offset = -grow_w
        y_offset = -grow_h // 2
    elif grow_direction == "up":
        # Bottom cameras grow up - offset up by full growth
        x_offset = -grow_w // 2
        y_offset = -grow_h
    else:
        x_offset = 0
        y_offset = 0

    return resized, x_offset, y_offset


def _clamp_mutual_emphasis(left_weight: float, right_weight: float,
                           max_single: float = 1.0, max_mutual: float = 0.5
                           ) -> Tuple[float, float]:
    """Clamp emphasis weights when both left and right cameras are emphasized.

    Prevents center collision by reducing both weights when simultaneous.

    Args:
        left_weight: Raw left camera emphasis (0.0-1.0)
        right_weight: Raw right camera emphasis (0.0-1.0)
        max_single: Max weight when only one side emphasized
        max_mutual: Max weight per side when both emphasized

    Returns:
        Tuple of (clamped_left, clamped_right)
    """
    if left_weight > 0.01 and right_weight > 0.01:
        # Both sides emphasized - cap each to max_mutual
        return min(left_weight, max_mutual), min(right_weight, max_mutual)
    return min(left_weight, max_single), min(right_weight, max_single)


def composite_frame(front: np.ndarray,
                    left_rep: Optional[np.ndarray] = None,
                    right_rep: Optional[np.ndarray] = None,
                    back: Optional[np.ndarray] = None,
                    left_pill: Optional[np.ndarray] = None,
                    right_pill: Optional[np.ndarray] = None,
                    cameras: Optional[Set[str]] = None,
                    layout: str = "grid",
                    emphasis: Optional['EmphasisState'] = None) -> np.ndarray:
    """
    Composite multiple camera frames into a single output frame.

    Adaptive layout based on selected cameras and layout mode:
    - Front only: Full screen 1920x1080
    - Front + back: Top/bottom split
    - Front + repeaters: Center + side panels
    - All cameras with layout="pip": Fullscreen front with PIP thumbnails
    - Multi-camera: Original 6-camera grid layout (default)

    Args:
        front: Front camera frame (required)
        left_rep: Left repeater camera frame
        right_rep: Right repeater camera frame
        back: Rear camera frame
        left_pill: Left pillar camera frame
        right_pill: Right pillar camera frame
        cameras: Set of camera names to include
        layout: "grid" (default 6-camera grid) or "pip" (fullscreen with thumbnails)
        emphasis: Optional EmphasisState for dynamic camera emphasis
    """
    canvas = np.zeros((OUTPUT_HEIGHT, OUTPUT_WIDTH, 3), dtype=np.uint8)

    # Default to all cameras if not specified (backward compatibility)
    if cameras is None:
        cameras = {"front", "back", "left_repeater", "right_repeater", "left_pillar", "right_pillar"}

    active_cams = cameras - {"front"}

    # --- LAYOUT 1: Front only (full screen) ---
    if len(active_cams) == 0:
        if front is not None:
            resized = _resize_frame(front, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
            canvas[:, :] = resized
        return canvas

    # --- LAYOUT 2: Front + back only (top/bottom split) ---
    if active_cams == {"back"}:
        if front is not None:
            resized = _resize_frame(front, (OUTPUT_WIDTH, SPLIT_HEIGHT))
            canvas[0:SPLIT_HEIGHT, :] = resized
        if back is not None:
            flipped = _flip_horizontal(back)
            back_emph = emphasis.back if emphasis else None
            if back_emph and back_emph.weight > 0.01:
                # Back camera grows upward when emphasized
                resized, x_off, y_off = _apply_emphasis_scale(
                    flipped, (OUTPUT_WIDTH, SPLIT_HEIGHT), back_emph.weight, "up")
                # Clamp placement to canvas
                bx = max(0, x_off)
                by = max(SPLIT_HEIGHT, SPLIT_HEIGHT + y_off)
                bw = min(resized.shape[1], OUTPUT_WIDTH - bx)
                bh = min(resized.shape[0], OUTPUT_HEIGHT - by)
                canvas[by:by+bh, bx:bx+bw] = resized[:bh, :bw]
                _maybe_draw_emphasis_border(canvas, 0, SPLIT_HEIGHT, OUTPUT_WIDTH, SPLIT_HEIGHT, back_emph)
            else:
                resized = _resize_frame(flipped, (OUTPUT_WIDTH, SPLIT_HEIGHT))
                canvas[SPLIT_HEIGHT:OUTPUT_HEIGHT, :] = resized
            _draw_text(canvas, "Back", (20, SPLIT_HEIGHT + 10), COLORS.WHITE, 16)
        return canvas

    # --- LAYOUT 3: Front + side repeaters ---
    if active_cams == {"left_repeater", "right_repeater"}:
        # Get emphasis weights with mutual clamping
        left_emph = emphasis.left_repeater if emphasis else None
        right_emph = emphasis.right_repeater if emphasis else None
        left_w = left_emph.weight if left_emph else 0.0
        right_w = right_emph.weight if right_emph else 0.0
        left_w, right_w = _clamp_mutual_emphasis(left_w, right_w)

        if front is not None:
            resized = _resize_frame(front, (REPEATER_LAYOUT_CENTER_WIDTH, OUTPUT_HEIGHT))
            canvas[:, REPEATER_LAYOUT_SIDE_WIDTH:REPEATER_LAYOUT_SIDE_WIDTH + REPEATER_LAYOUT_CENTER_WIDTH] = resized
        if left_rep is not None:
            if left_w > 0.01:
                resized, x_off, y_off = _apply_emphasis_scale(
                    left_rep, (REPEATER_LAYOUT_SIDE_WIDTH, OUTPUT_HEIGHT), left_w, "right")
                h, w = resized.shape[:2]
                # Left camera anchored at left edge, grows right
                canvas[max(0, y_off):max(0, y_off)+min(h, OUTPUT_HEIGHT), 0:min(w, REPEATER_LAYOUT_SIDE_WIDTH + int(REPEATER_LAYOUT_SIDE_WIDTH * EMPHASIS_MAX_SCALE_BOOST * left_w))] = resized[:min(h, OUTPUT_HEIGHT), :min(w, OUTPUT_WIDTH)]
                _maybe_draw_emphasis_border(canvas, 0, 0, REPEATER_LAYOUT_SIDE_WIDTH, OUTPUT_HEIGHT, left_emph)
            else:
                resized = _resize_frame(left_rep, (REPEATER_LAYOUT_SIDE_WIDTH, OUTPUT_HEIGHT))
                canvas[:, 0:REPEATER_LAYOUT_SIDE_WIDTH] = resized
            _draw_text(canvas, "L. Repeater", (10, 10), COLORS.WHITE, 12)
        if right_rep is not None:
            rx = REPEATER_LAYOUT_SIDE_WIDTH + REPEATER_LAYOUT_CENTER_WIDTH
            if right_w > 0.01:
                resized, x_off, y_off = _apply_emphasis_scale(
                    right_rep, (REPEATER_LAYOUT_SIDE_WIDTH, OUTPUT_HEIGHT), right_w, "left")
                h, w = resized.shape[:2]
                # Right camera anchored at right edge, grows left
                start_x = max(0, rx + x_off)
                canvas[max(0, y_off):max(0, y_off)+min(h, OUTPUT_HEIGHT), start_x:OUTPUT_WIDTH] = resized[:min(h, OUTPUT_HEIGHT), :OUTPUT_WIDTH-start_x]
                _maybe_draw_emphasis_border(canvas, rx, 0, REPEATER_LAYOUT_SIDE_WIDTH, OUTPUT_HEIGHT, right_emph)
            else:
                resized = _resize_frame(right_rep, (REPEATER_LAYOUT_SIDE_WIDTH, OUTPUT_HEIGHT))
                canvas[:, rx:] = resized
            _draw_text(canvas, "R. Repeater", (rx + 10, 10), COLORS.WHITE, 12)
        return canvas

    # --- LAYOUT 4: Front + back + repeaters (2x2 grid) ---
    if active_cams == {"back", "left_repeater", "right_repeater"}:
        # Get emphasis with mutual clamping for side cameras
        left_emph = emphasis.left_repeater if emphasis else None
        right_emph = emphasis.right_repeater if emphasis else None
        back_emph = emphasis.back if emphasis else None
        left_w = left_emph.weight if left_emph else 0.0
        right_w = right_emph.weight if right_emph else 0.0
        left_w, right_w = _clamp_mutual_emphasis(left_w, right_w)

        if front is not None:
            resized = _resize_frame(front, (GRID_2X2_HALF_WIDTH, GRID_2X2_HALF_HEIGHT))
            canvas[0:GRID_2X2_HALF_HEIGHT, 0:GRID_2X2_HALF_WIDTH] = resized
        if back is not None:
            flipped = _flip_horizontal(back)
            resized = _resize_frame(flipped, (GRID_2X2_HALF_WIDTH, GRID_2X2_HALF_HEIGHT))
            canvas[0:GRID_2X2_HALF_HEIGHT, GRID_2X2_HALF_WIDTH:] = resized
            _draw_text(canvas, "Back", (GRID_2X2_HALF_WIDTH + 20, 10), COLORS.WHITE, 16)
            _maybe_draw_emphasis_border(canvas, GRID_2X2_HALF_WIDTH, 0,
                                       GRID_2X2_HALF_WIDTH, GRID_2X2_HALF_HEIGHT, back_emph)
        if left_rep is not None:
            resized = _resize_frame(left_rep, (GRID_2X2_HALF_WIDTH, GRID_2X2_HALF_HEIGHT))
            canvas[GRID_2X2_HALF_HEIGHT:, 0:GRID_2X2_HALF_WIDTH] = resized
            _draw_text(canvas, "L. Repeater", (20, GRID_2X2_HALF_HEIGHT + 10), COLORS.WHITE, 16)
            _maybe_draw_emphasis_border(canvas, 0, GRID_2X2_HALF_HEIGHT,
                                       GRID_2X2_HALF_WIDTH, GRID_2X2_HALF_HEIGHT, left_emph)
        if right_rep is not None:
            resized = _resize_frame(right_rep, (GRID_2X2_HALF_WIDTH, GRID_2X2_HALF_HEIGHT))
            canvas[GRID_2X2_HALF_HEIGHT:, GRID_2X2_HALF_WIDTH:] = resized
            _draw_text(canvas, "R. Repeater", (GRID_2X2_HALF_WIDTH + 20, GRID_2X2_HALF_HEIGHT + 10),
                      COLORS.WHITE, 16)
            _maybe_draw_emphasis_border(canvas, GRID_2X2_HALF_WIDTH, GRID_2X2_HALF_HEIGHT,
                                       GRID_2X2_HALF_WIDTH, GRID_2X2_HALF_HEIGHT, right_emph)
        return canvas

    # --- LAYOUT 5: Front + left repeater (side by side) ---
    if active_cams == {"left_repeater"}:
        if left_rep is not None:
            resized = _resize_frame(left_rep, (SIDE_LAYOUT_SIDE_WIDTH, OUTPUT_HEIGHT))
            canvas[:, 0:SIDE_LAYOUT_SIDE_WIDTH] = resized
            _draw_text(canvas, "L. Repeater", (20, 10), COLORS.WHITE, 16)
            # Emphasis border for left repeater
            if emphasis:
                _maybe_draw_emphasis_border(canvas, 0, 0, SIDE_LAYOUT_SIDE_WIDTH, OUTPUT_HEIGHT,
                                           emphasis.get('left_repeater'))
        if front is not None:
            resized = _resize_frame(front, (SIDE_LAYOUT_FRONT_WIDTH, OUTPUT_HEIGHT))
            canvas[:, SIDE_LAYOUT_SIDE_WIDTH:] = resized
        return canvas

    # --- LAYOUT 6: Front + right repeater (side by side) ---
    if active_cams == {"right_repeater"}:
        if front is not None:
            resized = _resize_frame(front, (SIDE_LAYOUT_FRONT_WIDTH, OUTPUT_HEIGHT))
            canvas[:, 0:SIDE_LAYOUT_FRONT_WIDTH] = resized
        if right_rep is not None:
            resized = _resize_frame(right_rep, (SIDE_LAYOUT_SIDE_WIDTH, OUTPUT_HEIGHT))
            canvas[:, SIDE_LAYOUT_FRONT_WIDTH:] = resized
            _draw_text(canvas, "R. Repeater", (SIDE_LAYOUT_FRONT_WIDTH + 20, 10), COLORS.WHITE, 16)
            # Emphasis border for right repeater
            if emphasis:
                _maybe_draw_emphasis_border(canvas, SIDE_LAYOUT_FRONT_WIDTH, 0, SIDE_LAYOUT_SIDE_WIDTH, OUTPUT_HEIGHT,
                                           emphasis.get('right_repeater'))
        return canvas

    # --- LAYOUT 7: Front + left pillar (side by side) ---
    if active_cams == {"left_pillar"}:
        if left_pill is not None:
            resized = _resize_frame(left_pill, (SIDE_LAYOUT_SIDE_WIDTH, OUTPUT_HEIGHT))
            canvas[:, 0:SIDE_LAYOUT_SIDE_WIDTH] = resized
            _draw_text(canvas, "L. Pillar", (20, 10), COLORS.WHITE, 16)
            # Emphasis border for left pillar
            if emphasis:
                _maybe_draw_emphasis_border(canvas, 0, 0, SIDE_LAYOUT_SIDE_WIDTH, OUTPUT_HEIGHT,
                                           emphasis.get('left_pillar'))
        if front is not None:
            resized = _resize_frame(front, (SIDE_LAYOUT_FRONT_WIDTH, OUTPUT_HEIGHT))
            canvas[:, SIDE_LAYOUT_SIDE_WIDTH:] = resized
        return canvas

    # --- LAYOUT 8: Front + right pillar (side by side) ---
    if active_cams == {"right_pillar"}:
        if front is not None:
            resized = _resize_frame(front, (SIDE_LAYOUT_FRONT_WIDTH, OUTPUT_HEIGHT))
            canvas[:, 0:SIDE_LAYOUT_FRONT_WIDTH] = resized
        if right_pill is not None:
            resized = _resize_frame(right_pill, (SIDE_LAYOUT_SIDE_WIDTH, OUTPUT_HEIGHT))
            canvas[:, SIDE_LAYOUT_FRONT_WIDTH:] = resized
            _draw_text(canvas, "R. Pillar", (SIDE_LAYOUT_FRONT_WIDTH + 20, 10), COLORS.WHITE, 16)
            # Emphasis border for right pillar
            if emphasis:
                _maybe_draw_emphasis_border(canvas, SIDE_LAYOUT_FRONT_WIDTH, 0, SIDE_LAYOUT_SIDE_WIDTH, OUTPUT_HEIGHT,
                                           emphasis.get('right_pillar'))
        return canvas

    # --- LAYOUT 9: Fullscreen front with PIP thumbnails (bottom-anchored) ---
    # Triggered when all 5 secondary cameras are present AND layout is "pip"
    # Top row (pillars): smaller 280x158 thumbnails
    # Bottom row (repeaters + rear): larger 350x197 thumbnails (1.25x) for emphasis
    # Dynamic scaling: thumbnails grow when emphasized (up to EMPHASIS_MAX_SCALE_BOOST)
    # Growth direction: toward screen center, with top row shifting up to avoid overlap
    if layout == "pip" and active_cams == {"back", "left_repeater", "right_repeater", "left_pillar", "right_pillar"}:
        # Front camera fills entire background
        if front is not None:
            resized_front = _resize_frame(front, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
            canvas[:, :] = resized_front

        # Get emphasis states for scaling and border drawing
        left_pill_emph = emphasis.left_pillar if emphasis else None
        right_pill_emph = emphasis.right_pillar if emphasis else None
        left_rep_emph = emphasis.left_repeater if emphasis else None
        right_rep_emph = emphasis.right_repeater if emphasis else None
        back_emph = emphasis.back if emphasis else None

        def _calc_pip_scale(emph, max_boost=EMPHASIS_MAX_SCALE_BOOST):
            """Calculate scale factor from emphasis weight (1.0 to 1.0 + max_boost)."""
            if not emph or emph.weight <= EMPHASIS_VISIBILITY_THRESHOLD:
                return 1.0
            return 1.0 + emph.weight * max_boost

        # Calculate scales for all cameras first to determine layout shifts
        # Rear camera uses larger scale boost (50%) for better visibility when reversing
        left_rep_scale = _calc_pip_scale(left_rep_emph)
        right_rep_scale = _calc_pip_scale(right_rep_emph)
        back_scale = _calc_pip_scale(back_emph, EMPHASIS_REAR_SCALE_BOOST)
        left_pill_scale = _calc_pip_scale(left_pill_emph)
        right_pill_scale = _calc_pip_scale(right_pill_emph)

        # Calculate how much the bottom row grows upward (max of all bottom cameras)
        max_bottom_scale = max(left_rep_scale, right_rep_scale, back_scale)
        bottom_growth = int(PIP_BOTTOM_THUMB_HEIGHT * (max_bottom_scale - 1.0))

        # Shift top row up to avoid overlap with bottom row
        top_row_y_offset = bottom_growth

        # --- BOTTOM ROW (draw first so top row overlaps if needed) ---
        # Bottom-left repeater: anchored at bottom-left, grows UP and RIGHT
        if left_rep is not None:
            scaled_w = int(PIP_BOTTOM_THUMB_WIDTH * left_rep_scale)
            scaled_h = int(PIP_BOTTOM_THUMB_HEIGHT * left_rep_scale)
            # Anchor at bottom-left (grows up and right)
            x = PIP_BOTTOM_LEFT_X
            y = PIP_BOTTOM_ROW_Y + PIP_BOTTOM_THUMB_HEIGHT - scaled_h
            resized = _resize_frame(left_rep, (scaled_w, scaled_h))
            # Clip to canvas bounds
            y = max(0, y)
            draw_h = min(scaled_h, OUTPUT_HEIGHT - y)
            draw_w = min(scaled_w, OUTPUT_WIDTH - x)
            canvas[y:y+draw_h, x:x+draw_w] = resized[:draw_h, :draw_w]
            _draw_text(canvas, "L-REPEATER", (x + 110, y + scaled_h - 30))
            _maybe_draw_emphasis_border(canvas, x, y, draw_w, draw_h, left_rep_emph)

        # Rear camera: anchored at bottom-center, grows UP (centered horizontally)
        if back is not None:
            scaled_w = int(PIP_BOTTOM_THUMB_WIDTH * back_scale)
            scaled_h = int(PIP_BOTTOM_THUMB_HEIGHT * back_scale)
            # Anchor at bottom-center (grows up, centered)
            x = PIP_BOTTOM_CENTER_X + (PIP_BOTTOM_THUMB_WIDTH - scaled_w) // 2
            y = PIP_BOTTOM_ROW_Y + PIP_BOTTOM_THUMB_HEIGHT - scaled_h
            # Apply punch-in crop before resize to remove wide-angle vignetting
            cropped = _crop_center(back, PIP_REAR_CROP_PERCENT)
            resized = _flip_horizontal(_resize_frame(cropped, (scaled_w, scaled_h)))
            # Clip to canvas bounds
            y = max(0, y)
            x = max(0, x)
            draw_h = min(scaled_h, OUTPUT_HEIGHT - y)
            draw_w = min(scaled_w, OUTPUT_WIDTH - x)
            canvas[y:y+draw_h, x:x+draw_w] = resized[:draw_h, :draw_w]
            _draw_text(canvas, "REAR", (x + int(scaled_w * 0.41), y + scaled_h - 30))
            _maybe_draw_emphasis_border(canvas, x, y, draw_w, draw_h, back_emph)

        # Bottom-right repeater: anchored at bottom-right, grows UP and LEFT
        if right_rep is not None:
            scaled_w = int(PIP_BOTTOM_THUMB_WIDTH * right_rep_scale)
            scaled_h = int(PIP_BOTTOM_THUMB_HEIGHT * right_rep_scale)
            # Anchor at bottom-right (grows up and left)
            x = PIP_BOTTOM_RIGHT_X + PIP_BOTTOM_THUMB_WIDTH - scaled_w
            y = PIP_BOTTOM_ROW_Y + PIP_BOTTOM_THUMB_HEIGHT - scaled_h
            resized = _resize_frame(right_rep, (scaled_w, scaled_h))
            # Clip to canvas bounds
            x = max(0, x)
            y = max(0, y)
            draw_h = min(scaled_h, OUTPUT_HEIGHT - y)
            draw_w = min(scaled_w, OUTPUT_WIDTH - x)
            canvas[y:y+draw_h, x:x+draw_w] = resized[:draw_h, :draw_w]
            _draw_text(canvas, "R-REPEATER", (x + int(scaled_w * 0.30), y + scaled_h - 30))
            _maybe_draw_emphasis_border(canvas, x, y, draw_w, draw_h, right_rep_emph)

        # --- TOP ROW (draw after bottom row, with Y offset to avoid overlap) ---
        # Top-left pillar: anchored at bottom-left of its position, grows UP and RIGHT
        if left_pill is not None:
            scaled_w = int(PIP_TOP_THUMB_WIDTH * left_pill_scale)
            scaled_h = int(PIP_TOP_THUMB_HEIGHT * left_pill_scale)
            # Anchor at bottom-left of top row position (grows up and right)
            x = PIP_TOP_LEFT_X
            # Base Y is at bottom of top row slot, shifted up by bottom row growth
            base_bottom_y = PIP_TOP_ROW_Y + PIP_TOP_THUMB_HEIGHT - top_row_y_offset
            y = base_bottom_y - scaled_h
            resized = _resize_frame(left_pill, (scaled_w, scaled_h))
            # Clip to canvas bounds
            y = max(0, y)
            draw_h = min(scaled_h, OUTPUT_HEIGHT - y)
            draw_w = min(scaled_w, OUTPUT_WIDTH - x)
            canvas[y:y+draw_h, x:x+draw_w] = resized[:draw_h, :draw_w]
            _draw_text(canvas, "L-PILLAR", (x + 90, y + scaled_h - 30))
            _maybe_draw_emphasis_border(canvas, x, y, draw_w, draw_h, left_pill_emph)

        # Top-right pillar: anchored at bottom-right of its position, grows UP and LEFT
        if right_pill is not None:
            scaled_w = int(PIP_TOP_THUMB_WIDTH * right_pill_scale)
            scaled_h = int(PIP_TOP_THUMB_HEIGHT * right_pill_scale)
            # Anchor at bottom-right of top row position (grows up and left)
            x = PIP_TOP_RIGHT_X + PIP_TOP_THUMB_WIDTH - scaled_w
            # Base Y is at bottom of top row slot, shifted up by bottom row growth
            base_bottom_y = PIP_TOP_ROW_Y + PIP_TOP_THUMB_HEIGHT - top_row_y_offset
            y = base_bottom_y - scaled_h
            resized = _resize_frame(right_pill, (scaled_w, scaled_h))
            # Clip to canvas bounds
            x = max(0, x)
            y = max(0, y)
            draw_h = min(scaled_h, OUTPUT_HEIGHT - y)
            draw_w = min(scaled_w, OUTPUT_WIDTH - x)
            canvas[y:y+draw_h, x:x+draw_w] = resized[:draw_h, :draw_w]
            _draw_text(canvas, "R-PILLAR", (x + 90, y + scaled_h - 30))
            _maybe_draw_emphasis_border(canvas, x, y, draw_w, draw_h, right_pill_emph)

        return canvas

    # --- DEFAULT LAYOUT: Original 6-camera grid ---
    # Get emphasis states for border drawing
    left_pill_emph = emphasis.left_pillar if emphasis else None
    right_pill_emph = emphasis.right_pillar if emphasis else None
    left_rep_emph = emphasis.left_repeater if emphasis else None
    right_rep_emph = emphasis.right_repeater if emphasis else None
    back_emph = emphasis.back if emphasis else None

    # Front
    if front is not None:
        resized = _resize_frame(front, (FRONT_CAMERA_WIDTH, TOP_SECTION_HEIGHT))
        canvas[0:TOP_SECTION_HEIGHT, FRONT_CAMERA_X:FRONT_CAMERA_X + FRONT_CAMERA_WIDTH] = resized

    # Left Pillar
    if "left_pillar" in cameras:
        if left_pill is not None:
            resized = _resize_frame(left_pill, (PILLAR_WIDTH, PILLAR_HEIGHT))
            canvas[PILLAR_Y:PILLAR_Y + PILLAR_HEIGHT, 0:PILLAR_WIDTH] = resized
            _draw_text(canvas, "L. Pillar", (20, PILLAR_Y + 10), COLORS.WHITE, 16)
            _maybe_draw_emphasis_border(canvas, 0, PILLAR_Y, PILLAR_WIDTH, PILLAR_HEIGHT, left_pill_emph)
        else:
            _draw_rectangle(canvas, 0, PILLAR_Y, PILLAR_WIDTH, PILLAR_Y + PILLAR_HEIGHT,
                           COLORS.PLACEHOLDER_GREY, fill=True)

    # Right Pillar
    rx = FRONT_CAMERA_X + FRONT_CAMERA_WIDTH
    if "right_pillar" in cameras:
        if right_pill is not None:
            resized = _resize_frame(right_pill, (PILLAR_WIDTH, PILLAR_HEIGHT))
            canvas[PILLAR_Y:PILLAR_Y + PILLAR_HEIGHT, rx:rx + PILLAR_WIDTH] = resized
            _draw_text(canvas, "R. Pillar", (rx + 20, PILLAR_Y + 10), COLORS.WHITE, 16)
            _maybe_draw_emphasis_border(canvas, rx, PILLAR_Y, PILLAR_WIDTH, PILLAR_HEIGHT, right_pill_emph)
        else:
            _draw_rectangle(canvas, rx, PILLAR_Y, rx + PILLAR_WIDTH, PILLAR_Y + PILLAR_HEIGHT,
                           COLORS.PLACEHOLDER_GREY, fill=True)

    # Bottom Row
    bottom_cams = [
        ("left_repeater", left_rep, "L. Repeater", left_rep_emph),
        ("back", back, "Back", back_emph),
        ("right_repeater", right_rep, "R. Repeater", right_rep_emph)
    ]

    for i, (cam_name, cam, label, cam_emph) in enumerate(bottom_cams):
        if cam_name not in cameras:
            continue
        bx = i * BOTTOM_CAMERA_WIDTH

        if cam is not None:
            if cam_name == "back":
                cam = _flip_horizontal(cam)
            resized = _resize_frame(cam, (BOTTOM_CAMERA_WIDTH, BOTTOM_SECTION_HEIGHT))
            canvas[BOTTOM_SECTION_Y:BOTTOM_SECTION_Y + BOTTOM_SECTION_HEIGHT, bx:bx + BOTTOM_CAMERA_WIDTH] = resized
            _draw_text(canvas, label, (bx + 20, BOTTOM_SECTION_Y + 10), COLORS.WHITE, 16)
            _maybe_draw_emphasis_border(canvas, bx, BOTTOM_SECTION_Y,
                                       BOTTOM_CAMERA_WIDTH, BOTTOM_SECTION_HEIGHT, cam_emph)
        else:
            _draw_rectangle(canvas, bx, BOTTOM_SECTION_Y,
                           bx + BOTTOM_CAMERA_WIDTH, BOTTOM_SECTION_Y + BOTTOM_SECTION_HEIGHT,
                           COLORS.PLACEHOLDER_GREY, fill=True)
            _draw_text(canvas, label, (bx + 20, BOTTOM_SECTION_Y + 10), COLORS.GREY, 16)

    return canvas


def render_watermark(image_path: str, max_size: int = 150) -> Optional[np.ndarray]:
    """Load and resize a watermark image for overlay.

    Args:
        image_path: Path to watermark image (PNG, JPG, etc.)
        max_size: Maximum dimension (width or height) for the watermark

    Returns:
        RGB numpy array of the watermark, or None if loading fails.
        The watermark is resized to fit within max_size while preserving aspect ratio.
    """
    try:
        img = Image.open(image_path)

        # Convert to RGB (handle RGBA, grayscale, etc.)
        if img.mode == 'RGBA':
            # Composite over white background for consistent appearance
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])  # Use alpha channel as mask
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')

        # Resize preserving aspect ratio
        width, height = img.size
        if width > height:
            new_width = max_size
            new_height = int(height * max_size / width)
        else:
            new_height = max_size
            new_width = int(width * max_size / height)

        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        return np.array(img)
    except Exception as e:
        logger.warning(f"Failed to load watermark image: {e}")
        return None


def render_timestamp(canvas: np.ndarray, base_timestamp: str, elapsed_seconds: float,
                     scale: float = 1.0, position: Tuple[int, int] = (20, None)) -> None:
    """Render timestamp with running timecode on the canvas.

    Args:
        canvas: The video frame to draw on (modified in-place)
        base_timestamp: Base timestamp string (e.g., '2026-01-09 11:45:38')
        elapsed_seconds: Elapsed time in seconds from start of clip
        scale: Scaling factor for text size
        position: (x, y) position. If y is None, positions at bottom-left.
    """
    # Calculate timecode (HH:MM:SS)
    total_seconds = int(elapsed_seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60

    # Format: "2026-01-09 11:45:38 + 00:01:23"
    timecode = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    full_text = f"{base_timestamp} + {timecode}"

    # Get font
    font_size = int(18 * scale)
    font = _get_font(font_size)

    # Calculate text dimensions
    temp_img = Image.new('RGB', (1, 1))
    temp_draw = ImageDraw.Draw(temp_img)
    bbox = temp_draw.textbbox((0, 0), full_text, font=font)
    text_w = bbox[2] - bbox[0] + 16  # Padding
    text_h = bbox[3] - bbox[1] + 12

    # Position (default: bottom-left with margin)
    x = position[0]
    y = position[1] if position[1] is not None else (canvas.shape[0] - text_h - 20)

    # Create overlay with semi-transparent background
    overlay = Image.new('RGBA', (text_w, text_h), (0, 0, 0, 180))  # 70% opacity black
    draw = ImageDraw.Draw(overlay)

    # Draw text with shadow effect
    shadow_offset = max(1, int(1 * scale))
    draw.text((8 + shadow_offset, 6 + shadow_offset), full_text, fill=(0, 0, 0, 255), font=font)
    draw.text((8, 6), full_text, fill=COLORS.STEEL_BRIGHT + (255,), font=font)

    # Convert to RGB and blend onto canvas
    overlay_rgb = np.array(overlay.convert('RGB'))

    # Bounds check
    ch, cw = canvas.shape[:2]
    if x < 0 or y < 0 or x + text_w > cw or y + text_h > ch:
        return

    # Apply with alpha blending (overlay at 80% opacity)
    roi = canvas[y:y + text_h, x:x + text_w]
    alpha = 0.85
    blended = (roi.astype(np.float32) * (1 - alpha) + overlay_rgb.astype(np.float32) * alpha).astype(np.uint8)
    canvas[y:y + text_h, x:x + text_w] = blended
