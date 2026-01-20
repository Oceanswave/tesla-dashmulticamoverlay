"""
Constants for Tesla dashcam video processor.

Centralized definitions for layout dimensions, colors, and overlay settings.
"""

from typing import Tuple
from dataclasses import dataclass


# =============================================================================
# Output Video Dimensions
# =============================================================================

OUTPUT_WIDTH = 1920
OUTPUT_HEIGHT = 1080
OUTPUT_SIZE = (OUTPUT_WIDTH, OUTPUT_HEIGHT)


# =============================================================================
# Colors (RGB format for Pillow)
# =============================================================================

@dataclass(frozen=True)
class Colors:
    """Common colors in RGB format."""
    # Legacy colors (backward compatibility)
    WHITE: Tuple[int, int, int] = (255, 255, 255)
    BLACK: Tuple[int, int, int] = (0, 0, 0)
    GREEN: Tuple[int, int, int] = (0, 255, 0)
    RED: Tuple[int, int, int] = (255, 0, 0)
    BLUE: Tuple[int, int, int] = (0, 0, 255)
    GREY: Tuple[int, int, int] = (100, 100, 100)
    DARK_GREY: Tuple[int, int, int] = (30, 30, 30)
    PLACEHOLDER_GREY: Tuple[int, int, int] = (20, 20, 20)
    CYAN: Tuple[int, int, int] = (0, 255, 255)  # Path color on map

    # =========================================================================
    # Cybertruck-Inspired Palette
    # =========================================================================

    # Base metals (stainless steel tones)
    STEEL_BRIGHT: Tuple[int, int, int] = (200, 200, 205)   # Polished steel highlights
    STEEL_MID: Tuple[int, int, int] = (140, 145, 150)      # Primary steel
    STEEL_DARK: Tuple[int, int, int] = (80, 85, 90)        # Brushed steel
    GUNMETAL: Tuple[int, int, int] = (45, 50, 55)          # Dark gunmetal
    VOID_BLACK: Tuple[int, int, int] = (15, 18, 22)        # Deep background

    # Cybertruck orange accents (marker lights)
    CT_ORANGE: Tuple[int, int, int] = (255, 100, 0)        # Primary accent
    CT_ORANGE_GLOW: Tuple[int, int, int] = (255, 140, 50)  # Glow/highlight
    CT_ORANGE_DIM: Tuple[int, int, int] = (180, 70, 0)     # Inactive/dim

    # Status indicators
    CT_GREEN: Tuple[int, int, int] = (0, 255, 120)         # Success/eco (electric green)
    CT_AMBER: Tuple[int, int, int] = (255, 180, 0)         # Warning
    CT_RED: Tuple[int, int, int] = (255, 50, 50)           # Danger/brake

    # Speed zones (expanded gradient for smoother transitions)
    SPEED_ECO: Tuple[int, int, int] = (0, 255, 120)        # Bright green 0-25 (eco cruising)
    SPEED_CITY: Tuple[int, int, int] = (100, 255, 80)      # Yellow-green 25-40 (city driving)
    SPEED_SUBURBAN: Tuple[int, int, int] = (180, 230, 0)   # Yellow 40-55 (suburban)
    SPEED_HIGHWAY: Tuple[int, int, int] = (255, 180, 0)    # Amber 55-70 (highway)
    SPEED_FAST: Tuple[int, int, int] = (255, 120, 0)       # Orange 70-85 (fast highway)
    SPEED_VERY_FAST: Tuple[int, int, int] = (255, 70, 0)   # Red-orange 85-100 (very fast)
    SPEED_DANGER: Tuple[int, int, int] = (255, 40, 40)     # Red 100+ (danger)

    # Legacy speed zone aliases (backward compatibility)
    SPEED_LOW: Tuple[int, int, int] = (0, 255, 120)        # Green 0-40
    SPEED_MID: Tuple[int, int, int] = (255, 180, 0)        # Amber 40-80
    SPEED_HIGH: Tuple[int, int, int] = (255, 100, 0)       # Orange 80-100

    # G-Ball zones
    GBALL_SAFE: Tuple[int, int, int] = (0, 255, 120)       # Green < 0.3g
    GBALL_SPORT: Tuple[int, int, int] = (255, 100, 0)      # Orange 0.3-0.6g
    GBALL_LIMIT: Tuple[int, int, int] = (255, 50, 50)      # Red > 0.6g

    # Blinkers (signature orange)
    BLINKER_ACTIVE: Tuple[int, int, int] = (255, 100, 0)   # CT_ORANGE
    BLINKER_INACTIVE: Tuple[int, int, int] = (45, 50, 55)  # GUNMETAL


COLORS = Colors()


# =============================================================================
# Speed Zone Thresholds (MPH) - Expanded for smoother gradient
# =============================================================================

SPEED_ZONE_ECO = 25       # Below this: bright green (eco cruising)
SPEED_ZONE_CITY = 40      # Below this: yellow-green (city driving)
SPEED_ZONE_SUBURBAN = 55  # Below this: yellow (suburban)
SPEED_ZONE_HIGHWAY = 70   # Below this: amber (highway)
SPEED_ZONE_FAST = 85      # Below this: orange (fast highway)
SPEED_ZONE_VERY_FAST = 100  # Below this: red-orange; above: red (danger)

# Legacy thresholds (backward compatibility)
SPEED_ZONE_LOW = 40      # Below this: green (eco)
SPEED_ZONE_MID = 80      # Below this: amber
SPEED_ZONE_HIGH = 100    # Below this: orange; above: red (danger)


# =============================================================================
# G-Ball Zone Thresholds (G-force)
# =============================================================================

GBALL_ZONE_SAFE = 0.3    # Below this: green
GBALL_ZONE_SPORT = 0.6   # Below this: orange; above: red


# =============================================================================
# Dashboard Overlay Settings
# =============================================================================

DASHBOARD_WIDTH = 500  # Extended to fit G-ball indicator
DASHBOARD_HEIGHT = 200
DASHBOARD_SIZE = (DASHBOARD_WIDTH, DASHBOARD_HEIGHT)

# Speedometer gauge
SPEEDOMETER_CENTER = (70, 100)
SPEEDOMETER_RADIUS = 50
SPEEDOMETER_MAX_VALUE = 120  # MPH

# Steering wheel
STEERING_CENTER = (200, 100)
STEERING_RADIUS = 40
STEERING_THICKNESS = 4

# Pedals
PEDAL_CENTER = (330, 100)
PEDAL_BAR_WIDTH = 20
PEDAL_BAR_HEIGHT = 80

# G-Ball (acceleration indicator)
GBALL_CENTER = (450, 100)  # Position on dashboard
GBALL_RADIUS = 40          # Outer circle radius
GBALL_MAX_G = 1.0          # Max G-force for full deflection
GBALL_DOT_RADIUS = 6       # Size of the moving dot
MPS2_TO_G = 1.0 / 9.81     # Convert m/s² to G-force

# Blinker indicators (arrows at top of dashboard)
BLINKER_Y = 25             # Y position from top
BLINKER_LEFT_X = 50        # Left arrow X center
BLINKER_RIGHT_X = 450      # Right arrow X center
BLINKER_SIZE = 20          # Arrow size


# =============================================================================
# Map Overlay Settings
# =============================================================================

MAP_SIZE = 300
MAP_ZOOM_WINDOW = 0.002  # Degrees (~200m view) - default/fallback

# Dynamic map zoom based on speed (degrees, smaller = tighter zoom)
# Low-speed zones (expanded for parking/city detail)
MAP_ZOOM_PARKING = 0.0004     # 0-5 mph: ~30m view (parking maneuvers)
MAP_ZOOM_LOT = 0.0006         # 5-10 mph: ~45m view (parking lot)
MAP_ZOOM_RESIDENTIAL = 0.0008 # 10-15 mph: ~60m view (residential streets)
MAP_ZOOM_CITY_CRAWL = 0.0010  # 15-20 mph: ~75m view (heavy traffic)
MAP_ZOOM_CITY_SLOW = 0.0012   # 20-25 mph: ~100m view (city driving)
MAP_ZOOM_CITY_MODERATE = 0.0015  # 25-30 mph: ~120m view (moderate city)
# Higher-speed zones
MAP_ZOOM_CITY = 0.0018        # 30-45 mph: ~150m view (busy streets)
MAP_ZOOM_SUBURBAN = 0.0025    # 45-60 mph: ~220m view (suburban roads)
MAP_ZOOM_HIGHWAY = 0.0035     # 60-75 mph: ~300m view (highway)
MAP_ZOOM_FAST = 0.0045        # 75+ mph: ~400m view (fast highway)

# Legacy alias for backward compatibility
MAP_ZOOM_CREEPING = MAP_ZOOM_RESIDENTIAL

MAP_PADDING = 0.1
MAP_SUPERSAMPLE = 2  # Render at 2x resolution for sub-pixel smooth scrolling
MAP_ARROW_LENGTH = 15

# Map tile settings (for street/satellite modes)
MAP_TILE_ZOOM = 17  # ~150m view at zoom 17
MAP_TILE_CACHE_SIZE = 50  # Max cached tiles

# Rotation optimization: crop margin multiplier for crop-then-rotate strategy
# ROTATION_MARGIN = 1.5 means the crop radius is 1.5x larger (area is 2.25x)
# This ensures no corner clipping at 45° rotation (requires √2 ≈ 1.414x radius)
ROTATION_MARGIN = 1.5


# =============================================================================
# Overlay Positioning & Blending
# =============================================================================

OVERLAY_OPACITY = 0.8  # Overlay weight in blending
CANVAS_OPACITY = 0.2   # Background weight in blending

# Dashboard position (top center)
DASHBOARD_Y = 20
DASHBOARD_X_OFFSET = (OUTPUT_WIDTH - DASHBOARD_WIDTH) // 2

# Map position (top right)
MAP_Y = 20
MAP_X_MARGIN = 20  # Margin from right edge


# =============================================================================
# Layout: 6-Camera Grid (Default)
# =============================================================================

# Top section
TOP_SECTION_HEIGHT = 600
FRONT_CAMERA_WIDTH = 800
FRONT_CAMERA_X = (OUTPUT_WIDTH - FRONT_CAMERA_WIDTH) // 2

# Pillar cameras
PILLAR_WIDTH = 560
PILLAR_HEIGHT = 420
PILLAR_Y = (TOP_SECTION_HEIGHT - PILLAR_HEIGHT) // 2

# Bottom section
BOTTOM_SECTION_Y = 600
BOTTOM_SECTION_HEIGHT = 480
BOTTOM_CAMERA_WIDTH = 640


# =============================================================================
# Layout: Front Only
# =============================================================================

FRONT_ONLY_SIZE = (OUTPUT_WIDTH, OUTPUT_HEIGHT)


# =============================================================================
# Layout: Front + Back (Top/Bottom Split)
# =============================================================================

SPLIT_HEIGHT = 540  # Each half


# =============================================================================
# Layout: Front + Side Repeaters
# =============================================================================

REPEATER_LAYOUT_CENTER_WIDTH = 1280
REPEATER_LAYOUT_SIDE_WIDTH = 320  # (1920 - 1280) / 2


# =============================================================================
# Layout: 2x2 Grid (Front + Back + Repeaters)
# =============================================================================

GRID_2X2_HALF_WIDTH = 960
GRID_2X2_HALF_HEIGHT = 540


# =============================================================================
# Layout: Front + Single Side Camera (Repeater or Pillar)
# =============================================================================

# Side-by-side layout: Front takes 2/3, side camera takes 1/3
SIDE_LAYOUT_FRONT_WIDTH = 1280  # 2/3 of 1920
SIDE_LAYOUT_SIDE_WIDTH = 640    # 1/3 of 1920


# =============================================================================
# Layout: Fullscreen Front with PIP Thumbnails (Bottom-Anchored)
# =============================================================================
# Layout pushes thumbnails to bottom edge with larger bottom row for
# more important cameras (repeaters + rear). Bottom row is 1.25x larger.

# Top row (pillars) - standard size
PIP_TOP_THUMB_WIDTH = 280
PIP_TOP_THUMB_HEIGHT = 158  # Maintains ~16:9 aspect

# Bottom row (repeaters + rear) - 1.25x larger for emphasis
PIP_BOTTOM_THUMB_WIDTH = 350   # 280 * 1.25
PIP_BOTTOM_THUMB_HEIGHT = 197  # 158 * 1.25

PIP_EDGE_MARGIN = 40
PIP_ROW_GAP = 15  # Gap between top and bottom rows

# Row Y positions (calculated bottom-up for bottom-anchored layout)
# Bottom margin: 40px, bottom row height: 197px, gap: 15px, top row height: 158px
PIP_BOTTOM_ROW_Y = OUTPUT_HEIGHT - PIP_EDGE_MARGIN - PIP_BOTTOM_THUMB_HEIGHT  # 843
PIP_TOP_ROW_Y = PIP_BOTTOM_ROW_Y - PIP_ROW_GAP - PIP_TOP_THUMB_HEIGHT  # 670

# Top row X positions (280px thumbnails)
PIP_TOP_LEFT_X = PIP_EDGE_MARGIN  # 40
PIP_TOP_RIGHT_X = OUTPUT_WIDTH - PIP_EDGE_MARGIN - PIP_TOP_THUMB_WIDTH  # 1600

# Bottom row X positions (350px thumbnails)
PIP_BOTTOM_LEFT_X = PIP_EDGE_MARGIN  # 40
PIP_BOTTOM_CENTER_X = (OUTPUT_WIDTH - PIP_BOTTOM_THUMB_WIDTH) // 2  # 785
PIP_BOTTOM_RIGHT_X = OUTPUT_WIDTH - PIP_EDGE_MARGIN - PIP_BOTTOM_THUMB_WIDTH  # 1530

# Rear camera punch-in (crop to remove wide-angle vignetting)
PIP_REAR_CROP_PERCENT = 0.05  # 5% from each edge

# Legacy aliases for backward compatibility (deprecated)
PIP_THUMB_WIDTH = PIP_TOP_THUMB_WIDTH
PIP_THUMB_HEIGHT = PIP_TOP_THUMB_HEIGHT
PIP_LEFT_X = PIP_TOP_LEFT_X
PIP_RIGHT_X = PIP_TOP_RIGHT_X
PIP_CENTER_X = PIP_BOTTOM_CENTER_X


# =============================================================================
# Conversion Constants
# =============================================================================

MPS_TO_MPH = 2.23694  # meters per second to miles per hour
TESLA_DASHCAM_FPS = 30000 / 1001  # NTSC standard ~29.97 fps


# =============================================================================
# Valid Camera Names
# =============================================================================

ALL_CAMERAS = frozenset({
    "front",
    "back",
    "left_repeater",
    "right_repeater",
    "left_pillar",
    "right_pillar"
})


# =============================================================================
# Color Grading Presets
# =============================================================================
# Each preset defines adjustments for: brightness, contrast, saturation,
# gamma, shadows, highlights. Values are additive (except gamma is multiplicative).
# Presets can be stacked with manual CLI adjustments.

# =============================================================================
# Dynamic Camera Emphasis Settings
# =============================================================================

# Emphasis thresholds (G-force)
EMPHASIS_LATERAL_G_THRESHOLD = 0.2    # Minimum lateral G for turn emphasis
EMPHASIS_BRAKING_G_THRESHOLD = 0.3    # Minimum braking G for rear emphasis

# Smoothing factor for gradual transitions (matches map zoom)
EMPHASIS_SMOOTHING_FACTOR = 0.3       # 30% interpolation per frame

# Visual parameters
EMPHASIS_MAX_SCALE_BOOST = 0.15       # Max 15% size increase
EMPHASIS_BORDER_WIDTH = 4             # Border thickness in pixels
EMPHASIS_VISIBILITY_THRESHOLD = 0.05  # Minimum weight to show border

# Emphasis colors (from Cybertruck palette)
EMPHASIS_COLOR_BLINKER = COLORS.CT_ORANGE   # Turn signal emphasis
EMPHASIS_COLOR_BRAKE = COLORS.CT_RED        # Braking emphasis
EMPHASIS_COLOR_REVERSE = COLORS.CT_GREEN    # Reverse gear emphasis
EMPHASIS_COLOR_LATERAL = COLORS.CT_AMBER    # Lateral G (turn) emphasis


COLOR_PRESETS = {
    # Cinematic: Slight lift in shadows, boosted contrast, muted colors
    "cinematic": {
        "brightness": -0.05,
        "contrast": 0.15,
        "saturation": -0.1,
        "gamma": 1.1,
        "shadows": 0.08,
        "highlights": -0.05,
    },

    # Warm: Golden hour feel, boosted warmth (via saturation), lifted shadows
    "warm": {
        "brightness": 0.02,
        "contrast": 0.05,
        "saturation": 0.1,
        "gamma": 0.95,
        "shadows": 0.05,
        "highlights": 0.0,
    },

    # Cool: Blue hour feel, slightly desaturated, crisp contrast
    "cool": {
        "brightness": 0.0,
        "contrast": 0.08,
        "saturation": -0.05,
        "gamma": 1.05,
        "shadows": -0.02,
        "highlights": 0.02,
    },

    # Vivid: Punchy colors, high contrast, great for dashcam clarity
    "vivid": {
        "brightness": 0.05,
        "contrast": 0.2,
        "saturation": 0.35,
        "gamma": 0.98,
        "shadows": 0.0,
        "highlights": 0.0,
    },

    # Cybertruck: Cold steel aesthetic matching the UI theme
    # Desaturated, slight blue shift feel, industrial contrast
    "cybertruck": {
        "brightness": -0.02,
        "contrast": 0.12,
        "saturation": -0.15,
        "gamma": 1.08,
        "shadows": 0.02,
        "highlights": -0.03,
    },

    # Dramatic: High contrast, crushed blacks, for intense footage
    "dramatic": {
        "brightness": -0.08,
        "contrast": 0.25,
        "saturation": 0.05,
        "gamma": 1.15,
        "shadows": -0.1,
        "highlights": 0.05,
    },

    # Vintage: Faded look, lifted blacks, reduced contrast
    "vintage": {
        "brightness": 0.03,
        "contrast": -0.05,
        "saturation": -0.2,
        "gamma": 0.92,
        "shadows": 0.15,
        "highlights": -0.08,
    },

    # Natural: Subtle enhancement, minimal processing
    "natural": {
        "brightness": 0.0,
        "contrast": 0.05,
        "saturation": 0.05,
        "gamma": 1.0,
        "shadows": 0.02,
        "highlights": 0.0,
    },
}
