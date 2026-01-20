"""
GPS Map overlay renderer for Tesla dashcam.

Provides MapRenderer class for rendering GPS path maps with heading-up rotation
and optional satellite/street tile backgrounds.

Key design principles:
- In heading-up mode, the direction of travel ALWAYS points UP on screen
- The map rotates around you as you turn (like a car GPS)
- When you turn right, the world rotates left
- The "N" compass shows where north is relative to your heading
"""

import logging
import math
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from constants import (
    COLORS,
    MAP_SIZE, MAP_ZOOM_WINDOW, MAP_PADDING, MAP_ARROW_LENGTH, MAP_SUPERSAMPLE,
    SPEED_ZONE_ECO, SPEED_ZONE_CITY, SPEED_ZONE_SUBURBAN,
    SPEED_ZONE_HIGHWAY, SPEED_ZONE_FAST, SPEED_ZONE_VERY_FAST,
)

logger = logging.getLogger(__name__)


# Font cache
_font_cache: dict = {}
_font_path: Optional[str] = None


def _get_font(size: float = 12) -> ImageFont.FreeTypeFont:
    """Get a cached font instance."""
    global _font_path

    int_size = int(size)

    if int_size in _font_cache:
        return _font_cache[int_size]

    if _font_path is None:
        for font_name in ["DejaVuSans.ttf", "Arial.ttf", "Helvetica.ttf",
                          "/System/Library/Fonts/Helvetica.ttc"]:
            try:
                ImageFont.truetype(font_name, 12)
                _font_path = font_name
                break
            except (OSError, IOError):
                continue

    try:
        if _font_path:
            font = ImageFont.truetype(_font_path, int_size)
        else:
            font = ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    _font_cache[int_size] = font
    return font


class MapRenderer:
    """Renders GPS path map overlay with heading-up rotation.

    In heading-up mode:
    - Your direction of travel always points UP
    - The map rotates around you as you turn
    - North compass indicator shows where north is

    Args:
        scale: Scaling factor for overlay size (default 1.0)
        history: Initial GPS history points as [(lat, lon), ...]
        map_style: "simple" (vector), "street" (OSM tiles), or "satellite" (aerial)
        heading_up: If True, rotate map so direction of travel is always up
    """

    def __init__(self, scale: float = 1.0, history: List[Tuple[float, float]] = None,
                 map_style: str = "simple", heading_up: bool = True):
        self.scale = scale
        self.size = int(MAP_SIZE * scale)  # Final output size
        self.path: List[Tuple[float, float]] = history if history else []
        self.padding = MAP_PADDING
        self.zoom_window = MAP_ZOOM_WINDOW
        self.map_style = map_style
        self.heading_up = heading_up

        # Supersampling for sub-pixel smooth scrolling
        # By rendering at 2x and downscaling, sub-pixel movements become visible
        # as anti-aliased shifts instead of being rounded to 0
        self._supersample = MAP_SUPERSAMPLE
        self._internal_size = self.size * self._supersample

        # Scaled arrow and font (at internal resolution)
        self._arrow_length = int(MAP_ARROW_LENGTH * scale * self._supersample)
        self._font = _get_font(12 * scale * self._supersample)

        # Heading smoothing to reduce jerkiness
        self._smoothed_heading: Optional[float] = None
        self._heading_smooth_factor = 0.15  # Smooth rotations

        # Tile grid cache for street/satellite modes
        # Tiles are cached by their grid coordinates (z, x, y), not arbitrary positions
        self._tile_grid_cache: dict = {}  # (z, x, y) -> PIL Image (256x256)
        self._staticmap = None
        self._warned_fallback = False

        # Composite tile settings
        # We stitch together a grid of tiles into one big composite
        self._tile_size = 256  # Standard web map tile size
        # Fetch 11x11 grid (5 tiles in each direction) for:
        # - Rotation headroom (45° rotation needs √2 more coverage)
        # - Prefetch margin so tiles are ready before we scroll to edge
        self._composite_radius = 5  # 11x11 grid = 2816px composite
        self._composite: Optional[Image.Image] = None  # Stitched composite image
        self._composite_origin_x: int = 0  # Tile X coordinate of top-left of composite
        self._composite_origin_y: int = 0  # Tile Y coordinate of top-left of composite
        self._composite_zoom: int = 0  # Zoom level of composite

    def update(self, lat: float, lon: float, speed_mps: float = 0.0) -> None:
        """Add a GPS point to the path history.

        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees
            speed_mps: Vehicle speed in meters per second (for path coloring)
        """
        # Filter null island and obviously invalid coordinates
        if abs(lat) < 0.001 and abs(lon) < 0.001:
            return
        self.path.append((lat, lon, speed_mps))

    def _smooth_heading(self, heading_deg: float) -> float:
        """Apply exponential smoothing to heading to reduce jerkiness.

        Handles the 360/0 degree wraparound correctly.
        """
        if self._smoothed_heading is None:
            self._smoothed_heading = heading_deg
            return heading_deg

        # Handle wraparound: find the shortest angular distance
        diff = heading_deg - self._smoothed_heading

        # Normalize diff to [-180, 180]
        while diff > 180:
            diff -= 360
        while diff < -180:
            diff += 360

        # Apply smoothing
        self._smoothed_heading += diff * self._heading_smooth_factor

        # Normalize result to [0, 360)
        while self._smoothed_heading < 0:
            self._smoothed_heading += 360
        while self._smoothed_heading >= 360:
            self._smoothed_heading -= 360

        return self._smoothed_heading

    def _latlon_to_tile(self, lat: float, lon: float, zoom: int) -> Tuple[int, int]:
        """Convert lat/lon to tile coordinates at given zoom level.

        Returns (tile_x, tile_y) - integer tile coordinates in the global grid.
        """
        n = 2 ** zoom
        tile_x = int((lon + 180.0) / 360.0 * n)
        lat_rad = math.radians(lat)
        tile_y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
        return tile_x, tile_y

    def _tile_to_latlon(self, tile_x: int, tile_y: int, zoom: int) -> Tuple[float, float]:
        """Convert tile coordinates to lat/lon of the tile's NW corner."""
        n = 2 ** zoom
        lon = tile_x / n * 360.0 - 180.0
        lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * tile_y / n)))
        lat = math.degrees(lat_rad)
        return lat, lon

    def _latlon_to_pixel(self, lat: float, lon: float, zoom: int) -> Tuple[float, float]:
        """Convert lat/lon to absolute pixel coordinates at given zoom level.

        Returns (pixel_x, pixel_y) - floating point for sub-pixel precision.
        """
        n = 2 ** zoom
        pixel_x = (lon + 180.0) / 360.0 * n * self._tile_size
        lat_rad = math.radians(lat)
        pixel_y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n * self._tile_size
        return pixel_x, pixel_y

    def _fetch_single_tile(self, z: int, x: int, y: int) -> Optional[Image.Image]:
        """Fetch a single map tile by its grid coordinates.

        Returns the 256x256 tile image, or None on failure.
        Caches tiles by (z, x, y) for reuse.
        """
        cache_key = (z, x, y, self.map_style)
        if cache_key in self._tile_grid_cache:
            return self._tile_grid_cache[cache_key]

        try:
            import requests
            from io import BytesIO

            if self.map_style == "satellite":
                url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
            else:
                # OpenStreetMap
                url = f"https://tile.openstreetmap.org/{z}/{x}/{y}.png"

            headers = {"User-Agent": "TeslaDashcamOverlay/1.0"}
            resp = requests.get(url, headers=headers, timeout=5)
            resp.raise_for_status()

            img = Image.open(BytesIO(resp.content)).convert('RGB')
            self._tile_grid_cache[cache_key] = img

            # Limit cache size (keep ~100 tiles max)
            if len(self._tile_grid_cache) > 100:
                # Remove oldest entries
                keys = list(self._tile_grid_cache.keys())
                for k in keys[:20]:
                    del self._tile_grid_cache[k]

            return img

        except Exception as e:
            logger.debug(f"Failed to fetch tile {z}/{x}/{y}: {e}")
            return None

    def _build_composite(self, center_tile_x: int, center_tile_y: int, zoom: int) -> Optional[Image.Image]:
        """Build a composite image from a grid of tiles centered on given tile coords.

        Fetches tiles in a radius around the center and stitches them together.
        """
        radius = self._composite_radius
        grid_size = 2 * radius + 1  # e.g., radius=3 -> 7x7 grid
        composite_size = grid_size * self._tile_size  # e.g., 7*256 = 1792

        composite = Image.new('RGB', (composite_size, composite_size), COLORS.VOID_BLACK)

        tiles_fetched = 0
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                tile_x = center_tile_x + dx
                tile_y = center_tile_y + dy

                tile_img = self._fetch_single_tile(zoom, tile_x, tile_y)
                if tile_img is not None:
                    # Position in composite (top-left tile is at 0,0)
                    px = (dx + radius) * self._tile_size
                    py = (dy + radius) * self._tile_size
                    composite.paste(tile_img, (px, py))
                    tiles_fetched += 1

        if tiles_fetched == 0:
            return None

        return composite

    def _fetch_tile(self, lat: float, lon: float) -> Tuple[Optional[Image.Image], float, float]:
        """Fetch map tile composite for given coordinates using tile grid strategy.

        Returns (composite_image, origin_lat, origin_lon).

        Uses tile grid caching with predictive prefetching:
        1. Convert position to tile grid coordinates
        2. Check if current composite covers this position with margin
        3. If not, build new composite centered on current tile
        4. Return composite with its origin (NW corner) for offset calculation

        The composite is a stitched grid of standard 256px tiles, giving us
        a large seamless "virtual tile" with no boundary jumps.
        """
        from constants import MAP_TILE_ZOOM
        zoom = MAP_TILE_ZOOM

        # Convert current position to tile coordinates
        tile_x, tile_y = self._latlon_to_tile(lat, lon, zoom)

        # Check if current composite covers this position with enough margin
        # We want at least 2 tiles of margin before we need to rebuild
        # This ensures tiles are prefetched before we need them for rotation/scrolling
        if self._composite is not None and self._composite_zoom == zoom:
            margin = 2  # Rebuild when within 2 tiles of edge (gives rotation headroom)
            min_x = self._composite_origin_x + margin
            max_x = self._composite_origin_x + 2 * self._composite_radius - margin
            min_y = self._composite_origin_y + margin
            max_y = self._composite_origin_y + 2 * self._composite_radius - margin

            if min_x <= tile_x <= max_x and min_y <= tile_y <= max_y:
                # Still within safe zone of current composite
                origin_lat, origin_lon = self._tile_to_latlon(
                    self._composite_origin_x, self._composite_origin_y, zoom
                )
                return self._composite, origin_lat, origin_lon

        # Need to build/rebuild composite centered on current tile
        composite = self._build_composite(tile_x, tile_y, zoom)

        if composite is None:
            if not self._warned_fallback:
                logger.warning("Failed to fetch map tiles. Falling back to simple mode.")
                self._warned_fallback = True
            return None, lat, lon

        # Store composite and its origin
        self._composite = composite
        self._composite_origin_x = tile_x - self._composite_radius
        self._composite_origin_y = tile_y - self._composite_radius
        self._composite_zoom = zoom

        origin_lat, origin_lon = self._tile_to_latlon(
            self._composite_origin_x, self._composite_origin_y, zoom
        )
        return composite, origin_lat, origin_lon

    def render(self, heading_deg: float, current_lat: float, current_lon: float) -> np.ndarray:
        """Render map overlay with path and heading arrow.

        Uses sub-pixel precision for smooth scrolling: computes floating-point
        position offsets, draws at integer positions, then applies the fractional
        offset as a sub-pixel translation using bilinear interpolation.

        Args:
            heading_deg: Current heading in degrees (0=North, 90=East, clockwise)
            current_lat: Current latitude (pre-interpolated for smooth scrolling)
            current_lon: Current longitude (pre-interpolated for smooth scrolling)

        Returns:
            RGB numpy array of the rendered map overlay
        """
        # Apply heading smoothing for rotation
        smooth_heading = self._smooth_heading(heading_deg)

        # Calculate sub-pixel offset for smooth scrolling
        # We'll draw at integer positions, then shift by the fractional part
        geo_scale = (self._internal_size * (1 - 2 * self.padding)) / (self.zoom_window * 2)

        # Use a reference point (first path point) for stable rendering
        if self.path:
            ref_lat, ref_lon = self.path[0][0], self.path[0][1]
        else:
            ref_lat, ref_lon = current_lat, current_lon

        # Compute floating-point offset from reference
        float_dx = (current_lon - ref_lon) * geo_scale
        float_dy = -(current_lat - ref_lat) * geo_scale

        # Split into integer and fractional parts
        # Integer part: used for drawing positions
        # Fractional part: applied as sub-pixel shift at the end
        int_dx = int(float_dx)
        int_dy = int(float_dy)
        frac_dx = float_dx - int_dx
        frac_dy = float_dy - int_dy

        # Create base image at internal resolution (for supersampling)
        internal_size = self._internal_size

        # Track sub-pixel offset for smooth scrolling (will be applied via affine transform)
        tile_frac_x, tile_frac_y = 0.0, 0.0

        if self.map_style != "simple":
            composite, origin_lat, origin_lon = self._fetch_tile(current_lat, current_lon)
            if composite is not None:
                from constants import MAP_TILE_ZOOM

                # Calculate pixel positions in absolute tile coordinate space
                # This gives us sub-pixel precision for smooth scrolling
                origin_px, origin_py = self._latlon_to_pixel(origin_lat, origin_lon, MAP_TILE_ZOOM)
                current_px, current_py = self._latlon_to_pixel(current_lat, current_lon, MAP_TILE_ZOOM)

                # Offset from composite origin to current position (in pixels)
                # This is where our position is within the composite
                offset_x = current_px - origin_px
                offset_y = current_py - origin_py

                # Split into integer and fractional parts for sub-pixel precision
                int_offset_x = int(offset_x)
                int_offset_y = int(offset_y)
                frac_offset_x = offset_x - int_offset_x
                frac_offset_y = offset_y - int_offset_y

                # Composite is north-up, we need to rotate for heading-up
                if self.heading_up:
                    rotated = composite.rotate(smooth_heading, resample=Image.Resampling.BILINEAR,
                                               expand=False)
                    # Apply rotation to the integer offset vector for cropping
                    # Rotation is around composite center, so offset from center rotates
                    rad = math.radians(-smooth_heading)
                    cos_r, sin_r = math.cos(rad), math.sin(rad)

                    # Offset from composite center (not origin)
                    cx_offset = int_offset_x - composite.width // 2
                    cy_offset = int_offset_y - composite.height // 2
                    rotated_cx = cx_offset * cos_r - cy_offset * sin_r
                    rotated_cy = cx_offset * sin_r + cy_offset * cos_r

                    # Crop centered on rotated position
                    crop_cx = rotated.width // 2 + int(rotated_cx)
                    crop_cy = rotated.height // 2 + int(rotated_cy)
                    half = internal_size // 2
                    img = rotated.crop((crop_cx - half, crop_cy - half, crop_cx + half, crop_cy + half))

                    # Rotate fractional offset for sub-pixel shift (applied later)
                    tile_frac_x = frac_offset_x * cos_r - frac_offset_y * sin_r
                    tile_frac_y = frac_offset_x * sin_r + frac_offset_y * cos_r
                else:
                    # Crop directly at our position for north-up mode
                    half = internal_size // 2
                    img = composite.crop((int_offset_x - half, int_offset_y - half,
                                          int_offset_x + half, int_offset_y + half))

                    # Store fractional offset for sub-pixel shift
                    tile_frac_x = frac_offset_x
                    tile_frac_y = frac_offset_y

                img = img.convert('RGB')
            else:
                img = Image.new('RGB', (internal_size, internal_size), COLORS.VOID_BLACK)
        else:
            # Simple mode: draw a grid that scrolls with position
            img = Image.new('RGB', (internal_size, internal_size), COLORS.VOID_BLACK)
            self._draw_scrolling_grid(img, current_lat, current_lon, smooth_heading, geo_scale)

        draw = ImageDraw.Draw(img)

        # Draw path if we have points (using integer offset from reference)
        if len(self.path) > 1:
            self._draw_path_subpixel(draw, ref_lat, ref_lon, int_dx, int_dy, smooth_heading, geo_scale)

        # Draw current position arrow (always at center, points UP in heading-up mode)
        self._draw_position_arrow(draw, smooth_heading)

        # Draw compass indicator
        self._draw_compass(draw, smooth_heading)

        # Apply sub-pixel translation using affine transform
        # This shifts the entire image by the fractional pixel amount
        # with bilinear interpolation for smooth anti-aliased movement
        # For tile modes: use tile_frac_x/y (already rotated if heading-up)
        # For simple mode: use frac_dx/dy from path reference calculation
        if self.map_style != "simple":
            sub_px_x, sub_px_y = tile_frac_x, tile_frac_y
        else:
            # For simple mode, rotate the path-based fractional offset
            if self.heading_up:
                rad = math.radians(-smooth_heading)
                cos_r, sin_r = math.cos(rad), math.sin(rad)
                sub_px_x = frac_dx * cos_r - frac_dy * sin_r
                sub_px_y = frac_dx * sin_r + frac_dy * cos_r
            else:
                sub_px_x, sub_px_y = frac_dx, frac_dy

        if abs(sub_px_x) > 0.001 or abs(sub_px_y) > 0.001:
            # Affine matrix: [a, b, c, d, e, f] where new_x = a*x + b*y + c
            # To translate by (tx, ty), use: (1, 0, -tx, 0, 1, -ty)
            matrix = (1, 0, -sub_px_x, 0, 1, -sub_px_y)
            img = img.transform(img.size, Image.AFFINE, matrix, resample=Image.Resampling.BILINEAR)

        # Draw border AFTER sub-pixel shift (so it stays crisp)
        draw = ImageDraw.Draw(img)
        border_width = max(1, int(2 * self.scale * self._supersample))
        draw.rectangle([0, 0, internal_size - 1, internal_size - 1],
                       outline=COLORS.STEEL_DARK, width=border_width)

        # Downsample from internal resolution to output resolution
        if self._supersample > 1:
            img = img.resize((self.size, self.size), Image.Resampling.LANCZOS)

        return np.array(img)

    def _draw_scrolling_grid(self, img: Image.Image, current_lat: float, current_lon: float,
                             heading_deg: float, geo_scale: float) -> None:
        """Draw a grid of lines at fixed geographic coordinates that scroll with position.

        The grid is anchored to geographic coordinates, so when the vehicle moves,
        the grid lines appear to scroll in the opposite direction.
        """
        draw = ImageDraw.Draw(img)
        internal_size = self._internal_size
        center = internal_size // 2

        # Grid spacing in degrees (approximately every 50 meters ≈ 0.00045°)
        grid_spacing = 0.0005

        # Rotation for heading-up mode
        rotation_rad = math.radians(-heading_deg) if self.heading_up else 0
        cos_r = math.cos(rotation_rad)
        sin_r = math.sin(rotation_rad)

        # Calculate the range of grid lines needed to cover the visible area
        # Account for rotation - we need a larger area to cover corners
        visible_degrees = (internal_size / geo_scale) * 1.5  # 1.5x for rotation margin

        # Find the nearest grid line to our current position
        base_lat = round(current_lat / grid_spacing) * grid_spacing
        base_lon = round(current_lon / grid_spacing) * grid_spacing

        # Grid line color (subtle dark grey)
        grid_color = (40, 45, 50)  # Subtle but visible

        # Draw horizontal lines (constant latitude)
        for i in range(-10, 11):
            lat = base_lat + i * grid_spacing
            # Two points far apart on this latitude line
            lon1 = current_lon - visible_degrees
            lon2 = current_lon + visible_degrees

            # Convert to screen coordinates
            dx1 = (lon1 - current_lon) * geo_scale
            dy1 = -(lat - current_lat) * geo_scale
            dx2 = (lon2 - current_lon) * geo_scale
            dy2 = -(lat - current_lat) * geo_scale

            # Apply heading rotation
            if self.heading_up:
                x1 = center + int(dx1 * cos_r - dy1 * sin_r)
                y1 = center + int(dx1 * sin_r + dy1 * cos_r)
                x2 = center + int(dx2 * cos_r - dy2 * sin_r)
                y2 = center + int(dx2 * sin_r + dy2 * cos_r)
            else:
                x1, y1 = center + int(dx1), center + int(dy1)
                x2, y2 = center + int(dx2), center + int(dy2)

            draw.line([(x1, y1), (x2, y2)], fill=grid_color, width=1)

        # Draw vertical lines (constant longitude)
        for i in range(-10, 11):
            lon = base_lon + i * grid_spacing
            # Two points far apart on this longitude line
            lat1 = current_lat - visible_degrees
            lat2 = current_lat + visible_degrees

            # Convert to screen coordinates
            dx1 = (lon - current_lon) * geo_scale
            dy1 = -(lat1 - current_lat) * geo_scale
            dx2 = (lon - current_lon) * geo_scale
            dy2 = -(lat2 - current_lat) * geo_scale

            # Apply heading rotation
            if self.heading_up:
                x1 = center + int(dx1 * cos_r - dy1 * sin_r)
                y1 = center + int(dx1 * sin_r + dy1 * cos_r)
                x2 = center + int(dx2 * cos_r - dy2 * sin_r)
                y2 = center + int(dx2 * sin_r + dy2 * cos_r)
            else:
                x1, y1 = center + int(dx1), center + int(dy1)
                x2, y2 = center + int(dx2), center + int(dy2)

            draw.line([(x1, y1), (x2, y2)], fill=grid_color, width=1)

    def _get_speed_color(self, speed_mps: float) -> Tuple[int, int, int]:
        """Get color based on speed (m/s converted to mph internally).

        Uses expanded 7-zone gradient for smoother color transitions:
        - 0-25 mph: Bright green (eco cruising)
        - 25-40 mph: Yellow-green (city driving)
        - 40-55 mph: Yellow (suburban)
        - 55-70 mph: Amber (highway)
        - 70-85 mph: Orange (fast highway)
        - 85-100 mph: Red-orange (very fast)
        - 100+ mph: Red (danger)
        """
        speed_mph = speed_mps * 2.237  # Convert m/s to mph

        if speed_mph < SPEED_ZONE_ECO:
            return COLORS.SPEED_ECO          # Bright green (eco cruising)
        elif speed_mph < SPEED_ZONE_CITY:
            return COLORS.SPEED_CITY         # Yellow-green (city)
        elif speed_mph < SPEED_ZONE_SUBURBAN:
            return COLORS.SPEED_SUBURBAN     # Yellow (suburban)
        elif speed_mph < SPEED_ZONE_HIGHWAY:
            return COLORS.SPEED_HIGHWAY      # Amber (highway)
        elif speed_mph < SPEED_ZONE_FAST:
            return COLORS.SPEED_FAST         # Orange (fast)
        elif speed_mph < SPEED_ZONE_VERY_FAST:
            return COLORS.SPEED_VERY_FAST    # Red-orange (very fast)
        else:
            return COLORS.SPEED_DANGER       # Red (danger)

    def _blend_color(self, color: Tuple[int, int, int], fade: float) -> Tuple[int, int, int]:
        """Blend a color toward dark based on fade factor (0=full color, 1=dark)."""
        return tuple(int(c * (1 - fade * 0.7)) for c in color)

    def _draw_path_subpixel(self, draw: ImageDraw.Draw, ref_lat: float, ref_lon: float,
                             int_offset_x: int, int_offset_y: int, heading_deg: float,
                             geo_scale: float) -> None:
        """Draw the GPS path with sub-pixel precision for smooth scrolling.

        Path points are drawn relative to a reference point, with an integer offset
        applied. The fractional part of the offset is handled by the affine transform
        in render() for true sub-pixel smooth scrolling.

        Args:
            draw: ImageDraw context
            ref_lat, ref_lon: Reference point (typically first path point)
            int_offset_x, int_offset_y: Integer pixel offset from reference to current position
            heading_deg: Current heading for rotation
            geo_scale: Pixels per degree for coordinate conversion
        """
        # Rotation for heading-up mode
        rotation_rad = math.radians(-heading_deg) if self.heading_up else 0

        center = self._internal_size // 2

        def to_screen(lat: float, lon: float) -> Tuple[int, int]:
            """Convert lat/lon to screen coordinates relative to reference + offset."""
            # Position relative to reference point
            dx = (lon - ref_lon) * geo_scale
            dy = -(lat - ref_lat) * geo_scale

            # Apply integer offset (brings us to current position, integer part only)
            dx -= int_offset_x
            dy -= int_offset_y

            if self.heading_up:
                cos_r = math.cos(rotation_rad)
                sin_r = math.sin(rotation_rad)
                dx_rot = dx * cos_r - dy * sin_r
                dy_rot = dx * sin_r + dy * cos_r
                dx, dy = dx_rot, dy_rot

            return (int(center + dx), int(center + dy))

        # Filter to nearby points - use a generous window since we're at internal resolution
        # Estimate current position from reference + offset
        current_lat_approx = ref_lat - int_offset_y / geo_scale
        current_lon_approx = ref_lon + int_offset_x / geo_scale

        relevant_path = []
        for p in self.path:
            lat, lon = p[0], p[1]
            speed = p[2] if len(p) > 2 else 0.0
            if abs(lat - current_lat_approx) < 0.01 and abs(lon - current_lon_approx) < 0.01:
                relevant_path.append((lat, lon, speed))

        if len(relevant_path) < 2:
            return

        # Draw each segment with gradient fade and speed coloring
        n_segments = len(relevant_path) - 1
        line_width = max(2, int(3 * self.scale * self._supersample))

        for i in range(n_segments):
            p1 = relevant_path[i]
            p2 = relevant_path[i + 1]

            pt1 = to_screen(p1[0], p1[1])
            pt2 = to_screen(p2[0], p2[1])

            # Fade factor: 0 at newest (end), 1 at oldest (start)
            fade = 1.0 - (i / n_segments)

            # Use speed from the segment endpoint for color
            speed_color = self._get_speed_color(p2[2])

            # Apply fade to make older segments dimmer
            faded_color = self._blend_color(speed_color, fade)

            draw.line([pt1, pt2], fill=faded_color, width=line_width)

    def _draw_position_arrow(self, draw: ImageDraw.Draw, heading_deg: float) -> None:
        """Draw the current position indicator with dark border and glow effect."""
        center = self._internal_size // 2
        arrow_len = self._arrow_length

        if self.heading_up:
            # In heading-up mode, arrow ALWAYS points straight up
            tip = (center, center - arrow_len)
            tail = (center, center + arrow_len // 2)
        else:
            # In north-up mode, arrow rotates with heading
            rad = math.radians(heading_deg)
            tip = (center + int(arrow_len * math.sin(rad)),
                   center - int(arrow_len * math.cos(rad)))
            tail = (center - int((arrow_len // 2) * math.sin(rad)),
                    center + int((arrow_len // 2) * math.cos(rad)))

        line_width = max(2, int(3 * self.scale * self._supersample))
        head_size = int(8 * self.scale * self._supersample)
        border_color = COLORS.VOID_BLACK  # Dark border for contrast

        # Draw dark border/outline first (behind everything)
        border_width = line_width + int(4 * self.scale * self._supersample)
        border_head = head_size + int(4 * self.scale * self._supersample)
        draw.line([tail, tip], fill=border_color, width=border_width)
        if self.heading_up:
            left = (center - border_head // 2, center - arrow_len + border_head)
            right = (center + border_head // 2, center - arrow_len + border_head)
            draw.polygon([tip, left, right], fill=border_color)
        else:
            rad = math.radians(heading_deg)
            angle_left = rad + math.pi * 0.8
            angle_right = rad - math.pi * 0.8
            left = (int(tip[0] + border_head * math.sin(angle_left)),
                    int(tip[1] - border_head * math.cos(angle_left)))
            right = (int(tip[0] + border_head * math.sin(angle_right)),
                     int(tip[1] - border_head * math.cos(angle_right)))
            draw.polygon([tip, left, right], fill=border_color)

        # Create glow effect by drawing larger, dimmer versions behind the main arrow
        for glow_offset, glow_color in [(3, COLORS.CT_ORANGE_DIM), (2, COLORS.CT_ORANGE_DIM)]:
            glow_width = line_width + glow_offset * 2

            # Glow arrow body
            draw.line([tail, tip], fill=glow_color, width=glow_width)

            # Glow arrowhead (slightly larger)
            glow_head = head_size + glow_offset
            if self.heading_up:
                left = (center - glow_head // 2, center - arrow_len + glow_head)
                right = (center + glow_head // 2, center - arrow_len + glow_head)
                draw.polygon([tip, left, right], fill=glow_color)
            else:
                rad = math.radians(heading_deg)
                angle_left = rad + math.pi * 0.8
                angle_right = rad - math.pi * 0.8
                left = (int(tip[0] + glow_head * math.sin(angle_left)),
                        int(tip[1] - glow_head * math.cos(angle_left)))
                right = (int(tip[0] + glow_head * math.sin(angle_right)),
                         int(tip[1] - glow_head * math.cos(angle_right)))
                draw.polygon([tip, left, right], fill=glow_color)

        # Draw main arrow body on top
        draw.line([tail, tip], fill=COLORS.CT_ORANGE_GLOW, width=line_width)

        # Draw main arrowhead on top
        if self.heading_up:
            left = (center - head_size // 2, center - arrow_len + head_size)
            right = (center + head_size // 2, center - arrow_len + head_size)
            draw.polygon([tip, left, right], fill=COLORS.CT_ORANGE_GLOW)
        else:
            rad = math.radians(heading_deg)
            angle_left = rad + math.pi * 0.8
            angle_right = rad - math.pi * 0.8
            left = (int(tip[0] + head_size * math.sin(angle_left)),
                    int(tip[1] - head_size * math.cos(angle_left)))
            right = (int(tip[0] + head_size * math.sin(angle_right)),
                     int(tip[1] - head_size * math.cos(angle_right)))
            draw.polygon([tip, left, right], fill=COLORS.CT_ORANGE_GLOW)

    def _draw_compass(self, draw: ImageDraw.Draw, heading_deg: float) -> None:
        """Draw the north compass indicator with outlined text for visibility."""
        margin = int(20 * self.scale * self._supersample)
        radius_bg = int(12 * self.scale * self._supersample)
        internal_size = self._internal_size

        if self.heading_up:
            # Compass orbits around the edge to show where north is relative to travel direction
            # In heading-up mode, you always face UP on screen. N shows where geographic north is.
            # When heading=0 (facing North): N at top (you're facing north, north is ahead)
            # When heading=90 (facing East): N at left (north is to your left)
            # When heading=180 (facing South): N at bottom (north is behind you)
            # When heading=270 (facing West): N at right (north is to your right)
            # Standard math coords: 0°=right, 90°=top, counterclockwise positive
            compass_angle = math.radians(90 + heading_deg)
            orbit_radius = internal_size // 2 - margin
            compass_x = int(internal_size // 2 + orbit_radius * math.cos(compass_angle))
            compass_y = int(internal_size // 2 - orbit_radius * math.sin(compass_angle))

            # Clamp to stay within bounds
            compass_x = max(margin, min(internal_size - margin, compass_x))
            compass_y = max(margin, min(internal_size - margin, compass_y))
        else:
            # In north-up mode, N is always at top-right corner
            compass_x = internal_size - margin
            compass_y = margin

        # Draw background circle
        draw.ellipse([compass_x - radius_bg, compass_y - radius_bg,
                      compass_x + radius_bg, compass_y + radius_bg],
                     fill=COLORS.GUNMETAL)

        # Draw "N" with black outline for visibility on any background
        n_bbox = draw.textbbox((0, 0), "N", font=self._font)
        n_w = n_bbox[2] - n_bbox[0]
        n_h = n_bbox[3] - n_bbox[1]
        text_x = compass_x - n_w // 2
        text_y = compass_y - n_h // 2 - int(1 * self.scale)

        # Draw black outline by drawing text offset in 8 directions
        outline_offset = max(1, int(1.5 * self.scale * self._supersample))
        for dx in [-outline_offset, 0, outline_offset]:
            for dy in [-outline_offset, 0, outline_offset]:
                if dx != 0 or dy != 0:
                    draw.text((text_x + dx, text_y + dy), "N",
                              fill=COLORS.VOID_BLACK, font=self._font)

        # Draw white "N" on top
        draw.text((text_x, text_y), "N", fill=COLORS.WHITE, font=self._font)
