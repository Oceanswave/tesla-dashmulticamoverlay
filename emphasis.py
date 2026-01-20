"""
Dynamic camera emphasis based on driving context.

Calculates emphasis weights for side and rear cameras based on:
- Turn signals (blinker_on_left/right)
- Lateral G-force (turning)
- Braking G-force
- Reverse gear

Emphasis is expressed as a 0.0-1.0 weight with optional border color.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict
import dashcam_pb2

from constants import (
    COLORS,
    MPS2_TO_G,
    EMPHASIS_LATERAL_G_THRESHOLD,
    EMPHASIS_BRAKING_G_THRESHOLD,
    EMPHASIS_SMOOTHING_FACTOR,
    EMPHASIS_BORDER_WIDTH,
    EMPHASIS_COLOR_BLINKER,
    EMPHASIS_COLOR_BRAKE,
    EMPHASIS_COLOR_REVERSE,
    EMPHASIS_COLOR_LATERAL,
    EMPHASIS_VISIBILITY_THRESHOLD,
)


@dataclass
class CameraEmphasis:
    """Emphasis state for a single camera.

    Attributes:
        weight: Emphasis level from 0.0 (none) to 1.0 (full)
        border_color: RGB tuple for border, or None if no border
        border_width: Border thickness in pixels
    """
    weight: float = 0.0
    border_color: Optional[Tuple[int, int, int]] = None
    border_width: int = 0


@dataclass
class EmphasisState:
    """Emphasis state for all cameras that can be emphasized.

    Note: front camera is never emphasized (always primary view).
    """
    left_repeater: CameraEmphasis = field(default_factory=CameraEmphasis)
    right_repeater: CameraEmphasis = field(default_factory=CameraEmphasis)
    left_pillar: CameraEmphasis = field(default_factory=CameraEmphasis)
    right_pillar: CameraEmphasis = field(default_factory=CameraEmphasis)
    back: CameraEmphasis = field(default_factory=CameraEmphasis)

    def get(self, camera_name: str) -> CameraEmphasis:
        """Get emphasis for a camera by name.

        Args:
            camera_name: One of 'left_repeater', 'right_repeater',
                        'left_pillar', 'right_pillar', 'back'

        Returns:
            CameraEmphasis for the camera, or default (no emphasis) if not found
        """
        return getattr(self, camera_name, CameraEmphasis())


class EmphasisCalculator:
    """Calculates dynamic camera emphasis based on driving context.

    Priority order (highest first):
    1. Turn signals (blinker) - emphasize corresponding side cameras
    2. Reverse gear - emphasize rear camera
    3. Heavy braking - emphasize rear camera
    4. Lateral G-force (turns) - emphasize side cameras proportionally

    All emphasis transitions use smoothing for gradual visual changes.

    Args:
        lateral_g_threshold: Minimum lateral G for turn emphasis (default 0.2g)
        braking_g_threshold: Minimum braking G for rear emphasis (default 0.3g)
        smoothing: Interpolation factor per frame (default 0.3 = 30%)
    """

    # Camera keys for smoothing state
    CAMERAS = ['left_repeater', 'right_repeater', 'left_pillar',
               'right_pillar', 'back']

    def __init__(
        self,
        lateral_g_threshold: float = EMPHASIS_LATERAL_G_THRESHOLD,
        braking_g_threshold: float = EMPHASIS_BRAKING_G_THRESHOLD,
        smoothing: float = EMPHASIS_SMOOTHING_FACTOR
    ):
        # Validate thresholds to prevent division by zero
        if lateral_g_threshold >= 1.0:
            lateral_g_threshold = 0.9
        if braking_g_threshold >= 1.0:
            braking_g_threshold = 0.9

        self.lateral_g_threshold = lateral_g_threshold
        self.braking_g_threshold = braking_g_threshold
        self.smoothing = smoothing

        # Smoothed emphasis weights (persists across frames)
        self._smoothed: Dict[str, float] = {cam: 0.0 for cam in self.CAMERAS}

        # Last active colors for proper fade-out (persists until fully faded)
        self._last_colors: Dict[str, Optional[Tuple[int, int, int]]] = {
            cam: None for cam in self.CAMERAS
        }

    def compute(self, meta: Optional[dashcam_pb2.SeiMetadata]) -> EmphasisState:
        """Compute emphasis state from SEI metadata.

        Args:
            meta: SEI metadata from current frame, or None if unavailable

        Returns:
            EmphasisState with emphasis weights and colors for all cameras
        """
        # Target weights before smoothing
        targets: Dict[str, float] = {cam: 0.0 for cam in self.CAMERAS}

        # Colors for each camera (set by highest-priority trigger)
        colors: Dict[str, Optional[Tuple[int, int, int]]] = {
            cam: None for cam in self.CAMERAS
        }

        if meta is not None:
            # Extract values from metadata
            blinker_left = meta.blinker_on_left
            blinker_right = meta.blinker_on_right
            gear_state = meta.gear_state  # 0=P, 1=D, 2=R, 3=N

            # Convert acceleration to G-force
            # Positive X = turning right, Negative X = turning left
            # Positive Y = braking (deceleration), Negative Y = acceleration
            accel_x_g = meta.linear_acceleration_mps2_x * MPS2_TO_G
            accel_y_g = meta.linear_acceleration_mps2_y * MPS2_TO_G

            # --- Priority 1: Turn signals (highest priority for color) ---
            if blinker_left:
                targets['left_repeater'] = 1.0
                targets['left_pillar'] = 0.7  # Slightly less emphasis
                colors['left_repeater'] = EMPHASIS_COLOR_BLINKER
                colors['left_pillar'] = EMPHASIS_COLOR_BLINKER

            if blinker_right:
                targets['right_repeater'] = 1.0
                targets['right_pillar'] = 0.7
                colors['right_repeater'] = EMPHASIS_COLOR_BLINKER
                colors['right_pillar'] = EMPHASIS_COLOR_BLINKER

            # --- Priority 2: Reverse gear ---
            if gear_state == 2:  # Reverse
                targets['back'] = 1.0
                colors['back'] = EMPHASIS_COLOR_REVERSE

            # --- Priority 3: Heavy braking (only if not in reverse) ---
            elif accel_y_g > self.braking_g_threshold:
                # Proportional emphasis based on braking intensity
                # Scale from threshold to 1.0g for full emphasis
                brake_intensity = min(1.0,
                    (accel_y_g - self.braking_g_threshold) /
                    (1.0 - self.braking_g_threshold))
                targets['back'] = max(targets['back'], brake_intensity)
                if colors['back'] is None:
                    colors['back'] = EMPHASIS_COLOR_BRAKE

            # --- Priority 4: Lateral G-force (only if not already emphasized by blinker) ---
            if abs(accel_x_g) > self.lateral_g_threshold:
                # Proportional emphasis based on turn intensity
                lateral_intensity = min(1.0,
                    (abs(accel_x_g) - self.lateral_g_threshold) /
                    (1.0 - self.lateral_g_threshold))

                if accel_x_g < 0:  # Turning left (negative X = left turn)
                    if colors['left_repeater'] is None:
                        targets['left_repeater'] = max(
                            targets['left_repeater'], lateral_intensity)
                        colors['left_repeater'] = EMPHASIS_COLOR_LATERAL
                    if colors['left_pillar'] is None:
                        targets['left_pillar'] = max(
                            targets['left_pillar'], lateral_intensity * 0.7)
                        colors['left_pillar'] = EMPHASIS_COLOR_LATERAL
                else:  # Turning right (positive X = right turn)
                    if colors['right_repeater'] is None:
                        targets['right_repeater'] = max(
                            targets['right_repeater'], lateral_intensity)
                        colors['right_repeater'] = EMPHASIS_COLOR_LATERAL
                    if colors['right_pillar'] is None:
                        targets['right_pillar'] = max(
                            targets['right_pillar'], lateral_intensity * 0.7)
                        colors['right_pillar'] = EMPHASIS_COLOR_LATERAL

        # Apply smoothing to all cameras
        for cam in self.CAMERAS:
            current = self._smoothed[cam]
            target = targets[cam]
            # Exponential smoothing: new = old + factor * (target - old)
            self._smoothed[cam] = current + self.smoothing * (target - current)

            # Update last active color for proper fade-out
            # If a new color is set, use it; otherwise keep the old one during fade
            if colors[cam] is not None:
                self._last_colors[cam] = colors[cam]
            # Clear last color when fully faded out
            elif self._smoothed[cam] < EMPHASIS_VISIBILITY_THRESHOLD:
                self._last_colors[cam] = None

        # Build EmphasisState with smoothed weights
        # Use last_colors during fade-out so border color persists
        def get_color(cam: str) -> Optional[Tuple[int, int, int]]:
            if self._smoothed[cam] <= EMPHASIS_VISIBILITY_THRESHOLD:
                return None
            return colors[cam] if colors[cam] is not None else self._last_colors[cam]

        state = EmphasisState(
            left_repeater=CameraEmphasis(
                weight=self._smoothed['left_repeater'],
                border_color=get_color('left_repeater'),
                border_width=EMPHASIS_BORDER_WIDTH if self._smoothed['left_repeater'] > EMPHASIS_VISIBILITY_THRESHOLD else 0
            ),
            right_repeater=CameraEmphasis(
                weight=self._smoothed['right_repeater'],
                border_color=get_color('right_repeater'),
                border_width=EMPHASIS_BORDER_WIDTH if self._smoothed['right_repeater'] > EMPHASIS_VISIBILITY_THRESHOLD else 0
            ),
            left_pillar=CameraEmphasis(
                weight=self._smoothed['left_pillar'],
                border_color=get_color('left_pillar'),
                border_width=EMPHASIS_BORDER_WIDTH if self._smoothed['left_pillar'] > EMPHASIS_VISIBILITY_THRESHOLD else 0
            ),
            right_pillar=CameraEmphasis(
                weight=self._smoothed['right_pillar'],
                border_color=get_color('right_pillar'),
                border_width=EMPHASIS_BORDER_WIDTH if self._smoothed['right_pillar'] > EMPHASIS_VISIBILITY_THRESHOLD else 0
            ),
            back=CameraEmphasis(
                weight=self._smoothed['back'],
                border_color=get_color('back'),
                border_width=EMPHASIS_BORDER_WIDTH if self._smoothed['back'] > EMPHASIS_VISIBILITY_THRESHOLD else 0
            ),
        )

        return state

    def reset(self) -> None:
        """Reset smoothed state to zero (e.g., between clips)."""
        self._smoothed = {cam: 0.0 for cam in self.CAMERAS}
        self._last_colors = {cam: None for cam in self.CAMERAS}
