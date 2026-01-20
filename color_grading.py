"""
Color grading module for Tesla dashcam video processor.

Provides LUT-based color grading and adjustable parameters (brightness, contrast,
saturation, gamma, shadows, highlights). Designed for single 1080p frame processing
with ~3-5ms overhead per frame.
"""

import os
import logging
from typing import Optional, Dict, Any
from functools import lru_cache

import numpy as np

logger = logging.getLogger(__name__)

# LUT cache (module-level for persistence across ColorGrader instances)
_lut_cache: Dict[str, np.ndarray] = {}

# Pre-computed gamma lookup tables for common gamma values (faster than per-pixel pow)
_gamma_luts: Dict[float, np.ndarray] = {}


def load_cube_lut(path: str) -> Optional[np.ndarray]:
    """
    Load a .cube LUT file and return as a 3D numpy array.

    Args:
        path: Path to .cube file

    Returns:
        3D numpy array of shape (size, size, size, 3) with float32 values in [0,1],
        or None if loading fails
    """
    # Check cache first
    abs_path = os.path.abspath(path)
    if abs_path in _lut_cache:
        return _lut_cache[abs_path]

    if not os.path.exists(path):
        logger.warning(f"LUT file not found: {path}")
        return None

    try:
        size = None
        data_lines = []
        domain_min = [0.0, 0.0, 0.0]
        domain_max = [1.0, 1.0, 1.0]

        with open(path, 'r') as f:
            for line in f:
                line = line.strip()

                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue

                # Parse header
                if line.startswith('TITLE'):
                    continue
                elif line.startswith('DOMAIN_MIN'):
                    parts = line.split()[1:]
                    if len(parts) >= 3:
                        domain_min = [float(x) for x in parts[:3]]
                elif line.startswith('DOMAIN_MAX'):
                    parts = line.split()[1:]
                    if len(parts) >= 3:
                        domain_max = [float(x) for x in parts[:3]]
                elif line.startswith('LUT_3D_SIZE'):
                    size = int(line.split()[1])
                else:
                    # Data line - three floats
                    try:
                        values = [float(x) for x in line.split()[:3]]
                        if len(values) == 3:
                            data_lines.append(values)
                    except ValueError:
                        continue

        if size is None:
            logger.warning(f"LUT file missing LUT_3D_SIZE: {path}")
            return None

        expected_entries = size ** 3
        if len(data_lines) != expected_entries:
            logger.warning(f"LUT file has {len(data_lines)} entries, expected {expected_entries}: {path}")
            return None

        # Convert to numpy array and reshape to 3D
        # .cube format: R varies fastest, then G, then B
        # Shape: (B, G, R, 3) for fast indexing
        lut = np.array(data_lines, dtype=np.float32).reshape(size, size, size, 3)

        # Normalize from domain to [0,1] if needed
        for i in range(3):
            if domain_max[i] != domain_min[i] and (domain_min[i] != 0.0 or domain_max[i] != 1.0):
                lut[:, :, :, i] = (lut[:, :, :, i] - domain_min[i]) / (domain_max[i] - domain_min[i])

        # Cache the result
        _lut_cache[abs_path] = lut
        logger.debug(f"Loaded LUT: {path} ({size}³)")
        return lut

    except Exception as e:
        logger.warning(f"Failed to load LUT {path}: {e}")
        return None


def apply_lut(frame: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """
    Apply 3D LUT to frame using trilinear interpolation.

    Args:
        frame: BGR uint8 numpy array (OpenCV format)
        lut: 3D LUT array from load_cube_lut()

    Returns:
        Color-graded BGR uint8 numpy array
    """
    if lut is None:
        return frame

    size = lut.shape[0]
    max_idx = size - 1

    # Convert to float [0,1] for interpolation
    # OpenCV uses BGR, LUT expects RGB
    frame_rgb = frame[:, :, ::-1].astype(np.float32) / 255.0

    # Scale to LUT indices
    r_idx = frame_rgb[:, :, 0] * max_idx
    g_idx = frame_rgb[:, :, 1] * max_idx
    b_idx = frame_rgb[:, :, 2] * max_idx

    # Get integer indices for corners of interpolation cube
    r0 = np.clip(np.floor(r_idx).astype(np.int32), 0, max_idx)
    g0 = np.clip(np.floor(g_idx).astype(np.int32), 0, max_idx)
    b0 = np.clip(np.floor(b_idx).astype(np.int32), 0, max_idx)

    r1 = np.clip(r0 + 1, 0, max_idx)
    g1 = np.clip(g0 + 1, 0, max_idx)
    b1 = np.clip(b0 + 1, 0, max_idx)

    # Fractional parts for interpolation weights
    r_frac = r_idx - r0
    g_frac = g_idx - g0
    b_frac = b_idx - b0

    # Expand dims for broadcasting
    r_frac = r_frac[:, :, np.newaxis]
    g_frac = g_frac[:, :, np.newaxis]
    b_frac = b_frac[:, :, np.newaxis]

    # Trilinear interpolation (8 corners of the cube)
    # LUT indexing: [B, G, R, channel]
    c000 = lut[b0, g0, r0]
    c001 = lut[b0, g0, r1]
    c010 = lut[b0, g1, r0]
    c011 = lut[b0, g1, r1]
    c100 = lut[b1, g0, r0]
    c101 = lut[b1, g0, r1]
    c110 = lut[b1, g1, r0]
    c111 = lut[b1, g1, r1]

    # Interpolate along R
    c00 = c000 * (1 - r_frac) + c001 * r_frac
    c01 = c010 * (1 - r_frac) + c011 * r_frac
    c10 = c100 * (1 - r_frac) + c101 * r_frac
    c11 = c110 * (1 - r_frac) + c111 * r_frac

    # Interpolate along G
    c0 = c00 * (1 - g_frac) + c01 * g_frac
    c1 = c10 * (1 - g_frac) + c11 * g_frac

    # Interpolate along B
    result_rgb = c0 * (1 - b_frac) + c1 * b_frac

    # Convert back to BGR uint8
    result_bgr = (np.clip(result_rgb[:, :, ::-1], 0, 1) * 255).astype(np.uint8)

    return result_bgr


def apply_brightness(frame: np.ndarray, amount: float) -> np.ndarray:
    """
    Apply brightness adjustment.

    Args:
        frame: BGR uint8 numpy array
        amount: Adjustment value (-1.0 to 1.0, 0 = no change)

    Returns:
        Adjusted BGR uint8 numpy array
    """
    if abs(amount) < 0.001:
        return frame

    # Scale amount to pixel range
    adjustment = int(amount * 255)

    # Fast integer arithmetic with clipping
    result = np.clip(frame.astype(np.int16) + adjustment, 0, 255).astype(np.uint8)

    return result


def apply_contrast(frame: np.ndarray, amount: float) -> np.ndarray:
    """
    Apply contrast adjustment around midpoint (128).

    Args:
        frame: BGR uint8 numpy array
        amount: Adjustment value (-1.0 to 1.0, 0 = no change)

    Returns:
        Adjusted BGR uint8 numpy array
    """
    if abs(amount) < 0.001:
        return frame

    # Contrast factor (amount of 1.0 doubles contrast, -1.0 halves it)
    factor = 1.0 + amount

    # Apply contrast around midpoint
    # new_pixel = midpoint + (old_pixel - midpoint) * factor
    frame_float = frame.astype(np.float32)
    result = 128.0 + (frame_float - 128.0) * factor
    result = np.clip(result, 0, 255).astype(np.uint8)

    return result


def apply_saturation(frame: np.ndarray, amount: float) -> np.ndarray:
    """
    Apply saturation adjustment using Rec.709 luminance.

    Args:
        frame: BGR uint8 numpy array
        amount: Adjustment value (-1.0 to 1.0, 0 = no change)
                -1.0 = grayscale, positive values increase saturation

    Returns:
        Adjusted BGR uint8 numpy array
    """
    if abs(amount) < 0.001:
        return frame

    # Saturation factor
    factor = 1.0 + amount

    # Rec.709 luminance coefficients (BGR order)
    luma_coeffs = np.array([0.0722, 0.7152, 0.2126], dtype=np.float32)

    frame_float = frame.astype(np.float32)

    # Calculate luminance
    luminance = np.sum(frame_float * luma_coeffs, axis=2, keepdims=True)

    # Blend between luminance (grayscale) and original based on factor
    # factor=0 -> grayscale, factor=1 -> original, factor>1 -> boosted
    result = luminance + (frame_float - luminance) * factor
    result = np.clip(result, 0, 255).astype(np.uint8)

    return result


def apply_gamma(frame: np.ndarray, gamma: float) -> np.ndarray:
    """
    Apply gamma correction using pre-computed LUT for speed.

    Gamma adjustment: output = input^(1/gamma)
    - gamma < 1: darkens midtones (1/gamma > 1 lowers the curve)
    - gamma > 1: brightens midtones (1/gamma < 1 raises the curve)

    This follows the common image editing convention where higher gamma
    values produce brighter results.

    Args:
        frame: BGR uint8 numpy array
        gamma: Gamma value (1.0 = no change, <1 = darker, >1 = brighter)

    Returns:
        Adjusted BGR uint8 numpy array
    """
    if abs(gamma - 1.0) < 0.001:
        return frame

    # Round gamma for LUT caching (0.01 precision)
    gamma_key = round(gamma, 2)

    # Get or create gamma LUT
    # Using output = input^gamma directly (not inverse)
    # gamma < 1 -> lowers the curve -> darkens midtones
    # gamma > 1 -> raises the curve -> brightens midtones (inverted from standard)
    # We use inverse so that higher gamma = brighter (common in editing software)
    if gamma_key not in _gamma_luts:
        # Direct gamma: higher values brighten, lower values darken
        lut = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                       for i in range(256)], dtype=np.uint8)
        _gamma_luts[gamma_key] = lut

    gamma_lut = _gamma_luts[gamma_key]

    # Apply LUT to all channels
    return gamma_lut[frame]


def apply_shadows_highlights(frame: np.ndarray, shadows: float, highlights: float) -> np.ndarray:
    """
    Apply shadows and highlights adjustment.

    Shadows affects dark tones (lifts or crushes blacks).
    Highlights affects bright tones (lifts or crushes whites).

    Args:
        frame: BGR uint8 numpy array
        shadows: Shadow adjustment (-1.0 to 1.0, 0 = no change)
                 Positive lifts shadows, negative crushes them
        highlights: Highlight adjustment (-1.0 to 1.0, 0 = no change)
                    Positive lifts highlights, negative crushes them

    Returns:
        Adjusted BGR uint8 numpy array
    """
    if abs(shadows) < 0.001 and abs(highlights) < 0.001:
        return frame

    frame_float = frame.astype(np.float32) / 255.0

    # Luminance for tone targeting (Rec.709, BGR order)
    luma = 0.0722 * frame_float[:, :, 0] + 0.7152 * frame_float[:, :, 1] + 0.2126 * frame_float[:, :, 2]

    # Shadow mask: strong in darks, fades to zero in brights
    # Using smooth S-curve: 1 - luma² gives good falloff
    shadow_mask = (1.0 - luma ** 2)[:, :, np.newaxis]

    # Highlight mask: strong in brights, fades to zero in darks
    # Using luma² for smooth transition
    highlight_mask = (luma ** 2)[:, :, np.newaxis]

    # Apply adjustments
    result = frame_float.copy()

    if abs(shadows) >= 0.001:
        # Shadow adjustment (positive lifts, negative crushes)
        result = result + shadow_mask * shadows * 0.5

    if abs(highlights) >= 0.001:
        # Highlight adjustment (positive lifts, negative crushes)
        result = result + highlight_mask * highlights * 0.5

    result = np.clip(result * 255, 0, 255).astype(np.uint8)

    return result


class ColorGrader:
    """
    Main color grading class that applies LUT and parameter adjustments.

    Usage:
        grader = ColorGrader(preset='cinematic', brightness=0.1)
        if grader.is_active:
            frame = grader.grade(frame)
    """

    def __init__(
        self,
        preset: Optional[str] = None,
        lut_path: Optional[str] = None,
        brightness: float = 0.0,
        contrast: float = 0.0,
        saturation: float = 0.0,
        gamma: float = 1.0,
        shadows: float = 0.0,
        highlights: float = 0.0,
        presets: Optional[Dict[str, Dict[str, float]]] = None
    ):
        """
        Initialize the color grader.

        Args:
            preset: Name of a built-in preset (applied first)
            lut_path: Path to a .cube LUT file (applied after preset)
            brightness: Brightness adjustment (-1.0 to 1.0)
            contrast: Contrast adjustment (-1.0 to 1.0)
            saturation: Saturation adjustment (-1.0 to 1.0)
            gamma: Gamma value (0.1 to 3.0, default 1.0)
            shadows: Shadow adjustment (-1.0 to 1.0)
            highlights: Highlight adjustment (-1.0 to 1.0)
            presets: Dict of preset definitions (if None, uses COLOR_PRESETS from constants)
        """
        self.preset = preset
        self.lut_path = lut_path
        self.lut = None

        # Load preset definitions
        if presets is None:
            try:
                from constants import COLOR_PRESETS
                presets = COLOR_PRESETS
            except ImportError:
                presets = {}

        # Start with manual parameters
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.gamma = gamma
        self.shadows = shadows
        self.highlights = highlights

        # Apply preset values as base, then manual params override
        if preset and preset in presets:
            preset_values = presets[preset]
            # Preset provides defaults, manual params can override
            self.brightness = preset_values.get('brightness', 0.0) + brightness
            self.contrast = preset_values.get('contrast', 0.0) + contrast
            self.saturation = preset_values.get('saturation', 0.0) + saturation
            self.gamma = preset_values.get('gamma', 1.0) * gamma  # Multiplicative for gamma
            self.shadows = preset_values.get('shadows', 0.0) + shadows
            self.highlights = preset_values.get('highlights', 0.0) + highlights
            logger.debug(f"Applied preset '{preset}': {preset_values}")
        elif preset:
            logger.warning(f"Unknown preset '{preset}', ignoring")

        # Load LUT if specified
        # Check if lut_path is actually a preset name that looks like a path
        if lut_path:
            if lut_path.endswith('.cube') or os.path.exists(lut_path):
                self.lut = load_cube_lut(lut_path)
                if self.lut is not None:
                    logger.debug(f"Loaded LUT: {lut_path}")

        # Log effective settings
        if self.is_active:
            params = []
            if self.lut is not None:
                params.append(f"LUT={os.path.basename(lut_path or '')}")
            if abs(self.brightness) >= 0.001:
                params.append(f"brightness={self.brightness:+.2f}")
            if abs(self.contrast) >= 0.001:
                params.append(f"contrast={self.contrast:+.2f}")
            if abs(self.saturation) >= 0.001:
                params.append(f"saturation={self.saturation:+.2f}")
            if abs(self.gamma - 1.0) >= 0.001:
                params.append(f"gamma={self.gamma:.2f}")
            if abs(self.shadows) >= 0.001:
                params.append(f"shadows={self.shadows:+.2f}")
            if abs(self.highlights) >= 0.001:
                params.append(f"highlights={self.highlights:+.2f}")
            logger.debug(f"Color grading active: {', '.join(params)}")

    @property
    def is_active(self) -> bool:
        """Check if any color grading will be applied."""
        return (
            self.lut is not None or
            abs(self.brightness) >= 0.001 or
            abs(self.contrast) >= 0.001 or
            abs(self.saturation) >= 0.001 or
            abs(self.gamma - 1.0) >= 0.001 or
            abs(self.shadows) >= 0.001 or
            abs(self.highlights) >= 0.001
        )

    def grade(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply all color grading operations to a frame.

        Processing order:
        1. LUT (if loaded)
        2. Brightness
        3. Contrast
        4. Saturation
        5. Gamma
        6. Shadows/Highlights

        Args:
            frame: BGR uint8 numpy array (OpenCV format)

        Returns:
            Color-graded BGR uint8 numpy array
        """
        if not self.is_active:
            return frame

        result = frame

        # 1. Apply LUT first (major color transform)
        if self.lut is not None:
            result = apply_lut(result, self.lut)

        # 2. Brightness
        if abs(self.brightness) >= 0.001:
            result = apply_brightness(result, self.brightness)

        # 3. Contrast
        if abs(self.contrast) >= 0.001:
            result = apply_contrast(result, self.contrast)

        # 4. Saturation
        if abs(self.saturation) >= 0.001:
            result = apply_saturation(result, self.saturation)

        # 5. Gamma
        if abs(self.gamma - 1.0) >= 0.001:
            result = apply_gamma(result, self.gamma)

        # 6. Shadows/Highlights
        if abs(self.shadows) >= 0.001 or abs(self.highlights) >= 0.001:
            result = apply_shadows_highlights(result, self.shadows, self.highlights)

        return result


def create_color_grader(
    color_grade: Optional[str] = None,
    brightness: float = 0.0,
    contrast: float = 0.0,
    saturation: float = 0.0,
    gamma: float = 1.0,
    shadows: float = 0.0,
    highlights: float = 0.0
) -> Optional[ColorGrader]:
    """
    Factory function to create a ColorGrader from CLI-style arguments.

    The color_grade argument can be:
    - A preset name (e.g., 'cinematic', 'warm')
    - A path to a .cube LUT file (e.g., 'LUTs/NaturalBoost.cube')
    - None (only manual adjustments apply)

    Args:
        color_grade: Preset name or LUT file path
        brightness: Brightness adjustment (-1.0 to 1.0)
        contrast: Contrast adjustment (-1.0 to 1.0)
        saturation: Saturation adjustment (-1.0 to 1.0)
        gamma: Gamma value (0.1 to 3.0)
        shadows: Shadow adjustment (-1.0 to 1.0)
        highlights: Highlight adjustment (-1.0 to 1.0)

    Returns:
        ColorGrader instance, or None if no grading is configured
    """
    # Determine if color_grade is a preset or LUT path
    preset = None
    lut_path = None

    if color_grade:
        if color_grade.endswith('.cube') or os.path.exists(color_grade):
            lut_path = color_grade
        else:
            preset = color_grade

    grader = ColorGrader(
        preset=preset,
        lut_path=lut_path,
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        gamma=gamma,
        shadows=shadows,
        highlights=highlights
    )

    return grader if grader.is_active else None
