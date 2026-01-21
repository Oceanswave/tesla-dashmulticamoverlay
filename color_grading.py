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

# CRITICAL: Disable Numba internal threading BEFORE importing numba
# This prevents "workqueue threading layer" conflicts when called from ThreadPoolExecutor
# Must be set before any numba import
os.environ.setdefault('NUMBA_NUM_THREADS', '1')
os.environ.setdefault('NUMBA_THREADING_LAYER', 'workqueue')

import numpy as np

logger = logging.getLogger(__name__)

# LUT cache (module-level for persistence across ColorGrader instances)
_lut_cache: Dict[str, np.ndarray] = {}

# Pre-computed gamma lookup tables for common gamma values (faster than per-pixel pow)
_gamma_luts: Dict[float, np.ndarray] = {}

# Pre-computed combined LUTs for brightness+contrast+gamma (key: (brightness, contrast, gamma))
_combined_luts: Dict[tuple, np.ndarray] = {}

# Pre-computed shadows/highlights adjustment LUTs (key: (shadows, highlights))
# Maps luminance (0-255) -> adjustment amount to add to each channel
_shadow_highlight_luts: Dict[tuple, np.ndarray] = {}


def _get_shadow_highlight_lut(shadows: float, highlights: float) -> np.ndarray:
    """
    Get or create a LUT for shadows/highlights adjustment.

    Maps luminance (0-255) -> adjustment value to add to pixel channels.
    This avoids computing masks per-pixel by pre-computing the adjustment curve.

    Returns:
        float32 array of shape (256,) with adjustment values (can be negative)
    """
    key = (round(shadows, 2), round(highlights, 2))

    if key in _shadow_highlight_luts:
        return _shadow_highlight_luts[key]

    # Build LUT: for each luminance value, compute total adjustment
    luma_normalized = np.arange(256, dtype=np.float32) / 255.0

    # Shadow mask: strong in darks, fades to zero in brights (1 - luma²)
    shadow_weight = 1.0 - luma_normalized ** 2

    # Highlight mask: strong in brights, fades to zero in darks (luma²)
    highlight_weight = luma_normalized ** 2

    # Total adjustment for each luminance level
    adjustment = shadow_weight * shadows * 0.5 + highlight_weight * highlights * 0.5

    # Scale to 0-255 range
    adjustment = adjustment * 255.0

    _shadow_highlight_luts[key] = adjustment
    return adjustment


def _get_combined_lut(brightness: float, contrast: float, gamma: float) -> np.ndarray:
    """
    Get or create a combined LUT for brightness, contrast, and gamma.

    These are all per-channel operations that can be pre-computed into a single
    256-entry lookup table, reducing 3 passes to 1 fast array lookup.

    Processing order matches ColorGrader.grade(): brightness -> contrast -> gamma
    """
    # Round values for cache key (0.01 precision)
    key = (round(brightness, 2), round(contrast, 2), round(gamma, 2))

    if key in _combined_luts:
        return _combined_luts[key]

    # Build combined LUT: for each input value 0-255, compute final output
    lut = np.arange(256, dtype=np.float32)

    # 1. Apply brightness (if non-zero)
    if abs(brightness) >= 0.001:
        adjustment = brightness * 255
        lut = lut + adjustment

    # 2. Apply contrast (if non-zero)
    if abs(contrast) >= 0.001:
        factor = 1.0 + contrast
        lut = 128.0 + (lut - 128.0) * factor

    # 3. Apply gamma (if not 1.0)
    if abs(gamma - 1.0) >= 0.001:
        # Normalize to 0-1, apply gamma, scale back
        lut = np.clip(lut, 0, 255) / 255.0
        lut = np.power(lut, 1.0 / gamma) * 255.0

    # Clip and convert to uint8
    lut = np.clip(lut, 0, 255).astype(np.uint8)

    _combined_luts[key] = lut
    return lut


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


# Cache for expanded 3D LUTs (pre-computed to avoid per-frame interpolation)
_expanded_lut_cache: Dict[int, np.ndarray] = {}

# Numba JIT-compiled LUT application function (compiled on first use)
_numba_lut_func = None

# Numba JIT-compiled saturation function
_numba_saturation_func = None

# Numba JIT-compiled shadows/highlights function
_numba_shadow_highlight_func = None


def _get_numba_lut_function():
    """
    Get or create the numba JIT-compiled LUT application function.

    Returns a function that applies a 256³ LUT to a frame in ~3ms using
    JIT compilation. Uses non-parallel mode to be safe in multiprocess contexts.
    Falls back to numpy if numba is unavailable.
    """
    global _numba_lut_func

    if _numba_lut_func is not None:
        return _numba_lut_func

    try:
        from numba import njit

        # Use non-parallel mode to avoid threading conflicts in multiprocess contexts
        # The JIT compilation still provides ~50x speedup over pure numpy
        @njit(fastmath=True, cache=True)
        def _apply_lut_numba(frame, expanded_lut):
            """Apply 256³ LUT using numba JIT compilation."""
            height, width, _ = frame.shape
            result = np.empty((height, width, 3), dtype=np.uint8)

            for y in range(height):
                for x in range(width):
                    r = frame[y, x, 0]
                    g = frame[y, x, 1]
                    b = frame[y, x, 2]
                    result[y, x, 0] = expanded_lut[b, g, r, 0]
                    result[y, x, 1] = expanded_lut[b, g, r, 1]
                    result[y, x, 2] = expanded_lut[b, g, r, 2]

            return result

        _numba_lut_func = _apply_lut_numba
        logger.debug("Using numba JIT for LUT application (~10ms/frame)")
        return _numba_lut_func

    except ImportError:
        logger.debug("numba not available, using numpy fallback for LUT")

        def _apply_lut_numpy(frame, expanded_lut):
            """Numpy fallback for LUT application."""
            r = frame[:, :, 0]
            g = frame[:, :, 1]
            b = frame[:, :, 2]
            return expanded_lut[b, g, r]

        _numba_lut_func = _apply_lut_numpy
        return _numba_lut_func


def _expand_lut_to_256(lut: np.ndarray) -> np.ndarray:
    """
    Expand a 33x33x33 LUT to 256x256x256 via trilinear interpolation.

    This is done ONCE at load time, then lookups are direct array indexing.
    The expanded LUT trades memory (50MB) for speed (~165x faster apply).

    Args:
        lut: Original 3D LUT (typically 33x33x33x3)

    Returns:
        Expanded 256x256x256x3 uint8 LUT
    """
    cache_key = id(lut)
    if cache_key in _expanded_lut_cache:
        return _expanded_lut_cache[cache_key]

    size = lut.shape[0]
    logger.debug(f"Expanding {size}³ LUT to 256³ (one-time operation)...")

    # Create output grid
    expanded = np.zeros((256, 256, 256, 3), dtype=np.float32)

    # Scale factor from 256 to LUT indices
    scale = (size - 1) / 255.0

    # Generate all 256 values for each axis
    indices = np.arange(256) * scale

    # Floor and ceiling indices
    idx0 = np.floor(indices).astype(np.int32)
    idx1 = np.minimum(idx0 + 1, size - 1)
    frac = (indices - idx0).astype(np.float32)

    # Precompute interpolation weights
    w0 = 1.0 - frac
    w1 = frac

    # Vectorized trilinear interpolation for all 256^3 combinations
    # Process in chunks to avoid memory issues
    for b in range(256):
        b0, b1 = idx0[b], idx1[b]
        wb0, wb1 = w0[b], w1[b]

        for g in range(256):
            g0, g1 = idx0[g], idx1[g]
            wg0, wg1 = w0[g], w1[g]

            # Interpolate along R for all 256 R values at once
            r0, r1 = idx0, idx1

            # Get 8 corners for this (b,g) slice
            c000 = lut[b0, g0, r0]  # (256, 3)
            c001 = lut[b0, g0, r1]
            c010 = lut[b0, g1, r0]
            c011 = lut[b0, g1, r1]
            c100 = lut[b1, g0, r0]
            c101 = lut[b1, g0, r1]
            c110 = lut[b1, g1, r0]
            c111 = lut[b1, g1, r1]

            # Trilinear interpolation
            c00 = c000 * w0[:, np.newaxis] + c001 * w1[:, np.newaxis]
            c01 = c010 * w0[:, np.newaxis] + c011 * w1[:, np.newaxis]
            c10 = c100 * w0[:, np.newaxis] + c101 * w1[:, np.newaxis]
            c11 = c110 * w0[:, np.newaxis] + c111 * w1[:, np.newaxis]

            c0 = c00 * wg0 + c01 * wg1
            c1 = c10 * wg0 + c11 * wg1

            expanded[b, g, :, :] = c0 * wb0 + c1 * wb1

    # Convert to uint8
    expanded = (np.clip(expanded, 0, 1) * 255).astype(np.uint8)

    # Cache for reuse
    _expanded_lut_cache[cache_key] = expanded
    logger.debug(f"LUT expansion complete ({expanded.nbytes / 1024 / 1024:.1f} MB)")

    return expanded


def apply_lut(frame: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """
    Apply 3D LUT to frame using pre-expanded direct lookup with numba JIT.

    This is ~165x faster than per-pixel trilinear interpolation by:
    1. Pre-expanding the 33³ LUT to 256³ (done once at load time)
    2. Using numba JIT parallel compilation for direct indexing (~3ms/frame)

    Args:
        frame: RGB uint8 numpy array
        lut: 3D LUT array from load_cube_lut() (will be auto-expanded)

    Returns:
        Color-graded RGB uint8 numpy array
    """
    if lut is None:
        return frame

    # Expand LUT to 256³ if not already (cached after first call)
    expanded = _expand_lut_to_256(lut)

    # Get or create the numba JIT function (compiled once, cached)
    lut_func = _get_numba_lut_function()

    # Apply LUT using fast parallel function
    return lut_func(frame, expanded)


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


def _get_numba_saturation_function():
    """
    Get or create the numba JIT-compiled saturation function.

    Falls back to numpy if numba is unavailable.
    """
    global _numba_saturation_func

    if _numba_saturation_func is not None:
        return _numba_saturation_func

    try:
        from numba import njit

        # NOTE: parallel=False is required because workers are already parallelized
        # via multiprocessing. Nested parallelism causes "workqueue threading layer
        # is terminating: Concurrent access detected" errors.
        @njit(fastmath=True, cache=True)
        def _apply_saturation_numba(frame, factor):
            """Apply saturation using numba JIT."""
            height, width, _ = frame.shape
            result = np.empty((height, width, 3), dtype=np.uint8)

            for y in range(height):
                for x in range(width):
                    b = float(frame[y, x, 0])
                    g = float(frame[y, x, 1])
                    r = float(frame[y, x, 2])

                    # Rec.709 luminance
                    luma = 0.0722 * b + 0.7152 * g + 0.2126 * r

                    # Blend: result = luma + (original - luma) * factor
                    new_b = luma + (b - luma) * factor
                    new_g = luma + (g - luma) * factor
                    new_r = luma + (r - luma) * factor

                    # Clip and store
                    result[y, x, 0] = max(0, min(255, int(new_b + 0.5)))
                    result[y, x, 1] = max(0, min(255, int(new_g + 0.5)))
                    result[y, x, 2] = max(0, min(255, int(new_r + 0.5)))

            return result

        _numba_saturation_func = _apply_saturation_numba
        logger.debug("Using numba JIT for saturation")
        return _numba_saturation_func

    except ImportError:
        logger.debug("numba not available for saturation, using numpy")

        def _apply_saturation_numpy(frame, factor):
            """Numpy fallback for saturation."""
            frame_float = frame.astype(np.float32)
            luminance = (0.0722 * frame_float[:, :, 0] +
                         0.7152 * frame_float[:, :, 1] +
                         0.2126 * frame_float[:, :, 2])
            luma_3d = luminance[:, :, np.newaxis]
            result = luma_3d + (frame_float - luma_3d) * factor
            return np.clip(result, 0, 255).astype(np.uint8)

        _numba_saturation_func = _apply_saturation_numpy
        return _numba_saturation_func


def apply_saturation(frame: np.ndarray, amount: float) -> np.ndarray:
    """
    Apply saturation adjustment using Rec.709 luminance.

    Uses numba JIT compilation for ~5x speedup when available.

    Args:
        frame: BGR uint8 numpy array
        amount: Adjustment value (-1.0 to 1.0, 0 = no change)
                -1.0 = grayscale, positive values increase saturation

    Returns:
        Adjusted BGR uint8 numpy array
    """
    if abs(amount) < 0.001:
        return frame

    factor = 1.0 + amount
    sat_func = _get_numba_saturation_function()
    return sat_func(frame, factor)


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


def _get_numba_shadow_highlight_function():
    """
    Get or create the numba JIT-compiled shadows/highlights function.

    Falls back to numpy if numba is unavailable.
    """
    global _numba_shadow_highlight_func

    if _numba_shadow_highlight_func is not None:
        return _numba_shadow_highlight_func

    try:
        from numba import njit

        # NOTE: parallel=False is required because workers are already parallelized
        # via multiprocessing. Nested parallelism causes threading layer conflicts.
        @njit(fastmath=True, cache=True)
        def _apply_shadow_highlight_numba(frame, adjustment_lut):
            """Apply shadows/highlights using numba JIT."""
            height, width, _ = frame.shape
            result = np.empty((height, width, 3), dtype=np.uint8)

            for y in range(height):
                for x in range(width):
                    b = frame[y, x, 0]
                    g = frame[y, x, 1]
                    r = frame[y, x, 2]

                    # Compute luminance (integer approximation of Rec.709)
                    # luma = 0.0722*B + 0.7152*G + 0.2126*R ≈ (18*B + 183*G + 54*R) >> 8
                    luma = (18 * int(b) + 183 * int(g) + 54 * int(r)) >> 8
                    if luma > 255:
                        luma = 255

                    # Look up adjustment
                    adj = adjustment_lut[luma]

                    # Apply to all channels
                    new_b = float(b) + adj
                    new_g = float(g) + adj
                    new_r = float(r) + adj

                    # Clip and store
                    result[y, x, 0] = max(0, min(255, int(new_b + 0.5)))
                    result[y, x, 1] = max(0, min(255, int(new_g + 0.5)))
                    result[y, x, 2] = max(0, min(255, int(new_r + 0.5)))

            return result

        _numba_shadow_highlight_func = _apply_shadow_highlight_numba
        logger.debug("Using numba JIT for shadows/highlights")
        return _numba_shadow_highlight_func

    except ImportError:
        logger.debug("numba not available for shadows/highlights, using numpy")

        def _apply_shadow_highlight_numpy(frame, adjustment_lut):
            """Numpy fallback for shadows/highlights."""
            luma = ((18 * frame[:, :, 0].astype(np.int32) +
                     183 * frame[:, :, 1].astype(np.int32) +
                     54 * frame[:, :, 2].astype(np.int32)) >> 8).astype(np.uint8)
            adjustment = adjustment_lut[luma]
            result = frame.astype(np.float32) + adjustment[:, :, np.newaxis]
            return np.clip(result, 0, 255).astype(np.uint8)

        _numba_shadow_highlight_func = _apply_shadow_highlight_numpy
        return _numba_shadow_highlight_func


def apply_shadows_highlights(frame: np.ndarray, shadows: float, highlights: float) -> np.ndarray:
    """
    Apply shadows and highlights adjustment using pre-computed LUT.

    Uses numba JIT compilation for ~5x speedup when available.

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

    # Get pre-computed adjustment LUT
    adjustment_lut = _get_shadow_highlight_lut(shadows, highlights)

    # Use numba-accelerated function
    sh_func = _get_numba_shadow_highlight_function()
    return sh_func(frame, adjustment_lut)


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
        2. Brightness + Contrast + Gamma (combined into single LUT for speed)
        3. Saturation
        4. Shadows/Highlights

        Args:
            frame: BGR uint8 numpy array (OpenCV format)

        Returns:
            Color-graded BGR uint8 numpy array
        """
        if not self.is_active:
            return frame

        result = frame

        # 1. Apply color LUT first (major color transform)
        if self.lut is not None:
            result = apply_lut(result, self.lut)

        # 2. Combined brightness/contrast/gamma via pre-computed LUT (fast single pass)
        has_bcg = (abs(self.brightness) >= 0.001 or
                   abs(self.contrast) >= 0.001 or
                   abs(self.gamma - 1.0) >= 0.001)
        if has_bcg:
            combined_lut = _get_combined_lut(self.brightness, self.contrast, self.gamma)
            result = combined_lut[result]  # Single array lookup for all 3 operations

        # 3. Saturation (requires luminance calculation, can't be LUT'd)
        if abs(self.saturation) >= 0.001:
            result = apply_saturation(result, self.saturation)

        # 4. Shadows/Highlights
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


def warmup_color_grading(lut: Optional[np.ndarray] = None) -> None:
    """
    Pre-compile the numba JIT function for LUT application.

    Call this in each worker process before the main processing loop to ensure
    the JIT is warmed up and doesn't cause a 3+ second delay on first frame.

    The numba disk cache (cache=True) helps, but each process still needs to
    load and verify the cache on first call. This function ensures that overhead
    happens upfront rather than during frame processing.

    Args:
        lut: Optional LUT array. If provided, also pre-expands the LUT to 256³.
    """
    # Get/compile the numba function
    _ = _get_numba_lut_function()

    # If a LUT is provided, also expand it to 256³ (cached for reuse)
    if lut is not None:
        _ = _expand_lut_to_256(lut)

    # Run a tiny test frame to fully initialize the function
    # This triggers the actual JIT compilation/cache load
    test_frame = np.zeros((8, 8, 3), dtype=np.uint8)
    test_lut = np.zeros((256, 256, 256, 3), dtype=np.uint8)
    lut_func = _get_numba_lut_function()
    _ = lut_func(test_frame, test_lut)
