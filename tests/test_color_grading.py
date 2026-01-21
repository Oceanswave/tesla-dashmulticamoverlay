"""
Unit tests for the color grading module.

Tests cover:
- Individual adjustment functions (brightness, contrast, saturation, gamma, shadows/highlights)
- LUT loading and application
- ColorGrader class with presets and stacking
- Frame shape/dtype preservation
"""

import pytest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from color_grading import (
    load_cube_lut,
    apply_lut,
    apply_brightness,
    apply_contrast,
    apply_saturation,
    apply_gamma,
    apply_shadows_highlights,
    ColorGrader,
    create_color_grader,
)
from constants import COLOR_PRESETS


@pytest.fixture
def sample_frame():
    """Create a test frame with known values for predictable testing."""
    # 100x100 BGR frame with gradient
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    # Create a gradient from 0-255 across the width
    for x in range(100):
        frame[:, x, :] = int(x * 2.55)  # 0-255 range
    return frame


@pytest.fixture
def mid_gray_frame():
    """Create a frame filled with mid-gray (128) for testing around midpoint."""
    return np.full((100, 100, 3), 128, dtype=np.uint8)


@pytest.fixture
def full_hd_frame():
    """Create a 1080p frame for performance-representative testing."""
    return np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8)


@pytest.fixture
def lut_path():
    """Path to a real LUT file for integration testing."""
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "LUTs", "NaturalBoost.cube")
    if not os.path.exists(path):
        pytest.skip("LUT file not found for testing")
    return path


class TestBrightness:
    """Tests for brightness adjustment."""

    def test_zero_adjustment_no_change(self, sample_frame):
        """Zero brightness adjustment should return identical frame."""
        result = apply_brightness(sample_frame, 0.0)
        np.testing.assert_array_equal(result, sample_frame)

    def test_positive_brightens_frame(self, mid_gray_frame):
        """Positive brightness should increase pixel values."""
        result = apply_brightness(mid_gray_frame, 0.5)
        assert result.mean() > mid_gray_frame.mean()

    def test_negative_darkens_frame(self, mid_gray_frame):
        """Negative brightness should decrease pixel values."""
        result = apply_brightness(mid_gray_frame, -0.5)
        assert result.mean() < mid_gray_frame.mean()

    def test_clipping_at_max(self):
        """Brightness should clip at 255."""
        bright_frame = np.full((10, 10, 3), 250, dtype=np.uint8)
        result = apply_brightness(bright_frame, 0.5)
        assert result.max() == 255

    def test_clipping_at_min(self):
        """Brightness should clip at 0."""
        dark_frame = np.full((10, 10, 3), 5, dtype=np.uint8)
        result = apply_brightness(dark_frame, -0.5)
        assert result.min() == 0

    def test_preserves_shape_and_dtype(self, sample_frame):
        """Result should have same shape and dtype as input."""
        result = apply_brightness(sample_frame, 0.3)
        assert result.shape == sample_frame.shape
        assert result.dtype == np.uint8


class TestContrast:
    """Tests for contrast adjustment."""

    def test_zero_adjustment_no_change(self, sample_frame):
        """Zero contrast adjustment should return identical frame."""
        result = apply_contrast(sample_frame, 0.0)
        np.testing.assert_array_equal(result, sample_frame)

    def test_positive_increases_contrast(self, sample_frame):
        """Positive contrast should increase the range of values."""
        result = apply_contrast(sample_frame, 0.5)
        # Standard deviation should increase with contrast
        assert result.std() >= sample_frame.std()

    def test_negative_decreases_contrast(self, sample_frame):
        """Negative contrast should decrease the range of values."""
        result = apply_contrast(sample_frame, -0.5)
        # Standard deviation should decrease
        assert result.std() <= sample_frame.std()

    def test_midpoint_stays_same(self, mid_gray_frame):
        """Values at 128 should stay approximately the same."""
        result = apply_contrast(mid_gray_frame, 0.5)
        # Mid-gray should be close to 128 still
        assert abs(result.mean() - 128) < 5

    def test_preserves_shape_and_dtype(self, sample_frame):
        """Result should have same shape and dtype as input."""
        result = apply_contrast(sample_frame, 0.3)
        assert result.shape == sample_frame.shape
        assert result.dtype == np.uint8


class TestSaturation:
    """Tests for saturation adjustment."""

    def test_zero_adjustment_no_change(self, sample_frame):
        """Zero saturation adjustment should return identical frame."""
        result = apply_saturation(sample_frame, 0.0)
        np.testing.assert_array_equal(result, sample_frame)

    def test_negative_one_creates_grayscale(self):
        """Saturation of -1 should create grayscale image."""
        # Create a colorful frame
        colorful = np.zeros((10, 10, 3), dtype=np.uint8)
        colorful[:, :, 0] = 255  # Blue channel
        colorful[:, :, 2] = 100  # Red channel

        result = apply_saturation(colorful, -1.0)

        # In grayscale, all channels should be equal
        assert np.allclose(result[:, :, 0], result[:, :, 1], atol=1)
        assert np.allclose(result[:, :, 1], result[:, :, 2], atol=1)

    def test_positive_increases_saturation(self):
        """Positive saturation should increase color difference from gray."""
        # Frame with some color
        frame = np.full((10, 10, 3), 128, dtype=np.uint8)
        frame[:, :, 0] = 100  # Blue slightly different

        result = apply_saturation(frame, 0.5)

        # Color difference should increase
        input_diff = abs(frame[:, :, 0].mean() - frame[:, :, 1].mean())
        output_diff = abs(result[:, :, 0].mean() - result[:, :, 1].mean())
        assert output_diff >= input_diff

    def test_preserves_shape_and_dtype(self, sample_frame):
        """Result should have same shape and dtype as input."""
        result = apply_saturation(sample_frame, 0.3)
        assert result.shape == sample_frame.shape
        assert result.dtype == np.uint8


class TestGamma:
    """Tests for gamma correction."""

    def test_gamma_one_no_change(self, sample_frame):
        """Gamma of 1.0 should return identical frame."""
        result = apply_gamma(sample_frame, 1.0)
        np.testing.assert_array_equal(result, sample_frame)

    def test_gamma_less_than_one_darkens(self, mid_gray_frame):
        """Gamma < 1 should darken midtones (lower gamma = darker)."""
        result = apply_gamma(mid_gray_frame, 0.5)
        assert result.mean() < mid_gray_frame.mean()

    def test_gamma_greater_than_one_brightens(self, mid_gray_frame):
        """Gamma > 1 should brighten midtones (higher gamma = brighter)."""
        result = apply_gamma(mid_gray_frame, 2.0)
        assert result.mean() > mid_gray_frame.mean()

    def test_black_stays_black(self):
        """Black (0) should remain black regardless of gamma."""
        black_frame = np.zeros((10, 10, 3), dtype=np.uint8)
        result = apply_gamma(black_frame, 0.5)
        np.testing.assert_array_equal(result, black_frame)

    def test_white_stays_white(self):
        """White (255) should remain white regardless of gamma."""
        white_frame = np.full((10, 10, 3), 255, dtype=np.uint8)
        result = apply_gamma(white_frame, 2.0)
        np.testing.assert_array_equal(result, white_frame)

    def test_preserves_shape_and_dtype(self, sample_frame):
        """Result should have same shape and dtype as input."""
        result = apply_gamma(sample_frame, 1.5)
        assert result.shape == sample_frame.shape
        assert result.dtype == np.uint8


class TestShadowsHighlights:
    """Tests for shadows and highlights adjustment."""

    def test_zero_adjustments_no_change(self, sample_frame):
        """Zero adjustments should return identical frame."""
        result = apply_shadows_highlights(sample_frame, 0.0, 0.0)
        np.testing.assert_array_equal(result, sample_frame)

    def test_positive_shadows_lifts_darks(self):
        """Positive shadows should lift dark areas."""
        # Dark frame
        dark_frame = np.full((10, 10, 3), 30, dtype=np.uint8)
        result = apply_shadows_highlights(dark_frame, 0.5, 0.0)
        assert result.mean() > dark_frame.mean()

    def test_negative_shadows_crushes_darks(self):
        """Negative shadows should crush dark areas."""
        dark_frame = np.full((10, 10, 3), 30, dtype=np.uint8)
        result = apply_shadows_highlights(dark_frame, -0.5, 0.0)
        assert result.mean() < dark_frame.mean()

    def test_positive_highlights_lifts_brights(self):
        """Positive highlights should lift bright areas."""
        bright_frame = np.full((10, 10, 3), 220, dtype=np.uint8)
        result = apply_shadows_highlights(bright_frame, 0.0, 0.5)
        assert result.mean() > bright_frame.mean()

    def test_negative_highlights_crushes_brights(self):
        """Negative highlights should crush bright areas."""
        bright_frame = np.full((10, 10, 3), 220, dtype=np.uint8)
        result = apply_shadows_highlights(bright_frame, 0.0, -0.5)
        assert result.mean() < bright_frame.mean()

    def test_preserves_shape_and_dtype(self, sample_frame):
        """Result should have same shape and dtype as input."""
        result = apply_shadows_highlights(sample_frame, 0.2, -0.1)
        assert result.shape == sample_frame.shape
        assert result.dtype == np.uint8


class TestLUTLoading:
    """Tests for LUT file loading."""

    def test_load_valid_lut(self, lut_path):
        """Loading a valid .cube file should return a 3D array."""
        lut = load_cube_lut(lut_path)
        assert lut is not None
        assert lut.ndim == 4  # (size, size, size, 3)
        assert lut.shape[3] == 3  # RGB channels

    def test_load_nonexistent_file(self):
        """Loading a nonexistent file should return None."""
        lut = load_cube_lut("/nonexistent/path/to.cube")
        assert lut is None

    def test_lut_caching(self, lut_path):
        """LUT should be cached after first load."""
        lut1 = load_cube_lut(lut_path)
        lut2 = load_cube_lut(lut_path)
        # Should be the same object (cached)
        assert lut1 is lut2


class TestLUTApplication:
    """Tests for LUT application to frames."""

    def test_apply_lut_changes_frame(self, sample_frame, lut_path):
        """Applying a LUT should change the frame."""
        lut = load_cube_lut(lut_path)
        result = apply_lut(sample_frame, lut)
        # Should be different from input (unless LUT is identity)
        assert not np.array_equal(result, sample_frame)

    def test_apply_lut_preserves_shape(self, sample_frame, lut_path):
        """LUT application should preserve frame shape and dtype."""
        lut = load_cube_lut(lut_path)
        result = apply_lut(sample_frame, lut)
        assert result.shape == sample_frame.shape
        assert result.dtype == np.uint8

    def test_apply_none_lut_no_change(self, sample_frame):
        """Applying None LUT should return original frame."""
        result = apply_lut(sample_frame, None)
        np.testing.assert_array_equal(result, sample_frame)


class TestColorGrader:
    """Tests for the ColorGrader class."""

    def test_inactive_by_default(self):
        """ColorGrader with no settings should be inactive."""
        grader = ColorGrader()
        assert not grader.is_active

    def test_active_with_brightness(self):
        """ColorGrader with brightness adjustment should be active."""
        grader = ColorGrader(brightness=0.1)
        assert grader.is_active

    def test_active_with_preset(self):
        """ColorGrader with valid preset should be active."""
        grader = ColorGrader(preset="cinematic")
        assert grader.is_active

    def test_preset_applies_values(self):
        """Preset should set the adjustment values."""
        grader = ColorGrader(preset="cinematic")
        # Cinematic preset has contrast of 0.15
        assert grader.contrast == 0.15

    def test_manual_stacks_on_preset(self):
        """Manual adjustments should stack on preset values."""
        grader = ColorGrader(preset="cinematic", contrast=0.1)
        # Cinematic has 0.15 contrast, plus 0.1 manual = 0.25
        assert abs(grader.contrast - 0.25) < 0.01

    def test_unknown_preset_ignored(self):
        """Unknown preset name should be ignored (with warning)."""
        grader = ColorGrader(preset="nonexistent_preset", brightness=0.1)
        # Should still work, just without preset values
        assert grader.is_active
        assert grader.brightness == 0.1

    def test_grade_no_op_when_inactive(self, sample_frame):
        """Grading with inactive grader should return original frame."""
        grader = ColorGrader()
        result = grader.grade(sample_frame)
        np.testing.assert_array_equal(result, sample_frame)

    def test_grade_applies_adjustments(self, sample_frame):
        """Grading should apply configured adjustments."""
        grader = ColorGrader(brightness=0.2, contrast=0.1)
        result = grader.grade(sample_frame)
        # Frame should be modified
        assert not np.array_equal(result, sample_frame)

    def test_grade_preserves_shape_dtype(self, sample_frame):
        """Grading should preserve frame shape and dtype."""
        grader = ColorGrader(preset="vivid")
        result = grader.grade(sample_frame)
        assert result.shape == sample_frame.shape
        assert result.dtype == np.uint8

    def test_lut_path_loads_lut(self, lut_path):
        """Providing a LUT path should load and apply it."""
        grader = ColorGrader(lut_path=lut_path)
        assert grader.is_active
        assert grader.lut is not None


class TestCreateColorGrader:
    """Tests for the factory function."""

    def test_returns_none_when_inactive(self):
        """Factory should return None when no grading is configured."""
        grader = create_color_grader()
        assert grader is None

    def test_returns_grader_with_preset(self):
        """Factory should return ColorGrader with preset."""
        grader = create_color_grader(color_grade="warm")
        assert grader is not None
        assert grader.is_active

    def test_returns_grader_with_lut_path(self, lut_path):
        """Factory should return ColorGrader with LUT path."""
        grader = create_color_grader(color_grade=lut_path)
        assert grader is not None
        assert grader.lut is not None

    def test_returns_grader_with_manual_params(self):
        """Factory should return ColorGrader with manual parameters."""
        grader = create_color_grader(brightness=0.1, saturation=0.2)
        assert grader is not None
        assert grader.brightness == 0.1
        assert grader.saturation == 0.2


class TestPresets:
    """Tests for built-in presets."""

    def test_all_presets_exist(self):
        """All documented presets should exist in COLOR_PRESETS."""
        expected_presets = ["cinematic", "warm", "cool", "vivid", "cybertruck", "dramatic", "vintage", "natural"]
        for preset in expected_presets:
            assert preset in COLOR_PRESETS, f"Missing preset: {preset}"

    def test_presets_have_required_keys(self):
        """Each preset should have all adjustment keys."""
        required_keys = ["brightness", "contrast", "saturation", "gamma", "shadows", "highlights"]
        for name, values in COLOR_PRESETS.items():
            for key in required_keys:
                assert key in values, f"Preset '{name}' missing key '{key}'"

    def test_presets_values_in_range(self):
        """Preset values should be within valid ranges."""
        for name, values in COLOR_PRESETS.items():
            assert -1.0 <= values["brightness"] <= 1.0, f"Invalid brightness in {name}"
            assert -1.0 <= values["contrast"] <= 1.0, f"Invalid contrast in {name}"
            assert -1.0 <= values["saturation"] <= 1.0, f"Invalid saturation in {name}"
            assert 0.1 <= values["gamma"] <= 3.0, f"Invalid gamma in {name}"
            assert -1.0 <= values["shadows"] <= 1.0, f"Invalid shadows in {name}"
            assert -1.0 <= values["highlights"] <= 1.0, f"Invalid highlights in {name}"

    def test_cybertruck_preset_is_desaturated(self):
        """Cybertruck preset should have negative saturation for steel look."""
        assert COLOR_PRESETS["cybertruck"]["saturation"] < 0


class TestPerformance:
    """Performance-related tests."""

    def test_1080p_frame_processing(self, full_hd_frame):
        """Processing a 1080p frame should complete in reasonable time."""
        import time

        grader = ColorGrader(preset="cinematic", brightness=0.1)

        # Warm up run (first run may be slower due to cache misses)
        _ = grader.grade(full_hd_frame.copy())

        start = time.time()
        result = grader.grade(full_hd_frame)
        elapsed = time.time() - start

        # Should complete in under 200ms (generous for test stability across environments)
        # In practice, expect ~10-50ms on modern hardware
        assert elapsed < 0.2, f"Grading took {elapsed*1000:.1f}ms"
        assert result.shape == full_hd_frame.shape

    def test_multiple_frames_consistent(self, sample_frame):
        """Multiple frames should produce consistent results."""
        grader = ColorGrader(preset="warm")

        result1 = grader.grade(sample_frame.copy())
        result2 = grader.grade(sample_frame.copy())

        np.testing.assert_array_equal(result1, result2)


class TestMultiprocessingSafety:
    """Tests for multiprocessing safety (numba threading conflicts)."""

    def test_saturation_not_using_parallel(self):
        """Saturation numba function should not use parallel mode.

        Parallel mode in numba conflicts with multiprocessing workers,
        causing "Numba workqueue threading layer is terminating:
        Concurrent access has been detected" errors.
        """
        from color_grading import _get_numba_saturation_function

        func = _get_numba_saturation_function()

        # If using numba, check that parallel is disabled
        if hasattr(func, 'targetoptions'):
            assert not func.targetoptions.get('parallel', False), \
                "Saturation function should not use parallel=True (causes multiprocessing conflicts)"

    def test_shadow_highlight_not_using_parallel(self):
        """Shadows/highlights numba function should not use parallel mode."""
        from color_grading import _get_numba_shadow_highlight_function

        func = _get_numba_shadow_highlight_function()

        # If using numba, check that parallel is disabled
        if hasattr(func, 'targetoptions'):
            assert not func.targetoptions.get('parallel', False), \
                "Shadow/highlight function should not use parallel=True (causes multiprocessing conflicts)"

    def test_lut_function_not_using_parallel(self):
        """LUT numba function should not use parallel mode."""
        from color_grading import _get_numba_lut_function

        func = _get_numba_lut_function()

        # If using numba, check that parallel is disabled
        if hasattr(func, 'targetoptions'):
            assert not func.targetoptions.get('parallel', False), \
                "LUT function should not use parallel=True (causes multiprocessing conflicts)"
