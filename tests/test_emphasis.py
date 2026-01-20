"""
Unit tests for dynamic camera emphasis module.

Tests cover:
- Blinker triggers for side cameras
- Lateral G-force emphasis
- Braking emphasis for rear camera
- Reverse gear emphasis
- Smoothing transitions
- Priority ordering (blinker > reverse > braking > lateral)
- Overlap prevention (mutual emphasis clamping)
"""

import pytest
from unittest.mock import Mock, MagicMock

from emphasis import EmphasisCalculator, EmphasisState, CameraEmphasis
from constants import (
    COLORS,
    EMPHASIS_LATERAL_G_THRESHOLD,
    EMPHASIS_BRAKING_G_THRESHOLD,
    EMPHASIS_SMOOTHING_FACTOR,
    EMPHASIS_COLOR_BLINKER,
    EMPHASIS_COLOR_BRAKE,
    EMPHASIS_COLOR_REVERSE,
    EMPHASIS_COLOR_LATERAL,
    EMPHASIS_BORDER_WIDTH,
    MPS2_TO_G,
)


def create_mock_metadata(
    blinker_left: bool = False,
    blinker_right: bool = False,
    gear_state: int = 1,  # 0=P, 1=D, 2=R, 3=N
    accel_x: float = 0.0,  # m/s² (positive = right turn)
    accel_y: float = 0.0,  # m/s² (positive = braking)
) -> Mock:
    """Create mock SEI metadata for testing."""
    meta = Mock()
    meta.blinker_on_left = blinker_left
    meta.blinker_on_right = blinker_right
    meta.gear_state = gear_state
    meta.linear_acceleration_mps2_x = accel_x
    meta.linear_acceleration_mps2_y = accel_y
    return meta


class TestEmphasisCalculatorBasic:
    """Basic functionality tests."""

    def test_init_defaults(self):
        """Test default initialization."""
        calc = EmphasisCalculator()
        assert calc.lateral_g_threshold == EMPHASIS_LATERAL_G_THRESHOLD
        assert calc.braking_g_threshold == EMPHASIS_BRAKING_G_THRESHOLD
        assert calc.smoothing == EMPHASIS_SMOOTHING_FACTOR

    def test_no_metadata_returns_zero_emphasis(self):
        """When no metadata is provided, all emphasis should be zero."""
        calc = EmphasisCalculator()
        state = calc.compute(None)

        assert state.left_repeater.weight == 0.0
        assert state.right_repeater.weight == 0.0
        assert state.left_pillar.weight == 0.0
        assert state.right_pillar.weight == 0.0
        assert state.back.weight == 0.0

    def test_neutral_metadata_returns_zero_emphasis(self):
        """When metadata has no triggers, all emphasis should be zero."""
        calc = EmphasisCalculator()
        meta = create_mock_metadata()
        state = calc.compute(meta)

        # First frame, smoothing starts from 0
        assert state.left_repeater.weight == 0.0
        assert state.right_repeater.weight == 0.0
        assert state.back.weight == 0.0


class TestBlinkerEmphasis:
    """Tests for turn signal (blinker) emphasis."""

    def test_left_blinker_emphasizes_left_cameras(self):
        """Left blinker should emphasize left repeater and pillar."""
        calc = EmphasisCalculator(smoothing=1.0)  # No smoothing for immediate result
        meta = create_mock_metadata(blinker_left=True)
        state = calc.compute(meta)

        assert state.left_repeater.weight == 1.0
        assert state.left_pillar.weight == 0.7  # Reduced weight for pillar
        assert state.left_repeater.border_color == EMPHASIS_COLOR_BLINKER
        assert state.left_pillar.border_color == EMPHASIS_COLOR_BLINKER

        # Right side should be unaffected
        assert state.right_repeater.weight == 0.0
        assert state.right_pillar.weight == 0.0

    def test_right_blinker_emphasizes_right_cameras(self):
        """Right blinker should emphasize right repeater and pillar."""
        calc = EmphasisCalculator(smoothing=1.0)
        meta = create_mock_metadata(blinker_right=True)
        state = calc.compute(meta)

        assert state.right_repeater.weight == 1.0
        assert state.right_pillar.weight == 0.7
        assert state.right_repeater.border_color == EMPHASIS_COLOR_BLINKER
        assert state.right_pillar.border_color == EMPHASIS_COLOR_BLINKER

        # Left side should be unaffected
        assert state.left_repeater.weight == 0.0

    def test_both_blinkers_emphasize_both_sides(self):
        """Hazard lights (both blinkers) should emphasize both sides."""
        calc = EmphasisCalculator(smoothing=1.0)
        meta = create_mock_metadata(blinker_left=True, blinker_right=True)
        state = calc.compute(meta)

        assert state.left_repeater.weight == 1.0
        assert state.right_repeater.weight == 1.0


class TestReverseGearEmphasis:
    """Tests for reverse gear emphasis."""

    def test_reverse_gear_emphasizes_back_camera(self):
        """Reverse gear should emphasize back camera."""
        calc = EmphasisCalculator(smoothing=1.0)
        meta = create_mock_metadata(gear_state=2)  # 2 = Reverse
        state = calc.compute(meta)

        assert state.back.weight == 1.0
        assert state.back.border_color == EMPHASIS_COLOR_REVERSE

    def test_drive_gear_no_back_emphasis(self):
        """Drive gear should not emphasize back camera."""
        calc = EmphasisCalculator(smoothing=1.0)
        meta = create_mock_metadata(gear_state=1)  # 1 = Drive
        state = calc.compute(meta)

        assert state.back.weight == 0.0


class TestBrakingEmphasis:
    """Tests for braking emphasis."""

    def test_heavy_braking_emphasizes_back_camera(self):
        """Heavy braking should emphasize back camera."""
        calc = EmphasisCalculator(smoothing=1.0)
        # 0.5g braking in m/s² = 0.5 * 9.81 ≈ 4.9 m/s²
        braking_mps2 = 0.5 / MPS2_TO_G
        meta = create_mock_metadata(accel_y=braking_mps2)
        state = calc.compute(meta)

        assert state.back.weight > 0.0
        assert state.back.border_color == EMPHASIS_COLOR_BRAKE

    def test_light_braking_no_emphasis(self):
        """Light braking (below threshold) should not emphasize."""
        calc = EmphasisCalculator(smoothing=1.0)
        # 0.1g braking (below 0.3g threshold)
        braking_mps2 = 0.1 / MPS2_TO_G
        meta = create_mock_metadata(accel_y=braking_mps2)
        state = calc.compute(meta)

        assert state.back.weight == 0.0

    def test_reverse_overrides_braking(self):
        """Reverse gear should override braking for back camera color."""
        calc = EmphasisCalculator(smoothing=1.0)
        braking_mps2 = 0.5 / MPS2_TO_G
        meta = create_mock_metadata(gear_state=2, accel_y=braking_mps2)
        state = calc.compute(meta)

        # Back camera should be emphasized with reverse color, not brake color
        assert state.back.weight == 1.0
        assert state.back.border_color == EMPHASIS_COLOR_REVERSE


class TestLateralGEmphasis:
    """Tests for lateral G-force emphasis during turns."""

    def test_left_turn_emphasizes_left_cameras(self):
        """Left turn (negative lateral G) should emphasize left cameras."""
        calc = EmphasisCalculator(smoothing=1.0)
        # 0.4g left turn (negative X = left)
        lateral_mps2 = -0.4 / MPS2_TO_G
        meta = create_mock_metadata(accel_x=lateral_mps2)
        state = calc.compute(meta)

        assert state.left_repeater.weight > 0.0
        assert state.left_pillar.weight > 0.0
        assert state.left_repeater.border_color == EMPHASIS_COLOR_LATERAL

    def test_right_turn_emphasizes_right_cameras(self):
        """Right turn (positive lateral G) should emphasize right cameras."""
        calc = EmphasisCalculator(smoothing=1.0)
        # 0.4g right turn (positive X = right)
        lateral_mps2 = 0.4 / MPS2_TO_G
        meta = create_mock_metadata(accel_x=lateral_mps2)
        state = calc.compute(meta)

        assert state.right_repeater.weight > 0.0
        assert state.right_pillar.weight > 0.0
        assert state.right_repeater.border_color == EMPHASIS_COLOR_LATERAL

    def test_light_turn_no_emphasis(self):
        """Light turn (below threshold) should not emphasize."""
        calc = EmphasisCalculator(smoothing=1.0)
        # 0.1g turn (below 0.2g threshold)
        lateral_mps2 = 0.1 / MPS2_TO_G
        meta = create_mock_metadata(accel_x=lateral_mps2)
        state = calc.compute(meta)

        assert state.left_repeater.weight == 0.0
        assert state.right_repeater.weight == 0.0

    def test_blinker_overrides_lateral_g_color(self):
        """Blinker should override lateral G for color (priority)."""
        calc = EmphasisCalculator(smoothing=1.0)
        lateral_mps2 = -0.4 / MPS2_TO_G
        meta = create_mock_metadata(blinker_left=True, accel_x=lateral_mps2)
        state = calc.compute(meta)

        # Left cameras should use blinker color, not lateral color
        assert state.left_repeater.border_color == EMPHASIS_COLOR_BLINKER


class TestSmoothing:
    """Tests for emphasis transition smoothing."""

    def test_smoothing_gradual_increase(self):
        """Emphasis should increase gradually over multiple frames."""
        calc = EmphasisCalculator(smoothing=0.3)

        # Frame 1: Start from 0, blinker on
        meta = create_mock_metadata(blinker_left=True)
        state1 = calc.compute(meta)

        # With 30% smoothing, first frame should be 0.3
        assert 0.25 < state1.left_repeater.weight < 0.35

        # Frame 2: Continue
        state2 = calc.compute(meta)
        assert state2.left_repeater.weight > state1.left_repeater.weight

        # Frame 3: Continue
        state3 = calc.compute(meta)
        assert state3.left_repeater.weight > state2.left_repeater.weight

    def test_smoothing_gradual_decrease(self):
        """Emphasis should decrease gradually when trigger ends."""
        calc = EmphasisCalculator(smoothing=0.5)

        # First: Build up to full emphasis
        meta_on = create_mock_metadata(blinker_left=True)
        for _ in range(10):
            calc.compute(meta_on)

        # Now turn off blinker
        meta_off = create_mock_metadata(blinker_left=False)
        state1 = calc.compute(meta_off)

        # Should still have some emphasis (not instant drop to 0)
        assert state1.left_repeater.weight > 0.0

        # Continue decreasing
        state2 = calc.compute(meta_off)
        assert state2.left_repeater.weight < state1.left_repeater.weight

    def test_reset_clears_smoothed_state(self):
        """Reset should clear all smoothed state."""
        calc = EmphasisCalculator(smoothing=0.5)

        # Build up emphasis
        meta = create_mock_metadata(blinker_left=True)
        for _ in range(5):
            calc.compute(meta)

        # Reset
        calc.reset()

        # Next compute should start from zero
        state = calc.compute(None)
        assert state.left_repeater.weight == 0.0


class TestEmphasisState:
    """Tests for EmphasisState dataclass."""

    def test_get_valid_camera(self):
        """get() should return correct emphasis for valid camera names."""
        state = EmphasisState(
            left_repeater=CameraEmphasis(weight=0.5),
            back=CameraEmphasis(weight=0.8),
        )

        assert state.get('left_repeater').weight == 0.5
        assert state.get('back').weight == 0.8

    def test_get_invalid_camera_returns_default(self):
        """get() should return default CameraEmphasis for invalid names."""
        state = EmphasisState()
        result = state.get('invalid_camera')

        assert result.weight == 0.0
        assert result.border_color is None
        assert result.border_width == 0


class TestCameraEmphasis:
    """Tests for CameraEmphasis dataclass."""

    def test_default_values(self):
        """Default values should be no emphasis."""
        emph = CameraEmphasis()

        assert emph.weight == 0.0
        assert emph.border_color is None
        assert emph.border_width == 0

    def test_custom_values(self):
        """Custom values should be preserved."""
        emph = CameraEmphasis(
            weight=0.75,
            border_color=(255, 100, 0),
            border_width=4
        )

        assert emph.weight == 0.75
        assert emph.border_color == (255, 100, 0)
        assert emph.border_width == 4


class TestBorderThreshold:
    """Tests for border visibility threshold."""

    def test_low_weight_no_border(self):
        """Very low emphasis weight should not show border."""
        calc = EmphasisCalculator(smoothing=0.01)  # Very slow smoothing
        meta = create_mock_metadata(blinker_left=True)
        state = calc.compute(meta)

        # With 1% smoothing, first frame weight is ~0.01, too low for border
        assert state.left_repeater.border_color is None

    def test_high_weight_shows_border(self):
        """High emphasis weight should show border."""
        calc = EmphasisCalculator(smoothing=1.0)
        meta = create_mock_metadata(blinker_left=True)
        state = calc.compute(meta)

        assert state.left_repeater.border_color is not None
        assert state.left_repeater.border_width == EMPHASIS_BORDER_WIDTH


class TestProportionalEmphasis:
    """Tests for proportional emphasis based on G-force intensity."""

    def test_braking_proportional_to_intensity(self):
        """Braking emphasis should scale with G-force intensity."""
        calc = EmphasisCalculator(smoothing=1.0)

        # Moderate braking (0.4g)
        meta_moderate = create_mock_metadata(accel_y=0.4 / MPS2_TO_G)
        state_moderate = calc.compute(meta_moderate)
        calc.reset()

        # Hard braking (0.8g)
        meta_hard = create_mock_metadata(accel_y=0.8 / MPS2_TO_G)
        state_hard = calc.compute(meta_hard)

        assert state_hard.back.weight > state_moderate.back.weight

    def test_lateral_g_proportional_to_intensity(self):
        """Lateral G emphasis should scale with turn intensity."""
        calc = EmphasisCalculator(smoothing=1.0)

        # Gentle turn (0.25g)
        meta_gentle = create_mock_metadata(accel_x=-0.25 / MPS2_TO_G)
        state_gentle = calc.compute(meta_gentle)
        calc.reset()

        # Sharp turn (0.6g)
        meta_sharp = create_mock_metadata(accel_x=-0.6 / MPS2_TO_G)
        state_sharp = calc.compute(meta_sharp)

        assert state_sharp.left_repeater.weight > state_gentle.left_repeater.weight


class TestColorFadeOut:
    """Tests for color persistence during emphasis fade-out."""

    def test_color_persists_during_fadeout(self):
        """Border color should persist during fade-out, not disappear immediately."""
        calc = EmphasisCalculator(smoothing=0.5)

        # Build up emphasis with blinker
        meta_on = create_mock_metadata(blinker_left=True)
        for _ in range(10):  # Build to near-full emphasis
            calc.compute(meta_on)

        # Turn off blinker - color should persist during fade
        meta_off = create_mock_metadata(blinker_left=False)
        state1 = calc.compute(meta_off)

        # Weight is still high, color should persist (using last known color)
        assert state1.left_repeater.weight > 0.1
        assert state1.left_repeater.border_color == EMPHASIS_COLOR_BLINKER

        # Continue fading
        state2 = calc.compute(meta_off)
        assert state2.left_repeater.border_color == EMPHASIS_COLOR_BLINKER

    def test_color_cleared_when_fully_faded(self):
        """Border color should be cleared when weight drops below threshold."""
        calc = EmphasisCalculator(smoothing=0.9)  # Fast fade

        # Brief emphasis
        meta_on = create_mock_metadata(blinker_left=True)
        calc.compute(meta_on)

        # Fade out completely
        meta_off = create_mock_metadata()
        for _ in range(20):  # Fade to near zero
            state = calc.compute(meta_off)

        # Should be fully faded with no color
        assert state.left_repeater.weight < 0.05
        assert state.left_repeater.border_color is None


class TestThresholdValidation:
    """Tests for threshold validation in EmphasisCalculator."""

    def test_extreme_threshold_clamped(self):
        """Thresholds >= 1.0 should be clamped to prevent division by zero."""
        # This should not raise an exception
        calc = EmphasisCalculator(lateral_g_threshold=1.0, braking_g_threshold=1.0)

        # Thresholds should be clamped to 0.9
        assert calc.lateral_g_threshold == 0.9
        assert calc.braking_g_threshold == 0.9

    def test_normal_threshold_preserved(self):
        """Normal thresholds should be preserved."""
        calc = EmphasisCalculator(lateral_g_threshold=0.3, braking_g_threshold=0.4)

        assert calc.lateral_g_threshold == 0.3
        assert calc.braking_g_threshold == 0.4
