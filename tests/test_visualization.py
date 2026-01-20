"""
Tests for visualization rendering and compositing.

Tests dashboard rendering, map rendering, overlay blending, and frame composition.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from constants import (
    OUTPUT_WIDTH, OUTPUT_HEIGHT,
    DASHBOARD_WIDTH, DASHBOARD_HEIGHT,
    MAP_SIZE,
    COLORS,
    OVERLAY_OPACITY, CANVAS_OPACITY,
)
from visualization import (
    apply_overlay,
    DashboardRenderer,
    MapRenderer,
    composite_frame,
)


class TestApplyOverlay:
    """Tests for overlay blending function."""

    def test_overlay_applied(self):
        """Overlay should be blended onto canvas."""
        canvas = np.zeros((100, 100, 3), dtype=np.uint8)
        overlay = np.ones((20, 20, 3), dtype=np.uint8) * 255  # White overlay

        apply_overlay(canvas, overlay, 10, 10)

        # Check that the region was modified
        roi = canvas[10:30, 10:30]
        assert np.any(roi > 0), "Overlay should modify canvas"

    def test_overlay_outside_bounds_ignored(self):
        """Overlay outside canvas bounds should be ignored."""
        canvas = np.zeros((100, 100, 3), dtype=np.uint8)
        overlay = np.ones((20, 20, 3), dtype=np.uint8) * 255

        # Position overlay completely outside canvas
        apply_overlay(canvas, overlay, 200, 200)

        assert np.all(canvas == 0), "Canvas should be unchanged"

    def test_negative_position_ignored(self):
        """Negative positions should be ignored."""
        canvas = np.zeros((100, 100, 3), dtype=np.uint8)
        overlay = np.ones((20, 20, 3), dtype=np.uint8) * 255

        apply_overlay(canvas, overlay, -10, -10)

        assert np.all(canvas == 0), "Canvas should be unchanged"

    def test_partial_overlap_ignored(self):
        """Partial overlap (overlay extends past edge) should be ignored."""
        canvas = np.zeros((100, 100, 3), dtype=np.uint8)
        overlay = np.ones((20, 20, 3), dtype=np.uint8) * 255

        # Position at edge so overlay would extend past
        apply_overlay(canvas, overlay, 90, 90)

        assert np.all(canvas == 0), "Canvas should be unchanged for partial overlap"

    def test_custom_opacity(self):
        """Custom opacity values should be used."""
        canvas = np.ones((100, 100, 3), dtype=np.uint8) * 128
        overlay = np.ones((20, 20, 3), dtype=np.uint8) * 255

        apply_overlay(canvas, overlay, 10, 10, overlay_weight=0.5, canvas_weight=0.5)

        roi = canvas[10:30, 10:30]
        # With 50/50 blend: (128 * 0.5) + (255 * 0.5) = 191
        expected = int(128 * 0.5 + 255 * 0.5)
        assert np.allclose(roi, expected, atol=1), "Blend should respect weights"


class TestDashboardRenderer:
    """Tests for dashboard overlay rendering."""

    def test_render_returns_correct_size(self, mock_sei_metadata):
        """Dashboard render should return correct dimensions."""
        renderer = DashboardRenderer()
        result = renderer.render(mock_sei_metadata)

        assert result.shape == (DASHBOARD_HEIGHT, DASHBOARD_WIDTH, 3)

    def test_render_with_zero_speed(self):
        """Dashboard should render with zero speed."""
        from tests.conftest import MockSeiMetadata
        renderer = DashboardRenderer()
        meta = MockSeiMetadata(vehicle_speed_mps=0.0)
        result = renderer.render(meta)

        assert result is not None
        assert result.shape == (DASHBOARD_HEIGHT, DASHBOARD_WIDTH, 3)

    def test_render_with_max_speed(self):
        """Dashboard should handle high speeds."""
        from tests.conftest import MockSeiMetadata
        renderer = DashboardRenderer()
        meta = MockSeiMetadata(vehicle_speed_mps=50.0)  # ~112 mph
        result = renderer.render(meta)

        assert result is not None

    def test_render_with_steering_angle(self):
        """Dashboard should render steering angle."""
        from tests.conftest import MockSeiMetadata
        renderer = DashboardRenderer()
        meta = MockSeiMetadata(steering_wheel_angle=45.0)  # Match proto field name
        result = renderer.render(meta)

        assert result is not None


class TestMapRenderer:
    """Tests for map overlay rendering."""

    def test_render_returns_correct_size(self, mock_sei_metadata):
        """Map render should return correct dimensions."""
        renderer = MapRenderer()
        result = renderer.render(
            mock_sei_metadata.heading_deg,
            mock_sei_metadata.latitude_deg,
            mock_sei_metadata.longitude_deg
        )

        assert result.shape == (MAP_SIZE, MAP_SIZE, 3)

    def test_update_adds_to_path(self):
        """Update should add points to path."""
        renderer = MapRenderer()
        initial_len = len(renderer.path)

        renderer.update(37.7749, -122.4194)

        assert len(renderer.path) == initial_len + 1

    def test_history_preserved_from_init(self, sample_gps_history):
        """History from init should be preserved in path."""
        renderer = MapRenderer(history=sample_gps_history)

        assert len(renderer.path) == len(sample_gps_history)

    def test_render_with_history(self, sample_gps_history):
        """Map should render with existing history."""
        renderer = MapRenderer(history=sample_gps_history)
        result = renderer.render(45.0, 37.7749, -122.4194)

        assert result is not None
        assert result.shape == (MAP_SIZE, MAP_SIZE, 3)


class TestCompositeFrame:
    """Tests for multi-camera frame composition."""

    def test_front_only_composition(self, sample_frame, front_only_cameras):
        """Composition with only front camera."""
        result = composite_frame(
            front=sample_frame,
            cameras=front_only_cameras
        )

        assert result.shape == (OUTPUT_HEIGHT, OUTPUT_WIDTH, 3)

    def test_all_cameras_composition(self, sample_frame, all_cameras):
        """Composition with all cameras."""
        result = composite_frame(
            front=sample_frame,
            left_rep=sample_frame,
            right_rep=sample_frame,
            back=sample_frame,
            left_pill=sample_frame,
            right_pill=sample_frame,
            cameras=all_cameras
        )

        assert result.shape == (OUTPUT_HEIGHT, OUTPUT_WIDTH, 3)

    def test_missing_optional_cameras(self, sample_frame, all_cameras):
        """Composition should handle missing optional cameras."""
        result = composite_frame(
            front=sample_frame,
            left_rep=None,
            right_rep=None,
            back=None,
            cameras=all_cameras
        )

        assert result.shape == (OUTPUT_HEIGHT, OUTPUT_WIDTH, 3)

    def test_output_is_bgr(self, sample_frame, front_only_cameras):
        """Output should be 3-channel BGR."""
        result = composite_frame(
            front=sample_frame,
            cameras=front_only_cameras
        )

        assert len(result.shape) == 3
        assert result.shape[2] == 3
        assert result.dtype == np.uint8

    def test_front_left_repeater_layout(self, sample_frame):
        """Composition with front + left repeater should use side-by-side layout."""
        result = composite_frame(
            front=sample_frame,
            left_rep=sample_frame,
            cameras={"front", "left_repeater"}
        )

        assert result.shape == (OUTPUT_HEIGHT, OUTPUT_WIDTH, 3)

    def test_front_right_repeater_layout(self, sample_frame):
        """Composition with front + right repeater should use side-by-side layout."""
        result = composite_frame(
            front=sample_frame,
            right_rep=sample_frame,
            cameras={"front", "right_repeater"}
        )

        assert result.shape == (OUTPUT_HEIGHT, OUTPUT_WIDTH, 3)

    def test_front_left_pillar_layout(self, sample_frame):
        """Composition with front + left pillar should use side-by-side layout."""
        result = composite_frame(
            front=sample_frame,
            left_pill=sample_frame,
            cameras={"front", "left_pillar"}
        )

        assert result.shape == (OUTPUT_HEIGHT, OUTPUT_WIDTH, 3)

    def test_front_right_pillar_layout(self, sample_frame):
        """Composition with front + right pillar should use side-by-side layout."""
        result = composite_frame(
            front=sample_frame,
            right_pill=sample_frame,
            cameras={"front", "right_pillar"}
        )

        assert result.shape == (OUTPUT_HEIGHT, OUTPUT_WIDTH, 3)


class TestColors:
    """Tests for color constants."""

    def test_colors_are_rgb(self):
        """Colors should be in RGB format (3-tuple)."""
        assert len(COLORS.WHITE) == 3
        assert len(COLORS.BLACK) == 3
        assert len(COLORS.GREEN) == 3

    def test_white_is_max(self):
        """White should be (255, 255, 255)."""
        assert COLORS.WHITE == (255, 255, 255)

    def test_black_is_zero(self):
        """Black should be (0, 0, 0)."""
        assert COLORS.BLACK == (0, 0, 0)

    def test_green_is_rgb(self):
        """Green in RGB is (0, 255, 0)."""
        assert COLORS.GREEN == (0, 255, 0)

    def test_red_is_rgb(self):
        """Red in RGB is (255, 0, 0)."""
        assert COLORS.RED == (255, 0, 0)


class TestOutputDimensions:
    """Tests for output dimension constants."""

    def test_output_is_1080p(self):
        """Output should be 1920x1080."""
        assert OUTPUT_WIDTH == 1920
        assert OUTPUT_HEIGHT == 1080

    def test_dashboard_fits_in_output(self):
        """Dashboard should fit within output."""
        assert DASHBOARD_WIDTH < OUTPUT_WIDTH
        assert DASHBOARD_HEIGHT < OUTPUT_HEIGHT

    def test_map_fits_in_output(self):
        """Map should fit within output."""
        assert MAP_SIZE < OUTPUT_WIDTH
        assert MAP_SIZE < OUTPUT_HEIGHT

    def test_opacity_values_valid(self):
        """Opacity values should be between 0 and 1."""
        assert 0 <= OVERLAY_OPACITY <= 1
        assert 0 <= CANVAS_OPACITY <= 1


class TestDashboardRendererScaling:
    """Tests for dashboard scaling feature."""

    def test_scale_1x_matches_default(self, mock_sei_metadata):
        """Scale 1.0 should produce default dimensions."""
        renderer = DashboardRenderer(scale=1.0)
        result = renderer.render(mock_sei_metadata)

        assert result.shape == (DASHBOARD_HEIGHT, DASHBOARD_WIDTH, 3)

    def test_scale_2x_doubles_dimensions(self, mock_sei_metadata):
        """Scale 2.0 should double dimensions."""
        renderer = DashboardRenderer(scale=2.0)
        result = renderer.render(mock_sei_metadata)

        expected_height = int(DASHBOARD_HEIGHT * 2)
        expected_width = int(DASHBOARD_WIDTH * 2)
        assert result.shape == (expected_height, expected_width, 3)

    def test_scale_half_halves_dimensions(self, mock_sei_metadata):
        """Scale 0.5 should halve dimensions."""
        renderer = DashboardRenderer(scale=0.5)
        result = renderer.render(mock_sei_metadata)

        expected_height = int(DASHBOARD_HEIGHT * 0.5)
        expected_width = int(DASHBOARD_WIDTH * 0.5)
        assert result.shape == (expected_height, expected_width, 3)

    def test_width_and_height_properties_scaled(self):
        """Width and height properties should reflect scale."""
        renderer = DashboardRenderer(scale=1.5)

        assert renderer.width == int(DASHBOARD_WIDTH * 1.5)
        assert renderer.height == int(DASHBOARD_HEIGHT * 1.5)


class TestMapRendererScaling:
    """Tests for map scaling feature."""

    def test_scale_1x_matches_default(self, mock_sei_metadata):
        """Scale 1.0 should produce default dimensions."""
        renderer = MapRenderer(scale=1.0)
        result = renderer.render(
            mock_sei_metadata.heading_deg,
            mock_sei_metadata.latitude_deg,
            mock_sei_metadata.longitude_deg
        )

        assert result.shape == (MAP_SIZE, MAP_SIZE, 3)

    def test_scale_2x_doubles_dimensions(self, mock_sei_metadata):
        """Scale 2.0 should double dimensions."""
        renderer = MapRenderer(scale=2.0)
        result = renderer.render(
            mock_sei_metadata.heading_deg,
            mock_sei_metadata.latitude_deg,
            mock_sei_metadata.longitude_deg
        )

        expected_size = int(MAP_SIZE * 2)
        assert result.shape == (expected_size, expected_size, 3)

    def test_size_property_scaled(self):
        """Size property should reflect scale."""
        renderer = MapRenderer(scale=1.5)

        assert renderer.size == int(MAP_SIZE * 1.5)


class TestMapRendererHeadingUp:
    """Tests for heading-up map rotation feature."""

    def test_heading_up_default_true(self):
        """Heading-up should be enabled by default."""
        renderer = MapRenderer()

        assert renderer.heading_up is True

    def test_heading_up_can_be_disabled(self):
        """Heading-up can be disabled (north-up mode)."""
        renderer = MapRenderer(heading_up=False)

        assert renderer.heading_up is False

    def test_render_with_heading_up(self, mock_sei_metadata, sample_gps_history):
        """Map should render with heading-up mode."""
        renderer = MapRenderer(heading_up=True, history=sample_gps_history)
        result = renderer.render(45.0, 37.7749, -122.4194)

        assert result is not None
        assert result.shape == (MAP_SIZE, MAP_SIZE, 3)

    def test_render_with_north_up(self, mock_sei_metadata, sample_gps_history):
        """Map should render with north-up mode."""
        renderer = MapRenderer(heading_up=False, history=sample_gps_history)
        result = renderer.render(45.0, 37.7749, -122.4194)

        assert result is not None
        assert result.shape == (MAP_SIZE, MAP_SIZE, 3)

    def test_different_headings_produce_different_output_in_north_up(self, sample_gps_history):
        """Different headings should produce different arrow orientations in north-up mode."""
        renderer1 = MapRenderer(heading_up=False, history=sample_gps_history)
        renderer2 = MapRenderer(heading_up=False, history=sample_gps_history)

        result1 = renderer1.render(0.0, 37.7749, -122.4194)
        result2 = renderer2.render(90.0, 37.7749, -122.4194)

        # Results should differ due to different arrow orientation
        assert not np.array_equal(result1, result2)


class TestMapRendererMapStyle:
    """Tests for map tile style feature."""

    def test_simple_style_default(self):
        """Simple style should be default."""
        renderer = MapRenderer()

        assert renderer.map_style == "simple"

    def test_street_style_accepted(self):
        """Street style should be accepted."""
        renderer = MapRenderer(map_style="street")

        assert renderer.map_style == "street"

    def test_satellite_style_accepted(self):
        """Satellite style should be accepted."""
        renderer = MapRenderer(map_style="satellite")

        assert renderer.map_style == "satellite"

    def test_simple_style_renders_without_network(self, sample_gps_history):
        """Simple style should render without network."""
        renderer = MapRenderer(map_style="simple", history=sample_gps_history)
        result = renderer.render(45.0, 37.7749, -122.4194)

        assert result is not None
        assert result.shape == (MAP_SIZE, MAP_SIZE, 3)

    def test_tile_cache_initialized(self):
        """Tile grid cache should be initialized."""
        renderer = MapRenderer(map_style="street")

        assert hasattr(renderer, '_tile_grid_cache')
        assert isinstance(renderer._tile_grid_cache, dict)

    @patch('visualization.MapRenderer._fetch_tile')
    def test_fetch_tile_called_for_street_style(self, mock_fetch, sample_gps_history):
        """Fetch tile should be called for street style."""
        # Return tuple (tile, center_lat, center_lon) - None tile triggers fallback
        mock_fetch.return_value = (None, 37.775, -122.419)
        renderer = MapRenderer(map_style="street", history=sample_gps_history)
        renderer.render(45.0, 37.7749, -122.4194)

        mock_fetch.assert_called_once()

    @patch('visualization.MapRenderer._fetch_tile')
    def test_fallback_to_simple_when_tile_unavailable(self, mock_fetch, sample_gps_history):
        """Should fallback to simple mode when tiles unavailable."""
        # Return tuple (tile, center_lat, center_lon) - None tile triggers fallback
        mock_fetch.return_value = (None, 37.775, -122.419)
        renderer = MapRenderer(map_style="street", history=sample_gps_history)
        result = renderer.render(45.0, 37.7749, -122.4194)

        # Should still produce valid output
        assert result is not None
        assert result.shape == (MAP_SIZE, MAP_SIZE, 3)


class TestAcceleratorPedalHandling:
    """Tests for accelerator pedal value range handling."""

    def test_pedal_0_to_1_range(self, mock_sei_metadata):
        """Accelerator in 0-1 range should render correctly."""
        from tests.conftest import MockSeiMetadata
        renderer = DashboardRenderer()
        meta = MockSeiMetadata(accelerator_pedal_position=0.75)  # 0-1 range
        result = renderer.render(meta)

        assert result is not None

    def test_pedal_0_to_100_range(self):
        """Accelerator in 0-100 range should render correctly."""
        from tests.conftest import MockSeiMetadata
        renderer = DashboardRenderer()
        meta = MockSeiMetadata(accelerator_pedal_position=75.0)  # 0-100 range
        result = renderer.render(meta)

        assert result is not None

    def test_brake_applied_renders(self):
        """Brake applied should render the brake indicator."""
        from tests.conftest import MockSeiMetadata
        renderer = DashboardRenderer()
        meta = MockSeiMetadata(brake_applied=True)
        result = renderer.render(meta)

        assert result is not None


class TestGBallIndicator:
    """Tests for G-ball acceleration indicator."""

    def test_gball_renders_with_zero_acceleration(self):
        """G-ball should render with zero acceleration (centered dot)."""
        from tests.conftest import MockSeiMetadata
        renderer = DashboardRenderer()
        meta = MockSeiMetadata(
            linear_acceleration_mps2_x=0.0,
            linear_acceleration_mps2_y=0.0
        )
        result = renderer.render(meta)

        assert result is not None
        assert result.shape[0] == DASHBOARD_HEIGHT
        assert result.shape[1] == DASHBOARD_WIDTH

    def test_gball_renders_with_lateral_acceleration(self):
        """G-ball should render with lateral (turning) acceleration."""
        from tests.conftest import MockSeiMetadata
        renderer = DashboardRenderer()
        meta = MockSeiMetadata(
            linear_acceleration_mps2_x=5.0,  # ~0.5g turning
            linear_acceleration_mps2_y=0.0
        )
        result = renderer.render(meta)

        assert result is not None

    def test_gball_renders_with_longitudinal_acceleration(self):
        """G-ball should render with longitudinal (braking/accel) acceleration."""
        from tests.conftest import MockSeiMetadata
        renderer = DashboardRenderer()
        meta = MockSeiMetadata(
            linear_acceleration_mps2_x=0.0,
            linear_acceleration_mps2_y=-3.0  # ~0.3g braking
        )
        result = renderer.render(meta)

        assert result is not None

    def test_gball_clamps_extreme_values(self):
        """G-ball should clamp extreme acceleration values."""
        from tests.conftest import MockSeiMetadata
        renderer = DashboardRenderer()
        meta = MockSeiMetadata(
            linear_acceleration_mps2_x=50.0,  # Way over 1g
            linear_acceleration_mps2_y=50.0
        )
        result = renderer.render(meta)

        # Should still render without error
        assert result is not None


class TestBlinkerIndicators:
    """Tests for blinker indicators."""

    def test_blinkers_off_renders(self):
        """Dashboard should render with both blinkers off."""
        from tests.conftest import MockSeiMetadata
        renderer = DashboardRenderer()
        meta = MockSeiMetadata(
            blinker_on_left=False,
            blinker_on_right=False
        )
        result = renderer.render(meta)

        assert result is not None

    def test_left_blinker_on_renders(self):
        """Dashboard should render with left blinker on."""
        from tests.conftest import MockSeiMetadata
        renderer = DashboardRenderer()
        meta = MockSeiMetadata(
            blinker_on_left=True,
            blinker_on_right=False
        )
        result = renderer.render(meta)

        assert result is not None

    def test_right_blinker_on_renders(self):
        """Dashboard should render with right blinker on."""
        from tests.conftest import MockSeiMetadata
        renderer = DashboardRenderer()
        meta = MockSeiMetadata(
            blinker_on_left=False,
            blinker_on_right=True
        )
        result = renderer.render(meta)

        assert result is not None

    def test_hazards_on_renders(self):
        """Dashboard should render with hazards (both blinkers) on."""
        from tests.conftest import MockSeiMetadata
        renderer = DashboardRenderer()
        meta = MockSeiMetadata(
            blinker_on_left=True,
            blinker_on_right=True
        )
        result = renderer.render(meta)

        assert result is not None


class TestMapRendererDynamicZoom:
    """Tests for speed-based dynamic zoom feature."""

    def test_zoom_attributes_initialized(self):
        """Zoom attributes should be initialized to None."""
        renderer = MapRenderer()

        assert renderer._current_zoom is None
        assert renderer._current_crop_scale is None

    def test_parking_speed_zoom(self, sample_gps_history):
        """0-5 mph should use parking zoom (crop_scale 3.0)."""
        renderer = MapRenderer(history=sample_gps_history)
        speed_mps = 2.0 / 2.237  # ~2 mph

        zoom, crop_scale = renderer._get_dynamic_zoom(speed_mps)

        assert crop_scale == 3.0  # Ultra-tight for parking

    def test_lot_speed_zoom(self, sample_gps_history):
        """5-10 mph should use lot zoom (crop_scale 2.5)."""
        renderer = MapRenderer(history=sample_gps_history)
        speed_mps = 7.0 / 2.237  # ~7 mph

        zoom, crop_scale = renderer._get_dynamic_zoom(speed_mps)

        assert crop_scale == 2.5  # Parking lot

    def test_residential_speed_zoom(self, sample_gps_history):
        """10-15 mph should use residential zoom (crop_scale 2.2)."""
        renderer = MapRenderer(history=sample_gps_history)
        speed_mps = 12.0 / 2.237  # ~12 mph

        zoom, crop_scale = renderer._get_dynamic_zoom(speed_mps)

        assert crop_scale == 2.2  # Residential

    def test_city_crawl_speed_zoom(self, sample_gps_history):
        """15-20 mph should use city crawl zoom (crop_scale 1.7)."""
        renderer = MapRenderer(history=sample_gps_history)
        speed_mps = 17.0 / 2.237  # ~17 mph

        zoom, crop_scale = renderer._get_dynamic_zoom(speed_mps)

        assert crop_scale == 1.7  # City crawl

    def test_city_slow_speed_zoom(self, sample_gps_history):
        """20-25 mph should use city slow zoom (crop_scale 1.5)."""
        renderer = MapRenderer(history=sample_gps_history)
        speed_mps = 22.0 / 2.237  # ~22 mph

        zoom, crop_scale = renderer._get_dynamic_zoom(speed_mps)

        assert crop_scale == 1.5  # City slow

    def test_city_moderate_speed_zoom(self, sample_gps_history):
        """25-30 mph should use city moderate zoom (crop_scale 1.3)."""
        renderer = MapRenderer(history=sample_gps_history)
        speed_mps = 27.0 / 2.237  # ~27 mph

        zoom, crop_scale = renderer._get_dynamic_zoom(speed_mps)

        assert crop_scale == 1.3  # City moderate

    def test_city_speed_zoom(self, sample_gps_history):
        """30-45 mph should use city zoom (crop_scale 1.1)."""
        renderer = MapRenderer(history=sample_gps_history)
        speed_mps = 35.0 / 2.237  # ~35 mph

        zoom, crop_scale = renderer._get_dynamic_zoom(speed_mps)

        assert crop_scale == 1.1  # City

    def test_suburban_speed_zoom(self, sample_gps_history):
        """45-60 mph should use suburban zoom (crop_scale 0.8)."""
        renderer = MapRenderer(history=sample_gps_history)
        speed_mps = 50.0 / 2.237  # ~50 mph

        zoom, crop_scale = renderer._get_dynamic_zoom(speed_mps)

        assert crop_scale == 0.8  # Suburban

    def test_highway_speed_zoom(self, sample_gps_history):
        """60-75 mph should use highway zoom (crop_scale 0.55)."""
        renderer = MapRenderer(history=sample_gps_history)
        speed_mps = 65.0 / 2.237  # ~65 mph

        zoom, crop_scale = renderer._get_dynamic_zoom(speed_mps)

        assert crop_scale == 0.55  # Highway

    def test_fast_speed_zoom(self, sample_gps_history):
        """75+ mph should use fast zoom (crop_scale 0.4)."""
        renderer = MapRenderer(history=sample_gps_history)
        speed_mps = 80.0 / 2.237  # ~80 mph

        zoom, crop_scale = renderer._get_dynamic_zoom(speed_mps)

        assert crop_scale == 0.4  # Fast highway

    def test_zoom_smoothing(self, sample_gps_history):
        """Zoom should smooth transitions between speeds."""
        renderer = MapRenderer(history=sample_gps_history)

        # First call sets initial values
        renderer._get_dynamic_zoom(10.0 / 2.237)  # ~10 mph -> 2.2 crop_scale
        initial_crop_scale = renderer._current_crop_scale

        # Second call at different speed should smooth
        renderer._get_dynamic_zoom(50.0 / 2.237)  # ~50 mph -> 0.8 target
        smoothed_crop_scale = renderer._current_crop_scale

        # Should have moved toward 0.8 but not all the way
        assert smoothed_crop_scale < initial_crop_scale
        assert smoothed_crop_scale > 0.8


class TestMapRendererCompositeRadius:
    """Tests for fixed composite radius optimization."""

    def test_composite_radius_is_fixed(self):
        """Composite radius should be fixed at 8."""
        renderer = MapRenderer()

        assert renderer._composite_radius == 8

    def test_composite_radius_unchanged_by_speed(self, sample_gps_history):
        """Composite radius should not change based on speed."""
        renderer = MapRenderer(history=sample_gps_history)
        initial_radius = renderer._composite_radius

        # Call dynamic zoom at various speeds
        renderer._get_dynamic_zoom(5.0 / 2.237)   # Parking
        renderer._get_dynamic_zoom(50.0 / 2.237)  # Suburban
        renderer._get_dynamic_zoom(80.0 / 2.237)  # Fast highway

        # Radius should remain unchanged
        assert renderer._composite_radius == initial_radius
        assert renderer._composite_radius == 8


class TestMapRendererPathDrawing:
    """Tests for path drawing with early-exit optimization."""

    def test_path_renders_with_many_points(self):
        """Path should render correctly with many GPS points."""
        # Generate a long path (500 points)
        history = [(37.7749 + i * 0.0001, -122.4194 + i * 0.0001, 10.0)
                   for i in range(500)]
        renderer = MapRenderer(history=history)

        result = renderer.render(45.0, 37.7749 + 0.05, -122.4194 + 0.05)

        assert result is not None
        assert result.shape == (MAP_SIZE, MAP_SIZE, 3)

    def test_path_extends_to_visible_area(self, sample_gps_history):
        """Path should extend to fill visible map area."""
        renderer = MapRenderer(history=sample_gps_history)

        # Render should complete without error
        result = renderer.render(45.0, 37.7749, -122.4194)

        assert result is not None

    def test_empty_path_renders(self):
        """Map should render with empty path."""
        renderer = MapRenderer()

        result = renderer.render(45.0, 37.7749, -122.4194)

        assert result is not None
        assert result.shape == (MAP_SIZE, MAP_SIZE, 3)

    def test_single_point_path_renders(self):
        """Map should render with single point path."""
        renderer = MapRenderer(history=[(37.7749, -122.4194)])

        result = renderer.render(45.0, 37.7749, -122.4194)

        assert result is not None


class TestMapRendererCropRotate:
    """Tests for crop-then-rotate optimization."""

    def test_render_with_speed_parameter(self, sample_gps_history):
        """Render should accept speed_mps parameter for zoom."""
        renderer = MapRenderer(history=sample_gps_history)

        # Test with various speeds
        for speed_mph in [5, 25, 50, 75]:
            speed_mps = speed_mph / 2.237
            result = renderer.render(45.0, 37.7749, -122.4194, speed_mps=speed_mps)

            assert result is not None
            assert result.shape == (MAP_SIZE, MAP_SIZE, 3)

    def test_render_at_different_headings(self, sample_gps_history):
        """Render should work at all heading angles."""
        renderer = MapRenderer(history=sample_gps_history)

        for heading in [0, 45, 90, 135, 180, 225, 270, 315]:
            result = renderer.render(heading, 37.7749, -122.4194)

            assert result is not None
            assert result.shape == (MAP_SIZE, MAP_SIZE, 3)

    def test_heading_smoothing(self, sample_gps_history):
        """Heading should be smoothed between frames."""
        renderer = MapRenderer(history=sample_gps_history)

        # First call sets initial heading
        renderer.render(0.0, 37.7749, -122.4194)
        initial_smoothed = renderer._smoothed_heading

        # Large heading change should be smoothed
        renderer.render(90.0, 37.7749, -122.4194)
        smoothed = renderer._smoothed_heading

        # Should have moved toward 90 but not jumped all the way
        assert smoothed > initial_smoothed
        assert smoothed < 90.0
