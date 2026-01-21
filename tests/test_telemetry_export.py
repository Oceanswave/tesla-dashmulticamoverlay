"""
Tests for telemetry export module.

Tests data models, GPX writer, FIT writer, and exporter orchestration.
"""

import pytest
import io
import json
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from telemetry_export.data_models import (
    GearState,
    AutopilotState,
    TelemetryRecord,
    TelemetryTrack,
)
from telemetry_export.gpx_writer import write_gpx, _format_time, _gear_name, _autopilot_name
from telemetry_export.exporter import TelemetryExporter, parse_tesla_timestamp


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_timestamp():
    """Sample timestamp for testing."""
    return datetime(2026, 1, 9, 11, 45, 38, tzinfo=timezone.utc)


@pytest.fixture
def sample_telemetry_record(sample_timestamp):
    """Sample telemetry record with typical driving values."""
    return TelemetryRecord(
        timestamp=sample_timestamp,
        frame_seq_no=1234,
        latitude_deg=37.7749,
        longitude_deg=-122.4194,
        heading_deg=45.0,
        vehicle_speed_mps=20.0,
        linear_acceleration_x=0.5,
        linear_acceleration_y=-0.2,
        linear_acceleration_z=9.81,
        steering_wheel_angle=-5.0,
        accelerator_pedal_position=0.35,
        brake_applied=False,
        blinker_on_left=False,
        blinker_on_right=True,
        gear_state=GearState.DRIVE,
        autopilot_state=AutopilotState.AUTOSTEER,
    )


@pytest.fixture
def sample_track(sample_timestamp):
    """Sample track with multiple records for testing."""
    records = []
    base_lat, base_lon = 37.7749, -122.4194

    for i in range(10):
        records.append(TelemetryRecord(
            timestamp=sample_timestamp + timedelta(seconds=i),
            frame_seq_no=i,
            latitude_deg=base_lat + (i * 0.0001),
            longitude_deg=base_lon + (i * 0.0001),
            heading_deg=45.0 + i,
            vehicle_speed_mps=20.0 + i,
            linear_acceleration_x=0.1 * i,
            linear_acceleration_y=-0.05 * i,
            linear_acceleration_z=9.81,
            steering_wheel_angle=-5.0 + i,
            accelerator_pedal_position=0.3 + (i * 0.02),
            brake_applied=(i == 5),
            blinker_on_left=(i % 3 == 0),
            blinker_on_right=(i % 3 == 1),
            gear_state=GearState.DRIVE,
            autopilot_state=AutopilotState.AUTOSTEER if i > 3 else AutopilotState.NONE,
        ))

    return TelemetryTrack(
        name="2026-01-09_11-45-38",
        source_clips=["2026-01-09_11-45-38-front.mp4"],
        records=records,
    )


@pytest.fixture
def mock_sei_metadata():
    """Mock SEI metadata matching protobuf structure."""
    mock = MagicMock()
    mock.frame_seq_no = 1234
    mock.latitude_deg = 37.7749
    mock.longitude_deg = -122.4194
    mock.heading_deg = 45.0
    mock.vehicle_speed_mps = 20.0
    mock.linear_acceleration_mps2_x = 0.5
    mock.linear_acceleration_mps2_y = -0.2
    mock.linear_acceleration_mps2_z = 9.81
    mock.steering_wheel_angle = -5.0
    mock.accelerator_pedal_position = 0.35
    mock.brake_applied = False
    mock.blinker_on_left = False
    mock.blinker_on_right = True
    mock.gear_state = 1  # DRIVE
    mock.autopilot_state = 2  # AUTOSTEER
    return mock


# ============================================================================
# Data Models Tests
# ============================================================================

class TestGearState:
    """Tests for GearState enum."""

    def test_gear_values(self):
        """Gear states should have correct integer values."""
        assert GearState.PARK == 0
        assert GearState.DRIVE == 1
        assert GearState.REVERSE == 2
        assert GearState.NEUTRAL == 3

    def test_gear_names(self):
        """Gear states should have correct names."""
        assert GearState.PARK.name == "PARK"
        assert GearState.DRIVE.name == "DRIVE"
        assert GearState.REVERSE.name == "REVERSE"
        assert GearState.NEUTRAL.name == "NEUTRAL"


class TestAutopilotState:
    """Tests for AutopilotState enum."""

    def test_autopilot_values(self):
        """Autopilot states should have correct integer values."""
        assert AutopilotState.NONE == 0
        assert AutopilotState.SELF_DRIVING == 1
        assert AutopilotState.AUTOSTEER == 2
        assert AutopilotState.TACC == 3


class TestTelemetryRecord:
    """Tests for TelemetryRecord model."""

    def test_create_record(self, sample_telemetry_record):
        """Should create record with all fields."""
        assert sample_telemetry_record.latitude_deg == 37.7749
        assert sample_telemetry_record.vehicle_speed_mps == 20.0
        assert sample_telemetry_record.gear_state == GearState.DRIVE

    def test_speed_mph_conversion(self, sample_telemetry_record):
        """Should correctly convert m/s to mph."""
        expected_mph = 20.0 * 2.23694
        assert abs(sample_telemetry_record.speed_mph - expected_mph) < 0.01

    def test_speed_kmh_conversion(self, sample_telemetry_record):
        """Should correctly convert m/s to km/h."""
        expected_kmh = 20.0 * 3.6
        assert abs(sample_telemetry_record.speed_kmh - expected_kmh) < 0.01

    def test_lateral_g_conversion(self, sample_telemetry_record):
        """Should correctly convert lateral acceleration to G-force."""
        expected_g = 0.5 / 9.81
        assert abs(sample_telemetry_record.lateral_g - expected_g) < 0.001

    def test_longitudinal_g_conversion(self, sample_telemetry_record):
        """Should correctly convert longitudinal acceleration to G-force."""
        expected_g = -0.2 / 9.81
        assert abs(sample_telemetry_record.longitudinal_g - expected_g) < 0.001

    def test_from_sei_metadata(self, mock_sei_metadata, sample_timestamp):
        """Should create record from SEI metadata."""
        record = TelemetryRecord.from_sei_metadata(mock_sei_metadata, sample_timestamp)

        assert record.timestamp == sample_timestamp
        assert record.frame_seq_no == 1234
        assert record.latitude_deg == 37.7749
        assert record.vehicle_speed_mps == 20.0
        assert record.gear_state == GearState.DRIVE
        assert record.autopilot_state == AutopilotState.AUTOSTEER


class TestTelemetryTrack:
    """Tests for TelemetryTrack model."""

    def test_create_empty_track(self):
        """Should create empty track."""
        track = TelemetryTrack(name="test", source_clips=[])
        assert track.name == "test"
        assert len(track.records) == 0

    def test_duration_seconds(self, sample_track):
        """Should calculate correct duration."""
        # 10 records, 1 second apart = 9 seconds duration
        assert sample_track.duration_seconds == 9.0

    def test_duration_empty_track(self):
        """Empty track should have zero duration."""
        track = TelemetryTrack(name="test", source_clips=[])
        assert track.duration_seconds == 0.0

    def test_distance_meters(self, sample_track):
        """Should calculate approximate distance."""
        # With increasing speeds and 1-second intervals
        assert sample_track.distance_meters > 0

    def test_start_time(self, sample_track, sample_timestamp):
        """Should return first record timestamp."""
        assert sample_track.start_time == sample_timestamp

    def test_end_time(self, sample_track, sample_timestamp):
        """Should return last record timestamp."""
        expected_end = sample_timestamp + timedelta(seconds=9)
        assert sample_track.end_time == expected_end


# ============================================================================
# GPX Writer Tests
# ============================================================================

class TestGpxHelpers:
    """Tests for GPX helper functions."""

    def test_format_time(self, sample_timestamp):
        """Should format datetime as ISO 8601 UTC."""
        result = _format_time(sample_timestamp)
        assert result == "2026-01-09T11:45:38.000Z"

    def test_format_time_naive_datetime(self):
        """Should handle naive datetime by assuming UTC."""
        dt = datetime(2026, 1, 9, 11, 45, 38)
        result = _format_time(dt)
        assert result.endswith("Z")

    def test_gear_name(self):
        """Should convert gear enum to readable name."""
        assert _gear_name(GearState.PARK) == "PARK"
        assert _gear_name(GearState.DRIVE) == "DRIVE"
        assert _gear_name(GearState.REVERSE) == "REVERSE"
        assert _gear_name(GearState.NEUTRAL) == "NEUTRAL"

    def test_autopilot_name(self):
        """Should convert autopilot enum to readable name."""
        assert _autopilot_name(AutopilotState.NONE) == "MANUAL"
        assert _autopilot_name(AutopilotState.SELF_DRIVING) == "FSD"
        assert _autopilot_name(AutopilotState.AUTOSTEER) == "AUTOSTEER"
        assert _autopilot_name(AutopilotState.TACC) == "TACC"


class TestGpxWriter:
    """Tests for GPX file generation."""

    def test_write_gpx_creates_valid_xml(self, sample_track):
        """Should create valid XML document."""
        output = io.StringIO()
        write_gpx(sample_track, output)

        output.seek(0)
        content = output.read()

        # Should start with XML declaration
        assert content.startswith('<?xml version="1.0"')

        # Should be parseable XML
        root = ET.fromstring(content.split('\n', 1)[1])  # Skip declaration
        assert root.tag.endswith("gpx")

    def test_write_gpx_contains_trackpoints(self, sample_track):
        """Should contain trackpoints for each valid record."""
        output = io.StringIO()
        write_gpx(sample_track, output)

        output.seek(0)
        content = output.read()

        # Count trackpoints (trkpt elements)
        assert content.count("<trkpt") == len(sample_track.records)

    def test_write_gpx_has_namespaces(self, sample_track):
        """Should include required namespaces."""
        output = io.StringIO()
        write_gpx(sample_track, output)

        output.seek(0)
        content = output.read()

        assert "xmlns:gpxtpx" in content
        assert "xmlns:tesla" in content
        assert "http://www.topografix.com/GPX/1/1" in content

    def test_write_gpx_includes_tesla_extensions(self, sample_track):
        """Should include Tesla extension data."""
        output = io.StringIO()
        write_gpx(sample_track, output)

        output.seek(0)
        content = output.read()

        # Check for Tesla extension elements
        assert "TeslaExtension" in content
        assert "steering_angle" in content
        assert "accelerator" in content
        assert "gear" in content

    def test_write_gpx_skips_null_island(self, sample_timestamp):
        """Should skip records at (0, 0) coordinates."""
        track = TelemetryTrack(
            name="test",
            source_clips=["test.mp4"],
            records=[
                TelemetryRecord(
                    timestamp=sample_timestamp,
                    frame_seq_no=0,
                    latitude_deg=0.0,
                    longitude_deg=0.0,
                    heading_deg=0.0,
                    vehicle_speed_mps=0.0,
                    linear_acceleration_x=0.0,
                    linear_acceleration_y=0.0,
                    linear_acceleration_z=9.81,
                    steering_wheel_angle=0.0,
                    accelerator_pedal_position=0.0,
                    brake_applied=False,
                    blinker_on_left=False,
                    blinker_on_right=False,
                    gear_state=GearState.PARK,
                    autopilot_state=AutopilotState.NONE,
                )
            ]
        )

        output = io.StringIO()
        write_gpx(track, output)

        output.seek(0)
        content = output.read()

        # Should have no trackpoints
        assert "<trkpt" not in content


# ============================================================================
# Exporter Tests
# ============================================================================

class TestParseTeslaTimestamp:
    """Tests for Tesla timestamp parsing."""

    def test_valid_timestamp(self):
        """Should parse valid Tesla timestamp format."""
        result = parse_tesla_timestamp("2026-01-09_11-45-38")
        assert result is not None
        assert result.year == 2026
        assert result.month == 1
        assert result.day == 9
        assert result.hour == 11
        assert result.minute == 45
        assert result.second == 38
        assert result.tzinfo == timezone.utc

    def test_invalid_timestamp(self):
        """Should return None for invalid format."""
        assert parse_tesla_timestamp("invalid") is None
        assert parse_tesla_timestamp("") is None
        assert parse_tesla_timestamp("2026-01-09") is None


class TestTelemetryExporter:
    """Tests for TelemetryExporter class."""

    def test_init_with_clips(self):
        """Should initialize with clip list."""
        mock_clip = MagicMock()
        mock_clip.timestamp_prefix = "2026-01-09_11-45-38"
        mock_clip.front = "/path/to/front.mp4"

        exporter = TelemetryExporter([mock_clip])
        assert len(exporter.clips) == 1
        assert exporter.sample_rate == 1.0

    def test_init_with_sample_rate(self):
        """Should initialize with custom sample rate."""
        exporter = TelemetryExporter([], sample_rate=0.1)
        assert exporter.sample_rate == 0.1

    @patch('sei_parser.extract_sei_data')
    def test_extract_all_creates_track(self, mock_extract):
        """Should create track from clips."""
        # Setup mock
        mock_sei = MagicMock()
        mock_sei.frame_seq_no = 0
        mock_sei.latitude_deg = 37.7749
        mock_sei.longitude_deg = -122.4194
        mock_sei.heading_deg = 45.0
        mock_sei.vehicle_speed_mps = 20.0
        mock_sei.linear_acceleration_mps2_x = 0.0
        mock_sei.linear_acceleration_mps2_y = 0.0
        mock_sei.linear_acceleration_mps2_z = 9.81
        mock_sei.steering_wheel_angle = 0.0
        mock_sei.accelerator_pedal_position = 0.3
        mock_sei.brake_applied = False
        mock_sei.blinker_on_left = False
        mock_sei.blinker_on_right = False
        mock_sei.gear_state = 1
        mock_sei.autopilot_state = 0

        mock_extract.return_value = {0: mock_sei, 1: mock_sei}

        mock_clip = MagicMock()
        mock_clip.timestamp_prefix = "2026-01-09_11-45-38"
        mock_clip.front = "/path/to/front.mp4"

        exporter = TelemetryExporter([mock_clip])
        track = exporter.extract_all()

        assert track.name == "2026-01-09_11-45-38"
        assert len(track.records) == 2

    def test_export_gpx_writes_file(self, sample_track, tmp_path):
        """Should write GPX file."""
        exporter = TelemetryExporter([])
        exporter._track = sample_track

        output_path = tmp_path / "test.gpx"
        result = exporter.export_gpx(output_path)

        assert os.path.exists(result)
        with open(result) as f:
            content = f.read()
            assert "<?xml" in content
            assert "gpx" in content

    def test_export_json_writes_file(self, sample_track, tmp_path):
        """Should write JSON file with all fields."""
        exporter = TelemetryExporter([])
        exporter._track = sample_track

        output_path = tmp_path / "test.json"
        result = exporter.export_json(output_path)

        assert os.path.exists(result)
        with open(result) as f:
            data = json.load(f)
            assert data["name"] == sample_track.name
            assert len(data["records"]) == len(sample_track.records)
            # Check all fields present
            record = data["records"][0]
            assert "latitude_deg" in record
            assert "gear_state" in record
            assert "autopilot_state" in record
            assert "steering_wheel_angle" in record


# ============================================================================
# FIT Writer Tests (optional, only if fit-tool installed)
# ============================================================================

class TestFitWriter:
    """Tests for FIT file generation (requires fit-tool)."""

    @pytest.fixture
    def fit_available(self):
        """Check if fit-tool is available."""
        try:
            import fit_tool
            return True
        except ImportError:
            return False

    def test_import_fit_writer(self):
        """Should be able to import fit_writer module."""
        from telemetry_export.fit_writer import write_fit

    def test_write_fit_creates_binary(self, sample_track, tmp_path, fit_available):
        """Should create binary FIT file."""
        if not fit_available:
            pytest.skip("fit-tool not installed")

        from telemetry_export.fit_writer import write_fit

        output_path = tmp_path / "test.fit"
        with open(output_path, 'wb') as f:
            write_fit(sample_track, f)

        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0

        # FIT files start with header size byte (typically 14)
        with open(output_path, 'rb') as f:
            header = f.read(2)
            assert len(header) == 2

    def test_write_fit_empty_track_no_crash(self, tmp_path):
        """Should handle empty track gracefully."""
        from telemetry_export.fit_writer import write_fit

        track = TelemetryTrack(name="empty", source_clips=[], records=[])

        output_path = tmp_path / "empty.fit"
        with open(output_path, 'wb') as f:
            write_fit(track, f)

        # File should be empty or minimal (no records to write)
        assert os.path.exists(output_path)
