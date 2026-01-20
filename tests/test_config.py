"""
Tests for VideoConfig validation and camera parsing.

Tests configuration validation, camera set parsing, and ClipSet discovery.
"""

import pytest
from unittest.mock import patch, MagicMock
import tempfile
import os

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import (
    parse_cameras,
    ALL_CAMERAS,
    ClipSet,
    VideoConfig,
    discover_clips,
    find_siblings,
    build_camera_paths,
    extract_telemetry,
    CAMERA_MAPPING,
)


class TestParseCameras:
    """Tests for camera argument parsing."""

    def test_none_returns_all_cameras(self):
        """None input should return all cameras."""
        result = parse_cameras(None)
        assert result == ALL_CAMERAS

    def test_front_only(self):
        """Front-only should work."""
        result = parse_cameras("front")
        assert result == {"front"}

    def test_multiple_cameras(self):
        """Comma-separated cameras should be parsed."""
        result = parse_cameras("front,back,left_repeater")
        assert result == {"front", "back", "left_repeater"}

    def test_whitespace_handling(self):
        """Whitespace around camera names should be stripped."""
        result = parse_cameras(" front , back ")
        assert result == {"front", "back"}

    def test_case_insensitive(self):
        """Camera names should be case-insensitive."""
        result = parse_cameras("FRONT,Back,LEFT_REPEATER")
        assert result == {"front", "back", "left_repeater"}

    def test_missing_front_raises(self):
        """Missing front camera should raise ValueError."""
        with pytest.raises(ValueError, match="front.*required"):
            parse_cameras("back,left_repeater")

    def test_invalid_camera_raises(self):
        """Invalid camera name should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid camera"):
            parse_cameras("front,invalid_camera")

    def test_all_valid_cameras(self):
        """All valid camera names should be accepted."""
        all_names = ",".join(ALL_CAMERAS)
        result = parse_cameras(all_names)
        assert result == ALL_CAMERAS


class TestClipSet:
    """Tests for ClipSet dataclass."""

    def test_minimal_clipset(self):
        """ClipSet with only front camera."""
        clip = ClipSet(
            timestamp_prefix="2024-01-01_12-00-00",
            front="/path/to/front.mp4"
        )
        assert clip.front == "/path/to/front.mp4"
        assert clip.back is None
        assert clip.left_rep is None

    def test_full_clipset(self):
        """ClipSet with all cameras."""
        clip = ClipSet(
            timestamp_prefix="2024-01-01_12-00-00",
            front="/path/to/front.mp4",
            back="/path/to/back.mp4",
            left_rep="/path/to/left_repeater.mp4",
            right_rep="/path/to/right_repeater.mp4",
            left_pill="/path/to/left_pillar.mp4",
            right_pill="/path/to/right_pillar.mp4",
        )
        assert clip.front == "/path/to/front.mp4"
        assert clip.back == "/path/to/back.mp4"
        assert clip.left_rep == "/path/to/left_repeater.mp4"


class TestVideoConfig:
    """Tests for VideoConfig pydantic model."""

    def test_valid_config(self):
        """Valid configuration should be accepted."""
        clip = ClipSet(timestamp_prefix="test", front="/path/front.mp4")
        config = VideoConfig(
            playlist=[clip],
            output_file="output.mp4",
            overlay_scale=1.0,
            cameras={"front"}
        )
        assert config.overlay_scale == 1.0
        assert len(config.playlist) == 1

    def test_overlay_scale_minimum(self):
        """Overlay scale below minimum should raise."""
        clip = ClipSet(timestamp_prefix="test", front="/path/front.mp4")
        with pytest.raises(ValueError):
            VideoConfig(
                playlist=[clip],
                output_file="output.mp4",
                overlay_scale=0.05,  # Below 0.1 minimum
                cameras={"front"}
            )

    def test_default_cameras(self):
        """Default cameras should be all cameras."""
        clip = ClipSet(timestamp_prefix="test", front="/path/front.mp4")
        config = VideoConfig(
            playlist=[clip],
            output_file="output.mp4"
        )
        assert config.cameras == ALL_CAMERAS


class TestBuildCameraPaths:
    """Tests for camera path building."""

    def test_front_only(self):
        """Front-only camera set."""
        clip = ClipSet(
            timestamp_prefix="test",
            front="/path/front.mp4",
            back="/path/back.mp4"
        )
        paths = build_camera_paths(clip, {"front"})
        assert paths == {"front": "/path/front.mp4"}
        assert "back" not in paths

    def test_all_available_cameras(self):
        """All available cameras should be included."""
        clip = ClipSet(
            timestamp_prefix="test",
            front="/path/front.mp4",
            back="/path/back.mp4",
            left_rep="/path/left_rep.mp4"
        )
        cameras = {"front", "back", "left_repeater"}
        paths = build_camera_paths(clip, cameras)
        assert "front" in paths
        assert "back" in paths
        assert "left_rep" in paths

    def test_missing_file_excluded(self):
        """Cameras without files should be excluded."""
        clip = ClipSet(
            timestamp_prefix="test",
            front="/path/front.mp4",
            back=None  # No back camera file
        )
        cameras = {"front", "back"}
        paths = build_camera_paths(clip, cameras)
        assert "front" in paths
        assert "back" not in paths


class TestDiscoverClips:
    """Tests for clip discovery."""

    def test_nonexistent_path_raises(self):
        """Non-existent path should raise ValueError."""
        with pytest.raises(ValueError, match="not found"):
            discover_clips("/nonexistent/path")

    def test_single_file_discovery(self):
        """Single front file should discover siblings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create front file
            front_path = os.path.join(tmpdir, "2024-01-01_12-00-00-front.mp4")
            back_path = os.path.join(tmpdir, "2024-01-01_12-00-00-back.mp4")

            open(front_path, 'w').close()
            open(back_path, 'w').close()

            clips = discover_clips(front_path)
            assert len(clips) == 1
            assert clips[0].front == front_path
            assert clips[0].back == back_path

    def test_directory_discovery(self):
        """Directory should discover all front files and siblings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create two clip sets
            for i in range(2):
                prefix = f"2024-01-0{i+1}_12-00-00"
                open(os.path.join(tmpdir, f"{prefix}-front.mp4"), 'w').close()
                open(os.path.join(tmpdir, f"{prefix}-back.mp4"), 'w').close()

            clips = discover_clips(tmpdir)
            assert len(clips) == 2


class TestFindSiblings:
    """Tests for sibling file discovery."""

    def test_find_all_siblings(self):
        """All sibling files should be found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prefix = "2024-01-01_12-00-00"
            front_path = os.path.join(tmpdir, f"{prefix}-front.mp4")

            # Create all camera files
            suffixes = ["front", "back", "left_repeater", "right_repeater",
                       "left_pillar", "right_pillar"]
            for suffix in suffixes:
                open(os.path.join(tmpdir, f"{prefix}-{suffix}.mp4"), 'w').close()

            clip = find_siblings(tmpdir, prefix, front_path)

            assert clip.front == front_path
            assert clip.back is not None
            assert clip.left_rep is not None
            assert clip.right_rep is not None
            assert clip.left_pill is not None
            assert clip.right_pill is not None

    def test_missing_siblings_are_none(self):
        """Missing sibling files should be None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prefix = "2024-01-01_12-00-00"
            front_path = os.path.join(tmpdir, f"{prefix}-front.mp4")
            open(front_path, 'w').close()

            clip = find_siblings(tmpdir, prefix, front_path)

            assert clip.front == front_path
            assert clip.back is None
            assert clip.left_rep is None


class TestExtractTelemetry:
    """Tests for telemetry extraction."""

    def test_returns_tuple(self):
        """Extract telemetry should return tuple of (sei_data, gps_points, frame_count)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a minimal test file
            front_path = os.path.join(tmpdir, "test-front.mp4")
            open(front_path, 'w').close()

            clip = ClipSet(timestamp_prefix="test", front=front_path)

            # Mock extract_sei_data and get_video_frame_count
            with patch('main.extract_sei_data', return_value={}), \
                 patch('main.get_video_frame_count', return_value=100):
                sei_data, gps_points, frame_count = extract_telemetry(clip)

            assert isinstance(sei_data, dict)
            assert isinstance(gps_points, list)
            assert isinstance(frame_count, int)
            assert frame_count == 100

    def test_extracts_gps_from_sei(self):
        """GPS points should be extracted from SEI data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            front_path = os.path.join(tmpdir, "test-front.mp4")
            open(front_path, 'w').close()

            clip = ClipSet(timestamp_prefix="test", front=front_path)

            # Create mock SEI data with GPS coordinates
            mock_meta = MagicMock()
            mock_meta.latitude_deg = 37.7749
            mock_meta.longitude_deg = -122.4194

            mock_sei = {0: mock_meta, 1: mock_meta}

            with patch('main.extract_sei_data', return_value=mock_sei), \
                 patch('main.get_video_frame_count', return_value=1800):
                sei_data, gps_points, frame_count = extract_telemetry(clip)

            assert len(sei_data) == 2
            assert len(gps_points) == 2
            assert gps_points[0] == (37.7749, -122.4194)
            assert frame_count == 1800


class TestGetVideoFrameCount:
    """Tests for get_video_frame_count function."""

    @patch('main.subprocess.run')
    def test_returns_frame_count_from_ffprobe(self, mock_run):
        """Should return frame count from ffprobe output."""
        from main import get_video_frame_count

        mock_run.return_value = MagicMock(
            stdout="900\n",
            returncode=0
        )

        count = get_video_frame_count("/path/to/video.mp4")

        assert count == 900

    @patch('main.subprocess.run')
    def test_handles_na_frame_count(self, mock_run):
        """Should fallback to duration when nb_frames is N/A."""
        from main import get_video_frame_count

        # First call returns N/A, second call returns duration
        mock_run.side_effect = [
            MagicMock(stdout="N/A\n", returncode=0),
            MagicMock(stdout="60.0\n", returncode=0)  # 60 seconds
        ]

        count = get_video_frame_count("/path/to/video.mp4")

        # 60 seconds * 29.97 fps ≈ 1798
        assert count > 0
        assert abs(count - 1798) < 5

    @patch('main.subprocess.run')
    def test_returns_zero_on_failure(self, mock_run):
        """Should return 0 when both methods fail."""
        from main import get_video_frame_count
        import subprocess

        mock_run.side_effect = subprocess.CalledProcessError(1, "ffprobe")

        count = get_video_frame_count("/path/to/video.mp4")

        assert count == 0

    @patch('main.subprocess.run')
    def test_handles_empty_output(self, mock_run):
        """Should handle empty ffprobe output."""
        from main import get_video_frame_count

        # First call returns empty, second call returns duration
        mock_run.side_effect = [
            MagicMock(stdout="\n", returncode=0),
            MagicMock(stdout="30.0\n", returncode=0)
        ]

        count = get_video_frame_count("/path/to/video.mp4")

        assert count > 0


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_setup_logging_no_error(self):
        """setup_logging should not raise errors."""
        from main import setup_logging

        # Should not raise
        setup_logging(verbose=False)

    def test_setup_logging_verbose_no_error(self):
        """setup_logging with verbose should not raise errors."""
        from main import setup_logging

        # Should not raise
        setup_logging(verbose=True)


class TestConcatClips:
    """Tests for concat_clips function."""

    @patch('video_io.detect_hw_encoder')
    @patch('main.subprocess.Popen')
    @patch('main.subprocess.run')
    @patch('main.get_video_frame_count')
    @patch('main.create_concat_progress')
    def test_calls_ffmpeg_with_correct_args(self, mock_progress, mock_frame_count, mock_run, mock_popen, mock_hw_encoder):
        """concat_clips should call ffmpeg with correct arguments."""
        from main import concat_clips

        # Mock frame count
        mock_frame_count.return_value = 100

        # Mock hw encoder detection
        mock_hw_encoder.return_value = None

        # Mock subprocess.run (stream copy attempt) - fail to trigger fallback
        mock_run.return_value = MagicMock(returncode=1)

        # Mock progress context manager
        mock_progress_instance = MagicMock()
        mock_progress_instance.__enter__ = MagicMock(return_value=mock_progress_instance)
        mock_progress_instance.__exit__ = MagicMock(return_value=None)
        mock_progress_instance.add_task = MagicMock(return_value=1)
        mock_progress.return_value = mock_progress_instance

        # Mock subprocess Popen - return success for ffmpeg re-encode
        mock_process = MagicMock()
        mock_process.poll.return_value = 0  # Process finished with exit code 0
        mock_process.wait.return_value = 0
        mock_process.returncode = 0  # Explicitly set successful exit code
        mock_process.stderr = iter([])  # Empty iterator for stderr lines
        mock_popen.return_value = mock_process

        concat_clips(["/tmp/clip1.mp4", "/tmp/clip2.mp4"], "/tmp/output.mp4")

        # Verify Popen was called with ffmpeg command (re-encode path)
        mock_popen.assert_called_once()
        call_args = mock_popen.call_args[0][0]
        assert call_args[0] == "ffmpeg"
        assert "/tmp/output.mp4" in call_args

    @patch('video_io.detect_hw_encoder')
    @patch('main.subprocess.Popen')
    @patch('main.subprocess.run')
    @patch('main.get_video_frame_count')
    @patch('main.create_concat_progress')
    def test_handles_single_clip(self, mock_progress, mock_frame_count, mock_run, mock_popen, mock_hw_encoder):
        """concat_clips with single clip should still work."""
        from main import concat_clips

        mock_frame_count.return_value = 100

        # Mock hw encoder detection
        mock_hw_encoder.return_value = None

        # Mock subprocess.run (stream copy attempt) - fail to trigger fallback
        mock_run.return_value = MagicMock(returncode=1)

        # Mock progress context manager
        mock_progress_instance = MagicMock()
        mock_progress_instance.__enter__ = MagicMock(return_value=mock_progress_instance)
        mock_progress_instance.__exit__ = MagicMock(return_value=None)
        mock_progress_instance.add_task = MagicMock(return_value=1)
        mock_progress.return_value = mock_progress_instance

        # Mock subprocess Popen
        mock_process = MagicMock()
        mock_process.poll.return_value = 0  # Process finished
        mock_process.wait.return_value = 0
        mock_process.returncode = 0  # Explicitly set successful exit code
        mock_process.stderr = iter([])  # Empty iterator for stderr lines
        mock_popen.return_value = mock_process

        concat_clips(["/tmp/clip1.mp4"], "/tmp/output.mp4")

        # For single clip, should still call ffmpeg
        mock_popen.assert_called_once()
        call_args = mock_popen.call_args[0][0]
        assert "ffmpeg" in call_args[0]


class TestNullIslandFiltering:
    """Tests for GPS null island filtering."""

    def test_filters_exact_zero(self):
        """Should filter coordinates at exactly (0, 0)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            front_path = os.path.join(tmpdir, "test-front.mp4")
            open(front_path, 'w').close()

            clip = ClipSet(timestamp_prefix="test", front=front_path)

            # Create mock SEI data with null island coordinates
            mock_meta = MagicMock()
            mock_meta.latitude_deg = 0.0
            mock_meta.longitude_deg = 0.0

            mock_sei = {0: mock_meta}

            with patch('main.extract_sei_data', return_value=mock_sei), \
                 patch('main.get_video_frame_count', return_value=100):
                sei_data, gps_points, frame_count = extract_telemetry(clip)

            # Should be filtered out
            assert len(gps_points) == 0

    def test_filters_near_zero(self):
        """Should filter coordinates very close to (0, 0)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            front_path = os.path.join(tmpdir, "test-front.mp4")
            open(front_path, 'w').close()

            clip = ClipSet(timestamp_prefix="test", front=front_path)

            # Create mock SEI data with coordinates just under threshold
            mock_meta = MagicMock()
            mock_meta.latitude_deg = 0.0005  # Under 0.001 threshold
            mock_meta.longitude_deg = 0.0005

            mock_sei = {0: mock_meta}

            with patch('main.extract_sei_data', return_value=mock_sei), \
                 patch('main.get_video_frame_count', return_value=100):
                sei_data, gps_points, frame_count = extract_telemetry(clip)

            # Should be filtered out
            assert len(gps_points) == 0

    def test_keeps_valid_coordinates_near_equator(self):
        """Should keep valid coordinates even if one is near zero."""
        with tempfile.TemporaryDirectory() as tmpdir:
            front_path = os.path.join(tmpdir, "test-front.mp4")
            open(front_path, 'w').close()

            clip = ClipSet(timestamp_prefix="test", front=front_path)

            # Create mock SEI data with valid coordinates (latitude near 0, but longitude far from 0)
            mock_meta = MagicMock()
            mock_meta.latitude_deg = 0.5  # Near equator but valid
            mock_meta.longitude_deg = 100.0  # Far from prime meridian

            mock_sei = {0: mock_meta}

            with patch('main.extract_sei_data', return_value=mock_sei), \
                 patch('main.get_video_frame_count', return_value=100):
                sei_data, gps_points, frame_count = extract_telemetry(clip)

            # Should NOT be filtered - only one coordinate near zero
            assert len(gps_points) == 1
            assert gps_points[0] == (0.5, 100.0)


class TestVideoConfigMapStyle:
    """Tests for VideoConfig map_style validation."""

    def test_default_map_style(self):
        """Default map style should be 'simple'."""
        clip = ClipSet(timestamp_prefix="test", front="/path/front.mp4")
        config = VideoConfig(
            playlist=[clip],
            output_file="output.mp4"
        )
        assert config.map_style == "simple"

    def test_accepts_street_style(self):
        """Should accept 'street' map style."""
        clip = ClipSet(timestamp_prefix="test", front="/path/front.mp4")
        config = VideoConfig(
            playlist=[clip],
            output_file="output.mp4",
            map_style="street"
        )
        assert config.map_style == "street"

    def test_accepts_satellite_style(self):
        """Should accept 'satellite' map style."""
        clip = ClipSet(timestamp_prefix="test", front="/path/front.mp4")
        config = VideoConfig(
            playlist=[clip],
            output_file="output.mp4",
            map_style="satellite"
        )
        assert config.map_style == "satellite"


class TestVideoConfigNorthUp:
    """Tests for VideoConfig north_up option."""

    def test_default_north_up_false(self):
        """Default north_up should be False (heading-up mode)."""
        clip = ClipSet(timestamp_prefix="test", front="/path/front.mp4")
        config = VideoConfig(
            playlist=[clip],
            output_file="output.mp4"
        )
        assert config.north_up is False

    def test_can_set_north_up_true(self):
        """Should be able to set north_up to True."""
        clip = ClipSet(timestamp_prefix="test", front="/path/front.mp4")
        config = VideoConfig(
            playlist=[clip],
            output_file="output.mp4",
            north_up=True
        )
        assert config.north_up is True
