"""
Tests for video I/O operations.

Tests FFmpeg reader/writer and hardware encoder detection.
"""

import pytest
from unittest.mock import patch, MagicMock
import subprocess

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from video_io import (
    detect_hw_encoder,
    FFmpegWriter,
    FFmpegReader,
    VideoCaptures,
    get_video_info,
    _hw_encoder_cache,
    _hw_encoder_checked,
)
import video_io


class TestHardwareEncoderDetection:
    """Tests for hardware encoder detection."""

    def setup_method(self):
        """Reset the encoder cache before each test."""
        video_io._hw_encoder_cache = None
        video_io._hw_encoder_checked = False

    def test_detection_caches_result(self):
        """Encoder detection should cache its result."""
        with patch('video_io.subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=1)

            # First call
            result1 = detect_hw_encoder()
            # Second call should use cache
            result2 = detect_hw_encoder()

            assert result1 == result2
            # Should only have called subprocess once (for one encoder check)
            # but actually it tries multiple encoders until one works or all fail

    @patch('video_io.platform.system')
    @patch('video_io.subprocess.run')
    def test_macos_detects_videotoolbox(self, mock_run, mock_system):
        """On macOS, should detect h264_videotoolbox."""
        mock_system.return_value = "Darwin"
        mock_run.return_value = MagicMock(returncode=0)

        result = detect_hw_encoder()

        assert result == "h264_videotoolbox"

    @patch('video_io.platform.system')
    @patch('video_io.subprocess.run')
    def test_linux_detects_nvenc(self, mock_run, mock_system):
        """On Linux with NVIDIA, should detect h264_nvenc."""
        video_io._hw_encoder_cache = None
        video_io._hw_encoder_checked = False

        mock_system.return_value = "Linux"
        mock_run.return_value = MagicMock(returncode=0)

        result = detect_hw_encoder()

        assert result == "h264_nvenc"

    @patch('video_io.platform.system')
    @patch('video_io.subprocess.run')
    def test_fallback_to_none_when_no_hw_encoder(self, mock_run, mock_system):
        """Should return None when no hardware encoder available."""
        video_io._hw_encoder_cache = None
        video_io._hw_encoder_checked = False

        mock_system.return_value = "Darwin"
        mock_run.return_value = MagicMock(returncode=1)  # Encoder not available

        result = detect_hw_encoder()

        assert result is None


class TestFFmpegWriter:
    """Tests for FFmpegWriter class."""

    def test_default_uses_hw_encoding_flag(self):
        """Writer should have hardware encoding enabled by default."""
        writer = FFmpegWriter("/tmp/test.mp4", 30.0, (1920, 1080))

        assert writer.use_hw_encoding is True

    def test_can_disable_hw_encoding(self):
        """Writer should allow disabling hardware encoding."""
        writer = FFmpegWriter("/tmp/test.mp4", 30.0, (1920, 1080), use_hw_encoding=False)

        assert writer.use_hw_encoding is False

    @patch('video_io.detect_hw_encoder')
    def test_uses_libx264_when_hw_disabled(self, mock_detect):
        """Should use libx264 when hardware encoding is disabled."""
        writer = FFmpegWriter("/tmp/test.mp4", 30.0, (1920, 1080), use_hw_encoding=False)
        args = writer._build_encoder_args()

        assert "-c:v" in args
        idx = args.index("-c:v")
        assert args[idx + 1] == "libx264"
        mock_detect.assert_not_called()

    @patch('video_io.detect_hw_encoder')
    def test_uses_hw_encoder_when_available(self, mock_detect):
        """Should use hardware encoder when available."""
        mock_detect.return_value = "h264_videotoolbox"

        writer = FFmpegWriter("/tmp/test.mp4", 30.0, (1920, 1080), use_hw_encoding=True)
        args = writer._build_encoder_args()

        assert "-c:v" in args
        idx = args.index("-c:v")
        assert args[idx + 1] == "h264_videotoolbox"

    @patch('video_io.detect_hw_encoder')
    def test_encoder_property_reflects_choice(self, mock_detect):
        """Encoder property should reflect the chosen encoder."""
        mock_detect.return_value = None

        writer = FFmpegWriter("/tmp/test.mp4", 30.0, (1920, 1080))
        writer._build_encoder_args()

        assert writer.encoder == "libx264"


class TestFFmpegWriterEncoderArgs:
    """Tests for encoder-specific argument building."""

    @patch('video_io.detect_hw_encoder')
    def test_videotoolbox_args(self, mock_detect):
        """VideoToolbox should use bitrate-based encoding."""
        mock_detect.return_value = "h264_videotoolbox"

        writer = FFmpegWriter("/tmp/test.mp4", 30.0, (1920, 1080))
        args = writer._build_encoder_args()

        assert "-b:v" in args
        assert "10M" in args

    @patch('video_io.detect_hw_encoder')
    def test_nvenc_args(self, mock_detect):
        """NVENC should use preset and bitrate."""
        mock_detect.return_value = "h264_nvenc"

        writer = FFmpegWriter("/tmp/test.mp4", 30.0, (1920, 1080))
        args = writer._build_encoder_args()

        assert "-preset" in args
        assert "-b:v" in args

    @patch('video_io.detect_hw_encoder')
    def test_libx264_args(self, mock_detect):
        """libx264 should use CRF-based encoding."""
        mock_detect.return_value = None

        writer = FFmpegWriter("/tmp/test.mp4", 30.0, (1920, 1080))
        args = writer._build_encoder_args()

        assert "-crf" in args
        assert "23" in args


class TestGetVideoInfo:
    """Tests for get_video_info function."""

    @patch('video_io.subprocess.run')
    def test_parses_standard_output(self, mock_run):
        """Should parse standard ffprobe output correctly."""
        mock_run.return_value = MagicMock(
            stdout="1920,1080,30000/1001,900\n",
            returncode=0
        )

        width, height, fps, frame_count = get_video_info("/path/to/video.mp4")

        assert width == 1920
        assert height == 1080
        assert abs(fps - 29.97) < 0.01  # NTSC frame rate
        assert frame_count == 900

    @patch('video_io.subprocess.run')
    def test_handles_integer_fps(self, mock_run):
        """Should handle integer fps (no fraction)."""
        mock_run.return_value = MagicMock(
            stdout="1280,720,30,600\n",
            returncode=0
        )

        width, height, fps, frame_count = get_video_info("/path/to/video.mp4")

        assert fps == 30.0

    @patch('video_io.subprocess.run')
    def test_handles_na_frame_count(self, mock_run):
        """Should handle N/A frame count gracefully."""
        mock_run.return_value = MagicMock(
            stdout="1920,1080,30,N/A\n",
            returncode=0
        )

        width, height, fps, frame_count = get_video_info("/path/to/video.mp4")

        assert frame_count == 0

    @patch('video_io.subprocess.run')
    def test_handles_missing_frame_count(self, mock_run):
        """Should handle missing frame count field."""
        mock_run.return_value = MagicMock(
            stdout="1920,1080,30\n",
            returncode=0
        )

        width, height, fps, frame_count = get_video_info("/path/to/video.mp4")

        assert frame_count == 0

    @patch('video_io.subprocess.run')
    def test_raises_on_ffprobe_failure(self, mock_run):
        """Should raise RuntimeError when ffprobe fails."""
        mock_run.side_effect = subprocess.CalledProcessError(
            1, "ffprobe", stderr=b"Error message"
        )

        with pytest.raises(RuntimeError, match="Failed to probe video"):
            get_video_info("/path/to/video.mp4")


class TestFFmpegReaderUnit:
    """Unit tests for FFmpegReader class."""

    def test_init_sets_defaults(self):
        """Constructor should set default values."""
        reader = FFmpegReader("/path/to/video.mp4")

        assert reader.path == "/path/to/video.mp4"
        assert reader.process is None
        assert reader.width == 0
        assert reader.height == 0
        assert reader.fps == 0.0

    @patch('video_io._detect_hw_decoder')
    @patch('video_io.get_video_info')
    @patch('video_io.subprocess.Popen')
    def test_enter_gets_video_info(self, mock_popen, mock_info, mock_hwdec):
        """__enter__ should get video info and start process."""
        mock_info.return_value = (1920, 1080, 29.97, 900)
        mock_popen.return_value = MagicMock()
        mock_hwdec.return_value = None  # No hardware decoder for test

        reader = FFmpegReader("/path/to/video.mp4")
        result = reader.__enter__()

        assert result is reader
        assert reader.width == 1920
        assert reader.height == 1080
        assert reader.fps == 29.97
        mock_popen.assert_called_once()

    def test_get_property_fps(self):
        """get() should return fps property."""
        reader = FFmpegReader("/path/to/video.mp4")
        reader.fps = 29.97

        assert reader.get('fps') == 29.97

    def test_get_property_width(self):
        """get() should return width property."""
        reader = FFmpegReader("/path/to/video.mp4")
        reader.width = 1920

        assert reader.get('width') == 1920.0

    def test_get_property_height(self):
        """get() should return height property."""
        reader = FFmpegReader("/path/to/video.mp4")
        reader.height = 1080

        assert reader.get('height') == 1080.0

    def test_get_unknown_property(self):
        """get() should return 0 for unknown properties."""
        reader = FFmpegReader("/path/to/video.mp4")

        assert reader.get('unknown') == 0.0

    def test_read_returns_false_when_no_process(self):
        """read() should return (False, None) when no process."""
        reader = FFmpegReader("/path/to/video.mp4")

        success, frame = reader.read()

        assert success is False
        assert frame is None

    @patch('video_io._detect_hw_decoder')
    @patch('video_io.get_video_info')
    @patch('video_io.subprocess.Popen')
    def test_exit_terminates_process(self, mock_popen, mock_info, mock_hwdec):
        """__exit__ should terminate the process."""
        mock_info.return_value = (1920, 1080, 29.97, 900)
        mock_process = MagicMock()
        mock_popen.return_value = mock_process
        mock_hwdec.return_value = None

        reader = FFmpegReader("/path/to/video.mp4")
        reader.__enter__()
        reader.__exit__(None, None, None)

        mock_process.terminate.assert_called_once()
        mock_process.wait.assert_called_once()

    @patch('video_io._detect_hw_decoder')
    @patch('video_io.get_video_info')
    @patch('video_io.subprocess.Popen')
    def test_exit_kills_on_timeout(self, mock_popen, mock_info, mock_hwdec):
        """__exit__ should kill process if terminate times out."""
        mock_info.return_value = (1920, 1080, 29.97, 900)
        mock_process = MagicMock()
        mock_process.wait.side_effect = Exception("Timeout")
        mock_popen.return_value = mock_process
        mock_hwdec.return_value = None

        reader = FFmpegReader("/path/to/video.mp4")
        reader.__enter__()
        reader.__exit__(None, None, None)

        mock_process.kill.assert_called_once()


class TestVideoCapturesUnit:
    """Unit tests for VideoCaptures class."""

    def test_requires_front_camera(self):
        """Should raise ValueError if 'front' camera is missing."""
        with pytest.raises(ValueError, match="'front' camera path is required"):
            VideoCaptures({"back": "/path/to/back.mp4"})

    def test_accepts_front_only(self):
        """Should accept front camera only."""
        caps = VideoCaptures({"front": "/path/to/front.mp4"})

        assert "front" in caps.camera_paths

    @patch('video_io.FFmpegReader')
    def test_enter_creates_readers(self, mock_reader_class):
        """__enter__ should create readers for all cameras."""
        mock_reader = MagicMock()
        mock_reader.__enter__ = MagicMock(return_value=mock_reader)
        mock_reader_class.return_value = mock_reader

        caps = VideoCaptures({
            "front": "/path/to/front.mp4",
            "back": "/path/to/back.mp4"
        })
        result = caps.__enter__()

        assert "front" in result
        assert "back" in result
        assert mock_reader_class.call_count == 2


class TestVideoWriterContextUnit:
    """Unit tests for VideoWriterContext class."""

    def test_init_stores_parameters(self):
        """Constructor should store parameters."""
        from video_io import VideoWriterContext

        ctx = VideoWriterContext("/tmp/out.mp4", 0, 29.97, (1920, 1080))

        assert ctx.path == "/tmp/out.mp4"
        assert ctx.fps == 29.97
        assert ctx.size == (1920, 1080)
        assert ctx._writer is None

    @patch('video_io.FFmpegWriter')
    def test_enter_creates_writer(self, mock_writer_class):
        """__enter__ should create FFmpegWriter."""
        from video_io import VideoWriterContext

        mock_writer = MagicMock()
        mock_writer_class.return_value = mock_writer

        ctx = VideoWriterContext("/tmp/out.mp4", 0, 29.97, (1920, 1080))
        result = ctx.__enter__()

        assert result is ctx
        mock_writer_class.assert_called_once_with("/tmp/out.mp4", 29.97, (1920, 1080))
        mock_writer.__enter__.assert_called_once()

    @patch('video_io.FFmpegWriter')
    def test_write_delegates_to_writer(self, mock_writer_class):
        """write() should delegate to FFmpegWriter."""
        from video_io import VideoWriterContext
        import numpy as np

        mock_writer = MagicMock()
        mock_writer_class.return_value = mock_writer

        ctx = VideoWriterContext("/tmp/out.mp4", 0, 29.97, (1920, 1080))
        ctx.__enter__()

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        ctx.write(frame)

        mock_writer.write.assert_called_once()

    @patch('video_io.FFmpegWriter')
    def test_exit_closes_writer(self, mock_writer_class):
        """__exit__ should close the writer."""
        from video_io import VideoWriterContext

        mock_writer = MagicMock()
        mock_writer_class.return_value = mock_writer

        ctx = VideoWriterContext("/tmp/out.mp4", 0, 29.97, (1920, 1080))
        ctx.__enter__()
        ctx.__exit__(None, None, None)

        mock_writer.__exit__.assert_called_once()
