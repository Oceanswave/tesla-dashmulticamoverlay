"""
FFmpeg-based video I/O operations.

Provides context managers for reading and writing video frames via FFmpeg
subprocess pipes, eliminating OpenCV dependency and intermediate file issues.

Supports hardware-accelerated encoding when available (VideoToolbox on macOS,
NVENC on NVIDIA GPUs) with automatic fallback to software encoding.
"""

import subprocess
import logging
import platform
import numpy as np
from typing import Dict, Optional, Tuple, List

logger = logging.getLogger(__name__)


# Cache for detected hardware encoder
_hw_encoder_cache: Optional[str] = None
_hw_encoder_checked: bool = False


def detect_hw_encoder() -> Optional[str]:
    """
    Detect available hardware encoder for H.264.

    Checks for platform-specific hardware encoders in order of preference:
    - macOS: h264_videotoolbox (Apple VideoToolbox)
    - NVIDIA: h264_nvenc (NVIDIA NVENC)
    - AMD/Intel: h264_vaapi (VA-API on Linux)

    Returns:
        Encoder name if available, None if only software encoding available
    """
    global _hw_encoder_cache, _hw_encoder_checked

    if _hw_encoder_checked:
        return _hw_encoder_cache

    _hw_encoder_checked = True

    # Platform-specific encoder priority
    system = platform.system()
    if system == "Darwin":
        candidates = ["h264_videotoolbox"]
    elif system == "Linux":
        candidates = ["h264_nvenc", "h264_vaapi"]
    elif system == "Windows":
        candidates = ["h264_nvenc", "h264_qsv"]
    else:
        candidates = []

    # Test each encoder
    for encoder in candidates:
        try:
            cmd = [
                "ffmpeg", "-v", "error",
                "-f", "lavfi", "-i", "nullsrc=s=64x64:d=1",
                "-c:v", encoder,
                "-f", "null", "-"
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=5
            )
            if result.returncode == 0:
                logger.info(f"Hardware encoder detected: {encoder}")
                _hw_encoder_cache = encoder
                return encoder
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            continue

    logger.debug("No hardware encoder available, using libx264")
    return None


def get_video_info(path: str) -> Tuple[int, int, float, int]:
    """
    Get video metadata using ffprobe.

    Args:
        path: Path to video file

    Returns:
        Tuple of (width, height, fps, frame_count)
    """
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_frames",
        "-of", "csv=p=0",
        path
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        parts = result.stdout.strip().split(',')
        width = int(parts[0])
        height = int(parts[1])
        # Parse frame rate (e.g., "30000/1001" or "30")
        fps_parts = parts[2].split('/')
        fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else float(fps_parts[0])
        # nb_frames may be N/A for some containers
        try:
            frame_count = int(parts[3]) if len(parts) > 3 and parts[3] != 'N/A' else 0
        except (ValueError, IndexError):
            frame_count = 0
        return width, height, fps, frame_count
    except subprocess.CalledProcessError as e:
        logger.error(f"ffprobe failed for {path}: {e.stderr}")
        raise RuntimeError(f"Failed to probe video: {path}")


class FFmpegReader:
    """
    Context manager for reading video frames via FFmpeg pipe.

    Decodes video to raw RGB24 frames and provides them as numpy arrays.
    """

    def __init__(self, path: str):
        self.path = path
        self.process: Optional[subprocess.Popen] = None
        self.width = 0
        self.height = 0
        self.fps = 0.0
        self.frame_count = 0
        self._frame_size = 0

    def __enter__(self) -> 'FFmpegReader':
        # Get video info first
        self.width, self.height, self.fps, self.frame_count = get_video_info(self.path)
        self._frame_size = self.width * self.height * 3

        # Start FFmpeg process
        cmd = [
            "ffmpeg", "-v", "error",
            "-i", self.path,
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-"
        ]
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=self._frame_size * 2
        )
        return self

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read the next frame.

        Returns:
            Tuple of (success, frame) where frame is RGB numpy array or None
        """
        if self.process is None or self.process.stdout is None:
            return False, None

        raw = self.process.stdout.read(self._frame_size)
        if len(raw) != self._frame_size:
            return False, None

        frame = np.frombuffer(raw, dtype=np.uint8).reshape(self.height, self.width, 3)
        return True, frame

    def get(self, prop: str) -> float:
        """Get video property (for compatibility)."""
        if prop == 'fps':
            return self.fps
        elif prop == 'width':
            return float(self.width)
        elif prop == 'height':
            return float(self.height)
        return 0.0

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.process:
            try:
                if self.process.stdout:
                    self.process.stdout.close()
                if self.process.stderr:
                    self.process.stderr.close()
                self.process.terminate()
                self.process.wait(timeout=5)
            except Exception as e:
                logger.warning(f"Error closing FFmpeg reader for {self.path}: {e}")
                try:
                    self.process.kill()
                except Exception:
                    pass
        return False


class VideoCaptures:
    """
    Context manager for multiple video readers.

    Opens FFmpegReader objects for the provided camera paths and ensures all are
    properly released on exit, even if an exception occurs.
    """

    def __init__(self, camera_paths: Dict[str, str]):
        if 'front' not in camera_paths:
            raise ValueError("'front' camera path is required")
        self.camera_paths = camera_paths
        self.readers: Dict[str, FFmpegReader] = {}
        self._contexts: list = []

    def __enter__(self) -> Dict[str, FFmpegReader]:
        for key, path in self.camera_paths.items():
            try:
                reader = FFmpegReader(path)
                ctx = reader.__enter__()
                self._contexts.append(reader)
                self.readers[key] = ctx
            except Exception as e:
                if key == 'front':
                    # Clean up and raise
                    for r in self._contexts:
                        r.__exit__(None, None, None)
                    raise RuntimeError(f"Failed to open front camera: {path}") from e
                else:
                    logger.warning(f"Failed to open {key} camera: {path} - {e}")
        return self.readers

    def __exit__(self, exc_type, exc_val, exc_tb):
        for reader in self._contexts:
            try:
                reader.__exit__(exc_type, exc_val, exc_tb)
            except Exception as e:
                logger.warning(f"Error closing reader: {e}")
        self._contexts.clear()
        self.readers.clear()
        return False


class FFmpegWriter:
    """
    Context manager for writing video frames via FFmpeg pipe.

    Accepts RGB numpy arrays and encodes to H.264.

    Supports hardware-accelerated encoding when available:
    - macOS: VideoToolbox (h264_videotoolbox)
    - NVIDIA: NVENC (h264_nvenc)
    - Fallback: libx264 (software)

    Args:
        path: Output file path
        fps: Frame rate
        size: Video dimensions (width, height)
        use_hw_encoding: Try hardware encoding (default True, falls back to software)
    """

    def __init__(self, path: str, fps: float, size: Tuple[int, int],
                 use_hw_encoding: bool = True):
        self.path = path
        self.fps = fps
        self.width, self.height = size
        self.use_hw_encoding = use_hw_encoding
        self.process: Optional[subprocess.Popen] = None
        self._frame_count = 0
        self._encoder_used: str = "libx264"

    def _build_encoder_args(self) -> List[str]:
        """Build encoder-specific FFmpeg arguments."""
        hw_encoder = detect_hw_encoder() if self.use_hw_encoding else None

        if hw_encoder == "h264_videotoolbox":
            self._encoder_used = hw_encoder
            return [
                "-c:v", "h264_videotoolbox",
                "-b:v", "10M",  # Bitrate for VideoToolbox
                "-pix_fmt", "yuv420p",
            ]
        elif hw_encoder == "h264_nvenc":
            self._encoder_used = hw_encoder
            return [
                "-c:v", "h264_nvenc",
                "-preset", "fast",
                "-b:v", "10M",
                "-pix_fmt", "yuv420p",
            ]
        elif hw_encoder == "h264_vaapi":
            self._encoder_used = hw_encoder
            return [
                "-c:v", "h264_vaapi",
                "-b:v", "10M",
                "-pix_fmt", "vaapi_vld",
            ]
        elif hw_encoder == "h264_qsv":
            self._encoder_used = hw_encoder
            return [
                "-c:v", "h264_qsv",
                "-preset", "fast",
                "-b:v", "10M",
                "-pix_fmt", "yuv420p",
            ]
        else:
            # Software encoding fallback
            self._encoder_used = "libx264"
            return [
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-pix_fmt", "yuv420p",
            ]

    def __enter__(self) -> 'FFmpegWriter':
        encoder_args = self._build_encoder_args()

        cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{self.width}x{self.height}",
            "-r", str(self.fps),
            "-i", "-",
        ] + encoder_args + [
            "-movflags", "+faststart",
            self.path
        ]
        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=self.width * self.height * 3 * 2
        )
        logger.debug(f"Opened FFmpegWriter: {self.path} (encoder: {self._encoder_used})")
        return self

    @property
    def encoder(self) -> str:
        """Return the encoder being used (e.g., 'libx264', 'h264_videotoolbox')."""
        return self._encoder_used

    def write(self, frame: np.ndarray) -> None:
        """Write a frame (RGB numpy array)."""
        if self.process is None or self.process.stdin is None:
            raise RuntimeError("Writer not initialized")

        if frame.shape[:2] != (self.height, self.width):
            raise ValueError(f"Frame size mismatch: expected {self.width}x{self.height}, got {frame.shape[1]}x{frame.shape[0]}")

        self.process.stdin.write(frame.tobytes())
        self._frame_count += 1

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.process:
            try:
                if self.process.stdin:
                    self.process.stdin.close()
                self.process.wait(timeout=30)
                if self.process.returncode != 0:
                    stderr = self.process.stderr.read() if self.process.stderr else b''
                    logger.error(f"FFmpeg writer failed: {stderr.decode()}")
            except subprocess.TimeoutExpired:
                logger.warning(f"FFmpeg writer timeout, killing process")
                self.process.kill()
            except Exception as e:
                logger.warning(f"Error closing FFmpegWriter '{self.path}': {e}")
            logger.debug(f"Released FFmpegWriter: {self.path} ({self._frame_count} frames)")
        return False


class VideoWriterContext:
    """
    Compatibility wrapper for FFmpegWriter.

    Matches the old OpenCV-based interface for easier migration.
    """

    def __init__(self, path: str, fourcc: int, fps: float, size: Tuple[int, int]):
        # fourcc is ignored - we always use H.264
        self.path = path
        self.fps = fps
        self.size = size
        self._writer: Optional[FFmpegWriter] = None

    def __enter__(self) -> 'VideoWriterContext':
        self._writer = FFmpegWriter(self.path, self.fps, self.size)
        self._writer.__enter__()
        return self

    def write(self, frame: np.ndarray) -> None:
        if self._writer:
            self._writer.write(frame)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._writer:
            self._writer.__exit__(exc_type, exc_val, exc_tb)
        return False
