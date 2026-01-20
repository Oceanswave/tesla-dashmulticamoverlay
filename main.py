#!/usr/bin/env python3
import argparse
import sys
import os
import glob
import subprocess
import multiprocessing
import logging
import math
from typing import Optional, List, Tuple, Set, Callable
from dataclasses import dataclass
from pydantic import BaseModel, Field, field_validator

from sei_parser import extract_sei_data
from visualization import DashboardRenderer, MapRenderer, composite_frame, apply_overlay, render_watermark, render_timestamp
from emphasis import EmphasisCalculator
from constants import OUTPUT_WIDTH, OUTPUT_HEIGHT, DASHBOARD_WIDTH, MAP_SIZE, DASHBOARD_Y, MAP_Y, MAP_X_MARGIN, TESLA_DASHCAM_FPS
from color_grading import create_color_grader, ColorGrader
from video_io import VideoCaptures, VideoWriterContext
from rich_console import (
    console,
    setup_rich_logging,
    create_progress,
    create_scan_progress,
    create_concat_progress,
    create_render_progress,
    print_banner,
    print_config_summary,
    print_phase,
    print_completion_summary,
    print_error,
)

logger = logging.getLogger(__name__)

__version__ = "1.1.0"


def build_gps_interpolator(sei_data: dict) -> Callable[[int], Tuple[float, float, float]]:
    """Build a GPS interpolation function for buttery smooth map scrolling.

    Creates GPS anchor points where the position actually changed (>1 meter),
    then returns a function that interpolates between anchors for any frame.

    Args:
        sei_data: Dict mapping frame_idx to SEI metadata

    Returns:
        Function that takes frame_idx and returns (lat, lon, heading)
    """
    # Build list of frames where GPS actually changed (for interpolation anchors)
    anchors = []
    last_lat, last_lon = None, None

    for idx in sorted(sei_data.keys()):
        meta = sei_data[idx]
        # Skip null island
        if abs(meta.latitude_deg) < 0.001 and abs(meta.longitude_deg) < 0.001:
            continue

        # Only use as anchor if GPS position actually changed (>1 meter ≈ 0.00001°)
        if last_lat is not None:
            lat_diff = abs(meta.latitude_deg - last_lat)
            lon_diff = abs(meta.longitude_deg - last_lon)
            if lat_diff < 0.00001 and lon_diff < 0.00001:
                continue

        anchors.append((idx, meta.latitude_deg, meta.longitude_deg, meta.heading_deg))
        last_lat, last_lon = meta.latitude_deg, meta.longitude_deg

    if not anchors:
        # No valid GPS - return function that always returns zeros
        return lambda frame_idx: (0.0, 0.0, 0.0)

    def interpolate(frame_idx: int) -> Tuple[float, float, float]:
        """Interpolate GPS position for given frame index."""
        # Binary search for the anchor pair that brackets this frame
        lo, hi = 0, len(anchors) - 1

        # Handle edge cases
        if frame_idx <= anchors[0][0]:
            return (anchors[0][1], anchors[0][2], anchors[0][3])
        if frame_idx >= anchors[-1][0]:
            return (anchors[-1][1], anchors[-1][2], anchors[-1][3])

        # Find prev anchor (last anchor with index <= frame_idx)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if anchors[mid][0] <= frame_idx:
                lo = mid
            else:
                hi = mid - 1

        prev_anchor = anchors[lo]
        next_anchor = anchors[min(lo + 1, len(anchors) - 1)]

        # If same anchor, no interpolation needed
        if prev_anchor[0] == next_anchor[0]:
            return (prev_anchor[1], prev_anchor[2], prev_anchor[3])

        # Linear interpolation between anchors
        t = (frame_idx - prev_anchor[0]) / (next_anchor[0] - prev_anchor[0])

        lat = prev_anchor[1] + t * (next_anchor[1] - prev_anchor[1])
        lon = prev_anchor[2] + t * (next_anchor[2] - prev_anchor[2])

        # Interpolate heading with wraparound handling
        h1, h2 = prev_anchor[3], next_anchor[3]
        diff = h2 - h1
        if diff > 180:
            diff -= 360
        elif diff < -180:
            diff += 360
        heading = (h1 + t * diff) % 360

        return (lat, lon, heading)

    return interpolate


def setup_logging(verbose: bool = False) -> None:
    """
    Configure logging with Rich handler for beautiful output.

    Args:
        verbose: Enable DEBUG level logging
    """
    setup_rich_logging(verbose=verbose)

# Valid camera names matching Tesla file suffixes
ALL_CAMERAS = {"front", "back", "left_repeater", "right_repeater", "left_pillar", "right_pillar"}


def parse_cameras(camera_arg: Optional[str]) -> Set[str]:
    """Parse comma-separated camera argument into a validated set."""
    if camera_arg is None:
        return ALL_CAMERAS.copy()

    cameras = {c.strip().lower() for c in camera_arg.split(",") if c.strip()}

    # Validate camera names
    invalid = cameras - ALL_CAMERAS
    if invalid:
        raise ValueError(f"Invalid camera name(s): {', '.join(sorted(invalid))}. "
                        f"Valid cameras: {', '.join(sorted(ALL_CAMERAS))}")

    # Front is required for SEI telemetry
    if "front" not in cameras:
        raise ValueError("'front' camera is required (contains SEI telemetry data)")

    return cameras

@dataclass
class ClipSet:
    timestamp_prefix: str
    front: str
    left_rep: Optional[str] = None
    right_rep: Optional[str] = None
    back: Optional[str] = None
    left_pill: Optional[str] = None
    right_pill: Optional[str] = None

class VideoConfig(BaseModel):
    playlist: List[ClipSet]
    output_file: str
    overlay_scale: float = Field(default=1.0, ge=0.1)
    map_style: str = Field(default="simple")
    north_up: bool = Field(default=False)
    cameras: Set[str] = Field(default_factory=lambda: ALL_CAMERAS.copy())
    watermark_path: Optional[str] = Field(default=None)
    show_timestamp: bool = Field(default=False)
    layout: str = Field(default="grid")
    workers: Optional[int] = Field(default=None, ge=1)
    # Color grading options
    color_grade: Optional[str] = Field(default=None)
    brightness: float = Field(default=0.0, ge=-1.0, le=1.0)
    contrast: float = Field(default=0.0, ge=-1.0, le=1.0)
    saturation: float = Field(default=0.0, ge=-1.0, le=1.0)
    gamma: float = Field(default=1.0, ge=0.1, le=3.0)
    shadows: float = Field(default=0.0, ge=-1.0, le=1.0)
    highlights: float = Field(default=0.0, ge=-1.0, le=1.0)
    # Dynamic camera emphasis
    enable_emphasis: bool = Field(default=True)

    class Config:
        arbitrary_types_allowed = True

def discover_clips(input_path: str) -> List[ClipSet]:
    clips = []
    
    if os.path.isfile(input_path):
        front = input_path
        base = os.path.dirname(front)
        name = os.path.basename(front)
        if "-front.mp4" in name:
            prefix = name.split("-front.mp4")[0]
        else:
             prefix = os.path.splitext(name)[0]
        clips.append(find_siblings(base, prefix, front))
        
    elif os.path.isdir(input_path):
        front_files = sorted(glob.glob(os.path.join(input_path, "**/*-front.mp4"), recursive=True))
        if not front_files:
             front_files = sorted(glob.glob(os.path.join(input_path, "*-front.mp4")))
             
        for front in front_files:
            base = os.path.dirname(front)
            name = os.path.basename(front)
            prefix = name.split("-front.mp4")[0]
            clips.append(find_siblings(base, prefix, front))
            
    else:
        raise ValueError(f"Input path {input_path} not found.")
        
    return clips

def parse_clip_timestamp(timestamp_prefix: str) -> Optional[str]:
    """Parse Tesla dashcam timestamp prefix to human-readable format.

    Converts '2026-01-09_11-45-38' to '2026-01-09 11:45:38'
    Returns None if parsing fails.
    """
    try:
        # Tesla format: YYYY-MM-DD_HH-MM-SS
        if '_' in timestamp_prefix:
            date_part, time_part = timestamp_prefix.split('_')
            time_formatted = time_part.replace('-', ':')
            return f"{date_part} {time_formatted}"
    except Exception:
        pass
    return None


def find_siblings(base_dir: str, prefix: str, front_path: str) -> ClipSet:
    def get_path(suffix):
        p = os.path.join(base_dir, f"{prefix}-{suffix}.mp4")
        return p if os.path.exists(p) else None

    return ClipSet(
        timestamp_prefix=prefix,
        front=front_path,
        left_rep=get_path("left_repeater"),
        right_rep=get_path("right_repeater"),
        back=get_path("back"),
        left_pill=get_path("left_pillar"),
        right_pill=get_path("right_pillar")
    )


# Mapping from camera set names to ClipSet attributes and internal keys
CAMERA_MAPPING = {
    'front': ('front', 'front'),
    'left_repeater': ('left_rep', 'left_rep'),
    'right_repeater': ('right_rep', 'right_rep'),
    'back': ('back', 'back'),
    'left_pillar': ('left_pill', 'left_pill'),
    'right_pillar': ('right_pill', 'right_pill'),
}


def build_camera_paths(clip: ClipSet, cameras: Set[str]) -> dict:
    """Build a dict of camera keys to file paths for selected cameras with available files."""
    paths = {}
    for camera_name, (clip_attr, key) in CAMERA_MAPPING.items():
        if camera_name not in cameras:
            continue
        file_path = getattr(clip, clip_attr)
        if file_path:
            paths[key] = file_path
    return paths

def parse_args() -> VideoConfig:
    parser = argparse.ArgumentParser(description="Burn Tesla SEI metadata into connected video clips.")
    parser.add_argument("input_path", help="Path to input MP4 file or directory of clips")
    parser.add_argument("output_file", help="Path to output MP4 file")
    parser.add_argument("--overlay-scale", type=float, default=1.0,
                       help="Scale factor for dashboard/map overlays (default: 1.0)")
    parser.add_argument("--map-style", choices=["simple", "street", "satellite"], default="simple",
                       help="Map background style: simple (vector), street (OSM tiles), satellite (aerial)")
    parser.add_argument("--north-up", action="store_true",
                       help="Use north-up map orientation instead of heading-up (default: heading-up)")
    parser.add_argument("--cameras", type=str, default=None,
                       help="Comma-separated cameras to include: front,back,left_repeater,right_repeater,left_pillar,right_pillar (default: all)")
    parser.add_argument("--watermark", type=str, default=None, metavar="FILE",
                       help="Path to watermark image to overlay in lower-right corner")
    parser.add_argument("--timestamp", action="store_true",
                       help="Burn in date/time from dashcam filename in lower-left corner")
    parser.add_argument("--layout", choices=["grid", "pip"], default="grid",
                       help="Multi-camera layout: grid (6-camera grid, default) or pip (fullscreen front with PIP thumbnails)")
    parser.add_argument("--workers", "-j", type=int, default=None,
                       help="Number of parallel workers (default: CPU count)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose (debug) logging")
    # Color grading options
    parser.add_argument("--color-grade", type=str, default=None, metavar="PRESET|PATH",
                       help="Color grading preset name (cinematic, warm, cool, vivid, cybertruck, dramatic, vintage, natural) or path to .cube LUT file")
    parser.add_argument("--brightness", type=float, default=0.0,
                       help="Brightness adjustment (-1.0 to 1.0, default: 0)")
    parser.add_argument("--contrast", type=float, default=0.0,
                       help="Contrast adjustment (-1.0 to 1.0, default: 0)")
    parser.add_argument("--saturation", type=float, default=0.0,
                       help="Saturation adjustment (-1.0 to 1.0, default: 0)")
    parser.add_argument("--gamma", type=float, default=1.0,
                       help="Gamma correction (0.1 to 3.0, default: 1.0)")
    parser.add_argument("--shadows", type=float, default=0.0,
                       help="Shadow adjustment (-1.0 to 1.0, default: 0)")
    parser.add_argument("--highlights", type=float, default=0.0,
                       help="Highlight adjustment (-1.0 to 1.0, default: 0)")
    # Dynamic camera emphasis
    parser.add_argument("--no-emphasis", action="store_true",
                       help="Disable dynamic camera emphasis based on driving context")

    args = parser.parse_args()

    # Configure logging before any other operations
    setup_logging(verbose=args.verbose)

    try:
        cameras = parse_cameras(args.cameras)
        playlist = discover_clips(args.input_path)
        if not playlist:
            print_error("No clips found!", hint="Check the input path contains Tesla dashcam files")
            sys.exit(1)

        # Validate output path before processing
        output_dir = os.path.dirname(args.output_file) or "."
        if not os.path.exists(output_dir):
            print_error(f"Output directory does not exist: {output_dir}")
            sys.exit(1)
        if not os.access(output_dir, os.W_OK):
            print_error(f"No write permission for output directory: {output_dir}")
            sys.exit(1)

        # Validate watermark file if specified
        watermark_path = None
        if args.watermark:
            if not os.path.exists(args.watermark):
                print_error(f"Watermark file not found: {args.watermark}")
                sys.exit(1)
            watermark_path = os.path.abspath(args.watermark)

        logger.debug(f"Found {len(playlist)} clips to process.")
        logger.debug(f"Cameras: {', '.join(sorted(cameras))}")
        logger.debug(f"Overlay scale: {args.overlay_scale}, Map style: {args.map_style}, North-up: {args.north_up}")
        if watermark_path:
            logger.debug(f"Watermark: {watermark_path}")
        if args.timestamp:
            logger.debug("Timestamp burn-in: enabled")
        if args.color_grade:
            logger.debug(f"Color grade: {args.color_grade}")

        return VideoConfig(
            playlist=playlist,
            output_file=args.output_file,
            overlay_scale=args.overlay_scale,
            map_style=args.map_style,
            north_up=args.north_up,
            cameras=cameras,
            watermark_path=watermark_path,
            show_timestamp=args.timestamp,
            layout=args.layout,
            workers=args.workers,
            color_grade=args.color_grade,
            brightness=args.brightness,
            contrast=args.contrast,
            saturation=args.saturation,
            gamma=args.gamma,
            shadows=args.shadows,
            highlights=args.highlights,
            enable_emphasis=not args.no_emphasis
        )
    except Exception as e:
        print_error(str(e))
        sys.exit(1)

# Shared counter for real-time frame progress (initialized in main)
_shared_frame_counter = None


def _init_worker(counter):
    """Initialize worker process with shared counter."""
    global _shared_frame_counter
    _shared_frame_counter = counter


# Worker function must be at module level for multiprocessing
def process_clip_task(clip: ClipSet, output_temp: str, overlay_scale: float, map_style: str, north_up: bool,
                      history: List[Tuple[float, float]], cameras: Set[str], sei_data: dict,
                      watermark_path: Optional[str] = None, show_timestamp: bool = False,
                      layout: str = "grid", color_grader: Optional[ColorGrader] = None,
                      enable_emphasis: bool = True):
    """
    Worker to process a single clip.

    Uses context managers for video I/O to ensure proper cleanup on exceptions.
    Args:
        clip: The clip to process
        output_temp: Output file path
        overlay_scale: Scale factor for dashboard/map overlays
        map_style: Map background style ("simple", "street", "satellite")
        north_up: If True, use north-up map; if False, use heading-up
        history: GPS history points up to this clip
        cameras: Set of camera names to include
        sei_data: Pre-extracted SEI metadata dict (frame_idx -> metadata)
        watermark_path: Optional path to watermark image for lower-right corner
        show_timestamp: If True, burn in timestamp from clip filename
        layout: Multi-camera layout: "grid" (default) or "pip" (fullscreen with thumbnails)
        color_grader: Optional ColorGrader instance for color grading
        enable_emphasis: If True, enable dynamic camera emphasis based on driving context
    """
    global _shared_frame_counter

    # Build camera paths dict for the context manager
    camera_paths = build_camera_paths(clip, cameras)

    # Use context managers for safe resource cleanup
    with VideoCaptures(camera_paths) as caps:
        fps = caps['front'].fps
        # Tesla dashcam standard is 29.97 fps (NTSC: 30000/1001)
        # Using 30.0 causes cumulative drift at clip boundaries
        if fps == 0 or fps > 60:
            fps = 29.97
            logger.debug(f"FPS fallback for {clip.timestamp_prefix}: using {fps}")

        # fourcc is ignored by FFmpegWriter (always uses H.264)
        with VideoWriterContext(output_temp, 0, fps, (OUTPUT_WIDTH, OUTPUT_HEIGHT)) as out:
            # Initialize Renderers with scaling and map options
            dash_renderer = DashboardRenderer(scale=overlay_scale)
            map_renderer = MapRenderer(
                scale=overlay_scale,
                history=list(history),  # Copy history
                map_style=map_style,
                heading_up=not north_up
            )

            # Initialize emphasis calculator for dynamic camera highlighting
            emphasis_calculator = EmphasisCalculator() if enable_emphasis else None

            # Build GPS interpolator for buttery smooth map scrolling
            # This interpolates between GPS keyframes for every video frame
            get_interpolated_gps = build_gps_interpolator(sei_data)

            # Pre-render watermark if specified (render once, reuse for all frames)
            watermark_img = None
            if watermark_path:
                watermark_img = render_watermark(watermark_path, max_size=int(150 * overlay_scale))

            # Parse timestamp from clip filename if enabled (format: 2026-01-09_11-45-38)
            timestamp_str = None
            if show_timestamp:
                timestamp_str = parse_clip_timestamp(clip.timestamp_prefix)

            frame_idx = 0
            overlay_count = 0
            while True:
                frames = {}
                ret, frames['front'] = caps['front'].read()
                if not ret:
                    break

                for key in ['left_rep', 'right_rep', 'back', 'left_pill', 'right_pill']:
                    if key in caps:
                        _, frames[key] = caps[key].read()
                    else:
                        frames[key] = None

                meta = sei_data.get(frame_idx)

                # Compute camera emphasis based on driving context
                emphasis = emphasis_calculator.compute(meta) if emphasis_calculator else None

                canvas = composite_frame(
                    front=frames['front'],
                    left_rep=frames.get('left_rep'),
                    right_rep=frames.get('right_rep'),
                    back=frames.get('back'),
                    left_pill=frames.get('left_pill'),
                    right_pill=frames.get('right_pill'),
                    cameras=cameras,
                    layout=layout,
                    emphasis=emphasis
                )

                # Apply color grading to composited frame (before overlays)
                if color_grader is not None and color_grader.is_active:
                    canvas = color_grader.grade(canvas)

                # Get interpolated GPS for smooth map (works even without meta)
                interp_lat, interp_lon, interp_heading = get_interpolated_gps(frame_idx)

                if meta:
                    overlay_count += 1
                    # Update path history with raw GPS and speed (for gradient coloring)
                    map_renderer.update(meta.latitude_deg, meta.longitude_deg, meta.vehicle_speed_mps)
                    # Render dashboard with actual telemetry (requires meta for speed, pedals, etc.)
                    dash_img = dash_renderer.render(meta)
                    dash_x = (OUTPUT_WIDTH - dash_renderer.width) // 2
                    apply_overlay(canvas, dash_img, dash_x, DASHBOARD_Y)

                # Map overlay renders EVERY FRAME using interpolated GPS for smooth scrolling
                # This is outside the `if meta:` block so it updates every frame
                if interp_lat != 0.0 or interp_lon != 0.0:  # Skip if no GPS data at all
                    # Get current speed for dynamic zoom (use 0 if no metadata)
                    current_speed = meta.vehicle_speed_mps if meta else 0.0
                    map_img = map_renderer.render(interp_heading, interp_lat, interp_lon, current_speed)
                    map_x = OUTPUT_WIDTH - map_renderer.size - MAP_X_MARGIN
                    apply_overlay(canvas, map_img, map_x, MAP_Y)

                # Watermark (Lower Right) - applied every frame, even without SEI data
                if watermark_img is not None:
                    watermark_x = OUTPUT_WIDTH - watermark_img.shape[1] - 20
                    watermark_y = OUTPUT_HEIGHT - watermark_img.shape[0] - 20
                    apply_overlay(canvas, watermark_img, watermark_x, watermark_y)

                # Timestamp (Lower Left) - with running timecode
                if timestamp_str:
                    # Calculate timecode from frame index (HH:MM:SS:FF format)
                    seconds = frame_idx / fps
                    render_timestamp(canvas, timestamp_str, seconds, scale=overlay_scale)

                out.write(canvas)
                frame_idx += 1

                # Update shared counter every 10 frames for responsive progress
                if _shared_frame_counter is not None and frame_idx % 10 == 0:
                    with _shared_frame_counter.get_lock():
                        _shared_frame_counter.value += 10

            # Update any remaining frames not yet counted
            if _shared_frame_counter is not None:
                remainder = frame_idx % 10
                if remainder > 0:
                    with _shared_frame_counter.get_lock():
                        _shared_frame_counter.value += remainder

            # Debug: Show processing summary
            logger.debug(f"Clip {clip.timestamp_prefix}: Processed {frame_idx} frames, overlays applied to {overlay_count} frames")

    # Return tuple with clip info for progress tracking
    return (output_temp, clip.timestamp_prefix, frame_idx)

def get_video_frame_count(file_path: str) -> int:
    """Get frame count from video file using ffprobe (fast, no decode)."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=nb_frames",
                "-of", "csv=p=0",
                file_path
            ],
            capture_output=True,
            text=True,
            check=True
        )
        count = result.stdout.strip()
        if count and count != 'N/A':
            return int(count)
    except (subprocess.CalledProcessError, ValueError):
        pass

    # Fallback: estimate from duration
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=duration",
                "-of", "csv=p=0",
                file_path
            ],
            capture_output=True,
            text=True,
            check=True
        )
        duration = float(result.stdout.strip())
        return int(duration * TESLA_DASHCAM_FPS)  # Tesla dashcam uses NTSC ~29.97fps
    except (subprocess.CalledProcessError, ValueError):
        return 0


def extract_telemetry(clip: ClipSet) -> Tuple[dict, List[Tuple[float, float]], int]:
    """
    Extract SEI telemetry data, GPS points, and frame count from a clip.

    Returns:
        Tuple of (sei_data dict, gps_points list, frame_count int)
    """
    sei_data = extract_sei_data(clip.front)
    gps_points = []

    # Extract GPS points from SEI data
    for idx in sorted(sei_data.keys()):
        meta = sei_data[idx]
        # Skip null island (0,0) GPS errors - must check BOTH are near zero
        # Threshold of 0.001 degrees (~110m) catches initialization errors
        # while not filtering legitimate locations near (0,0) in the Gulf of Guinea
        if abs(meta.latitude_deg) < 0.001 and abs(meta.longitude_deg) < 0.001:
            continue
        gps_points.append((meta.latitude_deg, meta.longitude_deg))

    # Get frame count for progress tracking
    frame_count = get_video_frame_count(clip.front)

    return sei_data, gps_points, frame_count


def concat_clips(temp_files: List[str], output_file: str, fps: float = TESLA_DASHCAM_FPS):
    import re
    import time

    n = len(temp_files)

    # Get total frame count for progress tracking
    total_frames = sum(get_video_frame_count(f) for f in temp_files)
    logger.debug(f"Total frames to concatenate: {total_frames}")

    # Use concat FILTER (not demuxer) to properly reset timestamps
    # The demuxer can drop frames when timestamps don't align
    # The filter decodes, resets timestamps with setpts, and re-encodes

    # Build filter graph: reset timestamps for each input, then concat
    filter_parts = []

    # Reset video timestamps for each input
    for i in range(n):
        filter_parts.append(f"[{i}:v]setpts=PTS-STARTPTS[v{i}]")

    # Concat all video streams
    video_inputs = "".join(f"[v{i}]" for i in range(n))
    filter_parts.append(f"{video_inputs}concat=n={n}:v=1:a=0[outv]")

    filter_complex = ";".join(filter_parts)

    # Build command with all input files
    cmd = ["ffmpeg", "-y", "-fflags", "+genpts+igndts", "-progress", "pipe:2"]
    for temp_file in temp_files:
        cmd.extend(["-i", os.path.abspath(temp_file)])

    cmd.extend([
        "-filter_complex", filter_complex,
        "-map", "[outv]",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-r", str(fps),
        "-vsync", "cfr",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-video_track_timescale", "30000",  # Precise timescale
        "-threads", str(max(1, multiprocessing.cpu_count() // 2)),
        output_file
    ])
    logger.debug(f"FFmpeg command: {' '.join(cmd)}")

    # Patterns for parsing FFmpeg progress output
    frame_pattern = re.compile(r"frame=\s*(\d+)")
    fps_pattern = re.compile(r"fps=\s*([\d.]+)")
    speed_pattern = re.compile(r"speed=\s*([\d.]+)x")

    try:
        process = subprocess.Popen(
            cmd,
            stderr=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            text=True,
            bufsize=1
        )

        stderr_lines = []
        start_time = time.time()
        current_frame = 0
        current_fps = 0.0
        current_speed = 0.0
        has_started = False

        with create_concat_progress() as progress:
            task = progress.add_task(
                "Encoding",
                total=total_frames if total_frames > 0 else None,
                status=f"Loading {n} clips..."
            )

            for line in process.stderr:
                stderr_lines.append(line)

                # Parse frame count
                match = frame_pattern.search(line)
                if match:
                    current_frame = int(match.group(1))
                    has_started = True

                # Parse encoding FPS
                match = fps_pattern.search(line)
                if match:
                    current_fps = float(match.group(1))

                # Parse encoding speed
                match = speed_pattern.search(line)
                if match:
                    current_speed = float(match.group(1))

                # Build status string
                if has_started:
                    parts = []
                    if current_fps > 0:
                        parts.append(f"{current_fps:.0f} fps")
                    if current_speed > 0:
                        parts.append(f"{current_speed:.1f}x")
                    status = " • ".join(parts) if parts else ""
                else:
                    elapsed = time.time() - start_time
                    status = f"Initializing... ({elapsed:.0f}s)"

                progress.update(
                    task,
                    completed=current_frame,
                    status=status
                )

            # Wait for process to complete
            process.wait()

            # Mark complete with final status
            elapsed = time.time() - start_time
            if total_frames > 0:
                avg_fps = total_frames / elapsed if elapsed > 0 else 0
                progress.update(
                    task,
                    completed=total_frames,
                    status=f"Done in {elapsed:.1f}s ({avg_fps:.0f} fps avg)"
                )

        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd, stderr="".join(stderr_lines))

    except subprocess.CalledProcessError as e:
        print_error("FFmpeg concatenation failed", hint="Make sure ffmpeg is installed with libx264 support")
        logger.debug(f"Command: {' '.join(cmd)}")
        stderr_content = e.stderr if isinstance(e.stderr, str) else "".join(stderr_lines)
        if stderr_content:
            stderr_lines = stderr_content.strip().split('\n')
            relevant_lines = stderr_lines[-20:] if len(stderr_lines) > 20 else stderr_lines
            for line in relevant_lines:
                logger.debug(f"  {line}")
        sys.exit(1)
    except FileNotFoundError:
        print_error("ffmpeg not found", hint="Please install ffmpeg and ensure it's in your PATH")
        sys.exit(1)
            
import tempfile

# Wrapper for imap since it only accepts one argument
def process_clip_wrapper(args):
    return process_clip_task(*args)

def main():
    config = parse_args()

    if not config.playlist:
        return

    # Print startup banner and configuration
    print_banner(__version__)
    print_config_summary(
        clip_count=len(config.playlist),
        cameras=config.cameras,
        output_file=config.output_file,
        overlay_scale=config.overlay_scale,
        map_style=config.map_style,
        north_up=config.north_up,
        layout=config.layout,
        color_grade=config.color_grade,
        brightness=config.brightness,
        contrast=config.contrast,
        saturation=config.saturation,
        gamma=config.gamma,
        shadows=config.shadows,
        highlights=config.highlights,
    )

    # 1. Extract telemetry from all clips (SEI data + GPS points + frame counts)
    print_phase(1, 3, "Extracting telemetry")
    global_history = []
    clip_histories = []  # GPS history snapshot for each clip START
    clip_sei_data = []   # Cached SEI data for each clip
    clip_frame_counts = []  # Frame counts for progress tracking

    with create_scan_progress() as progress:
        task = progress.add_task("Extracting", total=len(config.playlist))
        for clip in config.playlist:
            clip_histories.append(list(global_history))
            sei_data, gps_points, frame_count = extract_telemetry(clip)
            clip_sei_data.append(sei_data)
            clip_frame_counts.append(frame_count)
            global_history.extend(gps_points)
            progress.advance(task)

    total_frames = sum(clip_frame_counts)
    logger.debug(f"Total GPS points: {len(global_history)}")
    logger.debug(f"Total frames to render: {total_frames:,}")
    logger.debug(f"Clips with telemetry: {sum(1 for s in clip_sei_data if s)}/{len(clip_sei_data)}")

    # 2. Parallel Processing (rendering only - SEI already extracted)
    num_processes = config.workers if config.workers else multiprocessing.cpu_count()
    print_phase(2, 3, f"Rendering frames ([highlight]{num_processes}[/] workers)")

    # Create color grader (shared across all workers)
    color_grader = create_color_grader(
        color_grade=config.color_grade,
        brightness=config.brightness,
        contrast=config.contrast,
        saturation=config.saturation,
        gamma=config.gamma,
        shadows=config.shadows,
        highlights=config.highlights
    )

    # Use a temporary directory for intermediate files
    with tempfile.TemporaryDirectory() as temp_dir:
        logger.debug(f"Using temporary directory: {temp_dir}")

        # Prepare args for all tasks (SEI data already extracted)
        args_list = []
        temp_files = []
        for i, clip in enumerate(config.playlist):
            temp_file = os.path.join(temp_dir, f"temp_clip_{i}.mp4")  # H.264 in MP4 container
            temp_files.append(temp_file)
            args_list.append((
                clip, temp_file, config.overlay_scale, config.map_style, config.north_up,
                clip_histories[i], config.cameras, clip_sei_data[i],
                config.watermark_path, config.show_timestamp, config.layout, color_grader,
                config.enable_emphasis
            ))

        # Create shared counter for real-time frame progress
        frame_counter = multiprocessing.Value('i', 0)
        import time

        with create_render_progress() as progress:
            task = progress.add_task(
                "Rendering",
                total=total_frames,
                status=f"Starting {num_processes} workers..."
            )

            clips_done = 0
            with multiprocessing.Pool(
                processes=num_processes,
                initializer=_init_worker,
                initargs=(frame_counter,)
            ) as pool:
                # Use imap_unordered for faster completion notification
                result_iter = pool.imap_unordered(process_clip_wrapper, args_list)
                start_time = time.time()

                # Poll for progress while processing
                failed_clips = []
                while clips_done < len(args_list):
                    # Check for completed clips (non-blocking with timeout)
                    try:
                        _ = result_iter.next(timeout=0.05)  # Faster polling
                        clips_done += 1
                    except StopIteration:
                        break
                    except multiprocessing.TimeoutError:
                        pass  # No result ready yet, just update progress
                    except Exception as e:
                        # Worker process failed - log error and continue with remaining clips
                        logger.error(f"Worker process failed: {e}")
                        failed_clips.append(str(e))
                        clips_done += 1  # Count as done to avoid hanging

                    # Update progress with current frame count
                    current_frames = frame_counter.value
                    elapsed = time.time() - start_time

                    # Show appropriate status based on progress
                    if current_frames == 0:
                        status = f"Initializing... ({elapsed:.0f}s)"
                    else:
                        render_fps = current_frames / elapsed if elapsed > 0 else 0
                        # Speed is rendering fps / video fps (29.97 for Tesla dashcam)
                        speed = render_fps / 29.97
                        status = f"{render_fps:.0f} fps • {speed:.1f}x"

                    progress.update(
                        task,
                        completed=current_frames,
                        status=status
                    )

        # 3. Concatenate
        print_phase(3, 3, "Concatenating clips")
        concat_clips(temp_files, config.output_file)

    # Print completion summary
    print_completion_summary(
        output_file=config.output_file,
        clip_count=len(config.playlist),
        gps_points=len(global_history),
    )

if __name__ == "__main__":
    main()
