#!/usr/bin/env python3
# CRITICAL: Set numba threading config BEFORE any imports
# Prevents "workqueue threading layer" conflicts with ThreadPoolExecutor
import os
os.environ['NUMBA_NUM_THREADS'] = '1'

import argparse
import sys
import glob
import subprocess
import multiprocessing
import logging
import math
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Tuple, Set, Callable, Union
from dataclasses import dataclass
from pydantic import BaseModel, Field, field_validator

from sei_parser import extract_sei_data
from visualization import DashboardRenderer, MapRenderer, composite_frame, apply_overlay, render_watermark, render_timestamp
from emphasis import EmphasisCalculator
from constants import OUTPUT_WIDTH, OUTPUT_HEIGHT, DASHBOARD_WIDTH, MAP_SIZE, DASHBOARD_Y, MAP_Y, MAP_X_MARGIN, TESLA_DASHCAM_FPS
from color_grading import create_color_grader, ColorGrader, warmup_color_grading
from video_io import VideoCaptures, VideoWriterContext, FrameBuffer
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
        # Find all front camera files
        front_files = glob.glob(os.path.join(input_path, "**/*-front.mp4"), recursive=True)
        if not front_files:
            front_files = glob.glob(os.path.join(input_path, "*-front.mp4"))

        # Sort by timestamp prefix (filename), not full path
        # This ensures chronological order regardless of directory structure
        # Tesla format: 2026-01-09_11-55-49-front.mp4
        def get_timestamp_key(path):
            filename = os.path.basename(path)
            # Extract timestamp prefix before "-front.mp4"
            prefix = filename.split("-front.mp4")[0]
            return prefix

        front_files = sorted(front_files, key=get_timestamp_key)
             
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

def parse_args() -> Union[VideoConfig, dict]:
    """
    Parse command line arguments.

    Returns:
        VideoConfig for video processing mode, or dict for export mode
    """
    parser = argparse.ArgumentParser(
        description="Burn Tesla SEI metadata into connected video clips, or export telemetry to GPX/FIT.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process video with overlays
  python main.py input/ output.mp4

  # Export telemetry to GPX
  python main.py input/ --export gpx

  # Export telemetry to FIT
  python main.py input/ --export fit -o my_drive.fit

  # Export all formats with reduced sample rate
  python main.py input/ --export all --export-sample-rate 0.1
"""
    )
    parser.add_argument("input_path", help="Path to input MP4 file or directory of clips")
    parser.add_argument("output_file", nargs="?", default=None,
                       help="Path to output MP4 file (not required for --export mode)")

    # Telemetry export options
    parser.add_argument("--export", choices=["gpx", "fit", "json", "all"], default=None,
                       metavar="FORMAT",
                       help="Export telemetry without video processing: gpx, fit, json, or all")
    parser.add_argument("-o", "--export-output", type=str, default=None, metavar="PATH",
                       help="Output path for telemetry export (auto-generated if not specified)")
    parser.add_argument("--export-sample-rate", type=float, default=1.0,
                       help="Fraction of frames to export (1.0=all, 0.1=every 10th frame)")
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

    # Handle export mode (telemetry only, no video processing)
    if args.export:
        return {
            "mode": "export",
            "input_path": args.input_path,
            "export_format": args.export,
            "output_path": args.export_output,
            "sample_rate": args.export_sample_rate,
            "verbose": args.verbose,
        }

    # Video processing mode requires output_file
    if args.output_file is None:
        print_error("output_file is required for video processing",
                   hint="Use --export for telemetry-only export, or provide an output file path")
        sys.exit(1)

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


def _process_frame_batch(args):
    """Process a single frame: compositing, overlays, color grading - all in one.

    This is a PURE FUNCTION designed to be called from a thread pool for parallel processing.
    All operations are stateless, using pre-computed values. Given the same inputs,
    it will always produce the same output.

    Args (unpacked from tuple):
        frames: Dict of camera frames (front, left_rep, etc.)
        cameras: Set of camera names to include
        layout: Layout mode (grid, pip)
        emphasis: Pre-computed EmphasisState for this frame
        color_grader: ColorGrader instance or None
        map_renderer: MapRenderer instance
        map_data: Pre-computed tuple (heading, lat, lon, smooth_heading, zoom_window, crop_scale, path_snapshot)
        dash_renderer: DashboardRenderer instance
        meta: SEI metadata for this frame
        overlay_positions: Tuple of (dash_x, dash_y, map_x, map_y)
        watermark_img: Pre-rendered watermark array or None
        timestamp_data: Tuple of (timestamp_str, fps, frame_idx) or None

    Returns:
        Fully composited and processed canvas ready for writing
    """
    import traceback as tb
    try:
        (frames, cameras, layout, emphasis, color_grader,
         map_renderer, map_data, dash_renderer, meta,
         overlay_positions, watermark_img, timestamp_data) = args

        dash_x, dash_y, map_x, map_y = overlay_positions

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

        # Render and apply dashboard with actual telemetry (if available)
        if meta:
            dash_img = dash_renderer.render(meta)
            apply_overlay(canvas, dash_img, dash_x, dash_y)

        # Render and apply map using pre-computed smoothed values (parallel-safe)
        if map_data is not None:
            heading, lat, lon, smooth_heading, zoom_window, crop_scale, path_snapshot = map_data
            if lat != 0.0 or lon != 0.0:
                map_img = map_renderer.render_stateless(
                    heading, lat, lon,
                    smooth_heading, zoom_window, crop_scale,
                    path_snapshot
                )
                apply_overlay(canvas, map_img, map_x, map_y)

        # Apply color grading to composited frame
        if color_grader is not None and color_grader.is_active:
            canvas = color_grader.grade(canvas)

        # Apply watermark (Lower Right)
        if watermark_img is not None:
            watermark_x = OUTPUT_WIDTH - watermark_img.shape[1] - 20
            watermark_y = OUTPUT_HEIGHT - watermark_img.shape[0] - 20
            apply_overlay(canvas, watermark_img, watermark_x, watermark_y)

        # Apply timestamp (Lower Left)
        if timestamp_data is not None:
            timestamp_str, fps, current_frame_idx = timestamp_data
            seconds = current_frame_idx / fps
            render_timestamp(canvas, timestamp_str, seconds, scale=1.0)  # Scale handled by dash_renderer

        return canvas
    except IndexError as e:
        # Log detailed info to help debug the index error
        logger.error(f"IndexError in _process_frame_batch: {e}")
        logger.error(f"map_data is None: {map_data is None}")
        if map_data is not None:
            logger.error(f"path_snapshot length: {len(map_data[6]) if map_data[6] else 'None'}")
        logger.error(f"Full traceback:\n{tb.format_exc()}")
        raise


# Number of threads for parallel frame processing within a clip
# Use most CPU cores since the work is CPU-bound
FRAME_PROCESSING_THREADS = max(4, multiprocessing.cpu_count() - 2)


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
    import traceback as tb
    try:
        return _process_clip_task_impl(clip, output_temp, overlay_scale, map_style, north_up,
                                       history, cameras, sei_data, watermark_path, show_timestamp,
                                       layout, color_grader, enable_emphasis)
    except Exception as e:
        logger.error(f"Clip {clip.timestamp_prefix} failed: {e}")
        logger.error(f"Full traceback:\n{tb.format_exc()}")
        raise


def _process_clip_task_impl(clip: ClipSet, output_temp: str, overlay_scale: float, map_style: str, north_up: bool,
                      history: List[Tuple[float, float]], cameras: Set[str], sei_data: dict,
                      watermark_path: Optional[str] = None, show_timestamp: bool = False,
                      layout: str = "grid", color_grader: Optional[ColorGrader] = None,
                      enable_emphasis: bool = True):
    """Internal implementation of process_clip_task with full error logging."""
    global _shared_frame_counter

    # Warm up numba JIT in this worker process to avoid 3+ second delay on first frame
    if color_grader is not None and color_grader.is_active and color_grader.lut is not None:
        warmup_color_grading(color_grader.lut)

    # Build camera paths dict for the context manager
    camera_paths = build_camera_paths(clip, cameras)

    # For PIP layout, read thumbnail cameras at reduced resolution
    # This significantly reduces I/O overhead (reading 6 cameras at full res is slow)
    # Thumbnails only need ~350x200, so we read at 640x360 (half res)
    camera_sizes = None
    if layout == "pip" and len(camera_paths) > 1:
        # Front camera needs full resolution for fullscreen
        # Other cameras are thumbnails - read at half resolution
        camera_sizes = {}
        for key in camera_paths:
            if key != 'front':
                # Half resolution: 640x360 instead of 1280x960
                # Still larger than needed (~350x200 thumbnails) but much faster
                camera_sizes[key] = (640, 360)

    # Use context managers for safe resource cleanup
    with VideoCaptures(camera_paths, camera_sizes=camera_sizes) as caps:
        # ALWAYS use TESLA_DASHCAM_FPS (29.97) for output regardless of source metadata
        # Tesla dashcam files have unreliable r_frame_rate values (e.g., 36/1, 10000/1)
        # but actual playback is ~30 fps. Using consistent FPS prevents sync issues
        # when concatenating clips with different metadata.
        fps = TESLA_DASHCAM_FPS

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

            # Pre-compute all smoothed map values for parallel rendering
            # This allows map rendering to be included in the parallel batch
            # Use actual frame count from ffprobe instead of SEI count estimation
            # SEI metadata entries are sparse - videos often have more frames than metadata
            # Add 10% safety margin since OpenCV may decode slightly more frames than ffprobe reports
            ffprobe_frames = get_video_frame_count(clip.front)
            if ffprobe_frames > 0:
                max_frames = int(ffprobe_frames * 1.1)  # 10% safety margin
            else:
                max_frames = len(sei_data) + 500  # Generous fallback if ffprobe fails
            frame_gps_data = []  # (heading, lat, lon, speed) per frame
            path_snapshots = []  # GPS history snapshot per frame
            current_path = list(history)  # Start with initial history

            for frame_idx in range(max_frames):
                interp_lat, interp_lon, interp_heading = get_interpolated_gps(frame_idx)
                meta = sei_data.get(frame_idx)
                speed = meta.vehicle_speed_mps if meta else 0.0

                frame_gps_data.append((interp_heading, interp_lat, interp_lon, speed))
                path_snapshots.append(list(current_path))  # Snapshot current path state

                # Update path for next frame (if valid GPS)
                if meta and (abs(meta.latitude_deg) >= 0.001 or abs(meta.longitude_deg) >= 0.001):
                    current_path.append((meta.latitude_deg, meta.longitude_deg, speed))

            # Pre-compute smoothed values (heading, zoom, crop_scale) for all frames
            smoothed_values = map_renderer.precompute_smoothed_values(frame_gps_data)

            # Pre-compute emphasis values for all frames (removes per-frame state dependency)
            precomputed_emphasis = emphasis_calculator.precompute_all(sei_data, max_frames) if emphasis_calculator else None

            # Pre-fetch map tiles for the entire route (prevents network latency during rendering)
            if map_style != 'simple':
                gps_points = [(lat, lon) for _, lat, lon, _ in frame_gps_data if lat != 0.0 or lon != 0.0]
                tiles_fetched = map_renderer.prefetch_tiles(gps_points)
                logger.debug(f"Pre-fetched {tiles_fetched} map tiles for route")

            frame_idx = 0
            overlay_count = 0

            # Batch size for parallel processing (larger batch = less synchronization overhead)
            # With read-ahead buffer, we can use larger batches without blocking
            # Increased from *4 to *8 for better throughput on multi-core systems
            BATCH_SIZE = max(FRAME_PROCESSING_THREADS * 8, 64)

            # Pre-compute overlay positions (constant for all frames)
            dash_x = (OUTPUT_WIDTH - dash_renderer.width) // 2
            map_x = OUTPUT_WIDTH - map_renderer.size - MAP_X_MARGIN
            overlay_positions = (dash_x, DASHBOARD_Y, map_x, MAP_Y)

            # Use FrameBuffer for read-ahead (overlaps I/O with computation)
            # Buffer size = 2x batch size to keep processing pipeline fed without excess memory
            with FrameBuffer(caps, buffer_size=BATCH_SIZE * 2) as frame_buffer:
                # Use ThreadPoolExecutor for parallel frame processing within a clip
                with ThreadPoolExecutor(max_workers=FRAME_PROCESSING_THREADS) as executor:
                    while True:
                        # Read a batch of frames from buffer (read-ahead hides latency)
                        frames_batch = frame_buffer.read_batch(BATCH_SIZE, timeout=2.0)
                        if not frames_batch:
                            break

                        # Build args for batch processing
                        batch_args = []
                        for frames in frames_batch:
                            current_idx = frame_idx + len(batch_args)
                            meta = sei_data.get(current_idx)

                            # Use pre-computed emphasis (removes per-frame state dependency)
                            emphasis = precomputed_emphasis[current_idx] if precomputed_emphasis and current_idx < len(precomputed_emphasis) else None

                            # Get pre-computed map data for this frame
                            # Defensive bounds checking on all arrays to prevent index errors
                            map_data = None
                            if (current_idx < len(smoothed_values) and
                                current_idx < len(frame_gps_data) and
                                current_idx < len(path_snapshots)):
                                smooth_heading, zoom_window, crop_scale = smoothed_values[current_idx]
                                heading, lat, lon, speed = frame_gps_data[current_idx]
                                path_snapshot = path_snapshots[current_idx]
                                map_data = (heading, lat, lon, smooth_heading, zoom_window, crop_scale, path_snapshot)

                            # Timestamp data (if enabled)
                            ts_data = (timestamp_str, fps, current_idx) if timestamp_str else None

                            # Build complete args tuple for pure function
                            batch_args.append((
                                frames, cameras, layout, emphasis, color_grader,
                                map_renderer, map_data, dash_renderer, meta,
                                overlay_positions, watermark_img, ts_data
                            ))

                            if meta:
                                overlay_count += 1

                        # Process batch in parallel - pure function returns complete frame
                        results = list(executor.map(_process_frame_batch, batch_args))

                        # Write results (only sequential operation remaining)
                        for canvas in results:
                            out.write(canvas)

                        frame_idx += len(batch_args)

                        # Update shared counter for progress
                        if _shared_frame_counter is not None:
                            with _shared_frame_counter.get_lock():
                                _shared_frame_counter.value += len(batch_args)

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
    from video_io import detect_hw_encoder

    n = len(temp_files)

    # Get total frame count for progress tracking
    total_frames = sum(get_video_frame_count(f) for f in temp_files)
    logger.debug(f"Total frames to concatenate: {total_frames}")

    # Try stream copy first (much faster, ~5x speedup)
    # This works when all clips have compatible codec/resolution/fps
    # which they should since we created them with the same settings
    concat_list_path = os.path.join(os.path.dirname(temp_files[0]), "concat_list.txt")
    with open(concat_list_path, 'w') as f:
        for temp_file in temp_files:
            # FFmpeg concat demuxer format
            f.write(f"file '{os.path.abspath(temp_file)}'\n")

    # Try stream copy with concat demuxer
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "concat", "-safe", "0",
        "-i", concat_list_path,
        "-c:v", "copy",  # Stream copy - no re-encoding!
        "-movflags", "+faststart",
        output_file
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            # Verify output has expected frame count
            output_frames = get_video_frame_count(output_file)
            if abs(output_frames - total_frames) < 10:  # Allow small variance
                logger.debug(f"Concatenation via stream copy successful: {output_frames} frames")
                os.remove(concat_list_path)
                return
            else:
                logger.debug(f"Stream copy frame count mismatch: {output_frames} vs {total_frames}")
    except (subprocess.TimeoutExpired, Exception) as e:
        logger.debug(f"Stream copy failed, falling back to re-encode: {e}")

    os.remove(concat_list_path)

    # Fallback: Use concat FILTER with re-encoding for timestamp alignment
    # This is slower but handles timestamp discontinuities
    logger.debug("Using concat filter with re-encoding (slower)")

    # Build filter graph: reset timestamps for each input, then concat
    filter_parts = []

    # Reset video timestamps for each input
    for i in range(n):
        filter_parts.append(f"[{i}:v]setpts=PTS-STARTPTS[v{i}]")

    # Concat all video streams
    video_inputs = "".join(f"[v{i}]" for i in range(n))
    filter_parts.append(f"{video_inputs}concat=n={n}:v=1:a=0[outv]")

    filter_complex = ";".join(filter_parts)

    # Use hardware encoder if available for faster re-encoding
    hw_encoder = detect_hw_encoder()
    if hw_encoder == "h264_videotoolbox":
        encoder_args = ["-c:v", "h264_videotoolbox", "-b:v", "10M"]
    elif hw_encoder == "h264_nvenc":
        encoder_args = ["-c:v", "h264_nvenc", "-preset", "fast", "-b:v", "10M"]
    else:
        encoder_args = ["-c:v", "libx264", "-preset", "fast", "-crf", "23"]

    # Build command with all input files
    cmd = ["ffmpeg", "-y", "-fflags", "+genpts+igndts", "-progress", "pipe:2"]
    for temp_file in temp_files:
        cmd.extend(["-i", os.path.abspath(temp_file)])

    cmd.extend([
        "-filter_complex", filter_complex,
        "-map", "[outv]",
    ] + encoder_args + [
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
import threading
from queue import Queue, Empty

# Wrapper for imap since it only accepts one argument
def process_clip_wrapper(args):
    return process_clip_task(*args)


def process_clips_parallel(clips_data: List[tuple], color_grader: Optional[ColorGrader],
                           num_workers: int, progress_callback) -> List[str]:
    """
    Process multiple clips with parallel reading and processing.

    Optimized architecture:
    1. Reader threads feed frames into a shared queue (overlapping I/O)
    2. ThreadPoolExecutor processes frames in parallel (all workers utilized)
    3. Results buffered per-clip, written after all processing done (no writer thread overhead)

    Args:
        clips_data: List of (clip, temp_file, overlay_scale, map_style, north_up,
                            history, cameras, sei_data, watermark_path, show_timestamp,
                            layout, enable_emphasis) tuples
        color_grader: Shared color grader
        num_workers: Number of processing threads
        progress_callback: Function to call with frames_done count

    Returns:
        List of output temp files
    """
    from video_io import VideoCaptures, VideoWriterContext
    from concurrent.futures import as_completed

    # Warm up color grading
    if color_grader and color_grader.is_active and color_grader.lut is not None:
        warmup_color_grading(color_grader.lut)

    # Shared state
    frame_queue = Queue(maxsize=num_workers * 8)  # Larger buffer for better overlap
    frames_done = [0]
    clips_done_reading = [0]

    def reader_thread(clip_idx, clip, camera_paths, cameras, sei_data, history,
                      overlay_scale, map_style, north_up, watermark_path, show_timestamp,
                      layout, enable_emphasis):
        """Read frames from one clip and push to shared queue."""
        # Setup per-clip renderers
        dash_renderer = DashboardRenderer(scale=overlay_scale)
        map_renderer = MapRenderer(
            scale=overlay_scale,
            history=list(history),
            map_style=map_style,
            heading_up=not north_up
        )
        emphasis_calculator = EmphasisCalculator() if enable_emphasis else None

        # Build GPS interpolator
        get_interpolated_gps = build_gps_interpolator(sei_data)

        # Pre-render watermark
        watermark_img = None
        if watermark_path:
            watermark_img = render_watermark(watermark_path, max_size=int(150 * overlay_scale))

        timestamp_str = None
        if show_timestamp:
            timestamp_str = parse_clip_timestamp(clip.timestamp_prefix)

        # Pre-compute smoothed values
        # Use actual frame count instead of SEI count estimation (same fix as process_clip_task)
        # Add 10% safety margin since OpenCV may decode slightly more frames than ffprobe reports
        ffprobe_frames = get_video_frame_count(clip.front)
        if ffprobe_frames > 0:
            max_frames = int(ffprobe_frames * 1.1)  # 10% safety margin
        else:
            max_frames = len(sei_data) + 500  # Generous fallback
        frame_gps_data = []
        path_snapshots = []
        current_path = list(history)

        for frame_idx in range(max_frames):
            interp_lat, interp_lon, interp_heading = get_interpolated_gps(frame_idx)
            meta = sei_data.get(frame_idx)
            speed = meta.vehicle_speed_mps if meta else 0.0
            frame_gps_data.append((interp_heading, interp_lat, interp_lon, speed))
            path_snapshots.append(list(current_path))
            if meta and (abs(meta.latitude_deg) >= 0.001 or abs(meta.longitude_deg) >= 0.001):
                current_path.append((meta.latitude_deg, meta.longitude_deg, speed))

        smoothed_values = map_renderer.precompute_smoothed_values(frame_gps_data)
        precomputed_emphasis = emphasis_calculator.precompute_all(sei_data, max_frames) if emphasis_calculator else None

        # Pre-fetch tiles
        if map_style != 'simple':
            gps_points = [(lat, lon) for _, lat, lon, _ in frame_gps_data if lat != 0.0 or lon != 0.0]
            map_renderer.prefetch_tiles(gps_points)

        # Overlay positions
        dash_x = (OUTPUT_WIDTH - dash_renderer.width) // 2
        map_x = OUTPUT_WIDTH - map_renderer.size - MAP_X_MARGIN
        overlay_positions = (dash_x, DASHBOARD_Y, map_x, MAP_Y)

        # For PIP layout, read thumbnails at reduced resolution
        camera_sizes = None
        if layout == "pip" and len(camera_paths) > 1:
            camera_sizes = {k: (640, 360) for k in camera_paths if k != 'front'}

        # Read frames
        with VideoCaptures(camera_paths, camera_sizes) as caps:
            fps = caps['front'].fps
            if fps == 0 or fps > 60:
                fps = 29.97

            frame_idx = 0
            while True:
                ret, frames = caps.read_all_parallel()
                if not ret:
                    break

                meta = sei_data.get(frame_idx)
                emphasis = precomputed_emphasis[frame_idx] if precomputed_emphasis and frame_idx < len(precomputed_emphasis) else None

                # Defensive bounds checking on all arrays to prevent index errors
                map_data = None
                if (frame_idx < len(smoothed_values) and
                    frame_idx < len(frame_gps_data) and
                    frame_idx < len(path_snapshots)):
                    smooth_heading, zoom_window, crop_scale = smoothed_values[frame_idx]
                    heading, lat, lon, speed = frame_gps_data[frame_idx]
                    path_snapshot = path_snapshots[frame_idx]
                    map_data = (heading, lat, lon, smooth_heading, zoom_window, crop_scale, path_snapshot)

                ts_data = (timestamp_str, fps, frame_idx) if timestamp_str else None

                # Push to queue
                frame_queue.put((
                    clip_idx, frame_idx, frames, cameras, layout, emphasis, color_grader,
                    map_renderer, map_data, dash_renderer, meta,
                    overlay_positions, watermark_img, ts_data
                ))
                frame_idx += 1

        # Signal this clip is done reading
        frame_queue.put((clip_idx, -1, None, None, None, None, None, None, None, None, None, None, None, None))

    def process_frame(args):
        """Process a single frame - pure function."""
        (clip_idx, frame_idx, frames, cameras, layout, emphasis, color_grader,
         map_renderer, map_data, dash_renderer, meta,
         overlay_positions, watermark_img, ts_data) = args

        dash_x, dash_y, map_x, map_y = overlay_positions

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

        if meta:
            dash_img = dash_renderer.render(meta)
            apply_overlay(canvas, dash_img, dash_x, dash_y)

        if map_data is not None:
            heading, lat, lon, smooth_heading, zoom_window, crop_scale, path_snapshot = map_data
            if lat != 0.0 or lon != 0.0:
                map_img = map_renderer.render_stateless(
                    heading, lat, lon, smooth_heading, zoom_window, crop_scale, path_snapshot
                )
                apply_overlay(canvas, map_img, map_x, map_y)

        if color_grader is not None and color_grader.is_active:
            canvas = color_grader.grade(canvas)

        if watermark_img is not None:
            watermark_x = OUTPUT_WIDTH - watermark_img.shape[1] - 20
            watermark_y = OUTPUT_HEIGHT - watermark_img.shape[0] - 20
            apply_overlay(canvas, watermark_img, watermark_x, watermark_y)

        if ts_data is not None:
            timestamp_str, fps, current_frame_idx = ts_data
            seconds = current_frame_idx / fps
            render_timestamp(canvas, timestamp_str, seconds, scale=1.0)

        return clip_idx, frame_idx, canvas

    # Start reader threads for all clips
    reader_threads = []
    temp_files = []
    for clip_idx, data in enumerate(clips_data):
        (clip, temp_file, overlay_scale, map_style, north_up, history, cameras,
         sei_data, watermark_path, show_timestamp, layout, enable_emphasis) = data
        temp_files.append(temp_file)
        camera_paths = build_camera_paths(clip, cameras)
        t = threading.Thread(target=reader_thread, args=(
            clip_idx, clip, camera_paths, cameras, sei_data, history,
            overlay_scale, map_style, north_up, watermark_path, show_timestamp,
            layout, enable_emphasis
        ))
        reader_threads.append(t)
        t.start()

    # Collect results per clip for ordered writing
    clip_results = [[] for _ in clips_data]

    import time as _time
    process_start = _time.time()

    # Process frames with thread pool
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {}  # future -> (clip_idx, frame_idx)

        while clips_done_reading[0] < len(clips_data) or futures:
            # Submit new work from queue
            try:
                while len(futures) < num_workers * 4:  # Keep workers busy
                    item = frame_queue.get(timeout=0.01)
                    clip_idx = item[0]
                    frame_idx = item[1]

                    if frame_idx == -1:  # End marker
                        clips_done_reading[0] += 1
                        continue

                    future = executor.submit(process_frame, item)
                    futures[future] = (clip_idx, frame_idx)
            except Empty:
                pass

            # Collect completed results
            done_futures = [f for f in futures if f.done()]
            for f in done_futures:
                clip_idx, frame_idx = futures.pop(f)
                canvas = f.result()[2]
                clip_results[clip_idx].append((frame_idx, canvas))
                frames_done[0] += 1
                progress_callback(frames_done[0])

    # Wait for readers to finish
    for t in reader_threads:
        t.join()

    logger.debug(f"Process phase: {_time.time() - process_start:.1f}s ({frames_done[0]} frames)")

    # Write all results in order (single sequential pass per clip)
    for clip_idx, results in enumerate(clip_results):
        results.sort(key=lambda x: x[0])
        temp_file = temp_files[clip_idx]
        with VideoWriterContext(temp_file, 0, TESLA_DASHCAM_FPS, (OUTPUT_WIDTH, OUTPUT_HEIGHT)) as out:
            for frame_idx, canvas in results:
                out.write(canvas)
        clip_results[clip_idx] = None

    return temp_files


def _handle_export_mode(config: dict) -> None:
    """
    Handle telemetry export mode (GPX, FIT, JSON) without video processing.

    Args:
        config: Dict with export configuration from parse_args()
    """
    from telemetry_export.exporter import TelemetryExporter

    print_banner(__version__)
    console.print(f"[bold cyan]Telemetry Export Mode[/] - Format: [highlight]{config['export_format'].upper()}[/]")
    console.print()

    # Discover clips
    clips = discover_clips(config["input_path"])
    if not clips:
        print_error("No clips found!", hint="Check the input path contains Tesla dashcam files")
        sys.exit(1)

    console.print(f"Found [highlight]{len(clips)}[/] clip(s)")

    # Create exporter with sample rate
    exporter = TelemetryExporter(clips, sample_rate=config["sample_rate"])

    # Extract telemetry with progress
    with create_scan_progress() as progress:
        task = progress.add_task("Extracting telemetry", total=len(clips))

        def progress_cb(idx, total):
            progress.update(task, completed=idx)

        track = exporter.extract_all(progress_callback=progress_cb)
        progress.update(task, completed=len(clips))

    console.print(f"Extracted [highlight]{len(track.records)}[/] records "
                 f"({track.duration_seconds:.1f}s, {track.distance_meters/1000:.2f}km)")
    console.print()

    # Determine output path
    output_path = config["output_path"]
    if output_path is None:
        # Auto-generate from first clip timestamp
        base_dir = os.path.dirname(config["input_path"]) if os.path.isfile(config["input_path"]) else config["input_path"]
        base_name = clips[0].timestamp_prefix
        output_path = os.path.join(base_dir, base_name)

    # Export based on format
    export_format = config["export_format"].lower()
    created_files = []

    try:
        if export_format == "all":
            created_files = exporter.export_all(output_path)
        elif export_format == "gpx":
            created_files.append(exporter.export_gpx(f"{output_path}.gpx"))
        elif export_format == "fit":
            created_files.append(exporter.export_fit(f"{output_path}.fit"))
        elif export_format == "json":
            created_files.append(exporter.export_json(f"{output_path}.json"))
    except ImportError as e:
        print_error(str(e), hint="Install missing dependencies with: pip install -r requirements.txt")
        sys.exit(1)

    # Print summary
    console.print()
    console.print("[bold green]\u2713 Export complete![/]")
    for f in created_files:
        console.print(f"  [dim]\u2192[/] {f}")


def main():
    config = parse_args()

    # Handle export mode (telemetry only, no video processing)
    if isinstance(config, dict) and config.get("mode") == "export":
        _handle_export_mode(config)
        return

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
    # Limit to 4 workers by default - more workers increase memory pressure
    # and can cause performance degradation due to context switching
    num_processes = config.workers if config.workers else min(4, multiprocessing.cpu_count())
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
            temp_file = os.path.join(temp_dir, f"temp_clip_{i}.mp4")
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

        # Use parallel ThreadPool architecture for better CPU utilization
        # (threads share memory, no serialization overhead)
        use_parallel = len(config.playlist) <= num_processes

        with create_render_progress() as progress:
            task = progress.add_task(
                "Rendering",
                total=total_frames,
                status=f"Starting {num_processes} workers..."
            )

            start_time = time.time()
            frames_done = [0]

            def update_progress(count):
                frames_done[0] = count
                elapsed = time.time() - start_time
                if count == 0:
                    status = f"Initializing... ({elapsed:.0f}s)"
                else:
                    render_fps = count / elapsed if elapsed > 0 else 0
                    speed = render_fps / 29.97
                    status = f"{render_fps:.0f} fps • {speed:.1f}x"
                progress.update(task, completed=count, status=status)

            if use_parallel:
                # ThreadPool-based parallel processing (better for few clips)
                # Remove color_grader from args_list for process_clips_parallel
                clips_data = []
                for i, clip in enumerate(config.playlist):
                    temp_file = os.path.join(temp_dir, f"temp_clip_{i}.mp4")
                    clips_data.append((
                        clip, temp_file, config.overlay_scale, config.map_style, config.north_up,
                        clip_histories[i], config.cameras, clip_sei_data[i],
                        config.watermark_path, config.show_timestamp, config.layout,
                        config.enable_emphasis
                    ))

                temp_files = process_clips_parallel(
                    clips_data, color_grader, num_processes, update_progress
                )
            else:
                # Original multiprocessing.Pool approach (better for many clips)
                clips_done = 0
                with multiprocessing.Pool(
                    processes=num_processes,
                    initializer=_init_worker,
                    initargs=(frame_counter,)
                ) as pool:
                    result_iter = pool.imap_unordered(process_clip_wrapper, args_list)

                    failed_clips = []
                    while clips_done < len(args_list):
                        try:
                            _ = result_iter.next(timeout=0.05)
                            clips_done += 1
                        except StopIteration:
                            break
                        except multiprocessing.TimeoutError:
                            pass
                        except Exception as e:
                            import traceback
                            logger.error(f"Worker process failed: {e}")
                            logger.debug(f"Full traceback: {traceback.format_exc()}")
                            failed_clips.append(str(e))
                            clips_done += 1

                        current_frames = frame_counter.value
                        update_progress(current_frames)

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
