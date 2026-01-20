# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Tesla dashcam video processor that extracts embedded SEI (Supplemental Enhancement Information) metadata from Tesla dashcam MP4 files and composites multiple camera angles into a single video with telemetry overlays (speedometer, steering wheel, pedal positions, G-ball acceleration indicator, blinker indicators, GPS map with heading-up rotation and optional satellite/street tiles).

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run (single file)
python3 main.py <path-to-front.mp4> output.mp4

# Run (directory of clips)
python3 main.py <directory-with-clips> output.mp4

# With custom font scaling (default is 1.5)
python3 main.py input/ output.mp4 --font-scale 2.0

# With overlay scaling (dashboard and map size)
python3 main.py input/ output.mp4 --overlay-scale 1.5

# Map styles: simple (default), street (OSM tiles), satellite (ESRI)
python3 main.py input/ output.mp4 --map-style satellite

# Select specific cameras (front is always required)
python3 main.py input/ output.mp4 --cameras front
python3 main.py input/ output.mp4 --cameras front,back
python3 main.py input/ output.mp4 --cameras front,left_repeater,right_repeater

# Verbose mode (shows debug output for troubleshooting)
python3 main.py input/ output.mp4 -v

# Regenerate protobuf bindings (if dashcam.proto changes)
protoc --python_out=. dashcam.proto
```

**External dependency**: FFmpeg with libx264 support (used for video I/O and concatenation). Hardware encoders (VideoToolbox, NVENC) used automatically when available.

## Architecture

### Processing Pipeline (main.py)

Three-phase processing:
1. **GPS Pre-scan**: Extracts GPS points from all clips to build route history
2. **Parallel Rendering**: Processes clips using multiprocessing (one process per CPU core), outputs H.264 directly via FFmpeg pipes
3. **Concatenation**: FFmpeg stream-copies intermediate H.264 clips into final output (very fast, no re-encode)

Clip discovery follows Tesla naming convention: `{timestamp}-front.mp4` with optional siblings (`-left_repeater`, `-right_repeater`, `-back`, `-left_pillar`, `-right_pillar`).

### SEI Parser (sei_parser.py)

Low-level MP4/H.264 parsing to extract Tesla's embedded telemetry:
- Navigates MP4 atom structure to find `mdat` (media data)
- Iterates H.264 NAL units, filtering for SEI type 6 with user data type 5
- Strips emulation prevention bytes (0x03 following 0x00 0x00) before protobuf parsing
- Returns `Dict[frame_index, SeiMetadata]` for frame-aligned metadata access

### Visualization (visualization.py)

Two overlay renderers:
- **DashboardRenderer** (500x200): Speedometer gauge, animated steering wheel, pedal indicators, G-ball acceleration indicator, blinker arrows, gear/autopilot status
- **MapRenderer** (300x300): GPS path polyline with heading arrow, heading-up rotation, optional street/satellite tiles via staticmap library

**G-Ball Indicator**: Shows lateral (X) and longitudinal (Y) acceleration as a moving dot. Converts m/s² to G-force, clamps to ±1g for full deflection.

**Blinker Indicators**: Dimmed arrow outlines drawn statically, bright green arrows overlaid when `blinker_on_left`/`blinker_on_right` metadata fields are true.

**Map Modes**:
- `simple`: Vector-drawn path on dark background (no network)
- `street`: OpenStreetMap tiles via staticmap
- `satellite`: ESRI World Imagery tiles

**Adaptive composite layouts** (1920x1080) based on `--cameras` flag:
- **Front only**: Full screen 1920x1080
- **Front + back**: Top/bottom split (540px each)
- **Front + both repeaters**: Center 1280px + side panels 320px each
- **Front + back + repeaters**: 2x2 grid (960x540 each)
- **Front + single side camera**: Side-by-side (front 1280px + side 640px)
  - Supports: left_repeater, right_repeater, left_pillar, right_pillar
- **All 6 cameras** (default): Original grid layout below

```
┌────────────┬─────────────────────────┬────────────┐
│ L. Pillar  │     Front (800x600)     │ R. Pillar  │  Top: 600px
│   560px    │                         │   560px    │
├────────────┼─────────────────────────┼────────────┤
│ L. Repeater│   Back (mirrored)       │ R. Repeater│  Bottom: 480px
│   640px    │       640px             │   640px    │
└────────────┴─────────────────────────┴────────────┘
```

### Data Schema (dashcam.proto)

Protobuf definition for Tesla telemetry with 16 fields including vehicle speed, gear, pedal positions, steering angle, autopilot state, GPS coordinates, and 3-axis acceleration.

## Key Patterns

- **Pydantic for config validation**: `VideoConfig` validates CLI args with type constraints
- **FFmpeg pipes for video I/O**: `video_io.py` uses subprocess pipes to read/write frames directly (no intermediate files with timestamp issues)
- **Hardware encoder auto-detection**: `detect_hw_encoder()` tests platform-specific encoders (VideoToolbox, NVENC, VA-API) and caches result
- **Pillow for drawing**: `visualization.py` uses Pillow's ImageDraw for overlays (speedometer, map, text labels)
- **Static/dynamic rendering split**: Base images with static elements (circles, outlines) pre-rendered once; only dynamic elements (values, positions) drawn per-frame
- **History snapshots for parallel processing**: Each clip receives GPS history up to its start time for map continuity
- **Overlay blending**: 80% overlay opacity over 20% canvas using numpy weighted addition
- **Temporary directory pattern**: Intermediate H.264 MP4s auto-cleaned after processing
- **Tile caching for maps**: Street/satellite tiles cached by rounded coordinates to minimize network requests
