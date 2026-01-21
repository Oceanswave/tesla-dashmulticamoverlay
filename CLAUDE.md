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

# Color grading with preset
python3 main.py input/ output.mp4 --color-grade cinematic

# Color grading with LUT file
python3 main.py input/ output.mp4 --color-grade LUTs/NaturalBoost.cube

# Manual color adjustments
python3 main.py input/ output.mp4 --brightness 0.1 --contrast 0.15 --saturation 0.2

# Combine preset with manual adjustments (values stack)
python3 main.py input/ output.mp4 --color-grade warm --brightness -0.05

# Disable dynamic camera emphasis
python3 main.py input/ output.mp4 --no-emphasis

# Custom worker count (default: min(4, cpu_count))
python3 main.py input/ output.mp4 --workers 2

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

**Dynamic Map Zoom**: Map automatically zooms based on vehicle speed with 10 distinct zones:

| Speed (mph) | Zone | crop_scale | View |
|-------------|------|------------|------|
| 0-5 | Parking | 3.0 | ~30m (ultra-tight) |
| 5-10 | Lot | 2.5 | ~45m |
| 10-15 | Residential | 2.2 | ~60m |
| 15-20 | City crawl | 1.7 | ~75m |
| 20-25 | City slow | 1.5 | ~100m |
| 25-30 | City moderate | 1.3 | ~120m |
| 30-45 | City | 1.1 | ~150m |
| 45-60 | Suburban | 0.8 | ~220m |
| 60-75 | Highway | 0.55 | ~300m |
| 75+ | Fast | 0.4 | ~400m |

Zoom transitions are smoothed (30% per frame) to prevent jarring changes.

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

### Dynamic Camera Emphasis (emphasis.py)

Context-aware camera highlighting that emphasizes relevant cameras based on driving conditions:

**EmphasisCalculator class**: Computes per-camera emphasis weights (0.0-1.0) based on SEI metadata triggers.

**Trigger priority** (highest first):
1. **Turn signals**: Blinker → side cameras with orange border (100% repeater, 70% pillar)
2. **Reverse gear**: Reverse → back camera with green border
3. **Heavy braking**: >0.3g deceleration → back camera with red border (proportional)
4. **Lateral G-force**: >0.2g lateral → side cameras with amber border (proportional)

**Visual effects**:
- Colored inset borders (4px, drawn inside camera frame to prevent overlap)
- Side cameras: up to 25% size increase when emphasized
- Rear camera: up to 50% size increase when emphasized (for braking and reverse)
- 30% smoothing per frame for gradual transitions
- In PIP layout, top row shifts upward when bottom row scales to avoid overlap

**Overlap prevention**:
- When both left AND right cameras emphasized simultaneously, each capped to 50% max scale
- Borders always drawn inset (never extend beyond camera bounds)

**CLI**: `--no-emphasis` disables the feature entirely (default: enabled)

### Color Grading (color_grading.py)

Post-composite color correction applied to the entire 1080p frame before overlays.

**Three approaches**:
1. **LUT-based**: Load standard .cube files (33³ grid) via trilinear interpolation
2. **Manual adjustments**: CLI flags for brightness, contrast, saturation, gamma, shadows, highlights
3. **Built-in presets**: Named styles with pre-tuned values

**Presets** (defined in `constants.py:COLOR_PRESETS`):
| Preset | Description |
|--------|-------------|
| `cinematic` | Lifted shadows, boosted contrast, muted colors |
| `warm` | Golden hour feel, lifted shadows |
| `cool` | Blue hour feel, crisp contrast |
| `vivid` | Punchy colors, high contrast for dashcam clarity |
| `cybertruck` | Cold steel aesthetic matching UI theme (desaturated, industrial) |
| `dramatic` | High contrast, crushed blacks |
| `vintage` | Faded look, lifted blacks, reduced contrast |
| `natural` | Subtle enhancement, minimal processing |

**LUT files available** in `./LUTs/`:
- BlueArchitecture.cube, BlueHour.cube, ColdChrome.cube, CrispAutumn.cube
- DarkAndSomber.cube, HardBoost.cube, LongBeachMorning.cube, LushGreen.cube
- MagicHour.cube, NaturalBoost.cube, OrangeAndBlue.cube, SoftBlackAndWhite.cube, Waves.cube

**Processing order**: LUT → Brightness → Contrast → Saturation → Gamma → Shadows/Highlights

**Performance**: ~3-5ms per 1080p frame (uses numpy vectorized ops and pre-computed gamma LUTs)

### Data Schema (dashcam.proto)

Protobuf definition for Tesla telemetry with 16 fields including vehicle speed, gear, pedal positions, steering angle, autopilot state, GPS coordinates, and 3-axis acceleration.

## Performance Characteristics

**Target**: Real-time (30fps) processing for all configurations.

| Configuration | FPS | Realtime | Notes |
|--------------|-----|----------|-------|
| Grid + Simple | 48 | 1.6x | Vector map, minimal overhead |
| Grid + Satellite | 28 | 0.9x | Tile fetching + compositing |
| Grid + Street | 30 | 1.0x | OSM tile fetching |
| PIP + Simple | 64 | 2.1x | Front fullscreen, smaller thumbnails |
| PIP + Satellite | 57 | 1.9x | Fastest tile-based config |
| + color grading | -1-2fps | LUT application overhead |

**Performance bottlenecks** (in order of impact):
1. **Video I/O** (~20ms): Reading 6 camera streams; parallelized via ThreadPoolExecutor
2. **Composite** (~17ms): Resizing 6 cameras to layout positions; uses OpenCV INTER_LINEAR
3. **Map rendering** (~2-3ms): Grid/tile drawing with heading-up rotation
4. **Dashboard** (~2-3ms): Pre-rendered base image with dynamic overlays

**Optimizations applied**:
- Parallel tile fetching (16 concurrent threads for satellite/street maps)
- Tile memory cache holds 500 tiles (full 17×17 composite + movement headroom)
- Frame resize skip when dimensions already match
- Emphasis scale fast-path when weight < 2%
- Read-ahead buffer (2x batch size) overlaps I/O with computation
- Pre-computed emphasis/GPS/map data enables parallel frame processing
- Composite reuse when position within 2-tile margin (avoids rebuild)

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
- **Crop-then-rotate optimization**: Map rotation crops a smaller region first (using `ROTATION_MARGIN=1.5`), then rotates ~0.8M pixels instead of full composite (~20× faster)
- **Fixed composite radius**: Tile composite uses fixed radius=8 (17×17 grid) to avoid rebuilds on speed changes; zoom handled by crop_scale instead
- **Path drawing early-exit**: Reverse iteration with early exit when 20+ consecutive segments are off-screen; path fills visible map area at current zoom
- **Frame count safety margin**: Use ffprobe frame count + 10% buffer to handle OpenCV/ffprobe count discrepancies
- **Defensive bounds checking**: All pre-computed array accesses check bounds to prevent worker crashes
- **Timestamp-based clip sorting**: Clips sorted by filename timestamp prefix, not full path, ensuring chronological order across directories
- **Thread-safe path rendering**: Map path drawing functions take local snapshots of shared state to prevent race conditions when multiple threads call render_stateless() simultaneously
- **FPS normalization**: Output always uses TESLA_DASHCAM_FPS (29.97) regardless of unreliable source metadata (Tesla files report incorrect r_frame_rate values like 36/1 or 10000/1); prevents sync issues when concatenating clips
