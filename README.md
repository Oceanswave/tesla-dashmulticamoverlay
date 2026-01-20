# Tesla Dashcam Multi-Camera Overlay

A Python tool that processes Tesla dashcam footage by extracting embedded SEI (Supplemental Enhancement Information) telemetry metadata and compositing multiple camera angles into a single video with real-time overlays.

## Features

- **Multi-Camera Composite**: Combines up to 6 camera angles (front, back, left/right repeaters, left/right pillars) into a single 1920x1080 video
- **Telemetry Overlays**: Extracts and displays embedded vehicle data:
  - Speedometer gauge with animated needle
  - Steering wheel indicator showing wheel position
  - Accelerator and brake pedal positions
  - **G-ball indicator** showing lateral and longitudinal acceleration
  - **Blinker/turn signal indicators** (left, right, hazards)
  - Gear indicator (P/R/N/D)
  - Autopilot status
  - GPS mini-map with route history and **heading-up rotation**
- **Map Styles**: Choose between simple vector, OpenStreetMap street tiles, or satellite imagery
- **Batch Processing**: Automatically discovers and processes multiple clips from a directory
- **Parallel Rendering**: Utilizes all CPU cores for faster processing
- **Hardware Acceleration**: Auto-detects and uses GPU encoding when available (VideoToolbox on macOS, NVENC on NVIDIA)
- **Camera Selection**: Choose which cameras to include in the output
- **Scalable Overlays**: Adjust overlay size with `--overlay-scale`

## Requirements

- Python 3.8+
- FFmpeg with libx264 support (must be in PATH)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/oceanswave/tesla-dashmulticamoverlay.git
   cd tesla-dashmulticamoverlay
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Verify FFmpeg is installed:
   ```bash
   ffmpeg -version
   ```

## Usage

### Basic Usage

Process a single front camera file (discovers sibling camera files automatically):
```bash
python3 main.py /path/to/2024-01-15_12-30-00-front.mp4 output.mp4
```

Process all clips in a directory:
```bash
python3 main.py /path/to/dashcam/folder/ output.mp4
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--overlay-scale` | Scale factor for dashboard/map overlays (0.5-2.0) | 1.0 |
| `--map-style` | Map rendering style: `simple`, `street`, or `satellite` | simple |
| `--cameras` | Comma-separated list of cameras to include | all |
| `--layout` | Multi-camera layout: `grid` (6-camera grid) or `pip` (fullscreen with thumbnails) | grid |
| `--workers`, `-j` | Number of parallel rendering workers | CPU count |
| `--watermark` | Path to watermark image to overlay in lower-right corner | none |
| `--timestamp` | Burn in date/time from dashcam filename in lower-left corner | off |
| `-v, --verbose` | Enable debug logging | off |

### Camera Selection

The front camera is always required (contains SEI telemetry). Available cameras:
- `front` (required)
- `back`
- `left_repeater`
- `right_repeater`
- `left_pillar`
- `right_pillar`

Examples:
```bash
# Front camera only (minimal output)
python3 main.py input/ output.mp4 --cameras front

# Front and back only
python3 main.py input/ output.mp4 --cameras front,back

# Front with side repeaters
python3 main.py input/ output.mp4 --cameras front,left_repeater,right_repeater
```

### Overlay Scaling

Scale the dashboard and map overlays:
```bash
# Larger overlays (1.5x)
python3 main.py input/ output.mp4 --overlay-scale 1.5

# Smaller overlays for less intrusive display
python3 main.py input/ output.mp4 --overlay-scale 0.75
```

### Map Styles

Choose between different map rendering styles:
```bash
# Simple vector map (default, no network required)
python3 main.py input/ output.mp4 --map-style simple

# OpenStreetMap street tiles (requires network)
python3 main.py input/ output.mp4 --map-style street

# Satellite imagery from ESRI (requires network)
python3 main.py input/ output.mp4 --map-style satellite
```

**Note**: Street and satellite modes fetch map tiles from the internet. If tiles are unavailable, the renderer automatically falls back to simple mode.

### Dynamic Map Zoom

The map automatically adjusts zoom level based on vehicle speed for optimal detail at each driving context:

| Speed | Zoom Level | Best For |
|-------|------------|----------|
| 0-5 mph | Ultra-tight (~30m) | Parking maneuvers |
| 5-10 mph | Tight (~45m) | Parking lot navigation |
| 10-15 mph | Close (~60m) | Residential streets |
| 15-20 mph | Near (~75m) | Heavy traffic |
| 20-25 mph | City slow (~100m) | Normal city driving |
| 25-30 mph | City (~120m) | Flowing traffic |
| 30-45 mph | Suburban (~150m) | Busy streets |
| 45-60 mph | Wide (~220m) | Suburban roads |
| 60-75 mph | Highway (~300m) | Highway driving |
| 75+ mph | Very wide (~400m) | Fast highway |

Zoom transitions are smoothed to prevent jarring changes when accelerating or braking.

### Watermark and Timestamp

Add branding or timestamp information to your videos:
```bash
# Add a logo/watermark to the lower-right corner
python3 main.py input/ output.mp4 --watermark logo.png

# Burn in the timestamp from the dashcam filename (lower-left corner)
python3 main.py input/ output.mp4 --timestamp

# Combine both features
python3 main.py input/ output.mp4 --watermark logo.png --timestamp
```

The timestamp displays the date/time from the dashcam filename (e.g., `2024-01-15 11:45:38`) plus a running timecode showing elapsed time within each clip.

**Supported watermark formats**: PNG, JPG, GIF, BMP. The watermark is automatically resized to fit within the overlay scale while preserving aspect ratio.

### Debugging

Enable verbose output to troubleshoot issues:
```bash
python3 main.py input/ output.mp4 -v
```

## Output Layouts

The composite video (1920x1080) adapts based on selected cameras:

### Layout 1: Front Only (`--cameras front`)

```
+-----------------------------------------------+
|                                               |
|                                               |
|              Front (1920x1080)                |
|                                               |
|                                               |
+-----------------------------------------------+
```

### Layout 2: Front + Back (`--cameras front,back`)

```
+-----------------------------------------------+
|                                               |
|              Front (1920x540)                 |
|                                               |
+-----------------------------------------------+
|                                               |
|           Back (mirrored, 1920x540)           |
|                                               |
+-----------------------------------------------+
```

### Layout 3: Front + Repeaters (`--cameras front,left_repeater,right_repeater`)

```
+--------+---------------------------+--------+
|        |                           |        |
|   L.   |                           |   R.   |
|  Rep   |     Front (1280x1080)     |  Rep   |
| 320px  |                           | 320px  |
|        |                           |        |
+--------+---------------------------+--------+
```

### Layout 4: 2x2 Grid (`--cameras front,back,left_repeater,right_repeater`)

```
+-----------------------+-----------------------+
|                       |                       |
|   Front (960x540)     |   Back (960x540)      |
|                       |      (mirrored)       |
+-----------------------+-----------------------+
|                       |                       |
| L. Repeater (960x540) | R. Repeater (960x540) |
|                       |                       |
+-----------------------+-----------------------+
```

### Layouts 5-8: Front + Single Side Camera

When pairing front with one side camera (`left_repeater`, `right_repeater`, `left_pillar`, or `right_pillar`), the layout places them side-by-side:

```
+----------------+--------------------------------+
|                |                                |
|  Side Camera   |        Front (1280x1080)       |
|    (640px)     |                                |
+----------------+--------------------------------+
```

Left cameras appear on the left; right cameras on the right. Examples:
- `--cameras front,left_repeater`
- `--cameras front,right_pillar`

### Layout 9: Full 6-Camera (default, `--layout grid`)

```
+------------+-------------------------+------------+
| L. Pillar  |     Front (800x600)     | R. Pillar  |  Top: 600px
|   560px    |                         |   560px    |
+------------+-------------------------+------------+
| L. Repeater|   Back (mirrored)       | R. Repeater|  Bottom: 480px
|   640px    |       640px             |   640px    |
+------------+-------------------------+------------+
```

### Layout 10: Fullscreen with PIP Thumbnails (`--layout pip`)

```
+-------------------------------------------------------+
|                                                       |
|              FRONT (fullscreen 1920x1080)             |
|                                                       |
|   +----------+                        +----------+    |
|   | L-PILLAR |                        | R-PILLAR |    |
|   +----------+                        +----------+    |
|                                                       |
|  +----------+    +----------+    +----------+         |
|  |L-REPEATER|    |   REAR   |    |R-REPEATER|         |
|  +----------+    +----------+    +----------+         |
+-------------------------------------------------------+
```

Use this layout when you want the front camera to fill the entire screen with other cameras as small picture-in-picture thumbnails:
```bash
python3 main.py input/ output.mp4 --layout pip
```

Thumbnail dimensions are 280x158 pixels each. The rear camera is mirrored so it matches what you'd see in a rearview mirror.

### Overlay Positions

Overlays are positioned consistently across all layouts:
- **Dashboard** (speedometer, steering, pedals, g-ball, blinkers, gear): Top center
- **GPS Map**: Top right corner

### Dashboard Layout

```
+------------------------------------------------------------------+
|  [<]     Speedometer    Steering    Pedals    G-ball       [>]   |
| Blinker     (gauge)     (wheel)     (P/B)   (accel dot)  Blinker |
|                                                                   |
|                    GEAR: D  |  MODE: Manual                       |
+------------------------------------------------------------------+
```

- **Blinkers**: Arrow indicators at top corners, green when active
- **Speedometer**: Animated gauge showing current speed in MPH
- **Steering Wheel**: Rotating wheel showing steering angle
- **Pedals**: P (accelerator) and B (brake) bar indicators
- **G-ball**: Acceleration indicator with dot showing lateral/longitudinal forces
- **Status**: Current gear and autopilot mode

## How It Works

### Processing Pipeline

1. **Clip Discovery**: Scans input for Tesla-formatted files (`{timestamp}-front.mp4`) and finds sibling camera files
2. **GPS Pre-scan**: Extracts GPS coordinates from all clips to build complete route history for the map
3. **Parallel Rendering**: Each clip is processed independently using multiprocessing
4. **Concatenation**: FFmpeg combines rendered clips into final output using the concat filter with timestamp reset

### Hardware Acceleration

The tool automatically detects and uses hardware-accelerated video encoding when available:

| Platform | Encoder | Hardware |
|----------|---------|----------|
| macOS | `h264_videotoolbox` | Apple Silicon / Intel Quick Sync |
| Linux/Windows | `h264_nvenc` | NVIDIA GPU |
| Linux | `h264_vaapi` | AMD/Intel (VA-API) |
| Windows | `h264_qsv` | Intel Quick Sync |

Hardware encoding provides 2-4x faster encoding with minimal quality difference. If no hardware encoder is available, the tool falls back to `libx264` (software encoding).

### SEI Metadata Extraction

Tesla embeds telemetry data in H.264 SEI (Supplemental Enhancement Information) NAL units within the front camera video. The parser:
1. Navigates MP4 atom structure to locate `mdat` (media data)
2. Iterates H.264 NAL units, filtering for SEI type 6 with user data type 5
3. Strips emulation prevention bytes before protobuf decoding
4. Returns frame-indexed metadata for overlay synchronization

### Telemetry Fields

| Field | Description |
|-------|-------------|
| `vehicle_speed_mps` | Current speed in m/s (converted to MPH for display) |
| `steering_wheel_angle` | Steering wheel position in degrees |
| `accelerator_pedal_position` | Accelerator pedal position (0-1 or 0-100) |
| `brake_applied` | Brake pedal engaged (boolean) |
| `gear_state` | Current gear (P=0, D=1, R=2, N=3) |
| `autopilot_state` | Autopilot mode (None=0, FSD=1, Autosteer=2, TACC=3) |
| `blinker_on_left` | Left turn signal active (boolean) |
| `blinker_on_right` | Right turn signal active (boolean) |
| `latitude_deg` | GPS latitude |
| `longitude_deg` | GPS longitude |
| `heading_deg` | Vehicle heading (0-360) |
| `linear_acceleration_mps2_x` | Lateral acceleration (m/s², used for G-ball) |
| `linear_acceleration_mps2_y` | Longitudinal acceleration (m/s², used for G-ball) |
| `linear_acceleration_mps2_z` | Vertical acceleration (m/s²) |

## File Structure

```
tesla-dashmulticamoverlay/
├── main.py              # CLI entry point and processing pipeline
├── sei_parser.py        # H.264 SEI metadata extraction
├── visualization.py     # Dashboard overlay and frame compositing
├── map_renderer.py      # GPS map with heading-up rotation and tiles
├── video_io.py          # FFmpeg-based video I/O
├── overlays.py          # Overlay application utilities
├── constants.py         # Layout dimensions and styling
├── rich_console.py      # Terminal output formatting
├── map_prototype.py     # Standalone map testing tool
├── dashcam.proto        # Protobuf schema for Tesla telemetry
├── dashcam_pb2.py       # Generated protobuf bindings
├── requirements.txt     # Python dependencies
└── tests/               # Test suite
```

## Troubleshooting

### No overlays appearing

- Verify the front camera file contains SEI metadata (run with `-v` to see frame counts)
- Some older Tesla firmware versions may not embed telemetry
- Sentry mode clips typically don't contain telemetry

### FFmpeg not found

Ensure FFmpeg is installed and in your PATH:
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html and add to PATH
```

### Video freezes at clip boundaries

This was a known issue with the concat demuxer. The current version uses the concat filter with `setpts=PTS-STARTPTS` to properly reset timestamps. If you experience this, ensure you're using the latest version.

### Out of memory

Processing uses ~2GB RAM per worker. Reduce parallel workers by modifying `num_processes` in `main.py`, or process fewer cameras with `--cameras`.

## Development

### Running Tests

```bash
pytest tests/ -v
```

### Regenerating Protobuf Bindings

If you modify `dashcam.proto`:
```bash
protoc --python_out=. dashcam.proto
```

## License

MIT License - See LICENSE file for details.

## Acknowledgments

- Tesla for embedding telemetry data in dashcam footage
- [OpenStreetMap](https://www.openstreetmap.org/) contributors for street map tiles
- [ESRI](https://www.esri.com/) for satellite imagery tiles
- [FFmpeg](https://ffmpeg.org/) for video processing
