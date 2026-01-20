#!/usr/bin/env python3
"""
Map rendering prototype for rapid testing and debugging.

This standalone script tests the MapRenderer with GPS data to verify
heading-up rotation behaves intuitively.

Usage:
    # Test at specific location and heading
    python map_prototype.py --lat 37.7749 --lon -122.4194 --heading 45

    # Use real GPS data from extracted JSON
    python map_prototype.py --from-json sample_gps_data.json

    # Replay real GPS data as animation
    python map_prototype.py --from-json sample_gps_data.json --replay

    # Simulate driving (animates heading changes)
    python map_prototype.py --simulate-drive

    # Generate rotation animation GIF
    python map_prototype.py --animate-rotation --lat 37.7749 --lon -122.4194

    # Interactive mode (requires tkinter)
    python map_prototype.py --interactive

    # Run all standard tests
    python map_prototype.py --all
"""

import math
import argparse
import time
import os
import sys
import json

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PIL import Image, ImageDraw, ImageFont
import numpy as np

from map_renderer import MapRenderer
from constants import COLORS


def load_gps_json(json_path: str) -> dict:
    """Load GPS data from extracted JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def generate_path_from_heading(lat: float, lon: float, heading_deg: float,
                                distance_deg: float = 0.003, points: int = 30) -> list:
    """Generate a path that leads TO the current position from the given heading direction.

    This simulates having driven FROM somewhere and arriving at (lat, lon) while
    heading in the direction of heading_deg.
    """
    path = []
    heading_rad = math.radians(heading_deg)

    # Work backwards from current position
    for i in range(points):
        t = (points - 1 - i) / (points - 1)  # 1.0 to 0.0
        # Heading 0 = North, 90 = East
        path_lat = lat - t * distance_deg * math.cos(heading_rad)
        path_lon = lon - t * distance_deg * math.sin(heading_rad)
        path.append((path_lat, path_lon))

    return path


def render_single_map(lat: float, lon: float, heading_deg: float,
                      map_style: str = "satellite", scale: float = 1.5,
                      add_annotations: bool = True) -> Image.Image:
    """Render a single map frame at the given position and heading."""

    # Generate a path leading to this position
    path = generate_path_from_heading(lat, lon, heading_deg)

    renderer = MapRenderer(
        scale=scale,
        history=path,
        map_style=map_style,
        heading_up=True
    )

    map_arr = renderer.render(heading_deg, lat, lon)
    img = Image.fromarray(map_arr)

    if add_annotations:
        draw = ImageDraw.Draw(img)

        # Try to load a font
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
            small_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 11)
        except:
            font = ImageFont.load_default()
            small_font = font

        # Add heading annotation
        heading_text = f"Heading: {heading_deg:.0f}°"
        draw.text((10, 10), heading_text, fill=COLORS.CT_ORANGE, font=font)

        # Add cardinal direction
        cardinals = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        cardinal_idx = int((heading_deg + 22.5) / 45) % 8
        cardinal = cardinals[cardinal_idx]
        draw.text((10, 28), f"Direction: {cardinal}", fill=COLORS.STEEL_BRIGHT, font=small_font)

        # Add coordinates
        coord_text = f"({lat:.4f}, {lon:.4f})"
        draw.text((10, img.height - 25), coord_text, fill=COLORS.STEEL_MID, font=small_font)

        # Add style indicator
        draw.text((10, 44), f"Style: {map_style}", fill=COLORS.STEEL_DARK, font=small_font)

        # Add "UP = Forward" reminder
        draw.text((img.width - 90, img.height - 25), "↑ = Forward",
                  fill=COLORS.CT_GREEN, font=small_font)

    return img


def test_single_location(lat: float, lon: float, heading: float,
                         output_dir: str = "map_test_output"):
    """Test map rendering at a specific location and heading."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"Rendering map at ({lat}, {lon}) heading {heading}°")

    for style in ["simple", "satellite"]:
        img = render_single_map(lat, lon, heading, map_style=style)

        filename = f"{output_dir}/location_{style}_h{int(heading):03d}.png"
        img.save(filename)
        print(f"  Saved: {filename}")


def simulate_drive(output_dir: str = "map_test_output"):
    """
    Simulate a driving scenario with turns to verify rotation feels natural.

    The rotation should feel like a car navigation system:
    - When you turn right, the map rotates left (counterclockwise)
    - When you turn left, the map rotates right (clockwise)
    - Your direction of travel is always UP
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("SIMULATED DRIVE")
    print("=" * 60)
    print("""
This simulation shows what happens when you drive and make turns.
The map should behave like a car navigation in heading-up mode:

- Starting heading North (0°): Road ahead is at TOP of map
- Turn right to East (90°): Map rotates LEFT, road still at TOP
- Turn right to South (180°): Map rotates LEFT again, road still at TOP
- etc.

Key verification: The orange path (where you came from) should
always appear BELOW the center arrow, extending downward.
""")

    # Start location
    lat, lon = 37.7749, -122.4194

    # Define a route: heading changes at each waypoint
    route = [
        (0, "Starting: Heading North"),
        (45, "Turning right toward NE"),
        (90, "Now heading East"),
        (90, "Continuing East"),
        (135, "Turning right toward SE"),
        (180, "Now heading South"),
        (180, "Continuing South"),
        (225, "Turning right toward SW"),
        (270, "Now heading West"),
        (315, "Turning right toward NW"),
        (0, "Back to heading North"),
    ]

    # Accumulated path
    all_path = []
    current_lat, current_lon = lat, lon

    images = []

    for i, (heading, description) in enumerate(route):
        print(f"\nFrame {i+1}: {description} (heading {heading}°)")

        # Move forward a bit in the heading direction
        if i > 0:
            heading_rad = math.radians(heading)
            current_lat += 0.0003 * math.cos(heading_rad)
            current_lon += 0.0003 * math.sin(heading_rad)

        all_path.append((current_lat, current_lon))

        # Create renderer with accumulated path
        renderer = MapRenderer(
            scale=1.5,
            history=all_path.copy(),
            map_style="simple",  # Use simple for faster iteration
            heading_up=True
        )

        map_arr = renderer.render(heading, current_lat, current_lon)
        img = Image.fromarray(map_arr)

        # Add annotations
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
        except:
            font = ImageFont.load_default()

        draw.text((10, 10), f"Frame {i+1}", fill=COLORS.CT_ORANGE, font=font)
        draw.text((10, 26), f"Heading: {heading}°", fill=COLORS.STEEL_BRIGHT, font=font)
        draw.text((10, 42), description, fill=COLORS.STEEL_MID, font=font)
        draw.text((10, img.height - 20), "Path = where you came from",
                  fill=COLORS.CT_ORANGE_DIM, font=font)

        images.append(img)

        filename = f"{output_dir}/drive_frame_{i:02d}.png"
        img.save(filename)

    # Create animated GIF
    gif_path = f"{output_dir}/simulated_drive.gif"
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=1000,  # 1 second per frame
        loop=0
    )
    print(f"\n✅ Animated GIF saved: {gif_path}")
    print("Open this GIF to see the driving simulation!")


def animate_rotation(lat: float, lon: float, output_dir: str = "map_test_output",
                     map_style: str = "satellite"):
    """
    Create an animation showing 360° rotation at a fixed position.

    This helps verify the rotation is smooth and intuitive.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nGenerating 360° rotation animation at ({lat}, {lon})...")
    print(f"Map style: {map_style}")

    images = []

    # Generate frames every 15° for 24 frames
    for heading in range(0, 360, 15):
        img = render_single_map(lat, lon, heading, map_style=map_style)
        images.append(img)
        print(f"  Frame: {heading}°")

    # Create GIF
    gif_path = f"{output_dir}/rotation_360_{map_style}.gif"
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=200,  # 200ms per frame = ~5 seconds total
        loop=0
    )
    print(f"\n✅ Rotation animation saved: {gif_path}")

    # Also create a static comparison grid
    create_rotation_grid(lat, lon, output_dir, map_style)


def create_rotation_grid(lat: float, lon: float, output_dir: str,
                         map_style: str = "satellite"):
    """Create a 3x3 grid showing different headings."""

    headings = [0, 45, 90, 135, 180, 225, 270, 315, 0]  # Full circle back to 0

    # Render each heading
    imgs = []
    for h in headings:
        img = render_single_map(lat, lon, h, map_style=map_style, scale=1.0)
        imgs.append(img)

    # Create 3x3 grid
    w, h = imgs[0].size
    grid = Image.new('RGB', (w * 3, h * 3), COLORS.VOID_BLACK)

    for i, img in enumerate(imgs):
        row = i // 3
        col = i % 3
        grid.paste(img, (col * w, row * h))

    filename = f"{output_dir}/rotation_grid_{map_style}.png"
    grid.save(filename)
    print(f"✅ Rotation grid saved: {filename}")


def replay_real_gps(json_path: str, output_dir: str = "map_test_output",
                    map_style: str = "satellite", frame_skip: int = 100,
                    max_frames: int = 50):
    """
    Replay real GPS data from extracted JSON as an animation.

    Args:
        json_path: Path to extracted GPS JSON file
        output_dir: Output directory for images
        map_style: Map style to use
        frame_skip: Sample every Nth frame (dashcam is 36fps, so 100 = ~3 sec)
        max_frames: Maximum frames in output animation
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nLoading GPS data from: {json_path}")
    gps_data = load_gps_json(json_path)

    track = gps_data.get('gps_track', [])
    if not track:
        print("No GPS track found in JSON!")
        return

    print(f"Total GPS points: {len(track)}")
    print(f"Center: ({gps_data['center']['lat']:.6f}, {gps_data['center']['lon']:.6f})")
    print(f"Sampling every {frame_skip} frames, max {max_frames} output frames")

    # Sample the track
    sampled_indices = list(range(0, len(track), frame_skip))[:max_frames]
    print(f"Will render {len(sampled_indices)} frames")

    # Build path incrementally
    path = []
    images = []

    for i, idx in enumerate(sampled_indices):
        point = track[idx]
        lat, lon = point['lat'], point['lon']
        heading = point['heading']
        speed = point['speed_mph']

        # Add to path
        path.append((lat, lon))

        print(f"  Frame {i+1}/{len(sampled_indices)}: "
              f"({lat:.5f}, {lon:.5f}) heading {heading:.0f}° @ {speed:.0f}mph")

        # Create renderer with path so far
        renderer = MapRenderer(
            scale=1.5,
            history=path.copy(),
            map_style=map_style,
            heading_up=True
        )

        map_arr = renderer.render(heading, lat, lon)
        img = Image.fromarray(map_arr)

        # Add annotations
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
            small_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 10)
        except:
            font = ImageFont.load_default()
            small_font = font

        # Heading and speed
        draw.text((10, 10), f"Heading: {heading:.0f}°", fill=COLORS.CT_ORANGE, font=font)
        draw.text((10, 26), f"Speed: {speed:.0f} mph", fill=COLORS.STEEL_BRIGHT, font=font)

        # Coordinates
        draw.text((10, img.height - 20), f"({lat:.5f}, {lon:.5f})",
                  fill=COLORS.STEEL_MID, font=small_font)

        # Frame counter
        draw.text((img.width - 70, 10), f"Frame {idx}",
                  fill=COLORS.STEEL_DARK, font=small_font)

        images.append(img)

    # Save individual frames
    for i, img in enumerate(images):
        filename = f"{output_dir}/real_gps_frame_{i:03d}.png"
        img.save(filename)

    # Create animated GIF
    gif_path = f"{output_dir}/real_gps_replay.gif"
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=200,  # 200ms per frame
        loop=0
    )
    print(f"\n✅ Animated GIF saved: {gif_path}")
    print(f"   {len(images)} frames at 200ms each = {len(images) * 0.2:.1f} seconds")


def test_with_real_gps(json_path: str, output_dir: str = "map_test_output"):
    """
    Test map rendering with a few samples from real GPS data.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nLoading GPS data from: {json_path}")
    gps_data = load_gps_json(json_path)

    track = gps_data.get('gps_track', [])
    if not track:
        print("No GPS track found!")
        return

    # Sample points at different headings
    print(f"\nFinding sample points with varied headings...")

    # Group by heading quadrant
    by_quadrant = {
        'N (315-45)': [],
        'E (45-135)': [],
        'S (135-225)': [],
        'W (225-315)': [],
    }

    for i, point in enumerate(track):
        h = point['heading']
        if h >= 315 or h < 45:
            by_quadrant['N (315-45)'].append((i, point))
        elif 45 <= h < 135:
            by_quadrant['E (45-135)'].append((i, point))
        elif 135 <= h < 225:
            by_quadrant['S (135-225)'].append((i, point))
        else:
            by_quadrant['W (225-315)'].append((i, point))

    print("\nHeading distribution:")
    for quadrant, points in by_quadrant.items():
        print(f"  {quadrant}: {len(points)} points")

    # Pick one from each quadrant
    samples = []
    for quadrant, points in by_quadrant.items():
        if points:
            # Pick one from the middle
            idx, point = points[len(points)//2]
            samples.append((quadrant, idx, point))

    print(f"\nRendering {len(samples)} sample points...")

    # Build full path for context
    full_path = [(p['lat'], p['lon']) for p in track]

    for quadrant, idx, point in samples:
        lat, lon = point['lat'], point['lon']
        heading = point['heading']
        speed = point['speed_mph']

        print(f"\n  {quadrant}:")
        print(f"    Position: ({lat:.5f}, {lon:.5f})")
        print(f"    Heading: {heading:.1f}°")
        print(f"    Speed: {speed:.1f} mph")

        # Use path up to this point
        path_so_far = [(p['lat'], p['lon']) for p in track[:idx+1]]

        for style in ["simple", "satellite"]:
            renderer = MapRenderer(
                scale=1.5,
                history=path_so_far,
                map_style=style,
                heading_up=True
            )

            map_arr = renderer.render(heading, lat, lon)
            img = Image.fromarray(map_arr)

            # Annotate
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
            except:
                font = ImageFont.load_default()

            draw.text((10, 10), f"Heading: {heading:.0f}° ({quadrant})",
                      fill=COLORS.CT_ORANGE, font=font)
            draw.text((10, 26), f"Speed: {speed:.0f} mph",
                      fill=COLORS.STEEL_BRIGHT, font=font)

            safe_quadrant = quadrant.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
            filename = f"{output_dir}/real_gps_{safe_quadrant}_{style}.png"
            img.save(filename)
            print(f"    Saved: {filename}")


def verify_intuitive_rotation():
    """
    Print explanation of what intuitive rotation means.
    """
    print("""
╔════════════════════════════════════════════════════════════════╗
║           WHAT "INTUITIVE ROTATION" MEANS                      ║
╠════════════════════════════════════════════════════════════════╣
║                                                                ║
║  In HEADING-UP mode (like car GPS navigation):                ║
║                                                                ║
║  1. YOUR FORWARD DIRECTION is always UP on the screen         ║
║     - The arrow in the center always points UP                 ║
║     - This represents "where you're going"                     ║
║                                                                ║
║  2. THE MAP ROTATES around you                                 ║
║     - When you turn RIGHT, the world rotates LEFT             ║
║     - When you turn LEFT, the world rotates RIGHT             ║
║     - This matches what you see out the windshield!            ║
║                                                                ║
║  3. THE "N" COMPASS shows where NORTH is                       ║
║     - Heading 0° (North): N is at TOP                         ║
║     - Heading 90° (East): N is on the LEFT                    ║
║     - Heading 180° (South): N is at BOTTOM                    ║
║     - Heading 270° (West): N is on the RIGHT                  ║
║                                                                ║
║  4. YOUR PATH (orange line) shows where you CAME FROM         ║
║     - Always extends DOWNWARD from center                      ║
║     - This is your trail, showing your history                 ║
║                                                                ║
║  VERIFICATION: Run --simulate-drive and watch the GIF.        ║
║  The path should always trail BEHIND you (downward).           ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
""")


def interactive_test(initial_lat: float = 37.7749, initial_lon: float = -122.4194):
    """Run an interactive test with keyboard controls."""
    try:
        import tkinter as tk
        from PIL import ImageTk
    except ImportError:
        print("tkinter not available - use command-line tests instead")
        print("Try: python map_prototype.py --simulate-drive")
        return

    class MapTester:
        def __init__(self, root):
            self.root = root
            self.root.title("Map Rotation Tester - Use Arrow Keys")

            self.lat = initial_lat
            self.lon = initial_lon
            self.heading = 0
            self.style = "satellite"
            self.path = [(self.lat, self.lon)]

            # Instructions
            instr = tk.Label(root, text=(
                "← → : Rotate heading | ↑ : Move forward | "
                "S/T : Toggle style | R : Reset"
            ), bg='#1a1a1a', fg='#888888')
            instr.pack(fill=tk.X)

            # Canvas for map
            self.canvas = tk.Canvas(root, width=450, height=450, bg='#0a0a0a')
            self.canvas.pack()

            # Status
            self.status_var = tk.StringVar()
            self.status = tk.Label(root, textvariable=self.status_var,
                                   bg='#1a1a1a', fg='#ff6400', font=('Helvetica', 12))
            self.status.pack(fill=tk.X)

            # Bind keys
            root.bind('<Left>', lambda e: self.rotate(-15))
            root.bind('<Right>', lambda e: self.rotate(15))
            root.bind('<Up>', lambda e: self.move_forward())
            root.bind('<Down>', lambda e: self.move_backward())
            root.bind('s', lambda e: self.toggle_style())
            root.bind('t', lambda e: self.toggle_style())
            root.bind('r', lambda e: self.reset())
            root.bind('<space>', lambda e: self.move_forward())

            self.photo = None
            self.render_map()

        def rotate(self, delta):
            self.heading = (self.heading + delta) % 360
            self.render_map()

        def move_forward(self):
            heading_rad = math.radians(self.heading)
            self.lat += 0.0002 * math.cos(heading_rad)
            self.lon += 0.0002 * math.sin(heading_rad)
            self.path.append((self.lat, self.lon))
            if len(self.path) > 100:
                self.path = self.path[-100:]
            self.render_map()

        def move_backward(self):
            heading_rad = math.radians(self.heading)
            self.lat -= 0.0001 * math.cos(heading_rad)
            self.lon -= 0.0001 * math.sin(heading_rad)
            self.render_map()

        def toggle_style(self):
            styles = ["simple", "satellite", "street"]
            idx = (styles.index(self.style) + 1) % len(styles)
            self.style = styles[idx]
            self.render_map()

        def reset(self):
            self.lat = initial_lat
            self.lon = initial_lon
            self.heading = 0
            self.path = [(self.lat, self.lon)]
            self.render_map()

        def render_map(self):
            renderer = MapRenderer(
                scale=1.5,
                history=self.path.copy(),
                map_style=self.style,
                heading_up=True
            )

            try:
                map_arr = renderer.render(self.heading, self.lat, self.lon)
                img = Image.fromarray(map_arr)

                # Add annotations
                draw = ImageDraw.Draw(img)
                try:
                    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
                except:
                    font = ImageFont.load_default()

                draw.text((10, 10), f"Heading: {self.heading}°",
                          fill=COLORS.CT_ORANGE, font=font)
                draw.text((10, 28), f"Style: {self.style}",
                          fill=COLORS.STEEL_BRIGHT, font=font)

                self.photo = ImageTk.PhotoImage(img)
                self.canvas.delete("all")
                self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

                # Update status
                cardinals = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
                cardinal_idx = int((self.heading + 22.5) / 45) % 8
                self.status_var.set(
                    f"Heading {self.heading}° ({cardinals[cardinal_idx]}) | "
                    f"({self.lat:.5f}, {self.lon:.5f})"
                )

            except Exception as e:
                print(f"Render error: {e}")

    root = tk.Tk()
    root.configure(bg='#1a1a1a')
    app = MapTester(root)
    print("\nInteractive mode started!")
    print("Use arrow keys to rotate (←→) and move forward (↑)")
    print("Press 'S' to toggle map style, 'R' to reset")
    root.mainloop()


def main():
    parser = argparse.ArgumentParser(
        description="Map rendering prototype for testing heading-up rotation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test at specific GPS coordinates
  python map_prototype.py --lat 37.7749 --lon -122.4194 --heading 90

  # Use real GPS data from extracted JSON (samples 4 directions)
  python map_prototype.py --from-json sample_gps_data.json

  # Replay real GPS data as animated GIF
  python map_prototype.py --from-json sample_gps_data.json --replay

  # Watch a simulated drive (creates animated GIF)
  python map_prototype.py --simulate-drive

  # Create 360° rotation animation
  python map_prototype.py --animate-rotation --lat 37.7749 --lon -122.4194

  # Interactive testing with keyboard
  python map_prototype.py --interactive

  # Explain what intuitive rotation means
  python map_prototype.py --explain
"""
    )

    parser.add_argument("--lat", type=float, default=37.7749,
                        help="Latitude (default: 37.7749 - San Francisco)")
    parser.add_argument("--lon", type=float, default=-122.4194,
                        help="Longitude (default: -122.4194)")
    parser.add_argument("--heading", type=float, default=0,
                        help="Heading in degrees, 0=North, 90=East (default: 0)")
    parser.add_argument("--style", choices=["simple", "satellite", "street"],
                        default="satellite", help="Map style (default: satellite)")

    # Real GPS data options
    parser.add_argument("--from-json", type=str, metavar="FILE",
                        help="Load GPS data from extracted JSON file")
    parser.add_argument("--replay", action="store_true",
                        help="Replay GPS data as animation (use with --from-json)")
    parser.add_argument("--frame-skip", type=int, default=100,
                        help="Sample every Nth frame for replay (default: 100)")
    parser.add_argument("--max-frames", type=int, default=50,
                        help="Maximum frames in replay animation (default: 50)")

    parser.add_argument("--simulate-drive", action="store_true",
                        help="Run simulated drive animation")
    parser.add_argument("--animate-rotation", action="store_true",
                        help="Create 360° rotation animation")
    parser.add_argument("--interactive", action="store_true",
                        help="Interactive mode with keyboard controls")
    parser.add_argument("--explain", action="store_true",
                        help="Explain what intuitive rotation means")
    parser.add_argument("--all", action="store_true",
                        help="Run all tests")

    parser.add_argument("--output-dir", default="map_test_output",
                        help="Output directory for images (default: map_test_output)")

    args = parser.parse_args()

    if args.explain:
        verify_intuitive_rotation()
        return

    if args.interactive:
        interactive_test(args.lat, args.lon)
        return

    # Handle real GPS data
    if args.from_json:
        if not os.path.exists(args.from_json):
            print(f"Error: JSON file not found: {args.from_json}")
            print("Run: python extract_sample_gps.py ./data/YOUR_DASHCAM_DIR")
            sys.exit(1)

        if args.replay:
            replay_real_gps(args.from_json, args.output_dir, args.style,
                           args.frame_skip, args.max_frames)
        else:
            test_with_real_gps(args.from_json, args.output_dir)
        return

    if args.simulate_drive or args.all:
        simulate_drive(args.output_dir)

    if args.animate_rotation or args.all:
        animate_rotation(args.lat, args.lon, args.output_dir, args.style)

    # If just coordinates provided, render single map
    if not args.simulate_drive and not args.animate_rotation and not args.all:
        test_single_location(args.lat, args.lon, args.heading, args.output_dir)

        # Also show verification info
        print("\n" + "-" * 50)
        print("VERIFICATION CHECKLIST:")
        print("-" * 50)
        print(f"✓ Heading {args.heading}° means you're facing ", end="")
        cardinals = ["North", "NE", "East", "SE", "South", "SW", "West", "NW"]
        cardinal_idx = int((args.heading + 22.5) / 45) % 8
        print(cardinals[cardinal_idx])
        print("✓ The arrow in the center should point UP (your forward direction)")
        print("✓ The orange path should extend DOWNWARD (where you came from)")

        if args.heading == 0:
            print("✓ N compass should be at TOP (you're facing North)")
        elif args.heading == 90:
            print("✓ N compass should be on LEFT (you're facing East, North is to your left)")
        elif args.heading == 180:
            print("✓ N compass should be at BOTTOM (you're facing South)")
        elif args.heading == 270:
            print("✓ N compass should be on RIGHT (you're facing West)")

        print("\nRun with --explain for detailed rotation explanation")
        print("Run with --simulate-drive for animated verification")


if __name__ == "__main__":
    main()
