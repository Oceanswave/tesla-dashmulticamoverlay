#!/usr/bin/env python3
"""
Extract GPS data from Tesla dashcam sample footage.

This script extracts GPS coordinates and headings from SEI metadata
embedded in Tesla dashcam videos, and exports them for use with the map prototype.
"""

import os
import sys
import json
import glob
from typing import List, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sei_parser import extract_sei_data


def extract_gps_from_clip(video_path: str) -> List[Dict[str, Any]]:
    """Extract GPS data from a single video clip.

    Returns list of dicts with frame_idx, lat, lon, heading, speed.
    """
    print(f"  Extracting: {os.path.basename(video_path)}")

    try:
        metadata = extract_sei_data(video_path)
    except Exception as e:
        print(f"    Error: {e}")
        return []

    gps_data = []
    for frame_idx, sei in sorted(metadata.items()):
        lat = sei.latitude_deg
        lon = sei.longitude_deg
        heading = sei.heading_deg
        speed = sei.vehicle_speed_mps

        # Filter out null island and invalid data
        if abs(lat) < 0.001 and abs(lon) < 0.001:
            continue
        if lat == 0 and lon == 0:
            continue

        gps_data.append({
            'frame': frame_idx,
            'lat': lat,
            'lon': lon,
            'heading': heading,
            'speed_mps': speed,
            'speed_mph': speed * 2.237,  # Convert m/s to mph
        })

    print(f"    Found {len(gps_data)} GPS points")
    return gps_data


def extract_from_directory(data_dir: str) -> Dict[str, Any]:
    """Extract GPS data from all front camera clips in a directory.

    Returns dict with metadata and combined GPS track.
    """
    # Find all front camera files
    front_files = sorted(glob.glob(os.path.join(data_dir, "*-front.mp4")))

    if not front_files:
        print(f"No front camera files found in {data_dir}")
        return {}

    print(f"Found {len(front_files)} front camera clips")

    all_gps = []
    clip_boundaries = []

    for video_path in front_files:
        clip_name = os.path.basename(video_path).replace("-front.mp4", "")
        clip_gps = extract_gps_from_clip(video_path)

        if clip_gps:
            clip_boundaries.append({
                'clip': clip_name,
                'start_idx': len(all_gps),
                'count': len(clip_gps),
            })
            all_gps.extend(clip_gps)

    if not all_gps:
        print("No GPS data found!")
        return {}

    # Calculate bounding box
    lats = [p['lat'] for p in all_gps]
    lons = [p['lon'] for p in all_gps]

    result = {
        'source_dir': data_dir,
        'total_points': len(all_gps),
        'clips': clip_boundaries,
        'bounds': {
            'min_lat': min(lats),
            'max_lat': max(lats),
            'min_lon': min(lons),
            'max_lon': max(lons),
        },
        'center': {
            'lat': sum(lats) / len(lats),
            'lon': sum(lons) / len(lons),
        },
        'gps_track': all_gps,
    }

    # Print summary
    print(f"\n{'='*60}")
    print("GPS EXTRACTION SUMMARY")
    print('='*60)
    print(f"Total GPS points: {len(all_gps)}")
    print(f"Center: ({result['center']['lat']:.6f}, {result['center']['lon']:.6f})")
    print(f"Bounds: lat [{result['bounds']['min_lat']:.6f}, {result['bounds']['max_lat']:.6f}]")
    print(f"        lon [{result['bounds']['min_lon']:.6f}, {result['bounds']['max_lon']:.6f}]")

    # Heading statistics
    headings = [p['heading'] for p in all_gps]
    print(f"Heading range: {min(headings):.1f}° to {max(headings):.1f}°")

    # Speed statistics
    speeds = [p['speed_mph'] for p in all_gps]
    print(f"Speed range: {min(speeds):.1f} to {max(speeds):.1f} mph")

    return result


def save_sample_data(gps_data: Dict[str, Any], output_path: str = "sample_gps_data.json"):
    """Save extracted GPS data to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(gps_data, f, indent=2)
    print(f"\n✅ Saved to: {output_path}")


def print_sample_points(gps_data: Dict[str, Any], count: int = 10):
    """Print sample GPS points for quick inspection."""
    track = gps_data.get('gps_track', [])

    print(f"\n{'='*60}")
    print(f"SAMPLE GPS POINTS (first {count})")
    print('='*60)
    print(f"{'Frame':>6} {'Latitude':>12} {'Longitude':>13} {'Heading':>8} {'Speed':>8}")
    print('-'*60)

    for point in track[:count]:
        print(f"{point['frame']:>6} {point['lat']:>12.6f} {point['lon']:>13.6f} "
              f"{point['heading']:>7.1f}° {point['speed_mph']:>7.1f}mph")

    if len(track) > count:
        print(f"... and {len(track) - count} more points")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Extract GPS data from Tesla dashcam footage")
    parser.add_argument("data_dir", nargs="?", default="./data/2026-01-09_11-55-49",
                        help="Directory containing dashcam clips")
    parser.add_argument("-o", "--output", default="sample_gps_data.json",
                        help="Output JSON file path")
    parser.add_argument("--sample", type=int, default=10,
                        help="Number of sample points to print")

    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        print(f"Error: Directory not found: {args.data_dir}")
        sys.exit(1)

    print(f"Extracting GPS data from: {args.data_dir}")
    print('='*60)

    gps_data = extract_from_directory(args.data_dir)

    if gps_data:
        print_sample_points(gps_data, args.sample)
        save_sample_data(gps_data, args.output)

        print(f"\n💡 To test with this data, run:")
        print(f"   python map_prototype.py --from-json {args.output}")


if __name__ == "__main__":
    main()
