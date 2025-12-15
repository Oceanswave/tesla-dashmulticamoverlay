#!/usr/bin/env python3
import argparse
import sys
import cv2
import os
import glob
import subprocess
import multiprocessing
from typing import Optional, List, Tuple
from dataclasses import dataclass
from pydantic import BaseModel, Field

from sei_parser import extract_sei_data
from visualization import DashboardRenderer, MapRenderer, composite_frame

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
    font_scale: float = Field(default=1.0, ge=0.1)

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

def parse_args() -> VideoConfig:
    parser = argparse.ArgumentParser(description="Burn Tesla SEI metadata into connected video clips.")
    parser.add_argument("input_path", help="Path to input MP4 file or directory of clips")
    parser.add_argument("output_file", help="Path to output MP4 file")
    parser.add_argument("--font-scale", type=float, default=1.0, help="Font scale for overlay text")
    
    args = parser.parse_args()
    
    try:
        playlist = discover_clips(args.input_path)
        if not playlist:
            print("No clips found!")
            sys.exit(1)
            
        print(f"Found {len(playlist)} clips to process.")
        
        return VideoConfig(
            playlist=playlist,
            output_file=args.output_file,
            font_scale=args.font_scale
        )
    except Exception as e:
        print(f"Configuration Error: {e}")
        sys.exit(1)

# Worker function must be at module level for multiprocessing
def process_clip_task(clip: ClipSet, output_temp: str, font_scale: float, history: List[Tuple[float, float]]):
    """
    Worker to process a single clip.
    """
    # Extract SEI
    sei_data = extract_sei_data(clip.front)
    
    # Open Caps
    caps = {}
    caps['front'] = cv2.VideoCapture(clip.front)
    if clip.left_rep: caps['left_rep'] = cv2.VideoCapture(clip.left_rep)
    if clip.right_rep: caps['right_rep'] = cv2.VideoCapture(clip.right_rep)
    if clip.back: caps['back'] = cv2.VideoCapture(clip.back)
    if clip.left_pill: caps['left_pill'] = cv2.VideoCapture(clip.left_pill)
    if clip.right_pill: caps['right_pill'] = cv2.VideoCapture(clip.right_pill)
    
    width = 1920
    height = 1080
    fps = caps['front'].get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30.0 # Fallback
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_temp, fourcc, fps, (width, height))
    
    # Initialize Renderers
    dash_renderer = DashboardRenderer()
    map_renderer = MapRenderer(history=list(history)) # Copy history
    
    frame_idx = 0
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
        
        canvas = composite_frame(
            front=frames['front'], 
            left_rep=frames.get('left_rep'), 
            right_rep=frames.get('right_rep'), 
            back=frames.get('back'),
            left_pill=frames.get('left_pill'), 
            right_pill=frames.get('right_pill')
        )
        
        if meta:
            map_renderer.update(meta.latitude_deg, meta.longitude_deg)
            dash_img = dash_renderer.render(meta)
            map_img = map_renderer.render(meta.heading_deg, meta.latitude_deg, meta.longitude_deg)
            
            # Dashboard Overlay (Top Center of Screen/Front Cam)
            dh, dw = dash_img.shape[:2]
            dy = 20
            dx = (1920 - dw) // 2
            
            if dy >= 0 and dx >= 0 and dy+dh <= 1080 and dx+dw <= 1920:
                 roi = canvas[dy:dy+dh, dx:dx+dw]
                 dst = cv2.addWeighted(roi, 0.2, dash_img, 0.8, 0)
                 canvas[dy:dy+dh, dx:dx+dw] = dst
            
            # Map Overlay (Top Right of Screen)
            mh, mw = map_img.shape[:2]
            my = 20
            mx = 1920 - mw - 20
            
            if my+mh <= 1080 and mx+mw <= 1920:
                roi_map = canvas[my:my+mh, mx:mx+mw]
                dst_map = cv2.addWeighted(roi_map, 0.2, map_img, 0.8, 0)
                canvas[my:my+mh, mx:mx+mw] = dst_map
            
        out.write(canvas)
        frame_idx += 1
        
    for cap in caps.values():
        cap.release()
    out.release()
    return output_temp

def extract_gps_points(clip: ClipSet) -> List[Tuple[float, float]]:
    """Quickly extract GPS points from a clip without rendering."""
    data = extract_sei_data(clip.front)
    points = []
    # Sort by frame index
    for idx in sorted(data.keys()):
        meta = data[idx]
        if abs(meta.latitude_deg) > 0.1:
            points.append((meta.latitude_deg, meta.longitude_deg))
    return points

def concat_clips(temp_files: List[str], output_file: str):
    print("Concatenating clips...")
    # Create list file
    list_path = "concat_list.txt"
    with open(list_path, "w") as f:
        for p in temp_files:
            abs_p = os.path.abspath(p)
            f.write(f"file '{abs_p}'\n")
            
    # Run ffmpeg with re-encoding to fix timestamp/seeking issues
    # Using libx264 for better compatibility than raw copy from opencv mp4v
    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", 
        "-i", list_path, 
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "copy", # Copy audio if any (likely none)
        output_file
    ]
    try:
        # Show ffmpeg output on error, but hide otherwise? 
        # Actually user might want to see progress or errors.
        # Let's verify if 'libx264' is available. Usually is.
        subprocess.run(cmd, check=True) # Allow stdout to show
        print(f"Successfully saved to {output_file}")
    except subprocess.CalledProcessError:
        print("Error: FFmpeg concatenation failed. Make sure ffmpeg is installed with libx264 support.")
        sys.exit(1)
    finally:
        # Cleanup
        if os.path.exists(list_path):
            os.remove(list_path)
            
import tempfile
from tqdm import tqdm

# Wrapper for imap since it only accepts one argument
def process_clip_wrapper(args):
    return process_clip_task(*args)

def main():
    config = parse_args()
    
    if not config.playlist:
        return
        
    print(f"Found {len(config.playlist)} clips.")

    # 1. Pre-scan GPS for history
    print("Step 1/3: Pre-scanning GPS data...")
    global_history = []
    clip_histories = [] # History snapshot for each clip START
    
    # Simple loop progress
    for clip in tqdm(config.playlist, desc="Scanning GPS", unit="clip"):
        clip_histories.append(list(global_history))
        points = extract_gps_points(clip)
        global_history.extend(points)
        
    print(f"Total GPS points: {len(global_history)}")
    
    # 2. Parallel Processing
    num_processes = min(multiprocessing.cpu_count(), len(config.playlist))
    print(f"Step 2/3: Rendering {len(config.playlist)} clips (Parallel Workers: {num_processes})...")
    
    # Use a temporary directory for intermediate files
    with tempfile.TemporaryDirectory() as temp_dir:
        # print(f"Using temporary directory: {temp_dir}")
        
        # Prepare args for all tasks
        args_list = []
        temp_files = []
        for i, clip in enumerate(config.playlist):
            temp_file = os.path.join(temp_dir, f"temp_clip_{i}.mp4")
            temp_files.append(temp_file)
            args_list.append((clip, temp_file, config.font_scale, clip_histories[i]))

        with multiprocessing.Pool(processes=num_processes) as pool:
            # Use imap to get results as they complete for the progress bar
            # We use the wrapper because imap only supports one argument
            results = list(tqdm(
                pool.imap(process_clip_wrapper, args_list), 
                total=len(args_list), 
                desc="Rendering", 
                unit="clip"
            ))
            
        # 3. Concatenate
        print("Step 3/3: Concatenating clips...")
        concat_clips(temp_files, config.output_file)
    
    print(f"\nDone! saved to {config.output_file}")

if __name__ == "__main__":
    main()
