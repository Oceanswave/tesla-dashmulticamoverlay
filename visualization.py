
import cv2
import numpy as np
import math
from typing import List, Tuple, Optional
import dashcam_pb2

class DashboardRenderer:
    def __init__(self, width: int = 400, height: int = 200, bg_color: Tuple[int, int, int] = (0, 0, 0)):
        self.width = width
        self.height = height
        self.bg_color = bg_color
        
        # Colors
        self.white = (255, 255, 255)
        self.green = (0, 255, 0)
        self.red = (0, 0, 255)
        self.blue = (255, 0, 0)
        self.grey = (100, 100, 100)
    
    def render(self, meta: dashcam_pb2.SeiMetadata) -> np.ndarray:
        # Create canvas with alpha channel support concept in mind, but returning BGR for now
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        if self.bg_color != (0, 0, 0):
            canvas[:] = self.bg_color

        # 1. Speedometer (Left side)
        speed_mph = meta.vehicle_speed_mps * 2.23694
        self._draw_gauge(canvas, center=(70, 100), radius=50, value=speed_mph, max_value=120, label="MPH")
        
        # 2. Steering Wheel (Center)
        self._draw_steering(canvas, center=(200, 100), angle=meta.steering_wheel_angle)
        
        # 3. Pedals (Right side)
        self._draw_pedals(canvas, center=(330, 100), accel=meta.accelerator_pedal_position, brake=meta.brake_applied)

        # 4. Status Text (Bottom)
        self._draw_status(canvas, meta)

        return canvas

    def _draw_gauge(self, img, center, radius, value, max_value, label):
        # Background arc
        start_angle = 135
        end_angle = 405
        cv2.ellipse(img, center, (radius, radius), 0, start_angle, end_angle, self.grey, 2)
        
        # Value arc
        val_angle = start_angle + (min(value, max_value) / max_value) * (end_angle - start_angle)
        cv2.ellipse(img, center, (radius, radius), 0, start_angle, val_angle, self.white, 3)
        
        # Text
        cv2.putText(img, f"{int(value)}", (center[0]-30, center[1]+10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, self.white, 2, cv2.LINE_AA)
        cv2.putText(img, label, (center[0]-15, center[1]+30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.grey, 1, cv2.LINE_AA)

    def _draw_steering(self, img, center, angle):
        # Simple circle with a cross bar
        radius = 40
        thickness = 4
        
        # Draw wheel rim
        cv2.circle(img, center, radius, self.grey, thickness)
        
        # Rotate a line for the steering angle
        # 0 degrees is straight? check calibration. Assuming typical automotive standard.
        # Convert to radians. usually + is left (CCW) or right? Telemetry varies.
        # Let's assume standard math: CCW. Visual check needed.
        rad = math.radians(angle)
        
        # Draw "spokes"
        # Center to Top (rotated)
        pt1 = (int(center[0] + radius * math.sin(rad)), int(center[1] - radius * math.cos(rad)))
        pt2 = (int(center[0] - radius * math.sin(rad)), int(center[1] + radius * math.cos(rad)))
        cv2.line(img, pt1, pt2, self.grey, thickness)
        
        # Horizontal bar
        pt3 = (int(center[0] + radius * math.cos(rad)), int(center[1] + radius * math.sin(rad)))
        pt4 = (int(center[0] - radius * math.cos(rad)), int(center[1] - radius * math.sin(rad)))
        cv2.line(img, pt3, pt4, self.grey, thickness)


    def _draw_pedals(self, img, center, accel, brake):
        # Accel bar
        bar_w = 20
        bar_h = 80
        x_accel = center[0] + 10
        y_bot = center[1] + bar_h // 2
        
        # Backgrounds
        cv2.rectangle(img, (x_accel, y_bot - bar_h), (x_accel + bar_w, y_bot), self.grey, 1)
        
        # Fill
        h_accel = int((accel / 100.0) * bar_h)
        if h_accel > 0:
            cv2.rectangle(img, (x_accel, y_bot - h_accel), (x_accel + bar_w, y_bot), self.green, -1)
        
        # Brake indicator box next to it
        x_brake = center[0] - 30
        cv2.rectangle(img, (x_brake, y_bot - bar_h), (x_brake + bar_w, y_bot), self.grey, 1)
        if brake:
             cv2.rectangle(img, (x_brake, y_bot - bar_h), (x_brake + bar_w, y_bot), self.red, -1)
             
        cv2.putText(img, "P", (x_accel+2, y_bot+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.grey, 1)
        cv2.putText(img, "B", (x_brake+2, y_bot+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.grey, 1)

    def _draw_status(self, img, meta):
        # Gear and AP state
        gear_map = {0:"P", 1:"D", 2:"R", 3:"N"}
        gear = gear_map.get(meta.gear_state, "?")
        
        ap_map = {0: "Manual", 1: "FSD", 2: "Autosteer", 3: "TACC"}
        ap = ap_map.get(meta.autopilot_state, "Unknown")
        
        text = f"GEAR: {gear}  |  MODE: {ap}"
        
        # Centered text
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        text_x = (self.width - text_size[0]) // 2
        
        cv2.putText(img, text, (text_x, self.height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.white, 1, cv2.LINE_AA)


class MapRenderer:
    def __init__(self, size: int = 300, history: List[Tuple[float, float]] = None):
        self.size = size
        self.path: List[Tuple[float, float]] = history if history else [] # (lat, lon)
        self.padding = 0.1 
        self.zoom_window = 0.002 # Degrees (~200m) window around current pos
        
        # Colors
        self.white = (255, 255, 255)
        self.green = (0, 255, 0)
        self.red = (0, 0, 255)

    def update(self, lat: float, lon: float):
        if abs(lat) < 0.1 and abs(lon) < 0.1:
            return  # Filter 0,0 GPS errors
        self.path.append((lat, lon))

    def render(self, heading_deg: float, current_lat: float, current_lon: float) -> np.ndarray:
        img = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        img[:] = (30, 30, 30) 
        
        if not self.path:
            return img

        # Zoomed Window
        min_lat = current_lat - self.zoom_window
        max_lat = current_lat + self.zoom_window
        min_lon = current_lon - self.zoom_window
        max_lon = current_lon + self.zoom_window
        
        lat_range = max_lat - min_lat
        lon_range = max_lon - min_lon
        
        scale_lat = (self.size * (1 - 2*self.padding)) / lat_range
        scale_lon = (self.size * (1 - 2*self.padding)) / lon_range
        scale = min(scale_lat, scale_lon) 
        
        # Center is current position
        center_lat = current_lat
        center_lon = current_lon
        
        def to_xy(lat, lon):
            # Clamp or just draw off-canvas? Draw all, simpler logic.
            x = int(self.size/2 + (lon - center_lon) * scale)
            y = int(self.size/2 - (lat - center_lat) * scale)
            return x, y

        # Optimize: only draw points relevant to window?
        # For simplicity, filter approx
        relevant_path = [p for p in self.path if abs(p[0]-current_lat) < 0.01 and abs(p[1]-current_lon) < 0.01]
        
        if len(relevant_path) > 1:
            pts = np.array([to_xy(lat, lon) for lat, lon in relevant_path], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(img, [pts], False, (0, 255, 255), 1, cv2.LINE_AA)
        
        # Draw current position arrow (Centered)
        curr = (self.size//2, self.size//2)
        arrow_len = 15
        
        dx = int(arrow_len * math.sin(math.radians(heading_deg)))
        dy = int(arrow_len * -math.cos(math.radians(heading_deg)))
        
        cv2.arrowedLine(img, (curr[0] - dx, curr[1] - dy), (curr[0] + dx, curr[1] + dy), (0, 0, 255), 2, cv2.LINE_AA, tipLength=0.5)
        
        # Add compass N
        cv2.putText(img, "N", (self.size - 20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.white, 1)

        return img


def composite_frame(front: np.ndarray, 
                    left_rep: Optional[np.ndarray] = None, 
                    right_rep: Optional[np.ndarray] = None, 
                    back: Optional[np.ndarray] = None,
                    left_pill: Optional[np.ndarray] = None,
                    right_pill: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Layout: 1920x1080
    
    Top Section (Height 600):
      [ Left Pillar ] [    Front Camera    ] [ Right Pillar ]
      Widths: 560px         800px              560px
      
    Bottom Section (Height 480):
      [ Left Repeater ] [      Back      ] [ Right Repeater ]
      Widths: 640px         640px            640px
    """
    canvas = np.zeros((1080, 1920, 3), dtype=np.uint8)
    
    # --- Dimensions ---
    top_h = 600
    
    # Front Cam (Center Top)
    front_w = 800
    front_x = (1920 - front_w) // 2
    
    # Side Pillars (Flanking Top)
    # Available width is front_x = 560.
    # Pillars are typically portrait-ish or 4:3. 
    # If we fit to width 560, height is 560*0.75 = 420.
    pillar_w = 560
    pillar_h = 420
    pillar_y = (top_h - pillar_h) // 2
    
    # Bottom Row
    bot_y = 600
    bot_h = 480
    bot_w = 640 # 1920 / 3

    # --- Draw Top Row ---
    
    # 1. Front
    if front is not None:
        resized = cv2.resize(front, (front_w, top_h))
        canvas[0:top_h, front_x:front_x+front_w] = resized
        
    # 2. Left Pillar
    if left_pill is not None:
        resized = cv2.resize(left_pill, (pillar_w, pillar_h))
        canvas[pillar_y:pillar_y+pillar_h, 0:pillar_w] = resized
        cv2.putText(canvas, "L. Pillar", (20, pillar_y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        # Placeholder
        cv2.rectangle(canvas, (0, pillar_y), (pillar_w, pillar_y+pillar_h), (20,20,20), -1)
        
    # 3. Right Pillar
    rx = front_x + front_w # 1360
    if right_pill is not None:
        resized = cv2.resize(right_pill, (pillar_w, pillar_h))
        canvas[pillar_y:pillar_y+pillar_h, rx:rx+pillar_w] = resized
        cv2.putText(canvas, "R. Pillar", (rx+20, pillar_y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
         cv2.rectangle(canvas, (rx, pillar_y), (rx+pillar_w, pillar_y+pillar_h), (20,20,20), -1)

    # --- Draw Bottom Row ---
    # Left Rep | Back | Right Rep
    bottom_cams = [
        (left_rep, "L. Repeater"),
        (back, "Back"),
        (right_rep, "R. Repeater")
    ]
    
    for i, (cam, label) in enumerate(bottom_cams):
        bx = i * bot_w
        
        if cam is not None:
            # Mirror Back cam (index 1 in this list)
            if i == 1:
                cam = cv2.flip(cam, 1)
                
            resized = cv2.resize(cam, (bot_w, bot_h))
            canvas[bot_y:bot_y+bot_h, bx:bx+bot_w] = resized
            cv2.putText(canvas, label, (bx+20, bot_y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            cv2.rectangle(canvas, (bx, bot_y), (bx+bot_w, bot_y+bot_h), (20,20,20), -1)
            cv2.putText(canvas, label, (bx+20, bot_y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 1)

    return canvas
