import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from sort import Sort
import math
import tempfile
import os
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from collections import defaultdict
import time
from datetime import datetime
try:
    import easyocr
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

st.set_page_config(page_title="Auto Rickshaw Behavior Analysis", layout="wide")

st.title("🚗 Auto Rickshaw Behavior Analysis")
st.markdown("Upload a video to analyze vehicle behavior using YOLO + SORT tracking")

# Sidebar for configuration
st.sidebar.header("⚙️ Model Configuration")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.45, 0.05, 
                                        help="Higher values = more accurate detections (0.45 recommended for accuracy)")
max_age = st.sidebar.slider("Tracker Max Age", 1, 100, 30, 5,
                            help="Frames to keep track alive without detection (30 = balanced accuracy)")
min_hits = st.sidebar.slider("Tracker Min Hits", 1, 10, 3, 
                             help="Minimum consecutive detections before confirming track (3 = more accurate)")
iou_threshold = st.sidebar.slider("IOU Threshold", 0.0, 1.0, 0.25, 0.05,
                                 help="Overlap threshold for matching (0.25 = optimal accuracy)")

st.sidebar.header("🎬 Video Processing")
frame_skip = st.sidebar.slider("Frame Skip (for faster processing)", 0, 10, 0, 1, 
                                help="Skip frames to speed up processing. 0=process every frame (recommended for full speed)")
resize_factor = st.sidebar.slider("Resize Video", 0.25, 1.0, 0.75, 0.25,
                                  help="Reduce resolution for faster processing (0.75 recommended)")

st.sidebar.header("🎯 Accuracy Settings")
behavior_smoothing = st.sidebar.slider("Behavior Smoothing Window", 3, 20, 10, 1,
                                      help="Number of frames to average for behavior detection (10 = more accurate)")
speed_smoothing = st.sidebar.slider("Speed Smoothing Window", 3, 15, 5, 1,
                                   help="Number of frames to average for speed calculation (5 = responsive & accurate)")
min_track_length = st.sidebar.slider("Min Track Length for Analysis", 5, 40, 20, 1,
                                    help="Minimum frames before analyzing behavior (20 = more reliable)")

st.sidebar.header("📊 Display Options")
show_speed = st.sidebar.checkbox("Show Speed", value=True)
show_trajectory = st.sidebar.checkbox("Show Trajectory", value=True)
show_heatmap = st.sidebar.checkbox("Show Activity Heatmap", value=False)
show_license_plates = st.sidebar.checkbox("Detect License Plates (OCR)", value=OCR_AVAILABLE, disabled=not OCR_AVAILABLE)
if not OCR_AVAILABLE:
    st.sidebar.info("📝 Install easyocr for license plate detection: `pip install easyocr`")
show_lane_suggestions = st.sidebar.checkbox("Show Lane Change Suggestions", value=True)
pixels_to_meters = st.sidebar.number_input("Pixels to Meters Ratio", min_value=0.01, max_value=2.0, value=0.10, step=0.01, 
                                           help="Calibration factor. Adjust if speeds seem too high or too low.")
st.sidebar.info("💡 **Speed Calibration Guide**:\n"
               "- If speeds show ~5x too high: Use 0.02-0.05\n"
               "- If speeds are accurate: Keep 0.10\n"
               "- If speeds show ~2x too low: Use 0.20-0.30\n"
               "- Test with known vehicle speed to calibrate")

st.sidebar.header("🚨 Alert Settings")
speed_alert_threshold = st.sidebar.number_input("Speed Alert Threshold (km/h)", min_value=0, max_value=100, value=30, step=5)
enable_alerts = st.sidebar.checkbox("Enable Speed Alerts", value=True)

st.sidebar.markdown("---")
show_detailed_analytics = st.sidebar.checkbox("📊 Show Detailed Analytics", value=False, 
                                             help="Display comprehensive analytics in an expandable section after processing (collapsible)")

st.sidebar.header("🛣️ Lane Configuration")
num_lanes = st.sidebar.number_input("Number of Lanes", min_value=1, max_value=6, value=2, step=1,
                                   help="Total number of lanes in the road")
lane_margin = st.sidebar.slider("Lane Edge Margin (%)", 0, 20, 10, 1,
                               help="Safety margin from road edges (% of frame width)")

def calculate_angle(p1, p2):
    """Calculate angle in degrees between two points."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle = math.degrees(math.atan2(dy, dx))
    return angle

def is_facing_front(bbox, positions_history, min_history=10):
    """Detect if vehicle is facing front/camera based on bbox aspect ratio and movement."""
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    
    # Avoid division by zero
    if height == 0 or width == 0:
        return False
    
    aspect_ratio = width / height
    
    # Front-facing vehicles appear significantly wider (stricter threshold)
    # aspect_ratio > 1.5 means width is 50% more than height
    is_wide = aspect_ratio > 1.5
    
    # Check movement direction - must be moving significantly towards camera
    moving_towards_camera = False
    moving_predominantly_vertical = False
    
    if len(positions_history) >= min_history:
        recent_positions = positions_history[-min_history:]
        y_positions = [pos[1] for pos in recent_positions]
        x_positions = [pos[0] for pos in recent_positions]
        
        if len(y_positions) >= 2:
            y_delta = y_positions[-1] - y_positions[0]
            x_delta = abs(x_positions[-1] - x_positions[0])
            
            # Moving down more than 30 pixels over history (stricter)
            # AND vertical movement should be dominant (more than horizontal)
            if y_delta > 30:
                moving_towards_camera = True
                # Check if movement is predominantly vertical
                if y_delta > x_delta * 1.5:  # Vertical movement 1.5x more than horizontal
                    moving_predominantly_vertical = True
    
    # Front-facing ONLY if: wide aspect ratio AND moving towards camera AND predominantly vertical movement
    return is_wide and moving_towards_camera and moving_predominantly_vertical

def smooth_values(values, window_size):
    """Apply moving average smoothing to values."""
    if len(values) < window_size:
        return np.mean(values) if values else 0
    return np.mean(values[-window_size:])

def calculate_acceleration(speeds, timestamps):
    """Calculate acceleration from speed history."""
    if len(speeds) < 2 or len(timestamps) < 2:
        return 0
    
    speed_diff = speeds[-1] - speeds[-2]
    time_diff = timestamps[-1] - timestamps[-2]
    
    if time_diff > 0:
        return speed_diff / time_diff
    return 0

def analyze_behavior_advanced(speed_history, track_id, behavior_window, speed_window, min_length, bbox=None):
    """Advanced behavior analysis with multi-factor decision making."""
    
    if len(speed_history[track_id]['positions']) < min_length:
        return "Initializing", 0.0, (128, 128, 128)  # Gray for initializing
    
    # Get smoothed speed
    speeds = speed_history[track_id]['speeds']
    if not speeds:
        return "Unknown", 0.0, (128, 128, 128)
    
    current_speed = smooth_values(speeds, speed_window)
    
    # Calculate speed variance (indicates erratic movement)
    if len(speeds) >= behavior_window:
        speed_variance = np.std(speeds[-behavior_window:])
        recent_speeds = speeds[-behavior_window:]
    else:
        speed_variance = 0
        recent_speeds = speeds
    
    # Calculate acceleration
    timestamps = speed_history[track_id]['timestamps']
    acceleration = calculate_acceleration(speeds, timestamps)
    
    # Calculate direction changes
    positions = speed_history[track_id]['positions']
    direction_changes = 0
    angle_variance = 0
    turn_deltas = []
    dominant_turn = 0
    turn_intensity = 0
    
    if len(positions) >= behavior_window:
        angles = []
        start_idx = max(0, len(positions) - behavior_window)
        for i in range(start_idx, len(positions) - 1):
            angle = calculate_angle(positions[i], positions[i + 1])
            angles.append(angle)
        
        if len(angles) >= 2:
            angle_variance = np.std(angles)
            # Count significant direction changes (>45 degrees) and preserve turn direction sign
            for i in range(1, len(angles)):
                angle_diff = angles[i] - angles[i-1]
                while angle_diff > 180:
                    angle_diff -= 360
                while angle_diff < -180:
                    angle_diff += 360
                turn_deltas.append(angle_diff)
                if abs(angle_diff) > 45:  # Increased threshold
                    direction_changes += 1

        if turn_deltas:
            significant_turns = [delta for delta in turn_deltas if abs(delta) > 30]
            if significant_turns:
                dominant_turn = float(np.mean(significant_turns))
                turn_intensity = float(np.mean([abs(delta) for delta in significant_turns]))
            else:
                dominant_turn = float(np.mean(turn_deltas))
                turn_intensity = float(np.mean([abs(delta) for delta in turn_deltas]))
    
    # Calculate movement consistency
    movement_consistency = 0
    recent_pixel_movement = 0
    if len(positions) >= 5:
        recent_distances = []
        for i in range(len(positions) - 5, len(positions) - 1):
            if i >= 0:
                dx = positions[i+1][0] - positions[i][0]
                dy = positions[i+1][1] - positions[i][1]
                dist = np.sqrt(dx*dx + dy*dy)
                recent_distances.append(dist)
        
        movement_consistency = np.std(recent_distances) if recent_distances else 0
        recent_pixel_movement = np.mean(recent_distances) if recent_distances else 0
    
    # Behavior classification with improved thresholds
    behavior = "Normal"
    confidence = 0.0
    color = (0, 255, 0)
    
    # Check if vehicle is facing front (HIGH ALERT PRIORITY)
    facing_front = False
    if bbox is not None:
        facing_front = is_facing_front(bbox, positions)
    
    # Multi-factor behavior classification for maximum accuracy
    # CRITICAL: Front-facing detection (highest priority)
    if facing_front:
        behavior = "FRONT FACING - HIGH ALERT"
        confidence = 0.99
        color = (255, 0, 0)  # Bright red
    
    # Use ONLY pixel movement for stopped detection - but also check speed
    # Vehicle is stopped if: very low pixel movement AND very low speed AND sufficient observation time
    elif recent_pixel_movement < 0.5 and current_speed < 2 and len(positions) >= 15:
        behavior = "Stopped"
        confidence = 0.95
        color = (0, 0, 255)
    
    elif current_speed > 45:  # High speed threshold
        behavior = "SPEEDING"
        confidence = min(1.0, (current_speed - 45) / 25 * 0.98)
        color = (255, 0, 0)
    
    # Multi-factor erratic behavior detection
    elif direction_changes >= 2 and current_speed > 8 and max(turn_intensity, angle_variance) > 40:
        turn_score = (direction_changes / 4) + (turn_intensity / 80) + (angle_variance / 120)
        confidence = min(1.0, max(0.6, turn_score * 0.9))
        if dominant_turn > 10:
            behavior = "Sudden Right Turn"
            color = (255, 140, 0)
        elif dominant_turn < -10:
            behavior = "Sudden Left Turn"
            color = (255, 215, 0)
        else:
            behavior = "Sudden Turn"
            color = (0, 165, 255)
    
    elif angle_variance > 70 and current_speed > 10 and speed_variance > 5:
        behavior = "Weaving"
        confidence = min(1.0, (angle_variance / 100 + speed_variance / 15) / 2 * 0.88)
        color = (0, 255, 255)
    
    elif speed_variance > 10 and current_speed > 10 and movement_consistency > 8:
        behavior = "Erratic"
        confidence = min(1.0, (speed_variance / 15 + movement_consistency / 12) / 2 * 0.82)
        color = (0, 255, 255)
    
    elif current_speed > 25:  # Fast but not speeding
        behavior = "Fast"
        confidence = 0.7
        color = (255, 0, 255)
    
    elif abs(acceleration) > 5:  # Higher acceleration threshold
        if acceleration > 0:
            behavior = "Accelerating"
        else:
            behavior = "Braking"
        confidence = min(1.0, abs(acceleration) / 8 * 0.7)
        color = (255, 165, 0)
    
    elif current_speed > 10:
        behavior = "Fast"
        confidence = min(1.0, current_speed / 20 * 0.65)
        color = (255, 0, 255)
    
    elif current_speed > 3:  # Clear movement threshold
        behavior = "Normal"
        confidence = 0.8
        color = (0, 255, 0)
    
    elif current_speed > 1 or recent_pixel_movement > 0.5:  # Slow but moving
        behavior = "Slow"
        confidence = 0.6
        color = (100, 200, 100)
    
    # If speed is very low AND pixel movement is very low, then stopped (backup check)
    elif current_speed < 1 and recent_pixel_movement < 0.5:
        behavior = "Stopped"
        confidence = 0.7
        color = (0, 0, 255)
    
    return behavior, confidence, color

def check_lane_clear(current_track, all_tracks, frame_width, direction='left', num_lanes=2, margin_percent=10):
    """Check if adjacent lane exists and is clear for lane change suggestion."""
    cx, cy = current_track['position']
    
    # Calculate usable road width (excluding margins)
    margin = (margin_percent / 100.0) * frame_width
    road_start = margin
    road_end = frame_width - margin
    usable_width = road_end - road_start
    lane_width = usable_width / num_lanes
    
    # Determine which lane the vehicle is currently in
    if cx < road_start or cx > road_end:
        # Vehicle outside road boundaries
        return False
    
    relative_x = cx - road_start
    current_lane = int(relative_x / lane_width)
    
    # Check if target lane exists
    if direction == 'left':
        target_lane = current_lane - 1
        if target_lane < 0:  # No lane on left (at leftmost edge)
            return False
    else:  # right
        target_lane = current_lane + 1
        if target_lane >= num_lanes:  # No lane on right (at rightmost edge)
            return False
    
    # Calculate target lane boundaries
    target_lane_start = road_start + (target_lane * lane_width)
    target_lane_end = target_lane_start + lane_width
    
    # Check if any vehicle is in the target lane (with safety margin)
    safety_y_margin = 150  # pixels - vertical safety distance
    safety_x_buffer = lane_width * 0.2  # 20% buffer within lane
    
    for track in all_tracks:
        if track['id'] == current_track['id']:
            continue
        tx, ty = track['position']
        
        # Check if vehicle is in target lane with safety margins
        if (target_lane_start - safety_x_buffer <= tx <= target_lane_end + safety_x_buffer and 
            abs(ty - cy) < safety_y_margin):
            return False
    
    return True

def get_autonomous_action(behavior, speed_kmh, position, frame_width, frame_height, current_track=None, all_tracks=None, num_lanes=2, margin_percent=10):
    """Determine autonomous vehicle action based on detected behavior."""
    actions = {
        'primary': '',
        'secondary': [],
        'priority': 'LOW',
        'icon': '',
        'color': '',
        'lane_change': None
    }
    
    # Calculate lane position more accurately
    x_pos = position[0]
    margin = (margin_percent / 100.0) * frame_width
    road_start = margin
    road_end = frame_width - margin
    usable_width = road_end - road_start
    
    if x_pos < road_start or x_pos > road_end:
        zone = "EDGE"  # Vehicle at edge/outside road
    else:
        lane_width = usable_width / num_lanes
        relative_x = x_pos - road_start
        current_lane = int(relative_x / lane_width)
        
        if current_lane == 0:
            zone = "LEFT"
        elif current_lane == num_lanes - 1:
            zone = "RIGHT"
        else:
            zone = "CENTER"
    
    # Determine actions based on behavior
    if behavior == "FRONT FACING - HIGH ALERT":
        actions['primary'] = "🚨🚨 EMERGENCY STOP - COLLISION IMMINENT"
        actions['secondary'] = ["Immediate evasive maneuver", "Vehicle facing front!", "Sound horn", "Brake hard"]
        actions['priority'] = "CRITICAL"
        actions['icon'] = "🚨"
        actions['color'] = "#FF0000"
        
    elif behavior == "SPEEDING":
        actions['primary'] = "⚠️ EMERGENCY BRAKE"
        actions['secondary'] = ["Slow down immediately", "Increase safe distance", "Prepare to stop"]
        actions['priority'] = "CRITICAL"
        actions['icon'] = "🚨"
        actions['color'] = "#FF0000"
        
    elif behavior in ("Sudden Turn", "Sudden Left Turn", "Sudden Right Turn"):
        turn_side = None
        if behavior == "Sudden Left Turn":
            turn_side = "LEFT"
            actions['icon'] = "↩️"
            actions['color'] = "#FFD27F"
        elif behavior == "Sudden Right Turn":
            turn_side = "RIGHT"
            actions['icon'] = "↪️"
            actions['color'] = "#FF8C00"
        else:
            actions['icon'] = "⚠️"
            actions['color'] = "#FFA500"

        preferred_escape = None
        if turn_side == "LEFT":
            preferred_escape = "RIGHT"
        elif turn_side == "RIGHT":
            preferred_escape = "LEFT"

        if all_tracks and current_track:
            left_clear = check_lane_clear(current_track, all_tracks, frame_width, 'left', num_lanes, margin_percent)
            right_clear = check_lane_clear(current_track, all_tracks, frame_width, 'right', num_lanes, margin_percent)

            if preferred_escape and ((preferred_escape == "LEFT" and left_clear) or (preferred_escape == "RIGHT" and right_clear)):
                actions['primary'] = f"🚗 MOVE {preferred_escape} - LANE CLEAR"
                actions['lane_change'] = preferred_escape
            elif zone == "CENTER" and left_clear and right_clear:
                actions['primary'] = "🚗 EITHER LANE CLEAR - CHOOSE SIDE"
                actions['lane_change'] = "BOTH"
            elif left_clear:
                actions['primary'] = "🚗 MOVE LEFT - LANE CLEAR"
                actions['lane_change'] = "LEFT"
            elif right_clear:
                actions['primary'] = "🚗 MOVE RIGHT - LANE CLEAR"
                actions['lane_change'] = "RIGHT"
            else:
                actions['primary'] = "⚠️ MAINTAIN DISTANCE - NO CLEAR LANE"
        else:
            if preferred_escape:
                actions['primary'] = f"🚗 CONSIDER MOVING {preferred_escape}"
            elif zone == "LEFT":
                actions['primary'] = "🚗 CONSIDER MOVING RIGHT"
            elif zone == "RIGHT":
                actions['primary'] = "🚗 CONSIDER MOVING LEFT"
            else:
                actions['primary'] = "⚠️ MAINTAIN DISTANCE"

        direction_text = {
            "Sudden Left Turn": "Vehicle made a sharp LEFT turn",
            "Sudden Right Turn": "Vehicle made a sharp RIGHT turn",
            "Sudden Turn": "Unpredictable turning behavior detected"
        }[behavior]

        actions['secondary'] = [direction_text, "Reduce speed", "Stay alert"]
        actions['priority'] = "HIGH"
        
    elif behavior == "Weaving" or behavior == "Erratic":
        actions['primary'] = "🛑 REDUCE SPEED & KEEP DISTANCE"
        actions['secondary'] = ["Erratic driving detected", "Maintain safe following distance", "Be prepared to brake"]
        actions['priority'] = "HIGH"
        actions['icon'] = "⚠️"
        actions['color'] = "#FFFF00"
    
    elif behavior == "Accelerating":
        actions['primary'] = "⚠️ MONITOR - VEHICLE ACCELERATING"
        actions['secondary'] = ["Vehicle picking up speed", "Maintain safe distance", "Be ready to adjust speed"]
        actions['priority'] = "MEDIUM"
        actions['icon'] = "⚡"
        actions['color'] = "#FFA500"
    
    elif behavior == "Braking":
        actions['primary'] = "🛑 PREPARE TO BRAKE"
        actions['secondary'] = ["Vehicle ahead is slowing", "Reduce speed", "Increase following distance"]
        actions['priority'] = "HIGH"
        actions['icon'] = "🛑"
        actions['color'] = "#FF6600"
        
    elif behavior == "Fast":
        if zone == "LEFT":
            actions['primary'] = "➡️ MOVE RIGHT - FAST VEHICLE"
        elif zone == "RIGHT":
            actions['primary'] = "⬅️ MOVE LEFT - FAST VEHICLE"
        else:
            actions['primary'] = "⬇️ SLOW DOWN - MAINTAIN DISTANCE"
        actions['secondary'] = ["Fast-moving vehicle ahead", "Increase following distance", "Monitor continuously"]
        actions['priority'] = "MEDIUM"
        actions['icon'] = "🔶"
        actions['color'] = "#FF00FF"
        
    elif behavior == "Stopped":
        actions['primary'] = "🛑 STOP OR CHANGE LANE"
        actions['secondary'] = ["Vehicle stopped ahead", "Prepare to brake", "Check for lane change opportunity"]
        actions['priority'] = "HIGH"
        actions['icon'] = "🛑"
        actions['color'] = "#FF0000"
    
    elif behavior == "Slow" or behavior == "Initializing":
        actions['primary'] = "⚠️ SLOW VEHICLE AHEAD"
        actions['secondary'] = ["Slow-moving vehicle", "Reduce speed", "Consider overtaking when safe"]
        actions['priority'] = "LOW"
        actions['icon'] = "🐌"
        actions['color'] = "#FFD700"
        
    else:  # Normal
        actions['primary'] = "✅ PROCEED NORMALLY"
        actions['secondary'] = ["Normal traffic flow", "Maintain current speed", "Stay in lane"]
        actions['priority'] = "LOW"
        actions['icon'] = "✅"
        actions['color'] = "#00FF00"
        
        # For normal behavior, check if lane change is beneficial (only if enabled)
        if all_tracks and current_track and num_lanes > 1:
            left_clear = check_lane_clear(current_track, all_tracks, frame_width, 'left', num_lanes, margin_percent)
            right_clear = check_lane_clear(current_track, all_tracks, frame_width, 'right', num_lanes, margin_percent)
            if left_clear or right_clear:
                lanes = []
                if left_clear:
                    lanes.append("LEFT")
                if right_clear:
                    lanes.append("RIGHT")
                actions['lane_change'] = " & ".join(lanes)
                actions['secondary'].append(f"💡 {actions['lane_change']} lane(s) clear for overtaking")
    
    return actions

def create_heatmap(frame_shape, positions_history):
    """Create a heatmap from position history."""
    heatmap = np.zeros((frame_shape[0], frame_shape[1]), dtype=np.float32)
    
    for track_positions in positions_history.values():
        for pos in track_positions:
            x, y = pos
            if 0 <= y < frame_shape[0] and 0 <= x < frame_shape[1]:
                cv2.circle(heatmap, (x, y), 15, 1, -1)
    
    heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)
    if heatmap.max() > 0:
        heatmap = (heatmap / heatmap.max() * 255).astype(np.uint8)
    
    return heatmap

def detect_license_plate(frame, bbox, reader=None):
    """Detect and read license plate from vehicle bounding box."""
    if reader is None or not OCR_AVAILABLE:
        return None
    
    x1, y1, x2, y2 = bbox
    # Expand bbox slightly for better plate detection
    h, w = frame.shape[:2]
    x1 = max(0, int(x1 - 5))
    y1 = max(0, int(y1 - 5))
    x2 = min(w, int(x2 + 5))
    y2 = min(h, int(y2 + 5))
    
    vehicle_roi = frame[y1:y2, x1:x2]
    
    try:
        # Read text from ROI
        results = reader.readtext(vehicle_roi, detail=0)
        # Look for license plate patterns (alphanumeric)
        for text in results:
            cleaned = ''.join(c for c in text if c.isalnum())
            if 4 <= len(cleaned) <= 12:  # Typical plate length
                return cleaned.upper()
    except:
        pass
    
    return None

def process_video(video_path, model_path, conf_threshold, max_age_val, min_hits_val, iou_thresh, 
                 px_to_m, show_spd, show_traj, show_heat, frame_skip_val, resize_val, speed_alert, enable_alert,
                 behavior_window, speed_window, min_track_len, show_plates=False, show_lane_suggest=True, 
                 num_lanes=2, lane_margin=10):
    """Process video and yield annotated frames."""
    # Load YOLO model
    model = YOLO(model_path)
    
    # Initialize OCR reader if needed
    ocr_reader = None
    if show_plates and OCR_AVAILABLE:
        try:
            ocr_reader = easyocr.Reader(['en'], gpu=False)
        except:
            show_plates = False
    
    # Load video
    cap = cv2.VideoCapture(video_path)
    
    # Initialize SORT tracker
    tracker = Sort(max_age=max_age_val, min_hits=min_hits_val, iou_threshold=iou_thresh)
    
    # Store past locations for speed and direction calculation
    speed_history = {}
    all_positions = {}
    license_plates = {}  # Store detected license plates
    unique_vehicles = set()  # Track all unique vehicle IDs
    
    # Analytics data
    behavior_counts = defaultdict(int)
    speed_data = defaultdict(list)
    track_data = []
    frame_data = []
    alerts = []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        fps = 30  # Default
    
    frame_count = 0
    processed_count = 0
    
    # Get frame dimensions
    ret, test_frame = cap.read()
    if not ret:
        return
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    orig_height, orig_width = test_frame.shape[:2]
    new_width = int(orig_width * resize_val)
    new_height = int(orig_height * resize_val)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Skip frames for faster processing
        if frame_skip_val > 0 and (frame_count - 1) % (frame_skip_val + 1) != 0:
            continue
        
        processed_count += 1
        
        # Resize frame if needed
        if resize_val < 1.0:
            frame = cv2.resize(frame, (new_width, new_height))
        
        # YOLO detection with optimized parameters for accuracy
        results = model.predict(
            source=frame, 
            conf=conf_threshold, 
            iou=0.5,  # NMS IOU threshold for removing duplicate detections
            max_det=50,  # Maximum detections per frame
            verbose=False,
            agnostic_nms=True  # Class-agnostic NMS for better accuracy
        )
        detections = results[0].boxes.xyxy.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        
        # Prepare detections for SORT tracker
        if len(detections) > 0:
            dets = np.hstack((detections, confidences.reshape(-1, 1)))
        else:
            dets = np.empty((0, 5))
        
        # Update SORT tracker
        tracks = tracker.update(dets)
        
        current_frame_tracks = []
        current_frame_track_info = []  # For lane checking
        autonomous_actions = []
        
        for track in tracks:
            x1, y1, x2, y2, track_id = track
            cx, cy = int((x1+x2)/2), int((y1+y2)/2)
            
            # Add to unique vehicles set
            unique_vehicles.add(int(track_id))
            
            if track_id not in speed_history:
                speed_history[track_id] = {
                    'positions': [],
                    'speeds': [],
                    'behaviors': [],
                    'timestamps': [],
                    'alerts': []
                }
                all_positions[track_id] = []
                
                # Try to detect license plate
                if show_plates and ocr_reader is not None:
                    plate = detect_license_plate(frame, (x1, y1, x2, y2), ocr_reader)
                    if plate:
                        license_plates[track_id] = plate
            
            speed_history[track_id]['positions'].append((cx, cy))
            all_positions[track_id].append((cx, cy))
            speed_history[track_id]['timestamps'].append(frame_count / fps)
            
            # Store track info for lane checking
            current_frame_track_info.append({
                'id': int(track_id),
                'position': (cx, cy),
                'bbox': (x1, y1, x2, y2)
            })
            
            speed_kmh = 0
            behavior = "Normal"
            color = (0, 255, 0)
            confidence = 0.0
            
            if len(speed_history[track_id]['positions']) >= 2:
                # Calculate instantaneous speed from last frame (more accurate)
                (x_prev, y_prev) = speed_history[track_id]['positions'][-1]
                dx = cx - x_prev
                dy = cy - y_prev
                dist_pixels = np.sqrt(dx*dx + dy*dy)
                dist_meters = dist_pixels * px_to_m
                
                # Time between frames (accounting for frame skip)
                time_diff = (1 / fps) * (frame_skip_val + 1)
                speed_ms = dist_meters / time_diff if time_diff > 0 else 0
                speed_kmh = speed_ms * 3.6
                
                # Apply minimum threshold - if moving very little, speed is 0
                if dist_pixels < 0.3:  # Less than 0.3 pixel movement = no movement
                    speed_kmh = 0
                
                # Store raw speed first
                speed_history[track_id]['speeds'].append(speed_kmh)
                
                # Apply median filter for smoothing (more accurate than exponential)
                if len(speed_history[track_id]['speeds']) >= 5:
                    # Use median of last 5 speeds to reduce noise
                    speed_kmh = np.median(speed_history[track_id]['speeds'][-5:])
                elif len(speed_history[track_id]['speeds']) >= 3:
                    # Use median of last 3 if not enough history
                    speed_kmh = np.median(speed_history[track_id]['speeds'][-3:])
                
                # Outlier filtering - cap unrealistic speeds
                if speed_kmh > 100:  # Cap at 100 km/h (realistic for auto rickshaw)
                    speed_kmh = 100
                
                # Use advanced behavior analysis
                behavior, confidence, color = analyze_behavior_advanced(
                    speed_history, track_id, behavior_window, speed_window, min_track_len,
                    bbox=(x1, y1, x2, y2)
                )
                
                # Override with speeding alert if enabled and threshold exceeded
                if enable_alert and speed_kmh > speed_alert and len(speed_history[track_id]['positions']) >= min_track_len:
                    behavior = "SPEEDING"
                    confidence = min(1.0, speed_kmh / (speed_alert * 1.5))
                    color = (255, 0, 0)
                    alerts.append({
                        'frame': frame_count,
                        'track_id': int(track_id),
                        'speed': speed_kmh,
                        'timestamp': frame_count / fps,
                        'confidence': confidence
                    })
                
                # Store behavior
                speed_history[track_id]['behaviors'].append(behavior)
                
                # Keep only necessary history
                max_history = max(behavior_window, speed_window) + 10
                if len(speed_history[track_id]['positions']) > max_history:
                    speed_history[track_id]['positions'] = speed_history[track_id]['positions'][-max_history:]
                    speed_history[track_id]['speeds'] = speed_history[track_id]['speeds'][-max_history:]
                    speed_history[track_id]['behaviors'] = speed_history[track_id]['behaviors'][-max_history:]
                    speed_history[track_id]['timestamps'] = speed_history[track_id]['timestamps'][-max_history:]
            else:
                # Not enough data yet
                behavior = "Initializing"
                confidence = 0.3
                color = (128, 128, 128)
            
            # Draw trajectory
            if show_traj and len(speed_history[track_id]['positions']) > 1:
                for i in range(1, len(speed_history[track_id]['positions'])):
                    pt1 = speed_history[track_id]['positions'][i-1]
                    pt2 = speed_history[track_id]['positions'][i]
                    cv2.line(frame, pt1, pt2, color, 2)
            
            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
            
            # Draw labels
            vehicle_id = license_plates.get(track_id, f"Vehicle {int(track_id)}")
            label = f"{vehicle_id} {behavior}"
            if show_spd and speed_kmh > 0:
                label += f" {speed_kmh:.1f}km/h"
            
            # Add confidence indicator
            if confidence > 0:
                label += f" ({int(confidence*100)}%)"
            
            # Draw extra warning for front-facing vehicles
            if behavior == "FRONT FACING - HIGH ALERT":
                # Draw flashing red border around vehicle
                thickness = 6
                cv2.rectangle(frame, (int(x1)-5, int(y1)-5), (int(x2)+5, int(y2)+5), (0, 0, 255), thickness)
                # Draw WARNING text above vehicle
                warning_text = "!!! COLLISION RISK !!!"
                (warn_w, warn_h), _ = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
                cv2.rectangle(frame, (int(x1), int(y1)-warn_h-35), 
                             (int(x1)+warn_w, int(y1)-25), (0, 0, 255), -1)
                cv2.putText(frame, warning_text, (int(x1), int(y1)-30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
            
            # Draw label with background
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (int(x1), int(y1)-label_height-10), 
                         (int(x1)+label_width, int(y1)), color, -1)
            cv2.putText(frame, label, (int(x1), int(y1)-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Update analytics
            behavior_counts[behavior] += 1
            speed_data[int(track_id)].append(speed_kmh)
        
        # Second pass: Get autonomous actions with lane info
        for track in tracks:
            x1, y1, x2, y2, track_id = track
            cx, cy = int((x1+x2)/2), int((y1+y2)/2)
            
            if track_id not in speed_history or len(speed_history[track_id]['behaviors']) == 0:
                continue
            
            behavior = speed_history[track_id]['behaviors'][-1]
            speed_kmh = speed_history[track_id]['speeds'][-1] if speed_history[track_id]['speeds'] else 0
            
            # Get current track info
            current_track_info = None
            for info in current_frame_track_info:
                if info['id'] == int(track_id):
                    current_track_info = info
                    break
            
            # Get autonomous vehicle action with lane checking
            if show_lane_suggest:
                action = get_autonomous_action(behavior, speed_kmh, (cx, cy), frame.shape[1], frame.shape[0], 
                                             current_track_info, current_frame_track_info, num_lanes, lane_margin)
            else:
                action = get_autonomous_action(behavior, speed_kmh, (cx, cy), frame.shape[1], frame.shape[0], 
                                             None, None, num_lanes, lane_margin)
            
            vehicle_id = license_plates.get(track_id, f"Vehicle {int(track_id)}")
            
            autonomous_actions.append({
                'track_id': int(track_id),
                'vehicle_id': vehicle_id,
                'behavior': behavior,
                'action': action['primary'],
                'priority': action['priority'],
                'icon': action['icon'],
                'color': action['color'],
                'details': action['secondary'],
                'lane_change': action.get('lane_change')
            })
            
            current_frame_tracks.append({
                'frame': frame_count,
                'track_id': int(track_id),
                'vehicle_id': license_plates.get(track_id, f"Vehicle {int(track_id)}"),
                'behavior': behavior,
                'speed': speed_kmh,
                'confidence': confidence,
                'x': cx,
                'y': cy
            })
        
        # Apply heatmap overlay if enabled
        if show_heat and all_positions:
            heatmap = create_heatmap(frame.shape, all_positions)
            heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            frame = cv2.addWeighted(frame, 0.7, heatmap_colored, 0.3, 0)
        
        # Store frame data
        frame_data.append({
            'frame': frame_count,
            'num_tracks': len(tracks),
            'total_unique': len(unique_vehicles),
            'timestamp': frame_count / fps
        })
        
        track_data.extend(current_frame_tracks)
        
        # Convert BGR to RGB for Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        yield frame_rgb, frame_count, total_frames, fps, len(tracks), len(unique_vehicles), behavior_counts, speed_data, track_data, frame_data, alerts, autonomous_actions
    
    cap.release()

# Main app
col1, col2 = st.columns([2, 1])

with col1:
    # File uploaders
    st.subheader("📁 Upload Files")
    uploaded_video = st.file_uploader("Upload Video File", type=["mp4", "avi", "mov", "mkv"])
    uploaded_model = st.file_uploader("Upload YOLO Model (best.pt)", type=["pt"])

with col2:
    st.subheader("📖 Instructions")
    st.markdown("""
    1. Upload your video file
    2. Upload YOLO model (or use default)
    3. Adjust parameters in sidebar
    4. Click 'Start Analysis'
    5. View real-time analytics
    """)

# Check if default model exists
default_model_path = r"best.pt"
model_exists = os.path.exists(default_model_path)

if uploaded_video is not None:
    # Save uploaded video to temp file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_video.read())
    video_path = tfile.name
    
    # Handle model
    if uploaded_model is not None:
        mfile = tempfile.NamedTemporaryFile(delete=False, suffix='.pt')
        mfile.write(uploaded_model.read())
        model_path = mfile.name
    elif model_exists:
        model_path = default_model_path
        st.info(f"Using default model: {default_model_path}")
    else:
        st.error("Please upload a YOLO model file or ensure 'best.pt' exists in the project directory.")
        st.stop()
    
    if st.button("🚀 Start Analysis", type="primary"):
        st.markdown("---")
        
        # Create placeholders with side-by-side layout
        col_video, col_control = st.columns([3, 2])
        
        with col_video:
            video_placeholder = st.empty()
            
            # Real-time charts below video
            st.markdown("### 📊 Live Analytics")
            chart_col1, chart_col2 = st.columns(2)
            behavior_chart_placeholder = chart_col1.empty()
            speed_chart_placeholder = chart_col2.empty()
        
        with col_control:
            st.markdown("### 🤖 Autonomous Control Panel")
            st.caption("*Real-time action recommendations*")
            control_placeholder = st.empty()
        
        st.markdown("---")
        progress_bar = st.progress(0)
        status_text = st.empty()
        stats_cols = st.columns(7)
        
        # Alert section
        alert_placeholder = st.empty()
        
        try:
            behavior_counts_final = defaultdict(int)
            speed_data_final = defaultdict(list)
            track_data_final = []
            frame_data_final = []
            alerts_final = []
            
            start_time = time.time()
            
            # Process video
            for frame_rgb, frame_num, total_frames, fps, num_tracks, total_unique_vehicles, behavior_counts, speed_data, track_data, frame_data, alerts, autonomous_actions in process_video(
                video_path, model_path, confidence_threshold, max_age, min_hits, iou_threshold, 
                pixels_to_meters, show_speed, show_trajectory, show_heatmap, frame_skip, resize_factor,
                speed_alert_threshold, enable_alerts, behavior_smoothing, speed_smoothing, min_track_length,
                show_license_plates, show_lane_suggestions, num_lanes, lane_margin
            ):
                # Update video display
                video_placeholder.image(frame_rgb, channels="RGB")
                
                # Update progress
                progress = frame_num / total_frames if total_frames > 0 else 0
                progress_bar.progress(progress)
                
                # Calculate processing stats
                elapsed_time = time.time() - start_time
                processing_fps = (frame_num / elapsed_time) if elapsed_time > 0 else 0
                
                # Update status
                status_text.text(f"Processing frame {frame_num}/{total_frames} | Speed: {processing_fps:.1f} FPS")
                
                # Update stats
                stats_cols[0].metric("Frame", f"{frame_num}/{total_frames}")
                stats_cols[1].metric("Video FPS", fps)
                stats_cols[2].metric("Process FPS", f"{processing_fps:.1f}")
                stats_cols[3].metric("Active Vehicles", num_tracks)
                stats_cols[4].metric("Total Unique", total_unique_vehicles)
                stats_cols[5].metric("Progress", f"{int(progress*100)}%")
                
                # Calculate average speed
                if speed_data:
                    all_speeds = [s for speeds in speed_data.values() for s in speeds if s > 0]
                    avg_speed = np.mean(all_speeds) if all_speeds else 0
                    stats_cols[6].metric("Avg Speed", f"{avg_speed:.1f} km/h")
                
                # Show recent alerts with priority for front-facing
                if alerts and len(alerts) > 0:
                    recent_alert = alerts[-1]
                    if recent_alert.get('type') == 'FRONT_FACING':
                        alert_placeholder.error(f"🚨🚨 CRITICAL ALERT: Vehicle {recent_alert['track_id']} FACING FRONT - COLLISION IMMINENT! Frame: {recent_alert['frame']} | Time: {recent_alert['timestamp']:.1f}s")
                    elif recent_alert.get('type') == 'SPEEDING':
                        alert_placeholder.warning(f"⚠️ SPEED ALERT: Vehicle {recent_alert['track_id']} - {recent_alert['speed']:.1f} km/h at {recent_alert['timestamp']:.1f}s")
                    else:
                        alert_placeholder.warning(f"⚠️ ALERT: Vehicle {recent_alert['track_id']} - {recent_alert['speed']:.1f} km/h at {recent_alert['timestamp']:.1f}s")
                
                # Update Autonomous Control Display
                if autonomous_actions:
                    # Sort by priority
                    priority_order = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
                    sorted_actions = sorted(autonomous_actions, key=lambda x: priority_order[x['priority']])
                    
                    with control_placeholder.container():
                        # Display top 3 priority actions
                        control_cols = st.columns(min(3, len(sorted_actions)))
                        
                        for idx, action in enumerate(sorted_actions[:3]):
                            with control_cols[idx]:
                                vehicle_display = action.get('vehicle_id', f"Vehicle {action['track_id']}")
                                # Create card-like display
                                st.markdown(f"""
                                <div style="
                                    border: 3px solid {action['color']};
                                    border-radius: 10px;
                                    padding: 15px;
                                    background-color: rgba{tuple(int(action['color'][i:i+2], 16) for i in (1, 3, 5)) + (0.1,)};
                                    margin-bottom: 10px;
                                ">
                                    <h4 style="color: {action['color']}; margin: 0;">
                                        {action['icon']} {vehicle_display}
                                    </h4>
                                    <p style="font-size: 12px; margin: 5px 0;">
                                        <strong>Priority:</strong> {action['priority']}
                                    </p>
                                    <p style="font-size: 11px; margin: 5px 0;">
                                        <strong>Behavior:</strong> {action['behavior']}
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                st.markdown(f"**{action['action']}**")
                                
                                # Show lane change suggestion if available
                                if action.get('lane_change'):
                                    st.success(f"🚦 {action['lane_change']} lane clear!")
                                
                                with st.expander("📋 Detailed Instructions"):
                                    for detail in action['details']:
                                        st.write(f"• {detail}")
                        
                        # Show all actions summary
                        if len(sorted_actions) > 3:
                            st.info(f"ℹ️ +{len(sorted_actions) - 3} more vehicles detected. Showing top 3 priority actions.")
                
                # Update live charts every 15 frames
                if frame_num % 15 == 0 and behavior_counts:
                    # Behavior pie chart
                    behavior_df = pd.DataFrame(list(behavior_counts.items()), columns=['Behavior', 'Count'])
                    fig_pie = px.pie(behavior_df, values='Count', names='Behavior', 
                                    title='Behavior Distribution',
                                    color='Behavior',
                                    color_discrete_map={
                                        'FRONT FACING - HIGH ALERT': '#FF0000',
                                        'Stopped': '#0000FF',
                                        'Normal': '#00FF00',
                                        'Fast': '#FF00FF',
                                            'Weaving': '#FFFF00',
                                            'Sudden Left Turn': '#FFB347',
                                            'Sudden Right Turn': '#FF8C00',
                                            'Sudden Turn': '#FFA500',
                                            'SPEEDING': '#8B0000'
                                    })
                    fig_pie.update_layout(height=250, margin=dict(l=0, r=0, t=30, b=0))
                    behavior_chart_placeholder.plotly_chart(fig_pie, use_container_width=True)
                    
                    # Speed histogram
                    if speed_data:
                        all_speeds = [s for speeds in speed_data.values() for s in speeds if s > 0]
                        if all_speeds:
                            fig_hist = go.Figure(data=[go.Histogram(x=all_speeds, nbinsx=20)])
                            fig_hist.update_layout(
                                title='Speed Distribution',
                                xaxis_title='Speed (km/h)',
                                yaxis_title='Count',
                                height=250,
                                margin=dict(l=0, r=0, t=30, b=0)
                            )
                            speed_chart_placeholder.plotly_chart(fig_hist, use_container_width=True)
                
                behavior_counts_final = behavior_counts.copy()
                speed_data_final = {k: v.copy() for k, v in speed_data.items()}
                track_data_final = track_data.copy()
                frame_data_final = frame_data.copy()
                alerts_final = alerts.copy()
            
            processing_time = time.time() - start_time
            st.success(f"✅ Video processing complete! Total time: {processing_time:.1f}s")
            
            # Show detailed analytics in expandable section if enabled
            if show_detailed_analytics:
                st.markdown("---")
                with st.expander("📊 **View Detailed Analytics**", expanded=False):
                    st.header("📊 Comprehensive Analysis Report")
                    
                    analytics_tabs = st.tabs(["Behavior", "Speed", "Timeline", "Vehicles", "Alerts", "Actions", "Advanced"])
                    
                    with analytics_tabs[0]:
                        st.subheader("Behavior Distribution")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if behavior_counts_final:
                                behavior_df = pd.DataFrame(list(behavior_counts_final.items()), columns=['Behavior', 'Count'])
                                behavior_df['Percentage'] = (behavior_df['Count'] / behavior_df['Count'].sum() * 100).round(2)
                                
                                fig_bar = px.bar(behavior_df, x='Behavior', y='Count', 
                                                title='Total Behavior Counts',
                                                color='Behavior',
                                                color_discrete_map={
                                                    'Stopped': '#FF0000',
                                                    'Normal': '#00FF00',
                                            'Fast': '#FF00FF',
                                            'Weaving': '#FFFF00',
                                            'Sudden Left Turn': '#FFB347',
                                            'Sudden Right Turn': '#FF8C00',
                                            'Sudden Turn': '#FFA500',
                                            'SPEEDING': '#8B0000'
                                                })
                                st.plotly_chart(fig_bar)
                        
                        with col2:
                            if behavior_counts_final:
                                fig_pie = px.pie(behavior_df, values='Count', names='Behavior',
                                                title='Behavior Percentage',
                                                color='Behavior',
                                                color_discrete_map={
                                                    'Stopped': '#FF0000',
                                                    'Normal': '#00FF00',
                                                    'Fast': '#FF00FF',
                                                    'Weaving': '#FFFF00',
                                                    'Sudden Left Turn': '#FFB347',
                                                    'Sudden Right Turn': '#FF8C00',
                                                    'Sudden Turn': '#FFA500',
                                                    'SPEEDING': '#8B0000'
                                                })
                                st.plotly_chart(fig_pie)
                                
                                st.dataframe(behavior_df)
                    
                    with analytics_tabs[1]:
                        st.subheader("Speed Analysis")
                        
                        if speed_data_final:
                            # Speed statistics by track
                            speed_stats = []
                            for track_id, speeds in speed_data_final.items():
                                if speeds:
                                    valid_speeds = [s for s in speeds if s > 0]
                                    if valid_speeds:
                                        speed_stats.append({
                                            'Vehicle ID': track_id,
                                            'Avg Speed (km/h)': np.mean(valid_speeds),
                                            'Max Speed (km/h)': np.max(valid_speeds),
                                            'Min Speed (km/h)': np.min(valid_speeds),
                                            'Std Dev': np.std(valid_speeds)
                                        })
                            
                            if speed_stats:
                                speed_stats_df = pd.DataFrame(speed_stats)
                                
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    # Box plot
                                    all_speed_data = []
                                    for track_id, speeds in speed_data_final.items():
                                        for speed in speeds:
                                            if speed > 0:
                                                all_speed_data.append({'Vehicle ID': str(track_id), 'Speed (km/h)': speed})
                                    
                                    if all_speed_data:
                                        speed_box_df = pd.DataFrame(all_speed_data)
                                        fig_box = px.box(speed_box_df, x='Track ID', y='Speed (km/h)',
                                                       title='Speed Distribution by Track')
                                        st.plotly_chart(fig_box)
                                
                                with col2:
                                    # Bar chart of average speeds
                                    fig_avg = px.bar(speed_stats_df, x='Track ID', y='Avg Speed (km/h)',
                                                   title='Average Speed by Track',
                                                   color='Avg Speed (km/h)',
                                                   color_continuous_scale='RdYlGn_r')
                                    st.plotly_chart(fig_avg)
                                
                                st.dataframe(speed_stats_df.style.format({
                                    'Avg Speed (km/h)': '{:.2f}',
                                    'Max Speed (km/h)': '{:.2f}',
                                    'Min Speed (km/h)': '{:.2f}',
                                    'Std Dev': '{:.2f}'
                                }))
                    
                    with analytics_tabs[2]:
                        st.subheader("Timeline Analysis")
                        
                        if track_data_final:
                            track_df = pd.DataFrame(track_data_final)
                            
                            # Speed over time
                            fig_timeline = px.scatter(track_df, x='frame', y='speed', 
                                                    color='track_id',
                                            title='Speed Over Time',
                                            labels={'frame': 'Frame', 'speed': 'Speed (km/h)', 'track_id': 'Vehicle ID'},
                                            hover_data=['behavior'])
                    fig_timeline.update_traces(marker=dict(size=3))
                    st.plotly_chart(fig_timeline)
                    
                    # Number of tracks over time
                    if frame_data_final:
                        frame_df = pd.DataFrame(frame_data_final)
                        fig_tracks = px.line(frame_df, x='frame', y='num_tracks',
                                           title='Number of Active Vehicles Over Time',
                                           labels={'frame': 'Frame', 'num_tracks': 'Active Vehicles'})
                        st.plotly_chart(fig_tracks)
            
            with analytics_tabs[3]:
                st.subheader("Vehicle Details")
                
                if track_data_final:
                    track_df = pd.DataFrame(track_data_final)
                    
                    # Select track to view
                    unique_tracks = sorted(track_df['track_id'].unique())
                    selected_track = st.selectbox("Select Vehicle", unique_tracks)
                    
                    if selected_track:
                        track_subset = track_df[track_df['track_id'] == selected_track]
                        
                        # Display track statistics
                        avg_confidence = track_subset['confidence'].mean()
                        avg_speed = track_subset[track_subset['speed'] > 0]['speed'].mean()
                        
                        stat_cols = st.columns(4)
                        stat_cols[0].metric("Avg Confidence", f"{avg_confidence*100:.1f}%")
                        stat_cols[1].metric("Avg Speed", f"{avg_speed:.1f} km/h" if not pd.isna(avg_speed) else "N/A")
                        stat_cols[2].metric("Total Frames", len(track_subset))
                        stat_cols[3].metric("Behaviors Detected", track_subset['behavior'].nunique())
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Speed and confidence over time
                            fig_track_speed = go.Figure()
                            fig_track_speed.add_trace(go.Scatter(
                                x=track_subset['frame'], 
                                y=track_subset['speed'],
                                mode='lines+markers',
                                name='Speed (km/h)',
                                line=dict(color='blue')
                            ))
                            fig_track_speed.add_trace(go.Scatter(
                                x=track_subset['frame'], 
                                y=track_subset['confidence'] * 50,  # Scale for visibility
                                mode='lines',
                                name='Confidence (scaled)',
                                line=dict(color='green', dash='dash')
                            ))
                            fig_track_speed.update_layout(
                                title=f'Track {selected_track} - Speed & Confidence Over Time',
                                xaxis_title='Frame',
                                yaxis_title='Value'
                            )
                            st.plotly_chart(fig_track_speed)
                        
                        with col2:
                            # Trajectory plot
                            fig_trajectory = px.scatter(track_subset, x='x', y='y',
                                                       title=f'Track {selected_track} - Trajectory',
                                                       labels={'x': 'X Position', 'y': 'Y Position'},
                                                       color='confidence',
                                                       color_continuous_scale='RdYlGn',
                                                       size='speed',
                                                       hover_data=['behavior', 'speed'])
                            fig_trajectory.update_yaxes(autorange="reversed")
                            st.plotly_chart(fig_trajectory)
                        
                        # Behavior summary for track
                        behavior_summary = track_subset['behavior'].value_counts()
                        st.write(f"**Behavior Summary for Track {selected_track}:**")
                        st.dataframe(behavior_summary)
            
            with analytics_tabs[4]:
                st.subheader("🚨 Alerts & Reports")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Speed Alerts Summary**")
                    if alerts_final:
                        alerts_df = pd.DataFrame(alerts_final)
                        st.metric("Total Speed Violations", len(alerts_final))
                        st.dataframe(alerts_df.style.format({
                            'speed': '{:.1f}',
                            'timestamp': '{:.2f}'
                        }))
                        
                        # Download alerts as CSV
                        csv = alerts_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Alerts Report",
                            data=csv,
                            file_name=f"speed_alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.info("No speed alerts detected during analysis.")
                
                with col2:
                    st.write("**Summary Statistics**")
                    if track_data_final:
                        track_df = pd.DataFrame(track_data_final)
                        
                        st.metric("Total Tracks Detected", len(track_df['track_id'].unique()))
                        st.metric("Total Frames Analyzed", len(frame_data_final))
                        st.metric("Processing Time", f"{processing_time:.1f}s")
                        
                        if speed_data_final:
                            all_speeds = [s for speeds in speed_data_final.values() for s in speeds if s > 0]
                            if all_speeds:
                                st.metric("Overall Avg Speed", f"{np.mean(all_speeds):.1f} km/h")
                                st.metric("Max Speed Recorded", f"{np.max(all_speeds):.1f} km/h")
                
                # Export full report
                if track_data_final:
                    st.markdown("---")
                    st.write("**Export Complete Analysis**")
                    full_df = pd.DataFrame(track_data_final)
                    csv_full = full_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Full Analysis Report",
                        data=csv_full,
                        file_name=f"full_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            
            with analytics_tabs[5]:
                st.markdown("## 🤖 Autonomous Vehicle Control Panel")
                st.markdown("*Real-time action recommendations based on detected vehicle behaviors*")
                
                if track_data_final:
                    track_df = pd.DataFrame(track_data_final)
                    
                    # Calculate action priorities based on behaviors
                    priority_counts = {
                        'CRITICAL': behavior_counts_final.get('SPEEDING', 0),
                        'HIGH': (
                            behavior_counts_final.get('Stopped', 0)
                            + behavior_counts_final.get('Sudden Turn', 0)
                            + behavior_counts_final.get('Sudden Left Turn', 0)
                            + behavior_counts_final.get('Sudden Right Turn', 0)
                            + behavior_counts_final.get('Weaving', 0)
                        ),
                        'MEDIUM': behavior_counts_final.get('Fast', 0) + behavior_counts_final.get('Slowing', 0),
                        'LOW': behavior_counts_final.get('Normal', 0) + behavior_counts_final.get('Initializing', 0)
                    }
                    
                    # Statistics cards
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("🔴 CRITICAL", priority_counts.get('CRITICAL', 0))
                    with col2:
                        st.metric("⚠️ HIGH", priority_counts.get('HIGH', 0))
                    with col3:
                        st.metric("🟡 MEDIUM", priority_counts.get('MEDIUM', 0))
                    with col4:
                        st.metric("✅ LOW", priority_counts.get('LOW', 0))
                    
                    st.divider()
                    
                    # Filter options
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        filter_priority = st.multiselect(
                            "🔍 Filter by Priority",
                            options=['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'],
                            default=['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']
                        )
                    with col2:
                        max_display = st.slider("Max Actions to Display", 5, 20, 10)
                    
                    st.markdown("---")
                    
                    # Create action recommendations from track data
                    action_recommendations = []
                    for _, row in track_df.iterrows():
                        behavior = row['behavior']
                        
                        # Map behavior to priority and action
                        if behavior == 'FRONT FACING - HIGH ALERT':
                            priority = 'CRITICAL'
                            action = '🚨🚨 EMERGENCY STOP - COLLISION IMMINENT'
                            details = 'Vehicle facing front/camera! Immediate evasive action required. Sound horn and brake hard!'
                        elif behavior == 'SPEEDING':
                            priority = 'CRITICAL'
                            action = '⚠️ EMERGENCY BRAKE'
                            details = 'Immediate deceleration required. Vehicle exceeding safe speed limit.'
                        elif behavior == 'Stopped':
                            priority = 'HIGH'
                            action = '🛑 STOP OR CHANGE LANE'
                            details = 'Vehicle stopped ahead. Prepare to brake or switch lanes safely.'
                        elif behavior == 'Sudden Left Turn':
                            priority = 'HIGH'
                            action = '↩️ WATCH LEFT & SHIFT RIGHT'
                            details = 'Vehicle veered sharply left. Maintain distance and shift to the right lane if safe.'
                        elif behavior == 'Sudden Right Turn':
                            priority = 'HIGH'
                            action = '↪️ WATCH RIGHT & SHIFT LEFT'
                            details = 'Vehicle veered sharply right. Maintain distance and favor the left lane if available.'
                        elif behavior == 'Sudden Turn':
                            priority = 'HIGH'
                            action = '🚗 MOVE LEFT/RIGHT'
                            details = 'Unpredictable movement detected. Adjust position and maintain distance.'
                        elif behavior == 'Weaving':
                            priority = 'HIGH'
                            action = '🛑 REDUCE SPEED & KEEP DISTANCE'
                            details = 'Erratic driving pattern. Increase following distance and reduce speed.'
                        elif behavior == 'Fast':
                            priority = 'MEDIUM'
                            action = '➡️ ADJUST LANE POSITION'
                            details = 'Fast-moving vehicle detected. Consider lane adjustment if needed.'
                        elif behavior == 'Slowing':
                            priority = 'MEDIUM'
                            action = '⚡ PREPARE TO SLOW DOWN'
                            details = 'Vehicle ahead is decelerating. Be ready to adjust speed.'
                        else:
                            priority = 'LOW'
                            action = '✅ PROCEED NORMALLY'
                            details = 'Normal traffic flow. Maintain current speed and lane.'
                        
                        action_recommendations.append({
                            'track_id': row['track_id'],
                            'priority': priority,
                            'behavior': behavior,
                            'action': action,
                            'details': details,
                            'speed': row['speed'],
                            'confidence': row['confidence']
                        })
                    
                    # Get unique tracks with their most critical actions
                    track_actions = {}
                    priority_order = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
                    
                    for action in action_recommendations:
                        track_id = action['track_id']
                        if track_id not in track_actions or priority_order[action['priority']] < priority_order[track_actions[track_id]['priority']]:
                            track_actions[track_id] = action
                    
                    # Convert to list and sort by priority
                    sorted_actions = sorted(track_actions.values(), key=lambda x: priority_order[x['priority']])
                    
                    # Filter by selected priorities
                    filtered_actions = [a for a in sorted_actions if a['priority'] in filter_priority][:max_display]
                    
                    # Display actions with interactive expandable cards
                    for idx, action in enumerate(filtered_actions):
                        priority_config = {
                            'CRITICAL': {'emoji': '🔴', 'color': '#FF0000', 'bg': '#FFE6E6'},
                            'HIGH': {'emoji': '⚠️', 'color': '#FF8C00', 'bg': '#FFF4E6'},
                            'MEDIUM': {'emoji': '🟡', 'color': '#FFD700', 'bg': '#FFFBE6'},
                            'LOW': {'emoji': '✅', 'color': '#32CD32', 'bg': '#E6FFE6'}
                        }
                        
                        config = priority_config.get(action['priority'], {'emoji': '⚪', 'color': '#808080', 'bg': '#F0F0F0'})
                        vehicle_display = action.get('vehicle_id', f"Track {action['track_id']}")
                        
                        # Create expandable section
                        with st.expander(f"{config['emoji']} **{vehicle_display}** | Priority: **{action['priority']}** | Action: **{action['action']}**", expanded=(idx < 3)):
                            
                            # Create two columns for better layout
                            info_col1, info_col2 = st.columns([1, 2])
                            
                            with info_col1:
                                st.markdown(f"""
                                <div style='background-color: {config['bg']}; padding: 15px; border-radius: 10px; border-left: 5px solid {config['color']}'>
                                    <h3 style='color: {config['color']}; margin: 0;'>{config['emoji']} {action['priority']}</h3>
                                    <p style='margin: 5px 0;'><b>Vehicle:</b> {vehicle_display}</p>
                                    <p style='margin: 5px 0;'><b>Behavior:</b> {action['behavior']}</p>
                                    <p style='margin: 5px 0;'><b>Speed:</b> {action.get('speed', 0):.1f} km/h</p>
                                    <p style='margin: 5px 0;'><b>Confidence:</b> {action.get('confidence', 0):.0f}%</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Lane change suggestion
                                if action.get('lane_change'):
                                    st.success(f"🚦 **{action['lane_change']}** lane clear for change!")
                                
                                # Action button styling
                                if action['priority'] in ['CRITICAL', 'HIGH']:
                                    st.error(f"🚨 **IMMEDIATE ACTION REQUIRED**")
                                else:
                                    st.success(f"✅ **ADVISORY**")
                            
                            with info_col2:
                                st.markdown(f"### {action['action']}")
                                st.markdown(f"**📋 Detailed Instructions:**")
                                for detail in action['details']:
                                    st.write(f"• {detail}")
                                
                                # Additional context based on behavior
                                if action['behavior'] == 'SPEEDING':
                                    st.warning("⚡ **Speed Alert:** Vehicle exceeding safe speed limits")
                                elif action['behavior'] == 'Weaving':
                                    st.warning("🌊 **Stability Alert:** Erratic movement pattern detected")
                                elif action['behavior'] == 'Sudden Left Turn':
                                    st.warning("↩️ **Left Turn Alert:** Vehicle made a sharp LEFT deviation")
                                elif action['behavior'] == 'Sudden Right Turn':
                                    st.warning("↪️ **Right Turn Alert:** Vehicle made a sharp RIGHT deviation")
                                elif action['behavior'] == 'Sudden Turn':
                                    st.warning("↪️ **Direction Alert:** Abrupt directional change detected")
                                elif action['behavior'] == 'Stopped':
                                    st.warning("🛑 **Traffic Alert:** Stationary vehicle ahead")
                            
                            # Progress bar for priority visualization
                            priority_value = {'CRITICAL': 100, 'HIGH': 75, 'MEDIUM': 50, 'LOW': 25}
                            st.progress(priority_value.get(action['priority'], 0) / 100)
                            
                            st.markdown("---")
                    
                    # Action summary section
                    st.markdown("### 📊 Action Summary & Analytics")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Pie chart of priorities
                        priority_df = pd.DataFrame(list(priority_counts.items()), columns=['Priority', 'Count'])
                        priority_df = priority_df[priority_df['Count'] > 0]
                        
                        if not priority_df.empty:
                            fig_priority = px.pie(priority_df, values='Count', names='Priority',
                                                title='Action Distribution by Priority',
                                                color='Priority',
                                                color_discrete_map={
                                                    'CRITICAL': '#FF0000',
                                                    'HIGH': '#FF8C00',
                                                    'MEDIUM': '#FFD700',
                                                    'LOW': '#32CD32'
                                                })
                            st.plotly_chart(fig_priority, use_container_width=True)
                    
                    with col2:
                        # Action type distribution
                        action_counts = {}
                        for action in action_recommendations:
                            action_counts[action['action']] = action_counts.get(action['action'], 0) + 1
                        
                        action_df = pd.DataFrame(list(action_counts.items()), columns=['Action', 'Count'])
                        action_df = action_df.sort_values('Count', ascending=False)
                        
                        fig_actions = px.bar(action_df, x='Action', y='Count',
                                           title='Recommended Actions Frequency',
                                           color='Count',
                                           color_continuous_scale='Reds')
                        st.plotly_chart(fig_actions, use_container_width=True)
                    
                    st.divider()
                    
                    # Comprehensive Action Guide Table
                    st.markdown("### 📖 Complete Action Guide Reference")
                    
                    action_guide = pd.DataFrame([
                        {
                            'Behavior': 'SPEEDING',
                            'Priority': '🔴 CRITICAL',
                            'Action': '⚠️ EMERGENCY BRAKE',
                            'Description': 'Immediate deceleration required. Vehicle exceeding safe speed limit.'
                        },
                        {
                            'Behavior': 'Stopped',
                            'Priority': '⚠️ HIGH',
                            'Action': '🛑 STOP OR CHANGE LANE',
                            'Description': 'Vehicle stopped ahead. Prepare to brake or switch lanes safely.'
                        },
                        {
                            'Behavior': 'Sudden Left Turn',
                            'Priority': '⚠️ HIGH',
                            'Action': '↩️ WATCH LEFT & SHIFT RIGHT',
                            'Description': 'Vehicle veered sharply left. Maintain distance and favor the right lane.'
                        },
                        {
                            'Behavior': 'Sudden Right Turn',
                            'Priority': '⚠️ HIGH',
                            'Action': '↪️ WATCH RIGHT & SHIFT LEFT',
                            'Description': 'Vehicle veered sharply right. Maintain distance and favor the left lane.'
                        },
                        {
                            'Behavior': 'Sudden Turn',
                            'Priority': '⚠️ HIGH',
                            'Action': '🚗 MOVE LEFT/RIGHT',
                            'Description': 'Unpredictable movement detected. Adjust position and maintain distance.'
                        },
                        {
                            'Behavior': 'Weaving',
                            'Priority': '⚠️ HIGH',
                            'Action': '🛑 REDUCE SPEED & KEEP DISTANCE',
                            'Description': 'Erratic driving pattern. Increase following distance and reduce speed.'
                        },
                        {
                            'Behavior': 'Fast',
                            'Priority': '🟡 MEDIUM',
                            'Action': '➡️ ADJUST LANE POSITION',
                            'Description': 'Fast-moving vehicle. Consider lane adjustment if needed.'
                        },
                        {
                            'Behavior': 'Slowing',
                            'Priority': '🟡 MEDIUM',
                            'Action': '⚡ PREPARE TO SLOW DOWN',
                            'Description': 'Vehicle decelerating. Be ready to adjust speed accordingly.'
                        },
                        {
                            'Behavior': 'Normal',
                            'Priority': '✅ LOW',
                            'Action': '✅ PROCEED NORMALLY',
                            'Description': 'Normal traffic flow. Maintain current speed and lane.'
                        }
                    ])
                    
                    st.dataframe(action_guide, use_container_width=True, height=300)
                    
                    # Download section with multiple options
                    st.markdown("### 💾 Export Options")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        # Download full action log
                        actions_df = pd.DataFrame(action_recommendations)
                        actions_csv = actions_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Complete Action Log",
                            data=actions_csv,
                            file_name=f"autonomous_actions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    
                    with col2:
                        # Download critical actions only
                        critical_actions = [a for a in action_recommendations if a['priority'] in ['CRITICAL', 'HIGH']]
                        if critical_actions:
                            critical_df = pd.DataFrame(critical_actions)
                            critical_csv = critical_df.to_csv(index=False)
                            st.download_button(
                                label="🚨 Download Critical Actions Only",
                                data=critical_csv,
                                file_name=f"critical_actions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                        else:
                            st.info("No critical actions detected")
                    
                    with col3:
                        # Download action guide reference
                        guide_csv = action_guide.to_csv(index=False)
                        st.download_button(
                            label="📄 Download Action Guide",
                            data=guide_csv,
                            file_name=f"action_guide_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                else:
                    st.info("🎬 No tracking data available yet. Upload and process a video to see real-time recommendations.")
                    st.markdown("""
                    **How the Autonomous Control Panel Works:**
                    1. 📤 Upload a video file with vehicle footage
                    2. 🔍 AI analyzes vehicle behaviors in real-time
                    3. 🎯 System generates priority-based action recommendations
                    4. 🤖 View interactive guidance for autonomous vehicle control
                    5. 💾 Export detailed action logs and reports
                    
                    **Priority Levels:**
                    - 🔴 **CRITICAL:** Immediate action required (e.g., emergency braking)
                    - ⚠️ **HIGH:** Important actions needed (e.g., lane changes, stopping)
                    - 🟡 **MEDIUM:** Advisory actions (e.g., speed adjustments)
                    - ✅ **LOW:** Normal operations (e.g., maintain current behavior)
                    """)
            
            with analytics_tabs[6]:
                st.subheader("📈 Advanced Graphs & Visualizations")
                
                if track_data_final:
                    track_df = pd.DataFrame(track_data_final)
                    
                    # Speed vs Confidence Scatter
                    st.markdown("### 🎯 Speed vs Confidence Analysis")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig_scatter = px.scatter(track_df, x='speed', y='confidence',
                                               color='behavior',
                                               size='speed',
                                               title='Speed vs Confidence by Behavior',
                                               labels={'speed': 'Speed (km/h)', 'confidence': 'Confidence Score'},
                                               hover_data=['track_id', 'frame'],
                                               color_discrete_map={
                                                   'Stopped': '#FF0000',
                                                   'Normal': '#00FF00',
                                                   'Fast': '#FF00FF',
                                                   'Weaving': '#FFFF00',
                                                   'Sudden Left Turn': '#FFB347',
                                                   'Sudden Right Turn': '#FF8C00',
                                                   'Sudden Turn': '#FFA500',
                                                   'SPEEDING': '#8B0000',
                                                   'Initializing': '#808080'
                                               })
                        st.plotly_chart(fig_scatter)
                    
                    with col2:
                        # Confidence distribution histogram
                        fig_conf_hist = px.histogram(track_df, x='confidence',
                                                    nbins=20,
                                                    title='Confidence Score Distribution',
                                                    labels={'confidence': 'Confidence Score', 'count': 'Frequency'},
                                                    color_discrete_sequence=['#1f77b4'])
                        fig_conf_hist.add_vline(x=track_df['confidence'].mean(), 
                                               line_dash="dash", 
                                               line_color="red",
                                               annotation_text=f"Mean: {track_df['confidence'].mean():.2f}")
                        st.plotly_chart(fig_conf_hist)
                    
                    # Behavior over time heatmap
                    st.markdown("### 🔥 Behavior Heatmap Over Time")
                    
                    # Create time bins
                    track_df['time_bin'] = (track_df['frame'] // 30).astype(int)  # 30 frames per bin
                    behavior_time = track_df.groupby(['time_bin', 'behavior']).size().reset_index(name='count')
                    behavior_pivot = behavior_time.pivot(index='behavior', columns='time_bin', values='count').fillna(0)
                    
                    fig_heatmap = px.imshow(behavior_pivot,
                                           title='Behavior Distribution Over Time',
                                           labels=dict(x="Time Period", y="Behavior", color="Count"),
                                           color_continuous_scale='YlOrRd',
                                           aspect='auto')
                    st.plotly_chart(fig_heatmap, use_container_width=True)
                    
                    # Speed distribution by behavior (violin plot)
                    st.markdown("### 🎻 Speed Distribution by Behavior (Violin Plot)")
                    fig_violin = px.violin(track_df[track_df['speed'] > 0], 
                                          x='behavior', 
                                          y='speed',
                                          box=True,
                                          points='all',
                                          title='Speed Distribution by Behavior Type',
                                          labels={'speed': 'Speed (km/h)', 'behavior': 'Behavior'},
                                          color='behavior',
                                          color_discrete_map={
                                              'Stopped': '#FF0000',
                                              'Normal': '#00FF00',
                                              'Fast': '#FF00FF',
                                              'Weaving': '#FFFF00',
                                              'Sudden Left Turn': '#FFB347',
                                              'Sudden Right Turn': '#FF8C00',
                                              'Sudden Turn': '#FFA500',
                                              'SPEEDING': '#8B0000'
                                          })
                    st.plotly_chart(fig_violin, use_container_width=True)
                    
                    # 3D scatter plot
                    st.markdown("### 🌐 3D Trajectory Analysis")
                    
                    # Limit to top 5 tracks for clarity
                    top_tracks = track_df['track_id'].value_counts().head(5).index.tolist()
                    track_df_filtered = track_df[track_df['track_id'].isin(top_tracks)]
                    
                    fig_3d = px.scatter_3d(track_df_filtered, 
                                          x='x', 
                                          y='y', 
                                          z='speed',
                                          color='track_id',
                                          size='confidence',
                                          title='3D Trajectory: Position (X,Y) vs Speed',
                                          labels={'x': 'X Position', 'y': 'Y Position', 'speed': 'Speed (km/h)'},
                                          hover_data=['behavior', 'frame'])
                    fig_3d.update_layout(scene=dict(
                        xaxis_title='X Position',
                        yaxis_title='Y Position',
                        zaxis_title='Speed (km/h)'
                    ))
                    st.plotly_chart(fig_3d, use_container_width=True)
                    
                    # Multi-track comparison
                    st.markdown("### 📊 Multi-Track Speed Comparison")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Line plot comparing speeds of multiple tracks
                        fig_multi = go.Figure()
                        for track_id in top_tracks:
                            track_subset = track_df[track_df['track_id'] == track_id]
                            fig_multi.add_trace(go.Scatter(
                                x=track_subset['frame'],
                                y=track_subset['speed'],
                                mode='lines',
                                name=f'Track {track_id}',
                                hovertemplate='<b>Frame</b>: %{x}<br><b>Speed</b>: %{y:.1f} km/h'
                            ))
                        
                        fig_multi.update_layout(
                            title='Speed Comparison Across Tracks',
                            xaxis_title='Frame',
                            yaxis_title='Speed (km/h)',
                            hovermode='x unified'
                        )
                        st.plotly_chart(fig_multi)
                    
                    with col2:
                        # Radar chart of behavior characteristics
                        behavior_stats = track_df.groupby('behavior').agg({
                            'speed': 'mean',
                            'confidence': 'mean',
                            'track_id': 'count'
                        }).reset_index()
                        behavior_stats.columns = ['Behavior', 'Avg_Speed', 'Avg_Confidence', 'Count']
                        
                        if len(behavior_stats) > 0:
                            fig_radar = go.Figure()
                            
                            for _, row in behavior_stats.iterrows():
                                fig_radar.add_trace(go.Scatterpolar(
                                    r=[row['Avg_Speed']/10, row['Avg_Confidence'], row['Count']/10],
                                    theta=['Speed', 'Confidence', 'Frequency'],
                                    fill='toself',
                                    name=row['Behavior']
                                ))
                            
                            fig_radar.update_layout(
                                polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
                                title='Behavior Characteristics (Normalized)',
                                showlegend=True
                            )
                            st.plotly_chart(fig_radar)
                    
                    # Sunburst chart for hierarchical behavior analysis
                    st.markdown("### ☀️ Hierarchical Behavior Analysis (Sunburst)")
                    
                    # Create hierarchy: Track -> Behavior -> Speed Range
                    track_df['speed_range'] = pd.cut(track_df['speed'], 
                                                     bins=[0, 5, 15, 30, 100],
                                                     labels=['0-5 km/h', '5-15 km/h', '15-30 km/h', '30+ km/h'])
                    
                    sunburst_data = track_df.groupby(['behavior', 'speed_range']).size().reset_index(name='count')
                    
                    fig_sunburst = px.sunburst(sunburst_data,
                                              path=['behavior', 'speed_range'],
                                              values='count',
                                              title='Behavior Breakdown by Speed Range',
                                              color='count',
                                              color_continuous_scale='RdYlGn_r')
                    st.plotly_chart(fig_sunburst, use_container_width=True)
                    
                    # Correlation heatmap
                    st.markdown("### 🔗 Feature Correlation Matrix")
                    
                    corr_data = track_df[['speed', 'confidence', 'x', 'y', 'frame']].corr()
                    
                    fig_corr = px.imshow(corr_data,
                                        title='Correlation Between Features',
                                        labels=dict(color="Correlation"),
                                        color_continuous_scale='RdBu_r',
                                        text_auto='.2f',
                                        aspect='auto')
                    st.plotly_chart(fig_corr)
                    
                    # Statistical summary
                    st.markdown("### 📋 Statistical Summary")
                    
                    summary_stats = track_df[['speed', 'confidence']].describe()
                    st.dataframe(summary_stats.style.format("{:.2f}"))
                    
                    # Download all graphs data
                    st.markdown("### 💾 Export Options")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Export track data with all features
                        export_df = track_df[['frame', 'track_id', 'behavior', 'speed', 'confidence', 'x', 'y']]
                        csv_export = export_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Complete Track Data",
                            data=csv_export,
                            file_name=f"complete_tracking_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    
                    with col2:
                        # Export behavior statistics
                        behavior_export = track_df.groupby('behavior').agg({
                            'speed': ['mean', 'std', 'min', 'max'],
                            'confidence': ['mean', 'std'],
                            'track_id': 'count'
                        }).round(2)
                        behavior_csv = behavior_export.to_csv()
                        st.download_button(
                            label="📥 Download Behavior Statistics",
                            data=behavior_csv,
                            file_name=f"behavior_statistics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                else:
                    st.info("Process a video to see advanced graphs and visualizations.")
            
        except Exception as e:
            st.error(f"Error processing video: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
        finally:
            # Cleanup temp files
            try:
                os.unlink(video_path)
                if uploaded_model is not None:
                    os.unlink(model_path)
            except:
                pass
else:
    st.info("👆 Please upload a video file to begin analysis")
    
    # Show legend
    st.markdown("---")
    st.subheader("🎨 Behavior Legend")
    legend_cols = st.columns(6)
    with legend_cols[0]:
        st.markdown("🟢 **Normal** - Steady movement")
    with legend_cols[1]:
        st.markdown("🔴 **Stopped** - No movement")
    with legend_cols[2]:
        st.markdown("🟡 **Weaving** - High speed movement")
    with legend_cols[3]:
        st.markdown("🟠 **Sudden Turn** - Sharp direction change")
    with legend_cols[4]:
        st.markdown("🟣 **Fast** - Above normal speed")
    with legend_cols[5]:
        st.markdown("🔴 **SPEEDING** - Exceeds alert threshold")

    extra_cols = st.columns(3)
    with extra_cols[0]:
        st.markdown("↩️ **Sudden Left Turn** - Sharp left deviation")
    with extra_cols[1]:
        st.markdown("↪️ **Sudden Right Turn** - Sharp right deviation")
    with extra_cols[2]:
        st.markdown("🚨 **FRONT FACING - HIGH ALERT** - Collision risk!")
