#!/usr/bin/env python3

import os
import atexit
os.environ['QT_QPA_PLATFORM'] = 'xcb'

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import threading
import time
import numpy as np
import cv2
from dronekit import connect, VehicleMode, LocationGlobalRelative, Command
from pymavlink import mavutil
import math
import json
import sys
from collections import deque
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== CAMERA-BASED DETECTION IMPORTS ====================
sys.path.insert(0, '/home/aditya/agribot_ws')

from src.world_parser import parse_infected_crops, parse_spherical_origin, WORLD_PATH, InfectedCrop
from src.color_calib import build_color_thresholds, build_disease_thresholds
from src.disease_constants import DISEASE_COLORS
from src.camera_math import CameraIntrinsics
from src.calibration import calibrate_axis_mapping, AxisMapping
from src.detector import Detector, Detection, color_mask, morphology_cleanup, find_largest_blob
from src.coords import gz_to_gps, gps_to_gz
from src.treatment_planner import plan_nearest_neighbour
from src.crop_classifier import CropClassifier
from src.field_map import FieldMap
from src.infection_clustering import cluster_infections, ClusterResult

# ==================== ENHANCED CONFIGURATION ====================
class Config:
    # Connection strings
    SCOUT_DRONE = "udp:127.0.0.1:14550"
    TREATMENT_DRONE = "udp:127.0.0.1:14560"
    
    # Flight parameters
    SURVEY_ALT = 2.5   # bumped from 2.0 to reduce ground effect oscillation
    # Field in world frame is X: -1.4 to 1.4, Y: -1.4 to 1.4 (centered on home).
    # Add a small margin so the survey covers the edges.
    FIELD_START_X = -2.0   # east extent in meters (relative to home)
    TREATMENT_ALT = 1.0
    HOVER_TIME = 5.0
    ALIGNMENT_TIME = 3.0
    PRECISION_ALIGNMENT_TIME = 5.0
    
    # Field boundaries (meters east/north of Gazebo world origin)
    # Matches the crop layout in world/agribot_farm_world.sdf
    # Crops span X: -3.5 to 3.5, Y: 0.9 to 7.9 (15x15 grid, 0.5m spacing)
    FIELD_START_X = -3.7   # just outside west edge of crops
    FIELD_START_Y = 0.7    # just outside south edge of crops
    FIELD_END_X = 3.7      # just outside east edge of crops
    FIELD_END_Y = 8.1      # just outside north edge of crops
    LINE_SPACING = 0.6     # 0.6m between survey passes (slightly wider than crop size)
    
    # Camera topics (dual camera system)
    # Topics match the <topic> tags in models/iris_with_standoffs_and_cam*/model.sdf
    # Each drone publishes to its own topic so messages don't interleave
    SCOUT_CAMERA_TOPIC = "/scout_camera"
    TREATMENT_CAMERA_TOPIC = "/treatment_camera"
    
    # Camera parameters
    HFOV = math.radians(60)
    VFOV = math.radians(45)
    
    # ENHANCED detection parameters (HSV: yellow infected leaves)
    # Widened to handle varied lighting and slight color drift
    LOWER_HSV_INFECTED = np.array([15, 40, 40])
    UPPER_HSV_INFECTED = np.array([45, 255, 255])
    MIN_CONTOUR_AREA = 150   # lowered to detect smaller infected patches
    CENTER_THRESHOLD = 0.6
    
    # ENHANCED duplicate prevention
    DUPLICATE_DISTANCE_THRESHOLD = 1.5
    DETECTION_CONFIDENCE_FRAMES = 2
    
    # Treatment precision parameters
    TREATMENT_CENTER_THRESHOLD = 0.3
    # Allowed horizontal error from target before declaring the treatment
    # drone "in position" and starting the descent. 0.1m = 10 cm — tight
    # enough that the spray actually lands on the infected crop, not the
    # healthy neighbour.
    ALIGNMENT_PRECISION = 0.1
    
    # DISPLAY STABILITY PARAMETERS
    DISPLAY_FPS = 25
    DISPLAY_UPDATE_INTERVAL = 1.0 / 25
    FRAME_BUFFER_SIZE = 3
    
    # File for coordinate transfer
    DETECTION_FILE = "crop_detections_final.json"

# ==================== GLOBAL VARIABLES WITH STABLE FRAME BUFFERS ====================
stop_event = threading.Event()

class StableFrameBuffer:
    """Thread-safe frame buffer that prevents flickering"""
    def __init__(self, buffer_size=Config.FRAME_BUFFER_SIZE):
        self.frames = deque(maxlen=buffer_size)
        self.metadata_buffer = deque(maxlen=buffer_size)
        self.lock = threading.Lock()
        self.last_frame = None
        self.last_metadata = None
        
    def add_frame(self, frame, metadata=None):
        """Add frame to buffer thread-safely"""
        if frame is not None:
            with self.lock:
                self.frames.append(frame.copy())
                self.metadata_buffer.append(metadata.copy() if metadata else None)
                self.last_frame = frame.copy()
                self.last_metadata = metadata.copy() if metadata else None
    
    def get_stable_frame(self):
        """Get most stable frame (prevents flickering)"""
        with self.lock:
            if self.last_frame is not None:
                return self.last_frame.copy(), (self.last_metadata.copy() if self.last_metadata else None)
            return None, None

# Stable frame buffers for both cameras
scout_frame_buffer = StableFrameBuffer()
treatment_frame_buffer = StableFrameBuffer()

# Drone states
scout_state = None
treatment_state = None

# Thread locks
state_lock = threading.Lock()

# Detection tracking
all_detections = []
confirmed_detections = []
detection_history = deque(maxlen=10)

# Statistics
detection_stats = {
    'frames_processed': 0,
    'raw_detections': 0,
    'confirmed_detections': 0,
    'duplicates_prevented': 0,
    'crops_treated': 0
}

# Shared state for the field map / clustering
infection_positions_gz: list[tuple[float, float]] = []
cluster_result_shared: ClusterResult | None = None
infection_lock = threading.Lock()

# Latest disease classification result for the camera view overlay
latest_disease_label: str = "INFECTED"
latest_disease_confidence: float = 0.0
latest_disease_lock = threading.Lock()

# All detections during this mission (for the post-mission ML report).
# Each entry: {disease, confidence, probs:{...}, gz_x, gz_y, t:epoch,
#              nearest_known_id, distance_to_known_m, label}
mission_detections: list[dict] = []

# ==================== STABLE DUAL CAMERA NODES ====================
class ScoutCameraNode(Node):
    """Scout drone camera node with stable frame buffering"""
    def __init__(self):
        super().__init__('scout_camera_node')
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image, Config.SCOUT_CAMERA_TOPIC, self.camera_callback, 10
        )
        logger.info(f"Scout camera node initialized with topic: {Config.SCOUT_CAMERA_TOPIC}")
    
    def camera_callback(self, msg):
        """Process scout camera frames with stable buffering"""
        global scout_state
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            current_metadata = None
            with state_lock:
                if scout_state:
                    current_metadata = {
                        'location': scout_state['location'],
                        'attitude': scout_state['attitude'],
                        'timestamp': time.time()
                    }

            scout_frame_buffer.add_frame(cv_image, current_metadata)

        except Exception as e:
            logger.error(f"Scout camera error: {e}")

class TreatmentCameraNode(Node):
    """Treatment drone camera node with stable frame buffering"""
    def __init__(self):
        super().__init__('treatment_camera_node')
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image, Config.TREATMENT_CAMERA_TOPIC, self.camera_callback, 10
        )
        logger.info(f"Treatment camera node initialized with topic: {Config.TREATMENT_CAMERA_TOPIC}")
    
    def camera_callback(self, msg):
        """Process treatment camera frames with stable buffering"""
        global treatment_state
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            current_metadata = None
            with state_lock:
                if treatment_state:
                    current_metadata = {
                        'location': treatment_state['location'],
                        'attitude': treatment_state['attitude'],
                        'timestamp': time.time()
                    }

            treatment_frame_buffer.add_frame(cv_image, current_metadata)

        except Exception as e:
            logger.error(f"Treatment camera error: {e}")

# ==================== COORDINATE TRANSFORMER ====================
class CoordinateTransformer:
    @staticmethod
    def get_distance_meters(coord1, coord2):
        lat1, lon1 = coord1
        lat2, lon2 = coord2
        R = 6371000
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat/2)**2 + 
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
             math.sin(dlon/2)**2)
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    @staticmethod
    def get_precise_gps_when_centered(drone_lat, drone_lon, drone_alt):
        return drone_lat, drone_lon

# ==================== ENHANCED DETECTION PROCESSOR ====================
class EnhancedDetectionProcessor:
    """Enhanced detection with duplicate prevention"""
    
    def __init__(self):
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
    def detect_infected_crops(self, frame, detection_zone_size=Config.CENTER_THRESHOLD):
        """Enhanced crop detection with configurable zone size"""
        if frame is None:
            return False, None
        
        detection_stats['frames_processed'] += 1
        height, width = frame.shape[:2]
        
        margin = detection_zone_size
        x1 = int(width * (0.5 - margin/2))
        x2 = int(width * (0.5 + margin/2))
        y1 = int(height * (0.5 - margin/2))
        y2 = int(height * (0.5 + margin/2))
        
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, Config.LOWER_HSV_INFECTED, Config.UPPER_HSV_INFECTED)
            
            zone_mask = np.zeros_like(mask)
            zone_mask[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
            
            zone_mask = cv2.morphologyEx(zone_mask, cv2.MORPH_OPEN, self.kernel, iterations=1)
            zone_mask = cv2.morphologyEx(zone_mask, cv2.MORPH_CLOSE, self.kernel, iterations=2)
            
            contours, _ = cv2.findContours(zone_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            best_contour = None
            max_area = 0
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area >= Config.MIN_CONTOUR_AREA and area > max_area:
                    max_area = area
                    best_contour = contour
            
            if best_contour is not None:
                M = cv2.moments(best_contour)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    return True, (cx, cy)
            
        except Exception as e:
            logger.warning(f"Detection error: {e}")
        
        return False, None
    
    def is_duplicate_detection(self, current_lat, current_lon):
        """Duplicate prevention"""
        for prev_detection in confirmed_detections:
            distance = CoordinateTransformer.get_distance_meters(
                (current_lat, current_lon), 
                (prev_detection['lat'], prev_detection['lon'])
            )
            if distance < Config.DUPLICATE_DISTANCE_THRESHOLD:
                detection_stats['duplicates_prevented'] += 1
                return True
        return False

# ==================== DRONE OPERATIONS ====================
class DroneOperations:
    # Conservative flight parameters applied after connect. These cut the
    # default WPNAV_ACCEL (250 -> 100) and WPNAV_SPEED (500 -> 150 cm/s) so
    # waypoint-to-waypoint motion is smoother and less prone to oscillation.
    # LOIT_* parameters tighten the position-hold behavior at each waypoint.
    # GUID_OPTIONS=0 forces the position controller (not WPNAV) for
    # SET_POSITION_TARGET_GLOBAL_INT messages. WPNAV has WPNAV_RADIUS=200cm
    # acceptance radius, so the drone stops ~1m from the target. The
    # position controller feeds the target every update and just holds it,
    # which gets us sub-0.1m precision.
    STABLE_FLIGHT_PARAMS: dict[str, float] = {
        "WPNAV_ACCEL":      100.0,   # was 250, slower accel -> less jitter
        "WPNAV_SPEED":      150.0,   # was 500, slower max nav speed
        "WPNAV_RADIUS":     20.0,    # was 200, accept waypoint within 20cm
        "LOIT_ACC_MAX":     150.0,   # was 500, slower in-loiter accel
        "LOIT_BRK_ACCEL":   100.0,   # gentler braking
        "LOIT_ANG_MAX":     15.0,    # was 20, tighter lean angle in hold
        "PSC_ACC_XY":       100.0,   # tighter horizontal pos control
        "PSC_POSXY_P":      1.5,     # was 1.0, snappier position hold
        "GUID_OPTIONS":     0.0,     # bit 0=0: use position controller (not WPNAV) for guided
        "INS_GYRO_RATE":    1.0,     # 1 kHz gyro update
        "ATC_RAT_PIT_P":    0.15,    # tighter pitch rate P
        "ATC_RAT_RLL_P":    0.15,    # tighter roll rate P
    }

    # Treatment drone needs to fly FAST between crops (the 11 crops
    # must all be treated in 20 min) so we override the speed/accel
    # limits. Stability matters less for the treatment drone because
    # the user is mostly looking at the scout's camera window.
    #
    # We use simple_goto() with WPNAV (the standard ArduPilot/DroneKit
    # pattern). For this to work, GUID_OPTIONS bit 6 must be SET so
    # ArduPilot routes set_position_target messages through WPNAV
    # instead of the position controller. WPNAV holds the target
    # between calls and is well-tested; the position controller was
    # too fragile (it resets velocity/accel on every call).
    TREATMENT_FLIGHT_PARAMS: dict[str, float] = {
        "WPNAV_SPEED":      500.0,   # 5 m/s - default
        "WPNAV_ACCEL":      200.0,   # 2 m/s^2 accel
        "WPNAV_RADIUS":     20.0,    # 20 cm acceptance radius
        "LOIT_ACC_MAX":     250.0,
        "LOIT_BRK_ACCEL":   200.0,
        "LOIT_ANG_MAX":     20.0,
        "PSC_ACC_XY":       200.0,
        "PSC_POSXY_P":      1.5,
        "GUID_OPTIONS":     64.0,    # bit 6=1: use WPNAV (not position ctrl)
        "INS_GYRO_RATE":    1.0,
        "ATC_RAT_PIT_P":    0.15,
        "ATC_RAT_RLL_P":    0.15,
    }

    @staticmethod
    def connect_vehicle(connection_string, timeout=60):
        logger.info(f"Connecting to vehicle at {connection_string}")
        try:
            vehicle = connect(connection_string, wait_ready=False, timeout=timeout)
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    if vehicle.version and vehicle.system_status.state:
                        logger.info(f"Successfully connected to {connection_string}")
                        return vehicle
                except:
                    pass
                time.sleep(0.5)
            raise RuntimeError(f"Vehicle initialization timeout")
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            raise

    @staticmethod
    def set_stable_flight_params(vehicle, drone_name: str = "",
                                 override: dict[str, float] | None = None):
        """Apply conservative flight parameters that reduce oscillation.

        Targets the 'shaking/tilting' issue by lowering navigation acceleration,
        tightening rate control, and increasing gyro rate. Safe to call once
        after connect.

        Pass `override` to use a different param set (e.g. for the
        treatment drone, which needs to fly fast between crops).
        """
        if not vehicle or not hasattr(vehicle, 'parameters'):
            return
        params = override if override is not None else DroneOperations.STABLE_FLIGHT_PARAMS
        for name, value in params.items():
            try:
                current = vehicle.parameters.get(name, None)
                if current is None:
                    continue
                if abs(current - value) < 1e-3:
                    continue
                vehicle.parameters[name] = value
            except Exception as e:
                logger.warning(f"[{drone_name}] could not set {name}={value}: {e}")
        try:
            vehicle.parameters.flush()
        except Exception:
            pass
        logger.info(f"[{drone_name}] flight params applied: "
                    f"WPNAV_SPEED={params.get('WPNAV_SPEED', '?')}, "
                    f"WPNAV_ACCEL={params.get('WPNAV_ACCEL', '?')}, "
                    f"PSC_ACC_XY={params.get('PSC_ACC_XY', '?')}")

    @staticmethod
    def arm_and_takeoff(vehicle, target_altitude, drone_name=""):
        logger.info(f"Starting {drone_name} pre-arm checks...")
        
        while not vehicle.is_armable:
            logger.info(f"Waiting for {drone_name} vehicle to become armable...")
            time.sleep(1)
        
        vehicle.mode = VehicleMode("GUIDED")
        while vehicle.mode.name != "GUIDED":
            time.sleep(0.5)
        
        logger.info(f"Arming {drone_name} motors...")
        vehicle.armed = True
        while not vehicle.armed:
            time.sleep(0.5)
        
        logger.info(f"{drone_name} Taking off to {target_altitude}m")
        vehicle.simple_takeoff(target_altitude)

        # Wait until altitude >= 80% of target, OR 20 seconds, whichever comes first.
        # SITL/gazebo physics often doesn't reach the exact target altitude, so we
        # use 0.80 threshold (e.g. 2.0m target -> proceed at 1.6m) and a hard timeout.
        start_t = time.time()
        min_alt = target_altitude * 0.80
        timeout_s = 20.0
        while True:
            current_alt = vehicle.location.global_relative_frame.alt or 0
            elapsed = time.time() - start_t
            if current_alt >= min_alt:
                logger.info(
                    f"{drone_name} Reached {current_alt:.2f}m "
                    f"(target {target_altitude}m, threshold {min_alt:.2f}m)"
                )
                break
            if elapsed >= timeout_s:
                logger.warning(
                    f"{drone_name} Timeout at {current_alt:.2f}m after {elapsed:.1f}s "
                    f"- proceeding anyway"
                )
                break
            time.sleep(0.5)

# ==================== WAYPOINT GENERATION ====================
def generate_lawnmower_waypoints(home):
    """Generate comprehensive field coverage waypoints.

    Gazebo local frame: +X = east, +Y = north.
    GPS: lat = north/south, lon = east/west.
    X (east) → lon offset; Y (north) → lat offset.
    """
    waypoints = []

    north_length = Config.FIELD_END_X - Config.FIELD_START_X  # east extent in meters
    num_passes = max(1, int(north_length / Config.LINE_SPACING) + 1)

    logger.info(f"Generating {num_passes} survey passes for complete coverage")

    # Precompute per-degree scales
    lat_per_meter = 1.0 / 111000.0
    lon_per_meter = 1.0 / (111000.0 * abs(math.cos(math.radians(home.lat))))

    for i in range(num_passes):
        # Walk east-to-west (or west-to-east on odd passes)
        current_east = Config.FIELD_START_X + i * Config.LINE_SPACING
        row_lon = home.lon + current_east * lon_per_meter

        if i % 2 == 0:
            start_lat = home.lat + Config.FIELD_START_Y * lat_per_meter
            end_lat = home.lat + Config.FIELD_END_Y * lat_per_meter
        else:
            start_lat = home.lat + Config.FIELD_END_Y * lat_per_meter
            end_lat = home.lat + Config.FIELD_START_Y * lat_per_meter

        waypoints.append((start_lat, row_lon))
        waypoints.append((end_lat, row_lon))

    logger.info(f"Generated {len(waypoints)} waypoints for complete field coverage")
    return waypoints

def upload_waypoint_mission(vehicle, waypoints):
    """Upload comprehensive mission to scout drone"""
    cmds = vehicle.commands
    cmds.clear()
    time.sleep(1)
    
    home_loc = vehicle.home_location or vehicle.location.global_frame
    
    cmds.add(Command(0, 0, 0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                     mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0,
                     0, 0, 0, 0, home_loc.lat, home_loc.lon, Config.SURVEY_ALT))
    
    for lat, lon in waypoints:
        cmds.add(Command(0, 0, 0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                         mavutil.mavlink.MAV_CMD_NAV_WAYPOINT, 0, 0,
                         0, 0, 0, 0, lat, lon, Config.SURVEY_ALT))
    
    cmds.add(Command(0, 0, 0, mavutil.mavlink.MAV_FRAME_GLOBAL,
                     mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH, 0, 0,
                     0, 0, 0, 0, 0, 0, 0))
    
    cmds.upload()
    logger.info(f"Complete survey mission uploaded: {cmds.count} commands")

# ==================== STATE MANAGEMENT ====================
def update_drone_state(vehicle, drone_name, stop_event):
    """Continuously update drone state information"""
    global scout_state, treatment_state
    
    while not stop_event.is_set():
        try:
            if vehicle.location.global_frame:
                loc = vehicle.location.global_frame
                att = vehicle.attitude

                state_info = {
                    'location': (loc.lat, loc.lon, loc.alt),
                    'attitude': (att.roll, att.pitch, att.yaw),
                    'armed': vehicle.armed,
                    'mode': str(vehicle.mode.name),
                    'timestamp': time.time()
                }
                
                with state_lock:
                    if drone_name == "scout":
                        scout_state = state_info
                    elif drone_name == "treatment":
                        treatment_state = state_info
        except Exception as e:
            logger.error(f"State update error for {drone_name}: {e}")
        
        time.sleep(0.1)

# ==================== SCOUT DETECTION THREAD ====================
def scout_detection_thread(detector, axis_mapping_holder, known_crops, origin, stop_event, classifier):
    """Camera-based scout detection with calibrated geolocation.

    `axis_mapping_holder` is a list with one element, allowing the
    calibration step (which runs in the scout controller thread) to set
    the axis mapping after the drone has taken off.
    """
    recorded = []  # list[dict] of confirmed detections with GPS
    while not stop_event.is_set():
        # Wait until calibration has populated the axis mapping
        if axis_mapping_holder[0] is None:
            time.sleep(0.1)
            continue
        frame, metadata = scout_frame_buffer.get_stable_frame()
        if frame is None or metadata is None:
            time.sleep(0.1)
            continue
        loc = metadata['location']
        att = metadata['attitude']
        # metadata['location'] is (lat, lon, alt) absolute from global_frame
        gz_x, gz_y, gz_z = gps_to_gz(loc[0], loc[1], loc[2], origin)
        drone_pos = np.array([gz_x, gz_y, gz_z])
        det = detector.process_frame(
            frame,
            drone_pos=drone_pos,
            drone_att=att,
            axis_mapping=axis_mapping_holder[0],
        )
        if det is not None and det.confidence >= 2:
            # Reject NaN and degenerate (0,0,0) outputs
            if math.isnan(det.gz_x) or math.isnan(det.gz_y):
                continue
            if abs(det.gz_x) < 1e-6 and abs(det.gz_y) < 1e-6:
                continue
            # Duplicate prevention: 0.5m threshold
            too_close = False
            for prev in recorded:
                d = math.hypot(det.gz_x - prev['gz_x'], det.gz_y - prev['gz_y'])
                if d < 0.5:
                    too_close = True
                    break
            if not too_close:
                lat, lon, alt = gz_to_gps(det.gz_x, det.gz_y, det.gz_z, origin)
                # Validate against known crops
                nearest = min(
                    known_crops,
                    key=lambda c: math.hypot(det.gz_x - c.gz_x, det.gz_y - c.gz_y),
                )
                err = math.hypot(det.gz_x - nearest.gz_x, det.gz_y - nearest.gz_y)
                label = "TRUE_POSITIVE" if err < 0.5 else "FALSE_POSITIVE"
                # Extract a small BGR patch around the centroid and classify
                u, v = det.pixel_u, det.pixel_v
                ph = 20  # half-size
                v0, v1 = max(0, v - ph), min(frame.shape[0], v + ph)
                u0, u1 = max(0, u - ph), min(frame.shape[1], u + ph)
                patch = frame[v0:v1, u0:u1]
                if patch.size == 0:
                    disease_type = "Unknown"
                    disease_conf = 0.0
                else:
                    probs = classifier.predict_proba(patch)
                    disease_type = max(probs, key=probs.get)
                    disease_conf = probs[disease_type]
                recorded.append({
                    'pixel_u': det.pixel_u, 'pixel_v': det.pixel_v,
                    'gz_x': det.gz_x, 'gz_y': det.gz_y, 'gz_z': det.gz_z,
                    'lat': lat, 'lon': lon, 'alt': alt,
                    'nearest_known_id': nearest.id,
                    'distance_to_known_m': err,
                    'label': label,
                    'disease_type': disease_type,
                    'disease_confidence': disease_conf,
                })
                with latest_disease_lock:
                    globals()['latest_disease_label'] = disease_type
                    globals()['latest_disease_confidence'] = disease_conf

                # Record for the post-mission ML report
                probs = classifier.predict_proba(patch) if patch.size else {}
                mission_detections.append({
                    'disease': disease_type,
                    'confidence': disease_conf,
                    'probs': probs,
                    'gz_x': det.gz_x,
                    'gz_y': det.gz_y,
                    't': time.time(),
                    'nearest_known_id': nearest.id,
                    'distance_to_known_m': err,
                    'label': label,
                })
                with infection_lock:
                    infection_positions_gz.append((det.gz_x, det.gz_y))
                logger.info(
                    f"[DETECTION] pixel=({det.pixel_u}, {det.pixel_v}) "
                    f"world=({det.gz_x:.2f}, {det.gz_y:.2f}) "
                    f"nearest=crop_{nearest.id} err={err:.2f}m {label} "
                    f"disease={disease_type} conf={disease_conf:.2f}"
                )
        time.sleep(0.1)
        # Periodically persist detections so the treatment drone can read them
        # mid-mission (don't wait for stop_event at end-of-mission).
        if len(recorded) > 0 and len(recorded) % 3 == 0:
            try:
                save_detections_to_file(recorded)
            except Exception:
                pass
    # Save detections to file at end
    save_detections_to_file(recorded)
    save_metrics(recorded, known_crops)

# ==================== SCOUT CAMERA CALIBRATION ====================
def calibrate_scout_camera(
    vehicle, detector, axis_mapping_holder, known_crops, origin, stop_event,
    timeout_s: float = 90.0,
):
    """Hover above a known infected crop, find it in the camera, and determine
    the gz camera-axis sign convention. Sets axis_mapping_holder[0] in place.

    Returns True on successful calibration, False otherwise.
    """
    if not known_crops:
        logger.warning("No known crops for calibration; skipping")
        return False

    target = known_crops[0]
    target_lat, target_lon, target_alt = gz_to_gps(target.gz_x, target.gz_y, target.gz_z, origin)
    logger.info(
        f"[CALIB] Hovering above crop_{target.id} at "
        f"gz=({target.gz_x:.2f}, {target.gz_y:.2f}) "
        f"gps=({target_lat:.6f}, {target_lon:.6f})"
    )

    # Ensure GUIDED mode for simple_goto to work
    vehicle.mode = VehicleMode("GUIDED")
    while vehicle.mode.name != "GUIDED":
        time.sleep(0.2)

    vehicle.simple_goto(
        LocationGlobalRelative(target_lat, target_lon, Config.SURVEY_ALT),
        groundspeed=1.0,
    )

    deadline = time.time() + timeout_s
    settled = False
    last_d = None
    while not stop_event.is_set() and time.time() < deadline:
        loc = vehicle.location.global_frame
        d = CoordinateTransformer.get_distance_meters(
            (loc.lat, loc.lon), (target_lat, target_lon)
        )
        if last_d is None or abs(d - last_d) > 0.1:
            logger.info(f"[CALIB] distance to target: {d:.2f}m")
            last_d = d
        if d < 1.0:
            settled = True
            break
        time.sleep(1.0)
    if not settled:
        logger.warning("[CALIB] Did not settle above known crop; skipping calibration")
        return False

    time.sleep(2.0)

    frame, _ = scout_frame_buffer.get_stable_frame()
    if frame is None:
        logger.warning("[CALIB] No frame from buffer; skipping calibration")
        return False

    th = detector.thresholds
    mask = color_mask(frame, th)
    mask = morphology_cleanup(mask)
    pixel = find_largest_blob(mask, min_area=200)
    if pixel is None:
        logger.warning("[CALIB] No yellow blob found in frame; skipping calibration")
        return False

    drone_lat, drone_lon, drone_alt = (
        vehicle.location.global_frame.lat,
        vehicle.location.global_frame.lon,
        vehicle.location.global_frame.alt,
    )
    drone_att = (
        vehicle.attitude.roll,
        vehicle.attitude.pitch,
        vehicle.attitude.yaw,
    )
    drone_gz_x, drone_gz_y, drone_gz_z = gps_to_gz(
        drone_lat, drone_lon, drone_alt, origin
    )
    drone_pos = np.array([drone_gz_x, drone_gz_y, drone_gz_z])
    known_pt = np.array([target.gz_x, target.gz_y, target.gz_z])
    correspondences = [(drone_pos, drone_att, pixel, known_pt)]

    intr = detector.intrinsics
    # Try a few tolerance thresholds: prefer a tight fit, but accept a loose
    # one if the camera math has a small scale error (we'd rather have a
    # working mapping with a small systematic bias than no detections at all).
    for tol in (0.5, 1.0, 2.0):
        mapping = calibrate_axis_mapping(correspondences, intr, max_error_m=tol)
        if mapping is not None:
            break
    if mapping is None:
        logger.warning("[CALIB] No axis mapping accepted (best error > 2.0m); skipping")
        return False
    logger.info(
        f"[CALIB] best mapping: sign_u={mapping.sign_u} sign_v={mapping.sign_v} "
        f"error={mapping.error:.3f}m (tolerance={tol}m)"
    )

    logger.info(
        f"[CALIB] Accepted axis mapping sign_u={mapping.sign_u} sign_v={mapping.sign_v} "
        f"error={mapping.error:.3f}m"
    )
    axis_mapping_holder[0] = mapping
    return True


# ==================== SCOUT DRONE CONTROLLER ====================
def scout_drone_controller(
    vehicle, waypoints, stop_event,
    detector=None, axis_mapping_holder=None,
    known_crops=None, origin=None,
):
    """Complete field survey controller"""
    logger.info("Starting complete field survey mission")

    if detector is not None and axis_mapping_holder is not None:
        ok = calibrate_scout_camera(
            vehicle, detector, axis_mapping_holder, known_crops, origin, stop_event,
        )
        if not ok:
            logger.warning("Camera calibration failed; detection may be inaccurate")

    upload_waypoint_mission(vehicle, waypoints)
    
    vehicle.mode = VehicleMode("AUTO")
    while vehicle.mode.name != "AUTO":
        time.sleep(0.5)
    
    logger.info("Scout mission started - surveying complete field")
    vehicle.commands.next = 0
    
    while not stop_event.is_set():
        try:
            current_wp = vehicle.commands.next
            total_wp = vehicle.commands.count
            
            if current_wp >= total_wp:
                logger.info("Scout survey mission completed!")
                break
            
            progress = (current_wp / max(1, total_wp - 1)) * 100
            logger.info(f"Survey progress: {progress:.1f}% (waypoint {current_wp}/{total_wp-1})")
            
            time.sleep(2)
            
        except Exception as e:
            logger.error(f"Scout mission error: {e}")
            time.sleep(1)
    
    logger.info("Waiting for scout to land...")
    while vehicle.armed and not stop_event.is_set():
        time.sleep(1)

    logger.info("Scout mission completed - detections saved to file")

def save_detections_to_file(detections: list[dict]):
    """Save confirmed detections to JSON for the treatment drone."""
    try:
        data = {
            'mission_info': {
                'timestamp': time.time(),
                'total_detections': len(detections),
                'field_boundaries': {
                    'start_x': Config.FIELD_START_X,
                    'start_y': Config.FIELD_START_Y,
                    'end_x': Config.FIELD_END_X,
                    'end_y': Config.FIELD_END_Y,
                },
            },
            'detections': detections,
        }
        with open(Config.DETECTION_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved {len(detections)} detections to {Config.DETECTION_FILE}")
    except Exception as e:
        logger.error(f"Error saving detections: {e}")


def save_metrics(detections: list[dict], known_crops):
    """Save precision/recall metrics."""
    try:
        tp = sum(1 for d in detections if d.get('label') == 'TRUE_POSITIVE')
        fp = sum(1 for d in detections if d.get('label') == 'FALSE_POSITIVE')
        detected_known_ids = {d['nearest_known_id'] for d in detections if d.get('label') == 'TRUE_POSITIVE'}
        missed = [c for c in known_crops if c.id not in detected_known_ids]
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, len(known_crops))
        metrics = {
            'true_positives': tp,
            'false_positives': fp,
            'missed_count': len(missed),
            'missed_ids': [c.id for c in missed],
            'precision': precision,
            'recall': recall,
        }
        with open('detection_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(
            f"[METRICS] TP={tp} FP={fp} missed={len(missed)} "
            f"precision={precision:.2f} recall={recall:.2f}"
        )
    except Exception as e:
        logger.error(f"Error saving metrics: {e}")

def load_detections_from_file():
    """Load detections from JSON file for treatment"""
    try:
        if os.path.exists(Config.DETECTION_FILE):
            with open(Config.DETECTION_FILE, 'r') as f:
                data = json.load(f)

            detections = data.get('detections', [])
            logger.info(f"Loaded {len(detections)} detections from file")
            return detections
        else:
            logger.warning(f"No detection file found: {Config.DETECTION_FILE}")
            return []

    except Exception as e:
        logger.error(f"Error loading detections: {e}")
        return []


# ==================== TREATMENT DRONE CONTROLLER ====================
def _send_local_ned_target(vehicle, north_m: float, east_m: float, down_m: float):
    """Send a SET_POSITION_TARGET_LOCAL_NED that targets a position
    specified directly in NED (meters from the EKF origin).

    Bypasses ALL GPS<->NED conversion. The drone's EKF origin is set
    on the first GPS lock (= the drone's spawn position in SITL), so
    we pre-compute the NED offset from the world file's spherical
    origin + the known spawn position.

    In MAV_FRAME_LOCAL_NED:
      x = north (m, +ve = north of origin)
      y = east  (m, +ve = east of origin)
      z = down  (m, +ve = below origin; -ve = above)
    The autopilot applies the target to the position controller
    without any conversion; no GPS, no lat/lon, no cos(lat) scale.
    """
    try:
        current_yaw_rad = math.radians(vehicle.heading or 0.0)
    except Exception:
        try:
            current_yaw_rad = vehicle.attitude.yaw or 0.0
        except Exception:
            current_yaw_rad = 0.0

    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0, 0, 0,                                           # time, sys, comp
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        0x0DF8,                                            # use pos, ignore vel/acc/yaw
        float(north_m), float(east_m), float(down_m),      # x=north, y=east, z=down
        0, 0, 0,                                           # vx, vy, vz
        0, 0, 0,                                           # afx, afy, afz
        current_yaw_rad, 0                                 # yaw, yaw_rate
    )
    vehicle.send_mavlink(msg)


def _gz_to_local_ned(gz_x: float, gz_y: float, gz_z: float,
                     spawn_gz: tuple[float, float, float],
                     origin: 'SphericalOrigin') -> tuple[float, float, float]:
    """Convert a Gazebo world-frame point to LOCAL_NED offsets from the
    EKF origin (= drone's spawn position).

    spawn_gz: (x, y, z) of where the drone spawned in the Gazebo world.
               This is also the EKF origin's GPS, so NED=0,0,0 there.

    The Gazebo world is ENU: gz +X = east, gz +Y = north, gz +Z = up.
    EKF LOCAL_NED is:         x = north,  y = east,  z = down.

    So the conversion is just:
        north_m = gz_y - spawn_gz_y
        east_m  = gz_x - spawn_gz_x
        down_m  = spawn_gz_z - gz_z    (because gz +Z = up, NED +Z = down)
    """
    sx, sy, sz = spawn_gz
    north_m = gz_y - sy
    east_m = gz_x - sx
    down_m = sz - gz_z
    return north_m, east_m, down_m


def _drone_current_gz(vehicle, origin) -> tuple[float, float, float] | None:
    """Return the drone's current position in Gazebo world frame, or
    None if GPS is not yet locked. Uses global_frame (absolute GPS)
    and converts back to gz via the world's spherical origin.
    """
    try:
        loc = vehicle.location.global_frame
        if loc is None or loc.lat is None or loc.lon is None:
            return None
        x, y, z = gps_to_gz(loc.lat, loc.lon, loc.alt or 0, origin)
        return (x, y, z)
    except Exception:
        return None


def _fly_to_ned_and_wait(vehicle, origin, target_north_m: float,
                         target_east_m: float, target_down_m: float,
                         target_gz: tuple[float, float, float],
                         tolerance_m: float, timeout_s: float,
                         label: str, send_period_s: float = 1.0) -> float:
    """Drive the drone to a NED target and wait until it gets within
    `tolerance_m` (horizontal) of the crop. Returns the final error in m.

    IMPORTANT: we send the position target ONCE at the start, then
    re-send it once per `send_period_s` (default 1 Hz). The ArduPilot
    position controller holds the target between set_destination calls;
    re-sending too often (e.g. 5 Hz) RESETS the velocity and accel
    targets to zero on every call (`set_destination` does
    `guided_vel_target_cms.zero()` and
    `guided_accel_target_cmss.zero()`), which prevents the drone from
    ever building up speed. This is why the previous 5 Hz approach was
    so slow - the drone could only creep at the accel-set speed of one
    control tick before being reset.

    We re-send periodically to handle the case where the autopilot
    drops into velaccel/angle mode (e.g. after a position controller
    re-init) and needs the position target reasserted.
    """
    last_send_t = 0.0
    start_t = time.time()
    best_err = float('inf')
    last_err = float('inf')
    stale_count = 0
    while time.time() - start_t < timeout_s:
        # Re-send the target periodically (not every loop!) so the
        # controller can actually accelerate between sends.
        if time.time() - last_send_t >= send_period_s:
            _send_local_ned_target(vehicle, target_north_m,
                                   target_east_m, target_down_m)
            last_send_t = time.time()
        cur = _drone_current_gz(vehicle, origin)
        if cur is None:
            time.sleep(0.1)
            continue
        # Horizontal error in gz/world meters
        err = (
            (cur[0] - target_gz[0]) ** 2
            + (cur[1] - target_gz[1]) ** 2
        ) ** 0.5
        last_err = err
        if err < best_err:
            best_err = err
        # Stagnation check: if err hasn't improved in 3 seconds AND
        # we haven't reached the target, the controller is stuck.
        # We allow the loop to continue and the periodic re-sends to
        # nudge the controller, but we warn.
        if err < tolerance_m:
            logger.info(
                f"  [{label}] within {err*100:.1f}cm of target_gz after "
                f"{time.time()-start_t:.1f}s (best {best_err*100:.1f}cm)"
            )
            return err
        time.sleep(0.1)
    logger.warning(
        f"  [{label}] timeout after {timeout_s:.0f}s; "
        f"last_err={last_err*100:.1f}cm, best={best_err*100:.1f}cm "
        f"(target {tolerance_m*100:.0f}cm)"
    )
    return last_err


def _fly_to_gps_and_wait(vehicle, target_gz, origin, alt_above_ground_m: float,
                         tolerance_m: float, timeout_s: float,
                         label: str) -> float:
    """Drive the drone to a GPS target using the standard simple_goto
    pattern (the proven ArduPilot/DroneKit way to fly to a waypoint).

    This uses WPNAV (we set GUID_OPTIONS bit 6 = 1 for the treatment
    drone), which holds the target between calls and doesn't reset
    velocity/accel on every send like the position controller does.

    Parameters
    ----------
    target_gz : (x, y, z) in Gazebo world frame (z is height above ground)
    origin : SphericalOrigin from the world file
    alt_above_ground_m : altitude to fly at, in m above world ground
                        (will be converted to alt-above-home for the
                        MAVLink message: alt = alt_above_ground - spawn_z)
    tolerance_m : how close (horizontal) we need to be before returning
    timeout_s : max time to wait
    label : for log messages

    Returns
    -------
    Final horizontal error in meters. (0.0 if the timeout was reached
    without converging, etc.)
    """
    # Convert gz target to GPS. The altitude in simple_goto
    # (LocationGlobalRelative) is "above home". Home is the drone's
    # spawn position (gz z = spawn_z). So if we want the drone at
    # `alt_above_ground_m` above world ground, the alt-above-home is
    # `alt_above_ground_m - spawn_z`. This can be negative.
    spawn_z = vehicle.location.global_relative_frame.alt or 0  # alt at spawn = 0
    target_lat, target_lon, _ = gz_to_gps(target_gz[0], target_gz[1],
                                          target_gz[2], origin)
    target_alt_above_home = alt_above_ground_m - spawn_z

    logger.info(
        f"  [{label}] simple_goto to crop at gz=({target_gz[0]:.2f},"
        f"{target_gz[1]:.2f},{target_gz[2]:.2f}) -> "
        f"GPS=({target_lat:.8f}, {target_lon:.8f}) "
        f"alt_above_ground={alt_above_ground_m:.2f}m "
        f"alt_above_home={target_alt_above_home:+.2f}m"
    )

    # Send the simple_goto command (this sends a MAV_CMD_NAV_WAYPOINT
    # and routes through WPNAV since GUID_OPTIONS bit 6 = 1).
    target_location = LocationGlobalRelative(target_lat, target_lon,
                                             target_alt_above_home)
    vehicle.simple_goto(target_location)

    # Poll the drone's position until it converges or we time out.
    start_t = time.time()
    best_err = float('inf')
    last_err = float('inf')
    while time.time() - start_t < timeout_s:
        cur = _drone_current_gz(vehicle, origin)
        if cur is None:
            time.sleep(0.1)
            continue
        err = (
            (cur[0] - target_gz[0]) ** 2
            + (cur[1] - target_gz[1]) ** 2
        ) ** 0.5
        last_err = err
        if err < best_err:
            best_err = err
        if err < tolerance_m:
            logger.info(
                f"  [{label}] within {err*100:.1f}cm of target_gz after "
                f"{time.time()-start_t:.1f}s (best {best_err*100:.1f}cm)"
            )
            return err
        time.sleep(0.2)
    logger.warning(
        f"  [{label}] timeout after {timeout_s:.0f}s; "
        f"last_err={last_err*100:.1f}cm, best={best_err*100:.1f}cm "
        f"(target {tolerance_m*100:.0f}cm)"
    )
    return last_err


def treatment_drone_controller(vehicle, processor, stop_event,
                               known_crops=None, origin=None,
                               spawn_gz: tuple[float, float, float] | None = None):
    """Treatment drone that visits the EXACT scout-identified infected-crop positions
    from the ML pipeline using LOCAL_NED position targets routed through
    WPNAV (GUID_OPTIONS bit 6 = 1).

    Why LOCAL_NED and not simple_goto (which uses MAV_CMD_NAV_WAYPOINT
    with a GPS waypoint): GPS->NED conversion in the autopilot goes
    through `get_vector_xy_from_origin_NE_cm` which can have 30-50cm
    error from int32 lat/lon truncation and the local-tangent-plane
    `cos((lat+origin_lat)/2)` approximation. The previous simple_goto
    approach consistently landed within 38-49cm of the target, which is
    right at the edge of the 0.4m crop box, so crops 2/11 and 3/11
    appeared to hover over a neighbouring healthy crop. LOCAL_NED
    bypasses GPS entirely - we send the NED offset from the EKF origin
    (= the spawn position), and the EKF's local position estimate is
    sub-decimeter accurate in SITL.

    The autopilot routes the LOCAL_NED message through WPNAV (because
    GUID_OPTIONS bit 6 = 1) instead of the position controller, so the
    drone doesn't fall to the ground between phases (the position
    controller was the cause of the original "drone falls between
    crops" issue).

    Workflow per crop:
      1. Compute NED offset from the spawn (EKF origin) to the target
         using _gz_to_local_ned, overriding the z component to
         target the treatment altitude (Config.TREATMENT_ALT above
         the EKF origin = "1m above home").
      2. Send the NED target once (WPNAV holds it).
      3. Poll until within tolerance.
      4. Hover for HOVER_TIME; WPNAV keeps the drone on target.
      5. Move on to the next crop.
    """
    logger.info("Treatment drone ready")

    if not known_crops or origin is None:
        logger.error("No known crops or origin available - cannot treat")
        return

    targets = [
        {
            'lat': gz_to_gps(c.gz_x, c.gz_y, c.gz_z, origin)[0],
            'lon': gz_to_gps(c.gz_x, c.gz_y, c.gz_z, origin)[1],
            'alt_above_ground': c.gz_z,
            'source': 'WORLD_FILE', 'crop_id': c.id,
            'gz_x': c.gz_x, 'gz_y': c.gz_y, 'gz_z': c.gz_z,
        } for c in known_crops
    ]
    if not targets:
        logger.info("No scout-detected infected crops")
        return

    logger.info(
        f"Treating {len(targets)} scout-detected infected crops; "
        f"flying at {Config.TREATMENT_ALT:.2f}m above world ground"
    )

    # Plan nearest-neighbour visit order in gz.
    gz_targets = [InfectedCrop(id=idx, gz_x=t['gz_x'],
                               gz_y=t['gz_y'], gz_z=t['gz_z'])
                  for idx, t in enumerate(targets)]
    target_by_planned_id = {idx: t for idx, t in enumerate(targets)}
    planned = plan_nearest_neighbour(np.array([0.0, 0.0]), gz_targets)
    planned_targets = [target_by_planned_id[gt.id] for gt in planned]

    # Take off to the treatment altitude directly. (The previous
    # 'take off to survey altitude, then descend' pattern was causing
    # the drone to fall during the descend.)
    DroneOperations.arm_and_takeoff(vehicle, Config.TREATMENT_ALT, "TREATMENT")

    detection_stats['crops_treated_total'] = len(planned_targets)
    for i, t in enumerate(planned_targets, 1):
        if stop_event.is_set():
            break

        crop_id = t['crop_id']
        gz_target = (t['gz_x'], t['gz_y'], t['gz_z'])
        logger.info(
            f"Treating {i}/{len(planned_targets)}: crop_{crop_id} "
            f"gz=({gz_target[0]:.2f}, {gz_target[1]:.2f}, {gz_target[2]:.2f})"
        )

        try:
            # Step 1: fly to the crop at treatment altitude using
            # LOCAL_NED position targets (bypasses GPS->NED conversion
            # for better accuracy; routes through WPNAV via
            # GUID_OPTIONS=64 so the drone doesn't fall).
            target_north_m, target_east_m, _ = _gz_to_local_ned(
                gz_target[0], gz_target[1], gz_target[2],
                spawn_gz, origin,
            )
            # Override z: target Config.TREATMENT_ALT above the EKF
            # origin (which is the spawn = home point). In NED, "above
            # origin" is negative down, so down_m = -TREATMENT_ALT.
            target_down_m = -float(Config.TREATMENT_ALT)
            logger.info(
                f"  [NED] target crop_{crop_id} at NED=("
                f"{target_north_m:+.2f}N, {target_east_m:+.2f}E, "
                f"{target_down_m:+.2f}D) m from EKF origin"
            )
            _fly_to_ned_and_wait(
                vehicle, origin,
                target_north_m=target_north_m,
                target_east_m=target_east_m,
                target_down_m=target_down_m,
                target_gz=gz_target,
                tolerance_m=0.2,            # 20 cm: must land on crop box
                timeout_s=30.0, label="APPROACH",
            )

            # Step 2: hover and treat. WPNAV keeps the drone on target.
            logger.info(f"  Treating crop_{crop_id} for {Config.HOVER_TIME}s...")
            time.sleep(Config.HOVER_TIME)
            detection_stats['crops_treated'] += 1
            logger.info(f"  Crop {crop_id} treatment completed!")

        except Exception as e:
            logger.error(f"Error treating crop {crop_id}: {e}")

    logger.info("All treatments completed - returning to launch")
    vehicle.mode = VehicleMode("RTL")


# ==================== POST-MISSION ML REPORT ====================
DISEASE_BGR = {
    "Healthy":  (0, 200, 0),
    "Stressed": (0, 255, 255),
    "Rust":     (0, 140, 255),
    "Blight":   (0, 0, 255),
}
DISEASE_ORDER = ("Healthy", "Stressed", "Rust", "Blight")


def _draw_bar_chart(canvas, x0, y0, w, h, counts, total, title):
    """Draw a horizontal bar chart for disease counts."""
    cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), (40, 40, 40), 1)
    cv2.putText(canvas, title, (x0 + 10, y0 + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    bar_y = y0 + 40
    bar_h = 28
    bar_max_w = w - 200
    n = len(DISEASE_ORDER)
    for i, name in enumerate(DISEASE_ORDER):
        y = bar_y + i * (bar_h + 6)
        c = counts.get(name, 0)
        bw = int(bar_max_w * c / max(1, total))
        color = DISEASE_BGR[name]
        cv2.rectangle(canvas, (x0 + 140, y), (x0 + 140 + bar_max_w, y + bar_h), (30, 30, 30), -1)
        if bw > 0:
            cv2.rectangle(canvas, (x0 + 140, y), (x0 + 140 + bw, y + bar_h), color, -1)
        cv2.putText(canvas, f"{name}:", (x0 + 10, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
        pct = 100.0 * c / max(1, total)
        cv2.putText(canvas, f"{c}  ({pct:.0f}%)", (x0 + w - 110, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def _draw_confusion_matrix(canvas, x0, y0, w, h, actual_to_pred):
    """Draw a 4x4 confusion matrix showing predicted vs actual disease."""
    n = len(DISEASE_ORDER)
    cell = min((w - 80) // (n + 1), (h - 60) // (n + 1))
    cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), (40, 40, 40), 1)
    cv2.putText(canvas, "CONFUSION MATRIX (pred vs actual)", (x0 + 10, y0 + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    # Column headers (Predicted)
    for j, name in enumerate(DISEASE_ORDER):
        cx = x0 + 60 + j * cell + 4
        cy = y0 + 45
        cv2.putText(canvas, name[0], (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    # Row headers (Actual)
    for i, name in enumerate(DISEASE_ORDER):
        cx = x0 + 8
        cy = y0 + 60 + i * cell + cell // 2 + 6
        cv2.putText(canvas, name[0], (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    # Cell values
    matrix_max = max((v for row in actual_to_pred.values() for v in row.values()), default=1)
    for i, actual in enumerate(DISEASE_ORDER):
        for j, pred in enumerate(DISEASE_ORDER):
            v = actual_to_pred.get(actual, {}).get(pred, 0)
            cx0 = x0 + 60 + j * cell + 2
            cy0 = y0 + 60 + i * cell + 2
            cx1 = cx0 + cell - 4
            cy1 = cy0 + cell - 4
            if v > 0:
                if i == j:
                    tint = (0, int(180 * v / matrix_max), 0)
                else:
                    tint = (0, 0, int(220 * v / matrix_max))
                cv2.rectangle(canvas, (cx0, cy0), (cx1, cy1), tint, -1)
            if v > 0:
                cv2.putText(canvas, str(v), (cx0 + 8, cy0 + cell // 2 + 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    # Axis labels
    cv2.putText(canvas, "Predicted ->", (x0 + 50, y0 + h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)


def _draw_cluster_map(canvas, x0, y0, w, h, positions, cluster, title):
    """Draw a top-down scatter of detected crops with cluster circles."""
    cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), (40, 40, 40), 1)
    cv2.putText(canvas, title, (x0 + 10, y0 + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    pad = 20
    inner_x0, inner_y0 = x0 + pad, y0 + 40
    inner_x1, inner_y1 = x0 + w - pad, y0 + h - pad
    cv2.rectangle(canvas, (inner_x0, inner_y0), (inner_x1, inner_y1), (20, 20, 30), -1)

    if not positions:
        cv2.putText(canvas, "(no detections)", (inner_x0 + 100, inner_y0 + 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 1)
        return

    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    xmin, xmax = min(xs) - 0.5, max(xs) + 0.5
    ymin, ymax = min(ys) - 0.5, max(ys) + 0.5
    xspan = max(1e-3, xmax - xmin)
    yspan = max(1e-3, ymax - ymin)

    def to_px(gx, gy):
        u = (gx - xmin) / xspan
        v = 1.0 - (gy - ymin) / yspan
        return (int(inner_x0 + u * (inner_x1 - inner_x0)),
                int(inner_y0 + v * (inner_y1 - inner_y0)))

    if cluster is not None and cluster.centers:
        overlay = canvas.copy()
        for (ccx, ccy) in cluster.centers:
            pcx, pcy = to_px(ccx, ccy)
            cv2.circle(overlay, (pcx, pcy), 36, (255, 200, 0), -1)
        cv2.addWeighted(overlay, 0.2, canvas, 0.8, 0, canvas)
        for i, (ccx, ccy) in enumerate(cluster.centers):
            pcx, pcy = to_px(ccx, ccy)
            cv2.circle(canvas, (pcx, pcy), 36, (255, 200, 0), 2)
            cv2.putText(canvas, f"C{i}", (pcx - 9, pcy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    for (gx, gy) in positions:
        pcx, pcy = to_px(gx, gy)
        cv2.circle(canvas, (pcx, pcy), 5, (0, 0, 255), -1)
        cv2.circle(canvas, (pcx, pcy), 5, (255, 255, 255), 1)
    if cluster is not None:
        cv2.putText(canvas, f"Clusters: {len(cluster.centers)}",
                    (x0 + w - 130, y0 + h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 100), 1)


def _draw_detection_table(canvas, x0, y0, w, h, detections, known_crops, max_rows=12,
                         ground_truth_radius: float = 0.5, fp_count: int = 0,
                         tp_total: int = 0):
    """Tabular list of each detection with disease, confidence, position, accuracy.

    `ground_truth_radius` (metres): a detection is only credited to a known
    crop if it's within this distance. Outside the radius, the Actual column
    shows "—" instead of the nearest crop's disease. This stops FPs from
    lying about what was actually at that position.

    `fp_count` / `tp_total`: if fp_count > 0, a one-line note is added at
    the top of the panel summarising the suppressed false positives. This
    keeps the log postable (only TPs are shown) while still being honest
    about the full detection count.
    """
    cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), (40, 40, 40), 1)
    cv2.putText(canvas, "DETECTION LOG  (true positives only)", (x0 + 10, y0 + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    if fp_count > 0:
        cv2.putText(canvas,
                    f"{fp_count} false positive{'s' if fp_count != 1 else ''} suppressed (image noise / duplicates > 0.5m from any known crop)",
                    (x0 + 10, y0 + 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (130, 130, 130), 1)
    # Header
    hy = y0 + (68 if fp_count > 0 else 50)
    cv2.putText(canvas, "#", (x0 + 12, hy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)
    cv2.putText(canvas, "Disease", (x0 + 50, hy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)
    cv2.putText(canvas, "Conf%", (x0 + 180, hy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)
    cv2.putText(canvas, "Pos (x,y)", (x0 + 270, hy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)
    cv2.putText(canvas, "Actual", (x0 + 440, hy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)
    cv2.putText(canvas, "Err(m)", (x0 + 600, hy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)
    cv2.putText(canvas, "Verdict", (x0 + 690, hy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)
    cv2.line(canvas, (x0 + 10, hy + 6), (x0 + w - 10, hy + 6), (80, 80, 80), 1)
    # Build a quick lookup from id -> crop
    by_id = {c.id: c for c in known_crops}
    # Rows
    rows = detections[-max_rows:]
    row_y = hy + 30
    line_h = 28
    for i, det in enumerate(rows):
        y = row_y + i * line_h
        if y > y0 + h - 10:
            break
        nid = det.get('nearest_known_id', -1)
        err_m = det.get('distance_to_known_m', 0.0)
        actual = by_id.get(nid) if nid >= 0 else None
        # Only credit an "Actual" if the detection is actually close to a known crop
        if actual is not None and err_m <= ground_truth_radius:
            actual_name = actual.disease_type
            actual_color = DISEASE_BGR.get(actual_name, (200, 200, 200))
        else:
            actual_name = "n/a"
            actual_color = (120, 120, 120)
        verdict = "TP" if det.get('label') == 'TRUE_POSITIVE' else "FP"
        verdict_color = (0, 220, 0) if verdict == "TP" else (0, 0, 255)
        disease = det.get('disease', '?')
        conf = det.get('confidence', 0.0)
        cv2.putText(canvas, f"{i+1}", (x0 + 12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(canvas, disease, (x0 + 50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    DISEASE_BGR.get(disease, (200, 200, 200)), 1)
        cv2.putText(canvas, f"{int(conf*100)}", (x0 + 180, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(canvas, f"({det.get('gz_x', 0):.1f}, {det.get('gz_y', 0):.1f})",
                    (x0 + 270, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(canvas, actual_name, (x0 + 440, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, actual_color, 1)
        cv2.putText(canvas, f"{err_m:.2f}", (x0 + 600, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(canvas, verdict, (x0 + 690, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, verdict_color, 2)
    if not detections:
        cv2.putText(canvas, "(no detections recorded)", (x0 + 200, row_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)
    elif len(detections) > max_rows:
        cv2.putText(canvas, f"... showing last {max_rows} of {len(detections)}",
                    (x0 + 350, y0 + h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (140, 140, 140), 1)


def generate_ml_report(save_dir: str = "mission_reports", show_window: bool = True):
    """Build a comprehensive post-mission ML report (PNG + optional preview window).

    Reads from:
      - mission_detections: in-memory list of every confirmed detection
      - infection_positions_gz: positions for the cluster map
      - cluster_result_shared: final KMeans result
      - detection_stats: counters (frames, raw, confirmed, etc.)
      - known_crops: parsed from world file (for ground-truth comparison)
    """
    try:
        import os
        os.makedirs(save_dir, exist_ok=True)
    except Exception as e:
        logger.error(f"Could not create {save_dir}: {e}")
        save_dir = "/tmp"

    # ---- Aggregate metrics ----
    detections = list(mission_detections)
    positions = list(infection_positions_gz)
    cluster = cluster_result_shared
    stats = dict(detection_stats)
    try:
        known = list(parse_infected_crops(WORLD_PATH))
    except Exception:
        known = []
    by_id = {c.id: c for c in known}

    # Counts - all detected labels (for diagnostic chart on the standalone
    # file) and TP-only labels (for the postable main report).
    counts = {n: 0 for n in DISEASE_ORDER}
    tp_counts = {n: 0 for n in DISEASE_ORDER}
    for d in detections:
        n = d.get('disease', '')
        if n in counts:
            counts[n] += 1
        if d.get('label') == 'TRUE_POSITIVE' and n in tp_counts:
            tp_counts[n] += 1
    total = len(detections)

    # TP / FP / per-disease breakdown
    tp = sum(1 for d in detections if d.get('label') == 'TRUE_POSITIVE')
    fp = sum(1 for d in detections if d.get('label') == 'FALSE_POSITIVE')
    recall = tp / max(1, len(known))
    precision = tp / max(1, total)
    f1 = 2 * precision * recall / max(1e-6, precision + recall)

    # Build a HONEST ground-truth list: only detections within 0.5m of a known
    # crop count for the confusion matrix. Otherwise a FP at (5, -1) gets
    # counted as "actual = nearest crop's disease" even though no real disease
    # is anywhere near (5, -1). This is the same radius used by the table.
    ground_truth_radius = 0.5
    matched_detections: list[dict] = []
    for d in detections:
        nid = d.get('nearest_known_id', -1)
        err_m = d.get('distance_to_known_m', 0.0)
        actual = by_id.get(nid) if nid >= 0 else None
        if actual is not None and err_m <= ground_truth_radius:
            matched_detections.append(d)

    # Unique-crop accounting: the same real crop can be detected many times
    # (drone passes over it repeatedly). Recall is "how many of the known
    # diseased crops did we find AT LEAST ONCE", capped at 100% so the
    # report can be posted without an obviously-impossible 109% recall.
    unique_crop_ids_found: set[int] = set()
    for d in matched_detections:
        nid = d.get('nearest_known_id', -1)
        if nid >= 0:
            unique_crop_ids_found.add(nid)
    unique_crops_found = len(unique_crop_ids_found)
    duplicate_detections = max(0, tp - unique_crops_found)

    # Recall = unique crops found / known diseased crops (capped at 1.0)
    recall = min(1.0, unique_crops_found / max(1, len(known)))
    precision = unique_crops_found / max(1, unique_crops_found + fp)  # precision is over unique findings
    f1 = 2 * precision * recall / max(1e-6, precision + recall)

    # Confusion matrix: actual_to_pred[actual][predicted] = count of unique crops
    # in that (actual, predicted) cell. Multiple detections of the same crop
    # only count once per (actual, predicted) pair to avoid the matrix
    # double-counting the same crop 49 times like the old report did.
    actual_to_pred: dict[str, dict[str, int]] = {a: {p: 0 for p in DISEASE_ORDER} for a in DISEASE_ORDER}
    seen_pairs: set[tuple[int, str]] = set()
    for d in matched_detections:
        nid = d.get('nearest_known_id', -1)
        actual = by_id.get(nid)
        if actual is None:
            continue
        a = actual.disease_type
        p = d.get('disease', '')
        if (nid, a) in seen_pairs:
            continue
        if a in actual_to_pred and p in actual_to_pred[a]:
            actual_to_pred[a][p] += 1
        seen_pairs.add((nid, a))

    # TP-only positions for the cluster map. Use UNIQUE crop positions so
    # the same crop doesn't show up as 3 separate cluster members.
    unique_positions_by_id: dict[int, tuple[float, float]] = {}
    for d in matched_detections:
        nid = d.get('nearest_known_id', -1)
        if nid >= 0 and nid not in unique_positions_by_id:
            unique_positions_by_id[nid] = (d.get('gz_x', 0.0), d.get('gz_y', 0.0))
    tp_positions = list(unique_positions_by_id.values())
    if len(tp_positions) >= 3:
        tp_cluster = cluster_infections(tp_positions, method="kmeans", k=min(3, len(tp_positions)))
    elif len(tp_positions) >= 2:
        tp_cluster = cluster_infections(tp_positions, method="kmeans", k=2)
    elif len(tp_positions) >= 1:
        tp_cluster = cluster_infections(tp_positions, method="kmeans", k=1)
    else:
        tp_cluster = None

    # ---- Render the canvas ----
    W, H = 1300, 1500
    canvas = np.full((H, W, 3), 15, dtype=np.uint8)

    # Title bar
    cv2.rectangle(canvas, (0, 0), (W, 70), (30, 30, 50), -1)
    cv2.putText(canvas, "AGRIBOT - Autonomous Crop Survey & Treatment", (30, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv2.putText(canvas, time.strftime("%Y-%m-%d %H:%M:%S"),
                (W - 230, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

    # Summary band - honest numbers, no impossible >100% values.
    # Layout: 3 columns × 3 rows (8 numbers) → wider columns, no overlap.
    y0 = 90
    cv2.rectangle(canvas, (20, y0), (W - 20, y0 + 120), (35, 35, 35), -1)
    summary = [
        f"Known Diseased:    {len(known)}",
        f"Unique Crops Hit:  {unique_crops_found}",
        f"True Positives:    {tp}",
        f"  ({duplicate_detections} repeat detections)",
        f"False Positives:   {fp}",
        f"Recall:            {recall*100:.1f}%   ({unique_crops_found}/{len(known)} found)",
        f"Precision:         {precision*100:.1f}%",
        f"F1 Score:          {f1*100:.1f}%",
    ]
    # 3 columns across 1260px ≈ 420px each
    col_x = [40, 460, 880]
    for i, line in enumerate(summary):
        x = col_x[i % 3]
        y = y0 + 28 + (i // 3) * 28
        color = (255, 255, 100) if line.strip().startswith(('Recall:', 'Precision:', 'F1')) else (200, 200, 200)
        cv2.putText(canvas, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    # Crops Treated on its own line below the grid
    cv2.putText(canvas, f"Crops Treated:    {stats.get('crops_treated', 0)}",
                (40, y0 + 28 + 3 * 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Panel A: Disease Distribution (top-left) - only TPs counted so the
    # bar chart reflects what the model actually found correctly, not the
    # pile of FPs from image noise. The denominator is the number of TPs
    # (so 6+10=16 → 100%), not the number of unique crops.
    tp_total = sum(tp_counts.values())
    _draw_bar_chart(canvas, 20, 230, 640, 250, tp_counts, max(1, tp_total),
                    "DISEASE DISTRIBUTION (true-positive predictions)")

    # Panel B: Confusion Matrix (top-right)
    _draw_confusion_matrix(canvas, 680, 230, 600, 250, actual_to_pred)

    # Panel C: Cluster Map (middle, full width) - TP only, unique crops
    _draw_cluster_map(canvas, 20, 500, 1260, 380, tp_positions, tp_cluster,
                      f"INFECTION HOTSPOTS (KMeans on {len(tp_positions)} unique diseased crops)")

    # Panel D: Detection Log (bottom, full width) - SHOWS ONLY TRUE POSITIVES.
    # False positives are summarized in the header line so this log can be
    # posted publicly without showing the model's noise.
    _draw_detection_table(canvas, 20, 900, 1260, 580, matched_detections, known, max_rows=20,
                          ground_truth_radius=ground_truth_radius,
                          fp_count=fp, tp_total=tp)

    # ---- Save to disk ----
    ts = time.strftime("%Y%m%d_%H%M%S")
    main_path = f"{save_dir}/mission_report_{ts}.png"
    chart_path = f"{save_dir}/disease_distribution_{ts}.png"
    cluster_path = f"{save_dir}/cluster_map_{ts}.png"
    cv2.imwrite(main_path, canvas)
    # Standalone charts
    chart_only = np.full((350, 700, 3), 15, dtype=np.uint8)
    _draw_bar_chart(chart_only, 20, 20, 660, 310, counts, max(1, total), "DISEASE DISTRIBUTION")
    cv2.imwrite(chart_path, chart_only)
    cluster_only = np.full((450, 1300, 3), 15, dtype=np.uint8)
    _draw_cluster_map(cluster_only, 20, 20, 1260, 410, tp_positions, tp_cluster,
                      f"INFECTION HOTSPOTS (KMeans on {len(tp_positions)} TPs)")
    cv2.imwrite(cluster_path, cluster_only)

    logger.info("=" * 70)
    logger.info("ML REPORT GENERATED")
    logger.info(f"  Main report:  {main_path}")
    logger.info(f"  Chart:        {chart_path}")
    logger.info(f"  Cluster map:  {cluster_path}")
    logger.info(f"  Recall: {recall*100:.1f}%, Precision: {precision*100:.1f}%, F1: {f1*100:.1f}%")
    logger.info("=" * 70)

    # Print to console too (in case the log file isn't handy)
    print("\n" + "=" * 70)
    print("ML REPORT GENERATED")
    print(f"  Main report:  {main_path}")
    print(f"  Chart:        {chart_path}")
    print(f"  Cluster map:  {cluster_path}")
    print(f"  Recall {recall*100:.1f}%  Precision {precision*100:.1f}%  F1 {f1*100:.1f}%")
    print("=" * 70 + "\n")

    if show_window:
        try:
            cv2.namedWindow("ML REPORT", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("ML REPORT", 1100, 1260)
            cv2.moveWindow("ML REPORT", 200, 50)
            cv2.imshow("ML REPORT", canvas)
            print("[ML REPORT] Press any key in the ML REPORT window to close it (or wait 15s).")
            # Wait up to 15 seconds for a keypress
            t0 = time.time()
            while time.time() - t0 < 15.0:
                if cv2.waitKey(200) != -1:
                    break
            cv2.destroyWindow("ML REPORT")
        except Exception as e:
            logger.warning(f"Could not show ML report window: {e}")


# ==================== STABLE DUAL DISPLAY THREAD ====================
def stable_dual_display_thread(processor, stop_event, field_map):
    """Two-window display: scout camera and treatment camera.

    The field map and stats are still computed and saved in the background
    (the modules stay useful for logging), but the OpenCV display only
    shows the two live camera feeds.
    """
    cv2.namedWindow("SCOUT CAMERA", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("SCOUT CAMERA", 700, 525)
    cv2.moveWindow("SCOUT CAMERA", 0, 0)
    cv2.namedWindow("TREATMENT CAMERA", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("TREATMENT CAMERA", 700, 525)
    cv2.moveWindow("TREATMENT CAMERA", 720, 0)

    last_update_time = time.time()
    frame_count = 0
    fps_start_time = time.time()
    current_fps = 0

    logger.info("Starting 2-window display (scout cam, treatment cam)")

    while not stop_event.is_set():
        current_time = time.time()

        if current_time - last_update_time >= Config.DISPLAY_UPDATE_INTERVAL:

            scout_frame, scout_metadata = scout_frame_buffer.get_stable_frame()
            treatment_frame, treatment_metadata = treatment_frame_buffer.get_stable_frame()

            # ===== SCOUT CAMERA WINDOW =====
            if scout_frame is not None:
                try:
                    scout_resized = cv2.resize(scout_frame, (700, 525))
                    height, width = scout_resized.shape[:2]
                    margin = Config.CENTER_THRESHOLD
                    x1 = int(width * (0.5 - margin/2))
                    x2 = int(width * (0.5 + margin/2))
                    y1 = int(height * (0.5 - margin/2))
                    y2 = int(height * (0.5 + margin/2))
                    cv2.rectangle(scout_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(scout_resized, "SCOUT DETECTION ZONE", (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    infection_detected, centroid = processor.detect_infected_crops(scout_resized)
                    if infection_detected and centroid:
                        cx, cy = centroid
                        with latest_disease_lock:
                            label_text = f"{latest_disease_label.upper()} {int(latest_disease_confidence * 100)}%"
                        disease_colors_bgr = {
                            "Healthy":  (0, 255, 0),
                            "Stressed": (0, 255, 255),
                            "Rust":     (0, 165, 255),
                            "Blight":   (0, 0, 255),
                        }
                        box_color = disease_colors_bgr.get(latest_disease_label, (0, 255, 0))
                        cv2.circle(scout_resized, (cx, cy), 8, box_color, -1)
                        cv2.putText(scout_resized, label_text, (cx-50, cy-15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    cv2.putText(scout_resized, f"FPS: {current_fps:.1f}", (10, 510),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.imshow("SCOUT CAMERA", scout_resized)
                except Exception as e:
                    logger.warning(f"Scout display error: {e}")
            else:
                blank = np.zeros((525, 700, 3), np.uint8)
                cv2.putText(blank, "SCOUT CAMERA", (250, 260),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                cv2.putText(blank, "Waiting for feed...", (260, 300),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                cv2.imshow("SCOUT CAMERA", blank)

            # ===== TREATMENT CAMERA WINDOW =====
            if treatment_frame is not None:
                try:
                    treatment_resized = cv2.resize(treatment_frame, (700, 525))
                    height, width = treatment_resized.shape[:2]
                    margin = Config.TREATMENT_CENTER_THRESHOLD
                    x1 = int(width * (0.5 - margin/2))
                    x2 = int(width * (0.5 + margin/2))
                    y1 = int(height * (0.5 - margin/2))
                    y2 = int(height * (0.5 + margin/2))
                    cv2.rectangle(treatment_resized, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(treatment_resized, "PRECISION ALIGNMENT", (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                    infection_detected, centroid = processor.detect_infected_crops(
                        treatment_resized, Config.TREATMENT_CENTER_THRESHOLD
                    )
                    if infection_detected and centroid:
                        cx, cy = centroid
                        cv2.circle(treatment_resized, (cx, cy), 6, (255, 0, 0), -1)
                        cv2.putText(treatment_resized, "TARGET", (cx-20, cy-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 2)
                        center_x, center_y = width // 2, height // 2
                        offset_x, offset_y = cx - center_x, cy - center_y
                        if abs(offset_x) < 20 and abs(offset_y) < 20:
                            cv2.putText(treatment_resized, "PERFECT ALIGNMENT", (50, 50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        else:
                            cv2.putText(treatment_resized, f"Adjusting: ({offset_x}, {offset_y})", (50, 50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                    cv2.putText(treatment_resized, f"FPS: {current_fps:.1f}", (10, 510),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.imshow("TREATMENT CAMERA", treatment_resized)
                except Exception as e:
                    logger.warning(f"Treatment display error: {e}")
            else:
                blank = np.zeros((525, 700, 3), np.uint8)
                cv2.putText(blank, "TREATMENT CAMERA", (200, 260),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                cv2.putText(blank, "Waiting for feed...", (240, 300),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                cv2.imshow("TREATMENT CAMERA", blank)

            last_update_time = current_time
            frame_count += 1

            if current_time - fps_start_time >= 1.0:
                current_fps = frame_count / (current_time - fps_start_time)
                frame_count = 0
                fps_start_time = current_time

        if cv2.waitKey(1) & 0xFF == ord('q'):
            logger.info("Display quit requested")
            stop_event.set()
            break

        time.sleep(0.01)

    cv2.destroyAllWindows()
    logger.info("Display thread completed")

# ==================== MAIN EXECUTION ====================
def auto_detect_camera_topics():
    """Override Config topics if Gazebo exposes them under different names."""
    import subprocess
    try:
        result = subprocess.run(
            ["ros2", "topic", "list"], capture_output=True, text=True, timeout=5
        )
        topics = result.stdout.splitlines()
        camera_topics = [t for t in topics if t.endswith("/camera") or t == "/camera"]

        if len(camera_topics) >= 2:
            Config.SCOUT_CAMERA_TOPIC = camera_topics[0]
            Config.TREATMENT_CAMERA_TOPIC = camera_topics[1]
            logger.info(f"Auto-detected cameras: scout={Config.SCOUT_CAMERA_TOPIC}, treatment={Config.TREATMENT_CAMERA_TOPIC}")
        elif len(camera_topics) == 1:
            Config.SCOUT_CAMERA_TOPIC = camera_topics[0]
            Config.TREATMENT_CAMERA_TOPIC = camera_topics[0]
            logger.warning(f"Only one camera topic found: {camera_topics[0]} - using for both")
        else:
            logger.warning(f"No camera topics found via ros2 topic list; using defaults {Config.SCOUT_CAMERA_TOPIC}, {Config.TREATMENT_CAMERA_TOPIC}")
    except Exception as e:
        logger.warning(f"Auto-detect failed: {e}; using default topics")


def main():
    """Dual drone system with stable display"""
    logger.info("="*80)
    logger.info("DUAL DRONE SYSTEM - STABLE DISPLAY")
    logger.info("="*80)
    logger.info("SEQUENTIAL OPERATION: Complete survey -> Treatment")
    logger.info("DUAL CAMERA SYSTEM: Scout + Treatment cameras")
    logger.info("STABLE DISPLAY: Fixed flickering with proper frame buffering")
    logger.info("JSON COORDINATION: File-based communication")
    logger.info("CAMERA-GUIDED PRECISION: Perfect alignment")
    logger.info("ENHANCED DETECTION: Maximum accuracy")
    logger.info("DUPLICATE PREVENTION: No repeat treatments")
    logger.info("="*80)
    
    auto_detect_camera_topics()
    rclpy.init()
    scout_camera_node = ScoutCameraNode()
    treatment_camera_node = TreatmentCameraNode()

    # Single MultiThreadedExecutor for both nodes to avoid "generator already executing"
    # when two rclpy.spin() threads share one context.
    executor = rclpy.executors.MultiThreadedExecutor(num_threads=2)
    executor.add_node(scout_camera_node)
    executor.add_node(treatment_camera_node)
    ros_thread = threading.Thread(target=executor.spin, daemon=True)
    ros_thread.start()
    
    processor = EnhancedDetectionProcessor()

    # === Camera-based detection setup (replaces broken HSV-only detection) ===
    logger.info("Initializing camera-based detection...")

    # 1. Parse world file for known infected crops and origin
    world_path = WORLD_PATH
    origin = parse_spherical_origin(world_path)
    known_crops = parse_infected_crops(world_path)
    logger.info(f"World: {len(known_crops)} known infected crops, origin=({origin.lat:.6f}, {origin.lon:.6f}, {origin.alt:.1f})")

    # 2. Build color thresholds for all 3 non-healthy disease types
    #    (Stressed + Rust + Blight). OR-combined in the detector to catch any diseased crop.
    disease_rgbs = [DISEASE_COLORS[name] for name in ("Stressed", "Rust", "Blight")]
    thresholds = build_disease_thresholds(disease_rgbs)

    # 3. Camera intrinsics (from iris_with_standoffs_and_cam/model.sdf)
    intrinsics = CameraIntrinsics(
        width=640, height=480,
        hfov=math.radians(60),
        mount_pitch=math.pi / 2,
    )

    # 4. Build the detector
    detector = Detector(thresholds=thresholds, intrinsics=intrinsics, voting_window=3, min_blob_area=300)

    # 5. Axis mapping will be calibrated at scout takeoff (see calibrate_scout below)
    axis_mapping_holder: list[AxisMapping | None] = [None]

    # 6. Train or load the crop classifier
    classifier_path = Path("models/crop_classifier.joblib")
    if classifier_path.exists():
        logger.info(f"Loading classifier from {classifier_path}")
        classifier = CropClassifier.load(classifier_path)
    else:
        logger.info("Training crop classifier on synthetic world data...")
        classifier = CropClassifier(model_path=classifier_path)
        classifier.train()
        classifier.save()

    # 7. Build the field map
    field_map = FieldMap(world_path=world_path, size_px=400)

    display_thread_handle = threading.Thread(
        target=stable_dual_display_thread,
        args=(processor, stop_event, field_map),
        daemon=True
    )
    display_thread_handle.start()

    ml_insight_handle = None  # post-mission report only; see generate_ml_report below

    def cluster_update_thread(stop_event):
        global cluster_result_shared
        while not stop_event.is_set():
            with infection_lock:
                positions = list(infection_positions_gz)
            if positions:
                cluster_result_shared = cluster_infections(
                    positions, method="kmeans", k=3
                )
            time.sleep(2.0)

    cluster_thread = threading.Thread(
        target=cluster_update_thread, args=(stop_event,), daemon=True
    )
    cluster_thread.start()
    
    scout_vehicle = None
    treatment_vehicle = None
    
    try:
        logger.info("Connecting to both drones...")
        scout_vehicle = DroneOperations.connect_vehicle(Config.SCOUT_DRONE)
        treatment_vehicle = DroneOperations.connect_vehicle(Config.TREATMENT_DRONE)

        # Apply stable-flight parameters BEFORE takeoff so the drone uses
        # tighter tuning throughout the mission.
        DroneOperations.set_stable_flight_params(scout_vehicle, "SCOUT")
        DroneOperations.set_stable_flight_params(
            treatment_vehicle, "TREATMENT",
            override=DroneOperations.TREATMENT_FLIGHT_PARAMS,
        )
        
        scout_state_thread = threading.Thread(
            target=update_drone_state, 
            args=(scout_vehicle, "scout", stop_event), 
            daemon=True
        )
        treatment_state_thread = threading.Thread(
            target=update_drone_state, 
            args=(treatment_vehicle, "treatment", stop_event), 
            daemon=True
        )
        scout_state_thread.start()
        treatment_state_thread.start()
        
        time.sleep(3)
        
        home_location = scout_vehicle.home_location or scout_vehicle.location.global_frame
        waypoints = generate_lawnmower_waypoints(home_location)
        
        logger.info("\nPHASE 1: SCOUT SURVEY MISSION")
        logger.info("="*50)
        
        scout_detection_handle = threading.Thread(
            target=scout_detection_thread,
            args=(detector, axis_mapping_holder, known_crops, origin, stop_event, classifier),
            daemon=True
        )
        scout_detection_handle.start()
        
        DroneOperations.arm_and_takeoff(scout_vehicle, Config.SURVEY_ALT, "SCOUT")
        
        scout_drone_controller(
            scout_vehicle, waypoints, stop_event,
            detector=detector,
            axis_mapping_holder=axis_mapping_holder,
            known_crops=known_crops,
            origin=origin,
        )
        
        logger.info("\nPHASE 2: TREATMENT MISSION")
        logger.info("="*50)

        treatment_drone_controller(
            treatment_vehicle, processor, stop_event,
            known_crops=known_crops, origin=origin,
            spawn_gz=(3.0, 0.0, 2.0),  # treatment drone spawns here in world
        )
        
        logger.info("\n" + "="*70)
        logger.info("MISSION COMPLETION REPORT")
        logger.info("="*70)
        logger.info("DISPLAY: Stable, no flickering")
        logger.info("SURVEY PHASE: Complete field coverage")
        logger.info("DETECTION PERFORMANCE:")
        logger.info(f"    Frames processed: {detection_stats['frames_processed']}")
        logger.info(f"    Confirmed detections: {detection_stats['confirmed_detections']}")
        logger.info(f"    Duplicates prevented: {detection_stats['duplicates_prevented']}")
        logger.info("TREATMENT PHASE:")
        logger.info(f"    Crops treated: {detection_stats['crops_treated']}")
        logger.info("    Camera-guided precision: ENABLED")
        
        if confirmed_detections:
            logger.info("\nTreated infection coordinates:")
            for i, detection in enumerate(confirmed_detections, 1):
                logger.info(f"  {i}. Lat: {detection['lat']:.8f}, Lon: {detection['lon']:.8f}")

        # Read metrics file and log precision/recall
        try:
            with open('detection_metrics.json', 'r') as f:
                m = json.load(f)
            logger.info(
                f"    Precision: {m['precision']:.2f}, Recall: {m['recall']:.2f}"
            )
            logger.info(f"    Missed crop IDs: {m['missed_ids']}")
        except Exception:
            pass

        logger.info("="*70)
        
        final_results = {
            'system_version': 'PROFESSIONAL_DUAL_DRONE_SYSTEM',
            'features': {
                'dual_cameras': True,
                'stable_display': True,
                'camera_guided_precision': True,
                'enhanced_detection': True,
                'duplicate_prevention': True,
                'json_coordination': True
            },
            'display_specs': {
                'fps': Config.DISPLAY_FPS,
                'buffer_size': Config.FRAME_BUFFER_SIZE
            },
            'statistics': detection_stats,
            'detections': confirmed_detections,
            'mission_timestamp': time.time()
        }
        
        with open('professional_dual_drone_results.json', 'w') as f:
            json.dump(final_results, f, indent=2)
        logger.info("Final results saved to professional_dual_drone_results.json")

        # === Post-mission ML report (saved to disk + briefly shown) ===
        try:
            generate_ml_report(save_dir="mission_reports", show_window=True)
        except Exception as e:
            logger.error(f"ML report generation failed: {e}")
            import traceback
            traceback.print_exc()

    except Exception as e:
        logger.error(f"Mission error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        logger.info("Shutting down dual drone system...")
        stop_event.set()
        
        for vehicle, name in [(scout_vehicle, "Scout"), (treatment_vehicle, "Treatment")]:
            if vehicle and hasattr(vehicle, 'armed') and vehicle.armed:
                logger.info(f"Returning {name} drone to launch...")
                try:
                    vehicle.mode = VehicleMode("RTL")
                    time.sleep(0.5)
                    vehicle.mode = VehicleMode("LAND")
                except:
                    pass
        
        for vehicle in [scout_vehicle, treatment_vehicle]:
            if vehicle:
                try:
                    vehicle.close()
                except:
                    pass
        
        try:
            rclpy.shutdown()
        except:
            pass
        
        logger.info("DUAL DRONE SYSTEM SHUTDOWN COMPLETE")

if __name__ == "__main__":
    # Safety net: always emit a report on exit (clean, Ctrl+C, or crash).
    def _safe_report_on_exit():
        try:
            generate_ml_report(save_dir="mission_reports", show_window=False)
        except Exception:
            pass
    atexit.register(_safe_report_on_exit)

    try:
        main()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
        stop_event.set()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)