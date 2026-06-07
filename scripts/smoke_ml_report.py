"""Smoke test: render a report from synthetic data so we can verify the
postable layout (TP-only log, capped recall, unique-crop cluster map)
without launching Gazebo.

We do this by exec-ing fly_drone.py with all the ROS/heavy deps stubbed
out, then seeding synthetic mission_detections data and calling
generate_ml_report() directly. Outputs land in /tmp/smoke_reports/.

Run:  PYTHONPATH=/home/aditya/agribot_ws python3 scripts/smoke_ml_report.py
"""
import os
import sys
import time
import types

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

sys.path.insert(0, '/home/aditya/agribot_ws')


# ============================================================
# Stub rclpy / sensor_msgs / std_msgs / geometry_msgs / etc.
# ============================================================
def _make_stub(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


class _FakeNode:
    def __init__(self, *a, **k): pass
    def get_logger(self):
        import logging
        return logging.getLogger('fake')
    def create_subscription(self, *a, **k): 
        return types.SimpleNamespace()
    def destroy_node(self): pass


class _Any:
    """Sentinel class that accepts any attribute access (mimics ROS msgs)."""
    def __init__(self, *a, **k): pass
    def __getattr__(self, k): return _Any()
    def __call__(self, *a, **k): return _Any()


class _StubMsg:
    """Stand-in for sensor_msgs.msg.Image etc."""
    def __init__(self, *a, **k): pass


# Make rclpy stubs
_make_stub('rclpy', init=lambda *a, **k: None, shutdown=lambda *a, **k: None,
           spin=lambda *a, **k: None, ok=lambda: True)
_make_stub('rclpy.node', Node=_FakeNode)

# Make sensor_msgs.msg with all msg classes as _StubMsg
sensor_mod = _make_stub('sensor_msgs')
sensor_msg_mod = _make_stub('sensor_msgs.msg')
for name in ['Image', 'CompressedImage', 'CameraInfo', 'PointCloud2', 'LaserScan',
             'Imu', 'NavSatFix', 'JointState', 'BatteryState', 'TimeReference']:
    setattr(sensor_msg_mod, name, _StubMsg)

# std_msgs
std_msg_mod = _make_stub('std_msgs.msg')
for name in ['String', 'Bool', 'Int32', 'Int64', 'Float32', 'Float64',
             'UInt8', 'UInt16', 'UInt32', 'Header', 'ColorRGBA', 'Time']:
    setattr(std_msg_mod, name, _StubMsg)

# geometry_msgs
geom_mod = _make_stub('geometry_msgs')
geom_msg_mod = _make_stub('geometry_msgs.msg')
for name in ['PoseStamped', 'Pose', 'Point', 'Quaternion', 'Vector3',
             'Twist', 'TwistStamped', 'TransformStamped', 'PointStamped']:
    setattr(geom_msg_mod, name, _StubMsg)

# nav_msgs
nav_mod = _make_stub('nav_msgs')
nav_msg_mod = _make_stub('nav_msgs.msg')
for name in ['Odometry', 'Path', 'OccupancyGrid']:
    setattr(nav_msg_mod, name, _StubMsg)

# mavros_msgs
mavros_mod = _make_stub('mavros_msgs')
mavros_msg_mod = _make_stub('mavros_msgs.msg')
for name in ['State', 'VFR_HUD', 'GlobalPositionTarget', 'PositionTarget',
             'RCIn', 'OverrideRCIn', 'MountControl', 'HomePosition']:
    setattr(mavros_msg_mod, name, _StubMsg)
mavros_srv_mod = _make_stub('mavros_msgs.srv')
for name in ['SetMode', 'CommandBool', 'CommandTOL', 'CommandLong',
             'ParamSet', 'ParamGet', 'StreamRate']:
    setattr(mavros_srv_mod, name, _StubMsg)

# cv_bridge
cv_bridge_mod = _make_stub('cv_bridge')
class _CvBridge:
    def imgmsg_to_cv2(self, *a, **k):
        import numpy as np
        return np.zeros((100, 100, 3), dtype=np.uint8)
    def cv2_to_imgmsg(self, *a, **k): return _StubMsg()
setattr(cv_bridge_mod, 'CvBridge', _CvBridge)

# message_filters
mf_mod = _make_stub('message_filters')
class _ApproximateTimeSynchronizer:
    def __init__(self, *a, **k): pass
    def registerCallback(self, *a, **k): pass
setattr(mf_mod, 'Subscriber', _StubMsg)
setattr(mf_mod, 'ApproximateTimeSynchronizer', _ApproximateTimeSynchronizer)
setattr(mf_mod, 'TimeSynchronizer', _ApproximateTimeSynchronizer)

# rosidl_runtime_types (for supported_type_name etc.)
_make_stub('rosidl_runtime_types')


# ============================================================
# Import the disease/mapping pieces that DON'T need ROS
# ============================================================
from src import disease_constants as DC  # noqa: F401


# ============================================================
# Bring in just the bits of fly_drone we need by exec-ing the file
# but replacing the ROS-dependent `import` lines with our stubs.
# ============================================================
src = open('/home/aditya/agribot_ws/fly_drone.py').read()
# Replace the `import rclpy` block with pass, so we can get past module load.
import re
src = re.sub(r"^import rclpy.*?^from .*? import .*?$", "# ros stubbed",
             src, count=20, flags=re.MULTILINE)

# Pre-define the globals the report function reads
import numpy as np
import cv2
import logging

logger = logging.getLogger("smoke")
logging.basicConfig(level=logging.INFO)


# Build the global state the report expects
mission_detections = []
infection_positions_gz = []
cluster_result_shared = None
detection_stats = {}


def _patch_logging(name, *a, **k):
    return logger

DC.LOGGER = logger  # in case


# Run the file body in a namespace with our stubs in place
ns = {
    '__name__': '__smoke_fly_drone__',
    '__file__': '/home/aditya/agribot_ws/fly_drone.py',
    'cv2': cv2,
    'np': np,
    'time': time,
    'logging': logging,
    'logger': logger,
    'math': __import__('math'),
    'os': os,
    'sys': sys,
    'json': __import__('json'),
    'threading': __import__('threading'),
    'random': __import__('random'),
    'collections': __import__('collections'),
    'atexit': __import__('atexit'),
    'subprocess': __import__('subprocess'),
    'signal': __import__('signal'),
    'itertools': __import__('itertools'),
}
# Run the file
exec(compile(src, '/home/aditya/agribot_ws/fly_drone.py', 'exec'), ns)


# ============================================================
# Seed synthetic data (must match the REAL world crop positions/IDs)
# ============================================================
known = [
    # (id, x, y, disease) - copied from real world
    (1,   -3.50, 0.90, 'Stressed'),
    (8,    0.00, 0.90, 'Stressed'),
    (30,   3.50, 1.40, 'Stressed'),
    (50,  -1.50, 2.40, 'Rust'),
    (80,  -1.50, 3.40, 'Rust'),
    (100,  1.00, 3.90, 'Stressed'),
    (119,  3.00, 4.40, 'Rust'),
    (157, -0.50, 5.90, 'Rust'),
    (158,  0.00, 5.90, 'Rust'),
    (175,  1.00, 6.40, 'Rust'),
    (224,  3.00, 7.90, 'Stressed'),
]

ns['mission_detections'].clear()
ns['infection_positions_gz'].clear()


def add_det(x, y, disease, conf, is_tp, closest_id=-1, err_m=0.0):
    ns['mission_detections'].append({
        'disease': disease,
        'confidence': conf,
        'gz_x': x,
        'gz_y': y,
        'label': 'TRUE_POSITIVE' if is_tp else 'FALSE_POSITIVE',
        'nearest_known_id': closest_id,
        'distance_to_known_m': err_m,
        'frame': 0,
        'timestamp': 0.0,
    })
    ns['infection_positions_gz'].append((x, y))


# 5 crops detected twice (dupes), 6 detected once = 16 TPs total
twice_ids = {50, 100, 80, 158, 175}
for cid, x, y, disease in known:
    add_det(x, y, disease, 0.92, True, cid, 0.0)
    if cid in twice_ids:
        add_det(x + 0.1, y + 0.05, disease, 0.85, True, cid, 0.11)

# 4 false positives far from any crop (err > 0.5m)
add_det(-3.2, 0.5, 'Stressed', 0.65, False, 1, 1.4)
add_det(0.8,  6.8, 'Rust',     0.58, False, 224, 1.7)
add_det(2.7,  0.9, 'Stressed', 0.61, False, 30, 1.2)
add_det(-1.9, 3.2, 'Rust',     0.55, False, 100,  1.1)

ns['detection_stats'].update({
    'frames': 1000,
    'raw_detections': 20,
    'confirmed': 20,
    'crops_treated': 11,
})

# Try to call generate_ml_report if it survived the import
generate_ml_report = ns.get('generate_ml_report')
if generate_ml_report is None:
    print("ERROR: generate_ml_report not defined in exec'd namespace")
    sys.exit(1)

os.makedirs('/tmp/smoke_reports', exist_ok=True)
print(f"Total detections: {len(ns['mission_detections'])}")
print(f"TP: {sum(1 for d in ns['mission_detections'] if d['label']=='TRUE_POSITIVE')}")
print(f"FP: {sum(1 for d in ns['mission_detections'] if d['label']=='FALSE_POSITIVE')}")
print(f"Known: {len(known)}")
print()

out = generate_ml_report(save_dir='/tmp/smoke_reports', show_window=False)
print(f"\nReport: {out}")
