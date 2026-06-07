# Agribot — Autonomous Dual-Drone Crop Survey & Treatment

Two drones, one mission: scout the field with a camera, detect diseased crops
using computer vision and ML, and fly the treatment drone precisely to each
infected plant for simulated spraying.

| Path | Purpose |
|------|---------|
| `fly_drone.py` | Main mission script (detection + survey + treatment + display) |
| `world/agribot_farm_world.sdf` | Gazebo world: 214 healthy + 11 infected crops, 2 camera-equipped drones |
| `models/` | Drone models (iris with camera gimbal, scout + treatment variants) |
| `src/` | Detection, coordinate conversion, ML classifier, world parsing |
| `tests/` | 88+ automated tests |

## Quick Start

```bash
cd /home/aditya/agribot_ws
./launch.sh
```

Opens Gazebo with SITL ArduPilot instances, camera bridges, then runs the
mission autonomously.

## How it works

1. **Scout survey** — The scout drone flies a lawnmower pattern over the field
   at ~2.5m altitude, streaming video through a ROS2 bridge.

2. **Real-time detection** — OpenCV color masking identifies stressed/yellow
   (H=[30,60]) and rust/orange (H=[0,30]) crops per frame, with multi-frame
   voting to suppress false positives.

3. **Disease classification** — A scikit-learn LogisticRegression classifier
   (trained on empirical RGB samples from the Gazebo world) labels each
   detection as Stressed, Rust, or Healthy.

4. **Treatment targeting** — World-file crop positions are matched to ML
   detections, deduplicated by proximity (0.5m radius), and the treatment
   drone flies to each using LOCAL_NED position targets routed through
   ArduPilot's WPNAV for sub-20cm accuracy.

5. **Post-mission report** — A 4-panel report (detection log, confusion matrix,
   bar chart, cluster map) is saved to `mission_reports/` and auto-launched.

## Requirements

- Ubuntu 22.04 / ROS2 Humble
- Gazebo Fortress (Gazebo Ignition)
- ArduPilot SITL (Copter 4.5+)
- Python 3.10+
- NVIDIA GPU recommended for smooth rendering

See `requirements.txt` for Python dependencies.

## Testing

```bash
pytest tests/ -v
```

88 tests covering coordinate conversion, color calibration, world parsing,
disease classification, ML report rendering, and detection logic.

## License

MIT — see LICENSE.
