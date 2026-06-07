#!/bin/bash
# Agribot - one-shot launcher for autonomous crop survey & treatment
# Starts Gazebo, 2 SITL instances, 2 ROS2 camera bridges, and the mission script.

set -e

WS=/home/aditya/agribot_ws
WORLD=$WS/world/agribot_farm_world.sdf
ARDUPILOT=/home/aditya/ardupilot
GAZEBO_MODELS=$WS/models

# Source ROS2 (try common distros)
for d in humble iron galactic foxy; do
  if [ -f /opt/ros/$d/setup.bash ]; then
    source /opt/ros/$d/setup.bash
    break
  fi
done

# Make our local models visible to gz sim
export GZ_SIM_RESOURCE_PATH="$GAZEBO_MODELS:$HOME/.gazebo/models:$HOME/ardupilot_gazebo/models:$HOME/ardupilot_gazebo/worlds/models:${GZ_SIM_RESOURCE_PATH:-}"

# Workaround for X11/GLX issues: force software rendering for the ogre2 render thread
# Prevents "BadValue (integer parameter out of range for operation)" on systems
# with broken or missing GPU drivers
export LIBGL_ALWAYS_SOFTWARE=${LIBGL_ALWAYS_SOFTWARE:-1}
export MESA_GL_VERSION_OVERRIDE=${MESA_GL_VERSION_OVERRIDE:-3.3}
export QT_QPA_PLATFORM=${QT_QPA_PLATFORM:-offscreen}

echo "=== Killing any leftover processes ==="
pkill -f sim_vehicle 2>/dev/null || true
pkill -f 'gz sim' 2>/dev/null || true
pkill -f ros_gz_bridge 2>/dev/null || true
sleep 2

echo "=== Starting Gazebo with agribot_farm_world.sdf ==="
# Headless by default. Pass --gui to open a Gazebo GUI window.
# Headless is safer on systems with broken GPU drivers / X11 GLX issues.
HEADLESS_FLAG="-s"
USE_GUI=""
if [ "$1" = "--gui" ]; then
  HEADLESS_FLAG=""
  USE_GUI=1
  echo "  (GUI mode: opening Gazebo window)"
else
  echo "  (headless mode: no GUI window, server only)"
fi

if [ -n "$USE_GUI" ]; then
  # Launch gz sim in background; GUI window will appear because we don't use -s
  gz sim -v4 -r $WORLD > /tmp/opencode/gz_sim.log 2>&1 &
  GZ_PID=$!
  echo "  (gz sim PID: $GZ_PID, GUI mode)"
else
  gz sim -v4 $HEADLESS_FLAG -r $WORLD > /tmp/opencode/gz_sim.log 2>&1 &
  GZ_PID=$!
  echo "  (gz sim PID: $GZ_PID, headless mode)"
fi

# Wait for plugin to bind
sleep 8
echo "  (waiting for plugin to bind...)"
for i in 1 2 3 4 5 6 7 8 9 10; do
  if ss -lunp 2>/dev/null | grep -q ":9002 "; then
    echo "  (plugin bound on 9002)"
    break
  fi
  sleep 1
done

echo "=== Starting scout SITL (instance 0, udp:127.0.0.1:14550) ==="
if command -v gnome-terminal >/dev/null 2>&1; then
  gnome-terminal -- bash -c "cd $ARDUPILOT && python3 Tools/autotest/sim_vehicle.py -v ArduCopter -f gazebo-iris --model JSON --out udp:127.0.0.1:14550 -I0 2>&1 | tee /tmp/opencode/sitl_0.log; exec bash"
else
  (cd $ARDUPILOT && python3 Tools/autotest/sim_vehicle.py -v ArduCopter -f gazebo-iris --model JSON --out udp:127.0.0.1:14550 -I0 > /tmp/opencode/sitl_0.log 2>&1 &)
fi
sleep 10

echo "=== Starting treatment SITL (instance 1, udp:127.0.0.1:14560) ==="
if command -v gnome-terminal >/dev/null 2>&1; then
  gnome-terminal -- bash -c "cd $ARDUPILOT && python3 Tools/autotest/sim_vehicle.py -v ArduCopter -f gazebo-iris --model JSON --out udp:127.0.0.1:14560 -I1 2>&1 | tee /tmp/opencode/sitl_1.log; exec bash"
else
  (cd $ARDUPILOT && python3 Tools/autotest/sim_vehicle.py -v ArduCopter -f gazebo-iris --model JSON --out udp:127.0.0.1:14560 -I1 > /tmp/opencode/sitl_1.log 2>&1 &)
fi
sleep 10

echo "=== Starting ROS2 camera bridges ==="
if command -v gnome-terminal >/dev/null 2>&1; then
  gnome-terminal -- bash -c "ros2 run ros_gz_bridge parameter_bridge /camera@sensor_msgs/msg/Image[gz.msgs.Image; exec bash"
else
  (ros2 run ros_gz_bridge parameter_bridge /camera@sensor_msgs/msg/Image[gz.msgs.Image &)
fi
sleep 1
if command -v gnome-terminal >/dev/null 2>&1; then
  gnome-terminal -- bash -c "ros2 run ros_gz_bridge parameter_bridge /treatment_camera@sensor_msgs/msg/Image[gz.msgs.Image; exec bash"
else
  (ros2 run ros_gz_bridge parameter_bridge /treatment_camera@sensor_msgs/msg/Image[gz.msgs.Image &)
fi
sleep 3

echo "=== Verifying camera topics ==="
ros2 topic list 2>/dev/null | grep -E 'camera|image' || echo "WARNING: no camera topics visible yet"

echo "=== Starting mission script ==="
cd $WS
python3 fly_drone.py
