# VR teleoperation startup, AI Worker SG2

Author: Dawit Chun
Verified working 2026-08-14.

Five terminals. Nothing can move until step 4, and nothing DOES move until step 6.

---

## 0. Pre-flight, before anything

* Workspace clear. Nobody within reach of either arm at full extension.
* E-stop in hand.
* Quest charged, proximity sensor **taped** (otherwise the session dies whenever you look at a screen).
* Stand on the taped floor mark. The VR origin is set from where you stand.

Check whether Zenoh is already up -- Cyclo Manager usually runs it, and a second daemon causes
confusing failures:

    docker ps | grep zenoh_daemon

If it is running, **skip** `rmw_zenohd` entirely. If not, in the ai_worker container: `zenohd`

---

## 1. VR publisher  [terminal 1]  -- no motion possible

    cd ~/robotis_applications/docker && ./container.sh enter
    ros2 launch robotis_vuer vr.launch.py model:=sg2

Leave it running. Keep hands off the grip buttons.

## 2. Quest browser  -- no motion possible

In the **Meta Quest Browser**, inside the headset:

    https://192.168.6.2:8012?ws=wss://192.168.6.2:8012

Accept the certificate (Advanced -> Proceed). Stand on the mark, elbows bent, hands forward.
Press **Enter VR**.

Ready when passthrough is on AND axis markers appear on **both** hands.

Then hang the headset around your neck.

> Any time the Vuer server restarts you MUST refresh this page and press Enter VR again. The failure
> is silent: the UI renders and nothing works.

## 3. Verify the input path  [terminal 2]  -- still no motion possible

    timeout 15 ros2 topic echo /vr_controller/left_squeeze > /tmp/sq.txt

Squeeze only the LEFT grip during those 15 s, then:

    grep "data" /tmp/sq.txt | sort -u | tail -3

Expect values rising to 1.0. A constant 0.0 means the VR session paused -- almost always the
proximity sensor, not a ROS fault.

---

## 4. Robot bringup  [terminal 3]  -- FIRST MOTION. The robot moves to its initial pose.

    cd ~/ai_worker && ./docker/container.sh enter
    ros2 launch ffw_bringup ffw_sg2_follower_ai.launch.py

Shortcut: `ffw_sg2_follower_ai`

Stand clear. This is the first step where the arms can move.

## 5. Motion controller  [terminal 4]  -- no motion by itself

    ros2 launch cyclo_motion_controller_ros ai_worker_controller.launch.py controller_type:=vr

Shortcut: `motion_controller controller_type:=vr`

Wait until the robot has reached its initial pose before starting this.

### 5b. Capture discovery data here, before activating  [terminal 5]

Everything is up and nothing is being commanded. This is the right moment:

    {
    echo "=== READY POSE ==="; timeout 3 ros2 topic echo /joint_states --once
    echo; echo "=== NODES ==="; ros2 node list
    echo; echo "=== ARM COMMAND PATH ==="
    ros2 topic info /leader/joint_trajectory_command_broadcaster_left/joint_trajectory --verbose
    ros2 topic info /leader/joint_trajectory_command_broadcaster_right/joint_trajectory --verbose
    echo; echo "=== TOPICS ==="; ros2 topic list
    } > /tmp/discovery.txt 2>&1

---

## 6. Activate  -- the robot follows you from here

**Match your arm posture to the robot's current pose first.** The system compares controller poses
to the robot's wrist poses and only starts if they are close; it then runs a 3-second check and a
5-second slow start.

1. Hold **both** squeeze/grip buttons (the deadman -- both are required, threshold 0.8)
2. Press **left X + right A** together

## 7. Pause and resume

Release the squeeze buttons to pause. To resume, re-align your posture to the robot's current pose
first, then repeat step 6. Resuming with your hands far from the last pose makes the robot move
quickly.

---

## Notes learned the hard way

* Docker Hub 401 on `ros:jazzy-ros-base` means a stale PAT in `~/.docker/config.json`. `docker
  logout` clears it; the file is safe to delete and is recreated on next login.
* "Enter VR" only exists in the headset's browser. On a desktop browser the page renders with no
  button, which looks like a fault and is not.
* The squeeze topic publishes at ~30 Hz mean but bursty, with gaps seen up to 167 ms. Log message
  timestamps rather than assuming a fixed rate.
