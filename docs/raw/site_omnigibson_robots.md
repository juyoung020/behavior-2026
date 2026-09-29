# Robots - BEHAVIOR

> 원본: https://behavior.stanford.edu/omnigibson/robots.html
> 받은 시각: 2026-09-29 18:22 KST (2026-09-29 09:22 UTC)
> 비고: 대회 페이지 본문이 링크하거나 대회 준비에 직접 쓰이는 사이트 문서
> 이 파일은 `_scrape.py` 가 자동으로 받은 원문 보관본이다. HTML→Markdown 변환 중 서식은 달라질 수 있으나 문장은 원문 그대로다.

---
# **Robots**

![Robot Family](https://behavior.stanford.edu/assets/robots/robot_family.png)

## Description

In **`OmniGibson`**, `Robot`s define agents that can interact with other objects in a given environment. Each robot can *interact* by deploying joint commands via its set of [`Controller`](https://behavior.stanford.edu/omnigibson/controllers.html)s, and can *perceive* its surroundings via its set of [`Sensor`](https://behavior.stanford.edu/omnigibson/sensors.html)s.

**`OmniGibson`** supports both navigation and manipulation robots, and allows for modular specification of individual controllers for controlling the different components of a given robot. For example, the `Fetch` robot is a mobile manipulator composed of a mobile (two-wheeled) base, two head joints, a trunk, seven arm joints, and two gripper finger joints. `Fetch` owns 4 controllers, one for controlling the base, the head, the trunk + arm, and the gripper. There are multiple options for each controller depending on the desired action space. For more information, check out our [robot examples](https://behavior.stanford.edu/getting_started/examples.html#robots).

It is important to note that robots are full-fledged `StatefulObject`s, and thus leverage the same APIs as normal scene objects and can be treated as such. Robots can be thought of as `StatefulObject`s that additionally own controllers (`robot.controllers`) and sensors (`robot.sensors`).

## Usage

### Importing

Robots can be added to a given `Environment` instance by specifying them in the config that is passed to the environment constructor via the `robots` key. This is expected to be a list of dictionaries, where each dictionary specifies the desired configuration for a single robot to be created. For each dict, the `type` key is required and specifies the desired robot class, and global `position` and `orientation` (in (x,y,z,w) quaternion form) can also be specified. Additional keys can be specified and will be passed directly to the specific robot class constructor. An example of a robot configuration is shown below in `.yaml` form:

> **single_fetch_config_example.yaml**
>
> |   | ``` robots: - type: Fetch position: [0, 0, 0] orientation: [0, 0, 0, 1] obs_modalities: [scan, rgb, depth] scale: 1.0 self_collision: false action_normalize: true action_type: continuous grasping_mode: physical rigid_trunk: false default_trunk_offset: 0.365 default_arm_pose: diagonal30 reset_joint_pos: tuck sensor_config: VisionSensor: sensor_kwargs: image_height: 128 image_width: 128 ScanSensor: sensor_kwargs: min_range: 0.05 max_range: 10.0 controller_config: base: name: DifferentialDriveController arm_0: name: InverseKinematicsController kv: 2.0 gripper_0: name: MultiFingerGripperController mode: binary camera: name: JointController use_delta_commands: False ``` |
> |---|---|

### Runtime

Usually, actions are passed to robots and observations retrieved via the `obs, info, terminated, truncated, done = env.step(action)`. However, actions can be directly deployed and observations retrieved from the robot using the following APIs:

- **Applying actions**: `robot.apply_action(action)` (1)
- **Retrieving observations**: `obs, info = robot.get_obs()` (2)

1. `action` is a 1D-numpy array. For more information, please see the [Controller](https://behavior.stanford.edu/omnigibson/controllers.html) section!
2. `obs` is a dict mapping observation name to observation data, and `info` is a dict of relevant metadata about the observations. For more information, please see the [Sensor](https://behavior.stanford.edu/omnigibson/sensors.html) section!

Controllers and sensors can be accessed directly via the `controllers` and `sensors` properties, respectively. And, like all objects in **`OmniGibson`**, common information such as joint data and object states can also be directly accessed from the `robot` class. Note that by default, control signals are updated and deployed every physics timestep via the robot's internal `step()` callback function. To disable controllers from automatically deploying control signals, set `robot.control_enabled = False`. This means that additional `step()` calls will **not** update the control signals sent, and so the most recent control signal will still be propagated until the user either re-enables control or manually sets the robot's joint positions / velocities / efforts.

## Types

**`OmniGibson`** currently supports 12 robots, consisting of 4 mobile robots, 3 manipulation robots, 4 mobile manipulation robots, and 1 anthropomorphic "robot" (a bimanual agent proxy used for VR teleoperation). Below, we provide a brief overview of each model:

### Mobile Robots

These are navigation-only robots (an instance of [`LocomotionRobot`](https://behavior.stanford.edu/reference/robots/locomotion_robot.md)) that solely consist of a base that can move.

| [**`Turtlebot`**](https://behavior.stanford.edu/reference/robots/turtlebot.md) The two-wheeled [Turtlebot 2](https://www.turtlebot.com/turtlebot2/) model with the Kobuki base. - *Controllers*: Base - *Sensors*: Camera, LIDAR | ![rgb](https://behavior.stanford.edu/assets/robots/Turtlebot.png) |
|---|---|
| [**`Locobot`**](https://behavior.stanford.edu/reference/robots/locobot.md) The two-wheeled, open-source [LoCoBot](http://www.locobot.org/) model. Note that in our model the arm is disabled and is fixed to the base. - *Controllers*: Base - *Sensors*: Camera, LIDAR | ![rgb](https://behavior.stanford.edu/assets/robots/Locobot.png) |
| [**`Husky`**](https://behavior.stanford.edu/reference/robots/husky.md) The four-wheeled [Husky UAV](https://clearpathrobotics.com/husky-unmanned-ground-vehicle-robot/) model from Clearpath Robotics. - *Controllers*: Base - *Sensors*: Camera, LIDAR | ![rgb](https://behavior.stanford.edu/assets/robots/Husky.png) |
| [**`Freight`**](https://behavior.stanford.edu/reference/robots/freight.md) The two-wheeled [Freight](https://docs.fetchrobotics.com/) model which serves as the base for the Fetch robot. - *Controllers*: Base - *Sensors*: Camera, LIDAR | ![rgb](https://behavior.stanford.edu/assets/robots/Freight.png) |

### Manipulation Robots

These are manipulation-only robots (an instance of [`ManipulationRobot`](https://behavior.stanford.edu/reference/robots/manipulation_robot.md)) that cannot move and solely consist of an actuated arm with a gripper attached to its end effector.

| [**`Franka`**](https://behavior.stanford.edu/reference/robots/franka.md) The popular 7-DOF [Franka Research 3](https://franka.de/) model equipped with a parallel jaw gripper. Note that OmniGibson also includes three alternative versions of Franka with dexterous hands: FrankaAllegro (equipped with an Allegro hand), FrankaLeap (equipped with a Leap hand) and FrankaInspire (equipped with an inspire hand). - *Controllers*: Arm, Gripper - *Sensors*: Wrist Camera | ![rgb](https://behavior.stanford.edu/assets/robots/FrankaPanda.png) |
|---|---|
| [**`VX300S`**](https://behavior.stanford.edu/reference/robots/vx300s.md) The 6-DOF [ViperX 300 6DOF](https://www.trossenrobotics.com/viperx-300-robot-arm-6dof.aspx) model from Trossen Robotics equipped with a parallel jaw gripper. - *Controllers*: Arm, Gripper - *Sensors*: Wrist Camera | ![rgb](https://behavior.stanford.edu/assets/robots/VX300S.png) |
| [**`A1`**](https://behavior.stanford.edu/reference/robots/a1.md) The 6-DOF A1 model equipped with a Inspire-Robots Dexterous Hand. - *Controllers*: Arm, Gripper - *Sensors*: Wrist Camera | ![rgb](https://behavior.stanford.edu/assets/robots/A1.png) |
| [**`Franka Mounted`**](https://behavior.stanford.edu/reference/robots/franka_mounted.md) Franka mounted on an aluminum extrusion frame cart. - *Controllers*: Arm, Gripper - *Sensors*: Wrist Camera | ![rgb](https://behavior.stanford.edu/assets/robots/FrankaMounted.png) |

### Mobile Manipulation Robots

These are robots that can both navigate and manipulate (and inherit from both [`LocomotionRobot`](https://behavior.stanford.edu/reference/robots/locomotion_robot.md) and [`ManipulationRobot`](https://behavior.stanford.edu/reference/robots/manipulation_robot.md)), and are equipped with both a base that can move as well as one or more gripper-equipped arms that can actuate.

| [**`Fetch`**](https://behavior.stanford.edu/reference/robots/fetch.md) The [Fetch](https://docs.fetchrobotics.com/) model, composed of a two-wheeled base, linear trunk, 2-DOF head, 7-DOF arm, and 2-DOF parallel jaw gripper. - *Controllers*: Base, Head, Arm, Gripper - *Sensors*: Head Camera, Wrist Camera, LIDAR | ![rgb](https://behavior.stanford.edu/assets/robots/Fetch.png) |
|---|---|
| [**`Tiago`**](https://behavior.stanford.edu/reference/robots/tiago.md) The bimanual [Tiago](https://pal-robotics.com/robots/tiago/) model from PAL robotics, composed of a holonomic base (which we model as a 3-DOF (x,y,rz) set of joints), linear trunk, 2-DOF head, x2 7-DOF arm, and x2 2-DOF parallel jaw grippers. - *Controllers*: Base, Head, Left Arm, Right Arm, Left Gripper, Right Gripper - *Sensors*: Head Camera, Wrist Cameras, Rear LIDAR, Front LIDAR | ![rgb](https://behavior.stanford.edu/assets/robots/Tiago.png) |
| [**`Stretch`**](https://behavior.stanford.edu/reference/robots/stretch.md) The [Stretch](https://hello-robot.com/stretch-3-product) model from Hello Robot, composed of a two-wheeled base, 2-DOF head, 5-DOF arm, and 1-DOF gripper. - *Controllers*: Base, Head, Arm, Gripper - *Sensors*: Head Camera, Wrist Camera | ![rgb](https://behavior.stanford.edu/assets/robots/Stretch.png) |
| [**`R1`**](https://behavior.stanford.edu/reference/robots/r1.md) The bimanual R1 model, composed of a holonomic base (which we model as a 3-DOF (x,y,rz) set of joints), 4-DOF torso, x2 6-DOF arm, and x2 2-DOF parallel jaw grippers. - *Controllers*: Base, Left Arm, Right Arm, Left Gripper, Right Gripper - *Sensors*: Head Camera, Wrist Cameras | ![rgb](https://behavior.stanford.edu/assets/robots/R1.png) |
| [**`R1 Pro`**](https://behavior.stanford.edu/reference/robots/r1pro.md) The bimanual R1Pro model, composed of a holonomic base (which we model as a 3-DOF (x,y,rz) set of joints), 4-DOF torso, x2 7-DOF arm, and x2 2-DOF parallel jaw grippers. - *Controllers*: Base, Left Arm, Right Arm, Left Gripper, Right Gripper - *Sensors*: Head Camera, Wrist Cameras | ![rgb](https://behavior.stanford.edu/assets/robots/R1Pro.png) |

### Additional Robots

| [**`BehaviorRobot`**](https://behavior.stanford.edu/reference/robots/behavior_robot.md#robots.behavior_robot.BehaviorRobot) A hand-designed model intended to be used exclusively for VR teleoperation. - *Controllers*: Base, Head, Left Arm, Right Arm, Left Gripper, Right Gripper - *Sensors*: Head Camera | ![rgb](https://behavior.stanford.edu/assets/robots/BehaviorRobot.png) |
|---|---|
