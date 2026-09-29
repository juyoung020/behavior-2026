# Tasks - BEHAVIOR

> 원본: https://behavior.stanford.edu/omnigibson/tasks.html
> 받은 시각: 2026-09-29 18:22 KST (2026-09-29 09:22 UTC)
> 비고: 대회 페이지 본문이 링크하거나 대회 준비에 직접 쓰이는 사이트 문서
> 이 파일은 `_scrape.py` 가 자동으로 받은 원문 보관본이다. HTML→Markdown 변환 중 서식은 달라질 수 있으나 문장은 원문 그대로다.

---
# **Tasks**

## Description

`Task`s define the high-level objectives that an agent must complete in a given `Environment`, subject to certain constraints (e.g. not flip over).

`Task`s have two important internal variables:

- `_termination_conditions`: a dict of {`str`: `TerminationCondition`} that define when an episode should be terminated. For each of the termination conditions, `termination_condition.step(...)` returns a tuple of `(done [bool], success [bool])`. If any of the termination conditions returns `done = True`, the episode is terminated. If any returns `success = True`, the episode is considered successful.
- `_reward_functions`: a dict of {`str`: `RewardFunction`} that define how the agent is rewarded. Each reward function has a `reward_function.step(...)` method that returns a tuple of `(reward [float], info [dict])`. The `reward` is a scalar value that is added to the agent's total reward for the current step. The `info` is a dictionary that can contain additional information about the reward.

`Task`s usually specify task-relevant observations (e.g. goal location for a navigation task) via the `_get_obs` method, which returns a tuple of `(low_dim_obs [dict], obs [dict])`, where the first element is a dict of low-dimensional observations that will be automatically flattened into a 1D array, and the second element is everything else that shouldn't be flattened. Different types of tasks should overwrite the `_get_obs` method to return the appropriate observations.

`Task`s also define the reset behavior (in-between episodes) of the environment via the `_reset_scene`, `_reset_agent`, and `_reset_variables` methods.

- `_reset_scene`: reset the scene for the next episode, default is `scene.reset()`.
- `_reset_agent`: reset the agent for the next episode, default is do nothing.
- `_reset_variables`: reset any internal variables as needed, default is do nothing.

Different types of tasks should overwrite these methods for the appropriate reset behavior, e.g. a navigation task might want to randomize the initial pose of the agent and the goal location.

## Usage

### Specifying

Every `Environment` instance includes a task, defined by its config that is passed to the environment constructor via the `task` key. This is expected to be a dictionary of relevant keyword arguments, specifying the desired task configuration to be created (e.g. reward type and weights, hyperparameters for reset behavior, etc). The `type` key is required and specifies the desired task class. Additional keys can be specified and will be passed directly to the specific task class constructor. An example of a task configuration is shown below in `.yaml` form:

> **point_nav_example.yaml**
>
> |   | ``` task: type: PointNavigationTask robot_idn: 0 floor: 0 initial_pos: null initial_quat: null goal_pos: null goal_tolerance: 0.36 # turtlebot bodywidth goal_in_polar: false path_range: [1.0, 10.0] visualize_goal: true visualize_path: false n_vis_waypoints: 25 reward_type: geodesic termination_config: max_collisions: 500 max_steps: 500 fall_height: 0.03 reward_config: r_potential: 1.0 r_collision: 0.1 r_pointgoal: 10.0 ``` |
> |---|---|

### Runtime

`Environment` instance has a `task` attribute that is an instance of the specified task class. Internally, `Environment`'s `reset` method will call the task's `reset` method, `step` method will call the task's `step` method, and the `get_obs` method will call the task's `get_obs` method.

## Types

**`OmniGibson`** currently supports 5 types of tasks, 7 types of termination conditions, and 5 types of reward functions.

### `Task`

| [**`DummyTask`**](https://behavior.stanford.edu/reference/tasks/dummy_task.html) Dummy task with trivial implementations. - `termination_conditions`: empty dict. - `reward_functions`: empty dict. - `_get_obs()`: empty dict. - `_reset_scene()`: default. - `_reset_agent()`: default. |
|---|
| [**`PointNavigationTask`**](https://behavior.stanford.edu/reference/tasks/point_navigation_task.html) PointGoal navigation task with fixed / randomized initial pose and goal location. - `termination_conditions`: `MaxCollision`, `Timeout`, `PointGoal`. - `reward_functions`: `PotentialReward`, `CollisionReward`, `PointGoalReward`. - `_get_obs()`: returns relative xy position to the goal, and the agent's current linear and angular velocities. - `_reset_scene()`: default. - `_reset_agent()`: sample initial pose and goal location. |
| [**`PointReachingTask`**](https://behavior.stanford.edu/reference/tasks/point_reaching_task.html) Similar to PointNavigationTask, except the goal is specified with respect to the robot's end effector. - `termination_conditions`: `MaxCollision`, `Timeout`, `PointGoal`. - `reward_functions`: `PotentialReward`, `CollisionReward`, `PointGoalReward`. - `_get_obs()`: returns the goal position and the end effector's position in the robot's frame, and the agent's current linear and angular velocities. - `_reset_scene()`: default. - `_reset_agent()`: sample initial pose and goal location. |
| [**`GraspTask`**](https://behavior.stanford.edu/reference/tasks/grasp_task.html) Grasp task for a single object. - `termination_conditions`: `Timeout`. - `reward_functions`: `GraspReward`. - `_get_obs()`: returns the object's pose in the robot's frame - `_reset_scene()`: reset pose for objects in `_objects_config`. - `_reset_agent()`: randomize the robot's pose and joint configurations. |
| [**`BehaviorTask`**](https://behavior.stanford.edu/reference/tasks/behavior_task.html) BEHAVIOR task of long-horizon household activity. - `termination_conditions`: `Timeout`, `PredicateGoal`. - `reward_functions`: `PotentialReward`. - `_get_obs()`: returns the existence, pose, and in-gripper information of all task relevant objects - `_reset_scene()`: default. - `_reset_agent()`: default. |

> **Follow our tutorial on BEHAVIOR tasks!**
>
> To better understand how to use / sample / load / customize BEHAVIOR tasks, please read our [BEHAVIOR tasks documentation](https://behavior.stanford.edu/behavior_components/behavior_tasks.html)!

### `TerminationCondition`

| [**`Timeout`**](https://behavior.stanford.edu/reference/termination_conditions/timeout.html) `FailureCondition`: episode terminates if `max_step` steps have passed. |
|---|
| [**`Falling`**](https://behavior.stanford.edu/reference/termination_conditions/falling.html) `FailureCondition`: episode terminates if the robot can no longer function (i.e.: falls below the floor height by at least `fall_height` or tilt too much by at least `tilt_tolerance`). |
| [**`MaxCollision`**](https://behavior.stanford.edu/reference/termination_conditions/max_collision.html) `FailureCondition`: episode terminates if the robot has collided more than `max_collisions` times. |
| [**`PointGoal`**](https://behavior.stanford.edu/reference/termination_conditions/point_goal.html) `SuccessCondition`: episode terminates if point goal is reached within `distance_tol` by the robot's base. |
| [**`ReachingGoal`**](https://behavior.stanford.edu/reference/termination_conditions/reaching_goal.html) `SuccessCondition`: episode terminates if reaching goal is reached within `distance_tol` by the robot's end effector. |
| [**`GraspGoal`**](https://behavior.stanford.edu/reference/termination_conditions/grasp_goal.html) `SuccessCondition`: episode terminates if target object has been grasped (by assistive grasping). |
| [**`PredicateGoal`**](https://behavior.stanford.edu/reference/termination_conditions/predicate_goal.html) `SuccessCondition`: episode terminates if all the goal predicates of `BehaviorTask` are satisfied. |

### `RewardFunction`

| [**`CollisionReward`**](https://behavior.stanford.edu/reference/reward_functions/collision_reward.html) Penalization of robot collision with non-floor objects, with a negative weight `r_collision`. |
|---|
| [**`PointGoalReward`**](https://behavior.stanford.edu/reference/reward_functions/point_goal_reward.html) Reward for reaching the goal with the robot's base, with a positive weight `r_pointgoal`. |
| [**`ReachingGoalReward`**](https://behavior.stanford.edu/reference/reward_functions/reaching_goal_reward.html) Reward for reaching the goal with the robot's end-effector, with a positive weight `r_reach`. |
| [**`PotentialReward`**](https://behavior.stanford.edu/reference/reward_functions/potential_reward.html) Reward for decreasing some arbitrary potential function value, with a positive weight `r_potential`. It assumes the task already has `get_potential` implemented. Generally low potential is preferred (e.g. a common potential for goal-directed task is the distance to goal). |
| [**`GraspReward`**](https://behavior.stanford.edu/reference/reward_functions/grasp_reward.html) Reward for grasping an object. It not only evaluates the success of object grasping but also considers various penalties and efficiencies. The reward is calculated based on several factors: - Grasping reward: A positive reward is given if the robot is currently grasping the specified object. - Distance reward: A reward based on the inverse exponential distance between the end-effector and the object. - Regularization penalty: Penalizes large magnitude actions to encourage smoother and more energy-efficient movements. - Position and orientation penalties: Discourages excessive movement of the end-effector. - Collision penalty: Penalizes collisions with the environment or other objects. |
