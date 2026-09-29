# Quickstart - BEHAVIOR

> 원본: https://behavior.stanford.edu/getting_started/quickstart.html
> 받은 시각: 2026-09-29 18:22 KST (2026-09-29 09:22 UTC)
> 비고: 대회 페이지 본문이 링크하거나 대회 준비에 직접 쓰이는 사이트 문서
> 이 파일은 `_scrape.py` 가 자동으로 받은 원문 보관본이다. HTML→Markdown 변환 중 서식은 달라질 수 있으나 문장은 원문 그대로다.

---
# **Quickstart**

Let's quickly create an environment programmatically!

**`OmniGibson`**'s workflow is straightforward: define the configuration of scene, object(s), robot(s), and task you'd like to load, and then instantiate our `Environment` class with that config.

Let's start with the following:

```
import omnigibson as og # (1)!
from omnigibson.macros import gm # (2)!

# Start with an empty configuration
cfg = dict()
```

1. All python scripts should start with this line! This allows access to key global variables through the top-level package.
2. Global macros (`gm`) can always be accessed directly and modified on the fly!

## **Defining a scene**

Next, let's define a scene:

```
cfg["scene"] = {
    "type": "Scene", # (1)!
    "floor_plane_visible": True, # (2)!
}
```

1. Our configuration gets parsed automatically and generates the appropriate class instance based on `type` (the string form of the class name). In this case, we're generating the most basic scene, which only consists of a floor plane. Check out [all of our available `Scene` classes](https://behavior.stanford.edu/reference/scenes/scene_base.html)!
2. In addition to specifying `type`, the remaining keyword-arguments get passed directly into the class constructor. So for the base [`Scene`](https://behavior.stanford.edu/reference/scenes/scene_base.html) class, you could optionally specify `"use_floor_plane"` and `"floor_plane_visible"`, whereas for the more powerful [`InteractiveTraversableScene`](https://behavior.stanford.edu/reference/scenes/interactive_traversable_scene.html) class (which loads a curated, preconfigured scene) you can additionally specify options for filtering objects, such as `"load_object_categories"` and `"load_room_types"`. You can see all available keyword-arguments by viewing the [individual `Scene` class](https://behavior.stanford.edu/reference/scenes/scene_base.html) you'd like to load!

## **Defining objects**

We can optionally define some objects to load into our scene:

```
cfg["objects"] = [ # (1)!
    {
        "type": "DatasetObject", # (2)!
        "name": "delicious_apple",
        "category": "apple", # (3)!
        "model": "agveuv", # (4)!
        "position": [0, 0, 1.0], # (5)!
    },
    {
        "type": "PrimitiveObject", # (6)!
        "name": "incredible_box",
        "primitive_type": "Cube", # (7)!
        "rgba": [0, 1.0, 1.0, 1.0], # (8)!
        "scale": [0.5, 0.5, 0.1], # (9)!
        "fixed_base": True, # (10)!
        "position": [-1.0, 0, 1.0],
        "orientation": [0, 0, 0.707, 0.707], # (11)!
    },
    {
        "type": "LightObject", # (12)!
        "name": "brilliant_light",
        "light_type": "Sphere", # (13)!
        "intensity": 50000, # (14)!
        "radius": 0.1, # (15)!
        "position": [3.0, 3.0, 4.0],
    },
]

# Note: USDObject loading is also supported if you have custom USD objects
# However, all objects in our dataset are encrypted and must be loaded via DatasetObject
# Below is an example of how you would load a custom USD object:
usd_cfg = {
    "type": "USDObject",
    "name": "awesome_custom_usd",
    "usd_path": "/path/to/your/custom.usd",
    "category": "my_category",
    "visual_only": True, # (17)!
    "scale": [1.0, 1.0, 1.0],
    "position": [1.0, 2.0, 1.0],
    "orientation": [0, 0, 0, 1.0],
},
```

1. Unlike the `"scene"` sub-config, we can define an arbitrary number of objects to load, so this is a `list` of `dict` istead of a single nested `dict`.
2. **`OmniGibson`** supports multiple object classes, and we showcase an instance of each core class here. A [`DatasetObject`](https://behavior.stanford.edu/reference/objects/dataset_object.html) is the default way to load objects from our **BEHAVIOR** dataset. It includes metadata and annotations not found on generic objects. Note that these assets are encrypted, and thus cannot be loaded via the `USDObject` class.
3. `category` is used by all object classes to assign semantic segmentation IDs.
4. Instead of explicitly defining the hardcoded path to the dataset USD model, `model` (in conjunction with `category`) is used to infer the exact dataset object to load. All objects **must** define the `name` argument! This is because **`OmniGibson`** enforces a global unique naming scheme.
5. `position` is used by all object classes and defines the initial (x,y,z) position of the object in the global frame.
6. A [`PrimitiveObject`](https://behavior.stanford.edu/reference/objects/primitive_object.html) is a programmatically generated object defining a convex primitive shape.
7. `primitive_type` defines what primitive shape to load -- see [`PrimitiveObject`](https://behavior.stanford.edu/reference/objects/primitive_object.html) for available options!
8. Because this object is programmatically generated, we can also specify the color to assign to this primitive object.
9. `scale` is used by all object classes and defines the global (x,y,z) relative scale of the object.
10. `fixed_base` is used by all object classes and determines whether the generated object is fixed relative to the world frame. Useful for fixing in place large objects, such as furniture or structures.
11. `orientation` is used by all object classes and defines the initial (x,y,z,w) quaternion orientation of the object in the global frame.
12. A [`LightObject`](https://behavior.stanford.edu/reference/objects/light_object.html) is a programmatically generated light source. It is used to directly illuminate the given scene.
13. `light_type` defines what light shape to load -- see [`LightObject`](https://behavior.stanford.edu/reference/objects/light_object.html) for available options!
14. `intensity` defines how bright the generated light source should be.
15. `radius` is used by `Sphere` lights and determines their relative size.
16. `visual_only` is used by all object classes and defines whether the object is subject to both gravity and collisions.

**Note on other object types:** In addition to the objects shown above, **`OmniGibson`** also supports [`USDObject`](https://behavior.stanford.edu/reference/objects/usd_object.html) for loading custom USD files. This is our most generic object class and generates an object sourced from a `usd_path` argument. However, since all objects in our dataset are encrypted, they must be loaded using `DatasetObject`. If you have your own custom USD objects, you can load them using `USDObject` by providing the path to your USD file. We've included a commented example at the end of the object list above to show how this would work.

## **Defining robots**

We can also optionally define robots to load into our scene:

```
cfg["robots"] = [ # (1)!
    {
        "type": "Fetch", # (2)!
        "name": "baby_robot",
        "obs_modalities": ["rgb", "depth"], # (3)!
    },
]
```

1. Like the `"objects"` sub-config, we can define an arbitrary number of robots to load, so this is a `list` of `dict`.
2. **`OmniGibson`** supports multiple robot classes, where each class represents a specific robot model. Check out our [`robots`](https://behavior.stanford.edu/reference/robots/robot_base.md) to view all available robot classes!
3. Execute `print(og.ALL_SENSOR_MODALITIES)` for a list of all available observation modalities!

## **Defining a task**

Lastly, we can optionally define a task to load into our scene. Since we're just getting started, let's load a "Dummy" task (which is the task that is loaded anyways even if we don't explicitly define a task in our config):

```
cfg["task"] = {
    "type": "DummyTask", # (1)!
    "termination_config": dict(), # (2)!
    "reward_config": dict(), # (3)!
}
```

1. Check out all of **`OmniGibson`**'s [available tasks](https://behavior.stanford.edu/reference/tasks/task_base.html)!
2. `termination_config` configures the termination conditions for this task. It maps specific [`TerminationCondition`](https://behavior.stanford.edu/reference/termination_conditions/termination_condition_base.html) arguments to their corresponding values to set.
3. `reward_config` configures the reward functions for this task. It maps specific [`RewardFunction`](https://behavior.stanford.edu/reference/reward_functions/reward_function_base.html) arguments to their corresponding values to set.

## **Creating the environment**

We're all set! Let's load the config and create our environment:

```
env = og.Environment(cfg)
```

Once the environment loads, we can interface with our environment similar to OpenAI's Gym interface:

```
obs, rew, terminated, truncated, info = env.step(env.action_space.sample())
```

> **What happens if we have no robot loaded?**
>
> Even if we have no robot loaded, we still need to define an "action" to pass into the environment. In this case, our action space is 0, so you can simply pass `[]` or `np.array([])` into the `env.step()` call!

> **my_first_env.py**
>
> |   | ``` import omnigibson as og from omnigibson.macros import gm cfg = dict() # Define scene cfg["scene"] = { "type": "Scene", "floor_plane_visible": True, } # Define objects cfg["objects"] = [ { "type": "DatasetObject", "name": "delicious_apple", "category": "apple", "model": "agveuv", "position": [0, 0, 1.0], }, { "type": "PrimitiveObject", "name": "incredible_box", "primitive_type": "Cube", "rgba": [0, 1.0, 1.0, 1.0], "scale": [0.5, 0.5, 0.1], "fixed_base": True, "position": [-1.0, 0, 1.0], "orientation": [0, 0, 0.707, 0.707], }, { "type": "LightObject", "name": "brilliant_light", "light_type": "Sphere", "intensity": 50000, "radius": 0.1, "position": [3.0, 3.0, 4.0], }, ] # Define robots cfg["robots"] = [ { "type": "Fetch", "name": "baby_robot", "obs_modalities": ["rgb", "depth"], }, ] # Define task cfg["task"] = { "type": "DummyTask", "termination_config": dict(), "reward_config": dict(), } # Create the environment env = og.Environment(cfg) # Allow camera teleoperation og.sim.enable_viewer_camera_teleoperation() # Step! for _ in range(10000): obs, rew, terminated, truncated, info = env.step(env.action_space.sample()) og.shutdown() ``` |
> |---|---|

## **Looking around**

Look around by:

- `Left-CLICK + Drag`: Tilt
- `Scroll-Wheel-CLICK + Drag`: Pan
- `Scroll-Wheel UP / DOWN`: Zoom

Interact with objects by:

- `Shift + Left-CLICK + Drag`: Apply force on selected object

Or, for more fine-grained control, run:

```
og.sim.enable_viewer_camera_teleoperation() # (1)!
```

1. This allows you to move the camera precisely with your keyboard, record camera poses, and dynamically modify lights!

Or, for programmatic control, directly set the viewer camera's global pose:

```
og.sim.viewer_camera.set_position_orientation(<POSITION>, <ORIENTATION>)
```

---

**Next:** Check out some of **`OmniGibson`**'s breadth of features from our [Modules](https://behavior.stanford.edu/omnigibson/overview.html) pages!
