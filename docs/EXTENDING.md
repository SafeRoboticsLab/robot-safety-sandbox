# Extending the sandbox

Everything in the sandbox is a plain-python dataclass config plus small torch
functions — no YAML layer, no hydra. To add a task you write (1) an env cfg
builder, (2) a margin function, (3) a `TaskSpec` registration; each part is a
few dozen lines patterned on an existing task. This guide walks the four
extension axes with worked examples taken from the shipped tasks.

The registry contract (`robot_safety_sandbox/registry.py`):

```python
register(TaskSpec(
  task_id="my_task",
  cfg_builder=my_env_cfg,          # (play: bool) -> ManagerBasedRlEnvCfg
  margin_fn=my_margins,            # (env) -> (g, l)   [None for kind="nominal"]
  default_algo="ReachAvoidPPO",    # SafetyPPO | ReachAvoidPPO | IsaacsPPO | PPO
  kind="safety",                   # "safety" (margins) | "nominal" (dense)
  supports_adversary=False,
))
```

Then `examples/train.py --task my_task` (safety) or `train_nominal.py`
(nominal) just work; `make_tensor("my_task", num_envs=2048)` builds the
GPU-resident env.

## 1. Margin functions (g and l)

A margin function maps the live batched env to two `(num_envs,)` tensors:
`g` (avoid: `g < 0` == failure) and `l` (reach: `l >= 0` == target reached),
signed and normalized to O(1). Compose them from the library in
`robot_safety_sandbox/margins.py`:

```python
from robot_safety_sandbox.margins import compose, g_terrain_relative, l_gap_completion

my_margins = compose(g_terrain_relative, l_gap_completion)   # (env) -> (g, l)
```

Writing a new term is ordinary torch over the mjlab scene API:

```python
def g_upright(env, sin_tilt_limit=0.94):
  """Stay upright: signed distance in projected-gravity tilt."""
  robot = env.scene["robot"]
  tilt = torch.norm(robot.data.projected_gravity_b[:, :2], dim=1)  # sin(angle)
  return (sin_tilt_limit - tilt) / 0.3          # O(1) normalization

def l_at_rest_past(env, x_goal=5.0):
  """Reach: at rest beyond x_goal (min of two conditions = AND)."""
  robot = env.scene["robot"]
  x_rel = robot.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
  speed = torch.norm(robot.data.root_link_lin_vel_w[:, :2], dim=1)
  return torch.minimum((x_rel - x_goal) / 0.5, (0.3 - speed) / 0.5)
```

Conventions that matter (each was learned the hard way — see
safety-stable-baselines/BEST_PRACTICES.md):
- **min = AND, max = OR** for combining conditions inside one margin.
- **Normalize every term to O(1)**; the l/g magnitude ratio is the implicit
  risk-tolerance dial (break-even attempt probability = |g|/(|g|+l)).
- **Check the g floor against your reset distribution's physics** — a margin
  violated by the spawn states themselves condemns that state space.
- The env must terminate when g < 0; the bridge's safety hook does this for
  registered tasks automatically.

## 2. Sensors and observations

Env cfgs are mjlab `ManagerBasedRlEnvCfg` dataclasses; builders typically
start from a shipped cfg and mutate it. Adding/retargeting sensors
(`envs/parkour/go2.py` does exactly this for Go2):

```python
from mjlab.sensor import ContactSensorCfg, RayCastSensorCfg

def my_env_cfg(play: bool = False):
  cfg = base_env_cfg(play=play)
  for sensor in cfg.scene.sensors or ():
    if isinstance(sensor, RayCastSensorCfg) and sensor.name == "terrain_scan":
      sensor.frame.name = "base_link"          # retarget to your robot's frame
  cfg.scene.sensors += (ContactSensorCfg(
    name="feet_ground_contact", primary="foot_collision", secondary="terrain",
    data=("found", "time"), reduce="netforce", num=4),)
  return cfg
```

Observation terms are entries in `cfg.observations[group].terms` — plain
functions `(env) -> tensor`, addable per group (`"proprioception"` for the
safety policy, `"actor"` for a nominal). Margins may read sensors directly
(`env.scene["feet_ground_contact"].data.current_contact_time`), and
certificate features should be **state-only** (no commands, no action
history) — see `features.py` for why (OOD at filter handover otherwise).

## 3. Terrains: heightfields, walls, gaps, obstacles

Custom terrain = a `SubTerrainCfg` dataclass whose `function` emits boxes /
heightfields for one tile (`envs/terrains/island.py` is the 40-line
reference):

```python
@dataclasses.dataclass
class MyGapTerrainCfg(SubTerrainCfg):
  gap_width_range: tuple[float, float] = (0.2, 0.6)

  def function(self, difficulty, spec, rng) -> TerrainOutput:
    body = spec.worldbody.add_body(name="terrain")
    geoms = []
    w = self.gap_width_range[0] + difficulty * (
        self.gap_width_range[1] - self.gap_width_range[0])
    _add_box(body, geoms, pos=(-1.0, 0, 0), size=(2.0, 4.0, 0.1))   # approach
    _add_box(body, geoms, pos=(w + 2.0, 0, 0), size=(2.0, 4.0, 0.1))  # far side
    # walls/obstacles are just more boxes; heightfields via spec.add_hfield
    return TerrainOutput(origin=(0.0, 0.0, 0.1), geoms=geoms)
```

Wire it via the terrain generator's `sub_terrains` dict in your cfg builder.
`difficulty` (0..1) is driven by the curriculum; pin it for eval by setting
`gap_width_range=(w, w)` (see `examples/eval_filter.py`). Curriculum
promotion predicates must measure **composed task success** — promoting on
timeouts alone gets exploited by standing still.

## 4. Contacts

Contact information enters three ways, all shown in shipped tasks:

- **Margins**: read a `ContactSensorCfg` (see §2) —
  `tasks/go2_crawl.py` gates its gait terms on per-foot
  `current_contact_time > 0` and excludes thigh/calf geoms from the illegal-
  contact term (a trunk-plant failure mode hid inside an over-broad contact
  margin; scope contact terms to the geoms that actually mean failure).
- **Robot collision geometry**: the robot cfg's `CollisionCfg`
  (`envs/assets_*/**_constants.py`) controls which geoms collide, condim,
  friction — Digit's `FEET_COLLISION` restricts collisions to toe geoms.
- **Sim budget**: dense contact scenes may need
  `cfg.sim.contact_sensor_maxmatch` raised (the parkour cfg sets 200).

## Checklist for a new robot

`envs/assets_go2/` (quadruped) and `envs/assets_digit/` (humanoid, with
closed kinematic loops and payload variants) are the two references:

1. `envs/assets_<robot>/xmls/<robot>.xml` + meshes; strip floor/lights (the
   sandbox owns terrain).
2. `<robot>_constants.py`: `EntityCfg` (actuators, default pose, collision,
   action scales), `get_<robot>_robot_cfg()`.
3. Env builders under `envs/<robot>_<task>/`; margins next to them or in a
   task module under `tasks/`.
4. `TaskSpec` registrations (+ a `kind="nominal"` dense twin under
   `nominal/` if you'll run filters).
5. Verify: import + `list_tasks()`, cfg construction both modes, one
   `make_tensor(..., num_envs=8)` reset/step on GPU.
