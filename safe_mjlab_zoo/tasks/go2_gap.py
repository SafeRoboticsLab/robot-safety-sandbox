"""Go2 gap-jumping benchmark family (parkour skill 1).

Pipeline (each stage warm-starts the next; the jump is NOT learnable in one
stage — it forms through the landing -> crossing reverse curriculum):

  go2_gap_landing   avoid-only   SafetyPPO     mid-air spawn -> soft land
  go2_gap_crossing  avoid-only   SafetyPPO     reverse curriculum launch->land
  go2_gap_chain     reach-avoid  ReachAvoidPPO arrival momentum -> safe rest
  go2_gap_chain (+adversary)     IsaacsPPO     two-player ISAACS game

The env cfgs are NATIVE to the zoo (envs/go2_gap/*, migrated phase-2).
"""

from __future__ import annotations

from ..margins import (compose, g_terrain_relative, l_gap_completion,
                       l_gap_foothold, l_launch_basin, l_rest, l_zero)
from ..registry import TaskSpec, register


def _cfgs():
  from safe_mjlab_zoo.envs.go2_gap.landing import unitree_go2_landing_env_cfg
  from safe_mjlab_zoo.envs.go2_gap.crossing import unitree_go2_crossing_env_cfg
  from safe_mjlab_zoo.envs.go2_gap.chain import (
    unitree_go2_crossing_chain_env_cfg,
    unitree_go2_crossing_chain_isaacs_env_cfg)
  return dict(landing=unitree_go2_landing_env_cfg,
              crossing=unitree_go2_crossing_env_cfg,
              chain=unitree_go2_crossing_chain_env_cfg,
              chain_isaacs=unitree_go2_crossing_chain_isaacs_env_cfg)


def single_gap_twin_env_cfg(play: bool = False):
  """SINGLE-GAP twin env (the terrain the jump actually exists on — the chain
  cluster terrain does NOT transfer; see the 2026-07-10 transfer probe).

  gap.py base env (origin AT the near edge, widths 0.1–1.0 across the patch
  grid, constant forward drive) with the ground-spawn stratum WIDENED to cover
  the approach corridor (x in [-1.8, -0.1], vx 0.3–2.5): the certificate must
  be accurate at filter-engagement states, not just jump-relevant ones."""
  from safe_mjlab_zoo.envs.go2_gap.gap import unitree_go2_gap_reach_avoid_env_cfg
  cfg = unitree_go2_gap_reach_avoid_env_cfg(play=play)
  rb = cfg.events["reset_base"]
  rb.params["ground_pose_range"]["x"] = (-1.8, -0.1)
  rb.params["ground_velocity_range"]["x"] = (0.3, 2.5)
  rb.params["midair_fraction"] = 0.35
  cfg.episode_length_s = 8.0        # the gauntlet horizon
  cfg.events.pop("push_robot", None)  # twins train without random pushes
  return cfg


def l_launch_single(env):
  """Launch-basin l for the single-gap terrain: gap face at x_rel = 0."""
  from ..margins import l_launch_basin
  return l_launch_basin(env, gap_x=0.0, band=0.35, v_launch=2.0)


# --- ISLAND twins: the terrain the 96.8% crossing ckpt actually lives on ------

def island_twin_env_cfg(play: bool = False):
  """Island-crossing terrain (origin AT the gap face; 1 m island with back
  pit; widths 0.05-0.6; long far platform) + BROAD spawns (ground-on-island +
  midair) replacing the reverse curriculum. Probe-first rule: the crossing
  ckpt must score high here before any twin trains."""
  import math
  from mjlab.managers.event_manager import EventTermCfg
  import safe_mjlab_zoo.envs.parkour.mdp as pmdp
  from safe_mjlab_zoo.envs.go2_gap.crossing import unitree_go2_crossing_env_cfg
  cfg = unitree_go2_crossing_env_cfg(play=play)
  cfg.curriculum = {}
  cfg.episode_length_s = 8.0
  cfg.events.pop("push_robot", None)
  cfg.events["reset_base"] = EventTermCfg(
    func=pmdp.reset_robot_midair_over_gaps, mode="reset",
    params={
      "midair_fraction": 0.35,
      "ground_pose_range": {"x": (-0.9, -0.1), "y": (-0.15, 0.15),
                            "yaw": (-0.15, 0.15)},
      "ground_velocity_range": {"x": (0.3, 2.5), "y": (-0.2, 0.2)},
      "midair_x_range": (0.0, 0.4), "midair_y_range": (-0.15, 0.15),
      "midair_z_range": (0.10, 0.45),
      "midair_vx_range": (1.0, 2.5), "midair_vy_range": (-0.2, 0.2),
      "midair_vz_range": (-1.0, 0.3),
      "midair_roll_range": (-math.radians(10.0), math.radians(10.0)),
      "midair_pitch_range": (-math.radians(10.0), math.radians(10.0)),
      "midair_yaw_range": (-0.2, 0.2),
    })
  return cfg


def _island_width(env):
  """Per-env gap width from the terrain row (island cfg: 0.05 + d*0.55)."""
  import torch
  levels = env.scene.terrain.terrain_levels.float()
  rows = 10.0
  return 0.05 + (levels / (rows - 1.0)).clamp(0.0, 1.0) * 0.55


def l_launch_island(env):
  """BANDED launch-basin l (v3). v_launch=2.2 keeps the basin inside the
  validated ballistic region. CRITICAL v3 fix: the basin is a THIN SHELL at
  the commitment boundary (x in [-0.35, +0.10]), not a half-space — v2's
  one-sided proximity let mid-flight states keep l>0, so the whole flight
  banked positive value, crashes were invisible to the backup, and the twins
  learned to charge-and-die (97.8% deaths). With the exit bound, flight and
  landing states revert to pure-avoid backup: the crash back-propagates into
  the landing skill while the commitment gradient stays intact."""
  import torch
  from ..margins import l_launch_basin
  base = l_launch_basin(env, gap_x=0.0, band=0.35, v_launch=2.2)
  robot = env.scene["robot"]
  x_rel = robot.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
  exit_bound = (0.10 - x_rel) / 0.3
  return torch.minimum(base, exit_bound)


def l_launch_gated(env):
  """THE SYNTHESIS l: launch basin GATED by feasibility, min-form.

  l = min(momentum, proximity, (W_JUMPABLE - width)/norm). At uncrossable
  widths the basin is UNREACHABLE -> V degrades to pure-avoid (stop) by the
  value function itself — fixing launch-l's feasibility-blindness (19.5% viol
  at w=0.7 on chain) while keeping its trap-free commitment property."""
  import torch
  base = l_launch_island(env)
  feasible = (0.55 - _island_width(env)) / 0.2
  return torch.minimum(base, feasible)


# --- FUNNEL COMPOSITION: l = calibrated MC-outcome certificate of the FROZEN
# crossing skill (V_hat), + bridge-level hybrid rollouts (base.py). The frozen
# critic was proven NOT to be a certificate (anti-correlated with success —
# 2026-07-10 funnel probe); V_hat is fit for purpose by outcome regression on
# frozen-skill rollouts (collect_vhat_data.py / fit_vhat.py -> vhat_cross.pt).

def funnel_twin_env_cfg(play: bool = False):
  """Island twin env with the CERTIFICATE DATASET's coverage: midair spawns
  extended into the home-curriculum committed arc (vx to 3.2) and gap widths
  extended to 0.95 across all rows — V_hat's inputs stay in-distribution and
  the twin trains against genuinely uncrossable widths (feasibility)."""
  from dataclasses import replace
  cfg = island_twin_env_cfg(play=play)
  rp = cfg.events["reset_base"].params
  rp["midair_x_range"] = (-0.15, 0.4)
  rp["midair_z_range"] = (0.05, 0.45)
  rp["midair_vx_range"] = (1.0, 3.2)
  rp["midair_vz_range"] = (-1.0, 0.8)
  rp["ground_velocity_range"]["x"] = (0.3, 3.0)
  gen = cfg.scene.terrain.terrain_generator
  subs = {k: replace(v, gap_width_range=(0.05, 0.95))
          for k, v in gen.sub_terrains.items()}
  cfg.scene.terrain.terrain_generator = replace(gen, sub_terrains=subs)
  cfg.scene.terrain.max_init_terrain_level = None
  return cfg


def lander_env_cfg(play: bool = False):
  """DEDICATED LANDER env (the viability WITNESS of the certified-launch
  target set): island terrain, 100% airborne ballistic spawns. Every spawn is
  committed (no stopping midair), so avoid-only g-forcing applies everywhere
  in the distribution — the mechanism that built the crossing skill, now
  aimed at maximizing the landing basin. The envelope deliberately covers
  ASCENT states (vz > 0, launch-fresh) and steep postures: the RA trainee
  banks EARLY in flight, so the witness must be strong there, not just at
  apex/descent (the island twins' midair vz was (-1.0, 0.3))."""
  import math
  from dataclasses import replace
  cfg = island_twin_env_cfg(play=play)
  rp = cfg.events["reset_base"].params
  rp["midair_fraction"] = 1.0
  rp["midair_x_range"] = (-0.30, 0.60)
  rp["midair_z_range"] = (0.05, 0.55)
  rp["midair_vx_range"] = (0.8, 3.2)
  rp["midair_vz_range"] = (-1.5, 2.0)
  rp["midair_roll_range"] = (-math.radians(15.0), math.radians(15.0))
  rp["midair_pitch_range"] = (-math.radians(25.0), math.radians(25.0))
  gen = cfg.scene.terrain.terrain_generator
  subs = {k: replace(v, gap_width_range=(0.05, 0.95))
          for k, v in gen.sub_terrains.items()}
  cfg.scene.terrain.terrain_generator = replace(gen, sub_terrains=subs)
  cfg.scene.terrain.max_init_terrain_level = None
  return cfg


def lander_sg_env_cfg(play: bool = False):
  """Lander CERTIFICATION env on SINGLE-GAP terrain (the jump's home terrain,
  and the terrain the cert twins train on — the witness must be certified on
  the deployment geometry, not its island birthplace). 100% airborne
  ballistic spawns, ascent included."""
  import math
  cfg = single_gap_twin_env_cfg(play=play)
  rp = cfg.events["reset_base"].params
  rp["midair_fraction"] = 1.0
  rp["midair_x_range"] = (-0.30, 0.60)
  rp["midair_z_range"] = (0.05, 0.55)
  rp["midair_vx_range"] = (0.8, 3.2)
  rp["midair_vz_range"] = (-1.5, 2.0)
  rp["midair_roll_range"] = (-math.radians(15.0), math.radians(15.0))
  rp["midair_pitch_range"] = (-math.radians(25.0), math.radians(25.0))
  return cfg


def goarc_sg_env_cfg(play: bool = False):
  """GO-ARC certification env (run-up pivot, v5): CROSSING skill rolled from
  committed GROUND states on single-gap. Its certified subset V_hat_go >= p*
  becomes the run-up trainee's reach target — the guard moves from airborne
  to pre-launch ground, so the trainee NEVER executes the launch and the
  launch death-gradient leaves its objective entirely (the poison that
  stopped v2-v4.1). Spawn (x, vx) spans the arc boundary in both directions
  so the certificate maps where commitment succeeds vs fails."""
  cfg = single_gap_twin_env_cfg(play=play)
  rp = cfg.events["reset_base"].params
  rp["midair_fraction"] = 0.0
  rp["ground_pose_range"]["x"] = (-0.7, -0.05)
  rp["ground_velocity_range"]["x"] = (1.2, 3.5)
  return cfg


def launcher_sg_env_cfg(play: bool = False):
  """LAUNCHER (v5a): the missing competence. The go-arc certification measured
  the crossing skill at 0.8% from ground states on single-gap (died 130k/133k
  — charges and dies; best committed cell 8.6%): GROUND LAUNCH never existed
  in any component. Build it avoid-only via forcing: 50% midair ballistic
  (forces landing) + 50% truly-committed ground, vx 2.8-3.5 at the lip where
  braking is physically infeasible (forces launching). No walk-in stratum —
  this skill's world starts committed; the run-up trainee delivers into it."""
  import math
  cfg = single_gap_twin_env_cfg(play=play)
  rp = cfg.events["reset_base"].params
  rp["midair_fraction"] = 0.5
  rp["midair_x_range"] = (-0.1, 0.6)
  rp["midair_z_range"] = (0.05, 0.55)
  rp["midair_vx_range"] = (1.5, 3.5)
  rp["midair_vz_range"] = (-1.0, 2.0)
  rp["midair_roll_range"] = (-math.radians(15.0), math.radians(15.0))
  rp["midair_pitch_range"] = (-math.radians(25.0), math.radians(25.0))
  rp["ground_committed_fraction"] = 0.5
  rp["ground_committed_pose_range"] = {"x": (-0.3, -0.05), "y": (-0.1, 0.1),
                                       "yaw": (-0.1, 0.1)}
  rp["ground_committed_velocity_range"] = {"x": (2.8, 3.5), "y": (-0.1, 0.1)}
  return cfg


def launcher_nw_env_cfg(play: bool = False):
  """LAUNCHER v5b — NARROW WIDTHS (0.1-0.4). v5a at uniform widths 0.1-1.0
  hit 7.3% in its committed cell: over half its forced launches were at gaps
  beyond Go2's demonstrated envelope (proven jumps: 0.3 m, 26/26), so failed
  -jump experience dominated even under forcing. Competence first at feasible
  widths; width generalization is a later curriculum, and the R-CBF claim
  needs a decision contrast at SOME feasible width, not athletics."""
  from dataclasses import replace
  cfg = launcher_sg_env_cfg(play=play)
  gen = cfg.scene.terrain.terrain_generator
  subs = {k: replace(v, gap_width_range=(0.1, 0.4))
          for k, v in gen.sub_terrains.items()}
  cfg.scene.terrain.terrain_generator = replace(gen, sub_terrains=subs)
  cfg.scene.terrain.max_init_terrain_level = None
  return cfg


def goarc_nw_env_cfg(play: bool = False):
  """Ground-arc certification env at the narrow-width envelope (0.1-0.4) —
  the v5b launcher's operating widths."""
  from dataclasses import replace
  cfg = goarc_sg_env_cfg(play=play)
  gen = cfg.scene.terrain.terrain_generator
  subs = {k: replace(v, gap_width_range=(0.1, 0.4))
          for k, v in gen.sub_terrains.items()}
  cfg.scene.terrain.terrain_generator = replace(gen, sub_terrains=subs)
  cfg.scene.terrain.max_init_terrain_level = None
  return cfg


def launcher_gb_env_cfg(play: bool = False):
  """LAUNCHER v6 — GAIT-BANK committed spawns (narrow widths). v5b launched
  61.6% from teleport-committed spawns but 0.4% from real arrivals: the skill
  was anchored to default-pose spawn states. Spawns here: 50% midair
  ballistic (landing) + 50% full sim states harvested mid-stride at 2.3-3.6
  m/s, placed at the lip — committed by position+momentum AND physically
  real. Gate: composed (run-up delivery -> launch) success, not spawn-cell."""
  import math
  from mjlab.managers.event_manager import EventTermCfg
  import safe_mjlab_zoo.envs.parkour.mdp as pmdp
  cfg = launcher_nw_env_cfg(play=play)
  old = cfg.events["reset_base"].params
  cfg.events["reset_base"] = EventTermCfg(
    func=pmdp.reset_robot_gait_bank_over_gaps, mode="reset",
    params={
      "bank_path": "~/SAFE/gait_bank.pt",
      "bank_fraction": 0.5,
      "bank_x_range": (-0.35, -0.05),
      "bank_y_range": (-0.1, 0.1),
      "midair_fraction": 1.0,
      "ground_pose_range": old["ground_pose_range"],
      "ground_velocity_range": old["ground_velocity_range"],
      "midair_x_range": old["midair_x_range"],
      "midair_y_range": old["midair_y_range"],
      "midair_z_range": old["midair_z_range"],
      "midair_vx_range": old["midair_vx_range"],
      "midair_vy_range": old["midair_vy_range"],
      "midair_vz_range": old["midair_vz_range"],
      "midair_roll_range": old["midair_roll_range"],
      "midair_pitch_range": old["midair_pitch_range"],
      "midair_yaw_range": old["midair_yaw_range"],
    })
  return cfg


def certified_single_env_cfg(play: bool = False):
  """v4: certified-launch arms on SINGLE-GAP terrain (user call, 2026-07-10:
  flat ground behind the gap — no island back-pit artifacts in the approach,
  unlimited runway, and the terrain where the jump demonstrably exists).
  Strata: 65% walk-in approach (stoppable), 25% committed launch-arc
  (trainee-EXECUTED launches bridge the ground->air value boundary — the v2
  flat-latch lesson), 10% midair seeds (banked-value anchor)."""
  cfg = single_gap_twin_env_cfg(play=play)
  cfg.episode_length_s = 10.0
  rp = cfg.events["reset_base"].params
  # approach_length is 2.0 m (origin AT the near edge): spawns deeper than
  # x=-1.9 fall off the BACK of the platform and die at spawn (the v4.0
  # ep_len=39 bug: ~19% of episodes were instant deaths teaching a spurious
  # "deep approach = doomed" value). 1.8 m of runway still reaches ~3 m/s
  # from a standing start — momentum buildup stays learnable.
  rp["ground_pose_range"]["x"] = (-1.8, -0.1)
  rp["ground_velocity_range"]["x"] = (0.3, 2.5)
  rp["ground_committed_fraction"] = 0.25
  rp["ground_committed_pose_range"] = {"x": (-0.5, -0.05), "y": (-0.15, 0.15),
                                       "yaw": (-0.1, 0.1)}
  rp["ground_committed_velocity_range"] = {"x": (2.0, 3.2), "y": (-0.1, 0.1)}
  rp["midair_fraction"] = 0.10
  rp["midair_vx_range"] = (1.5, 3.2)
  rp["midair_vz_range"] = (-0.5, 1.5)
  return cfg


_VHAT_CACHE: dict = {}


def _load_vhat(device):
  import os
  import torch
  import torch.nn as nn
  path = os.environ.get("VHAT_PATH", os.path.expanduser("~/SAFE/vhat_cross.pt"))
  key = (path, str(device))
  if key not in _VHAT_CACHE:
    st = torch.load(path, map_location=device, weights_only=True)
    mlp = nn.Sequential(nn.Linear(st["in_dim"], 256), nn.ReLU(),
                        nn.Linear(256, 256), nn.ReLU(),
                        nn.Linear(256, 1)).to(device)
    mlp.load_state_dict(st["state_dict"])
    mlp.eval()
    _VHAT_CACHE[key] = (mlp, st["obs_mean"].to(device).float(),
                        st["obs_var"].to(device).float(), float(st["p_star"]))
    print(f"[l_vhat] certificate loaded from {path} (p*={st['p_star']:.2f})")
  return _VHAT_CACHE[key]


def l_vhat(env):
  """Funnel-mouth reach margin: l = (V_hat(obs) - p*) / 0.2, clamped [-1, 1].

  V_hat = calibrated p(frozen skill crosses AND survives from here), evaluated
  on the FROZEN skill's own normalizer. Uses the bridge-cached actor obs
  (env._zoo_last_obs) because this hook runs INSIDE env.step, before this
  step's observations are computed — a deliberate 1-step (20 ms) lag; the
  certificate is smooth at that scale and the latch in base.py is sticky."""
  import torch
  obs = getattr(env, "_zoo_last_obs", None)
  n = env.scene["robot"].data.root_link_pos_w.shape[0]
  if obs is None:
    return torch.full((n,), -1.0, device=env.device)
  mlp, mean, var, p_star = _load_vhat(env.device)
  with torch.no_grad():
    nobs = torch.clamp((obs - mean) / torch.sqrt(var + 1e-8), -10.0, 10.0)
    p = torch.sigmoid(mlp(nobs).squeeze(-1))
  return ((p - p_star) / 0.2).clamp(-1.0, 1.0)


# --- CERTIFIED-LAUNCH twins (L3): the RA target set is the LANDER's certified
# basin — T = {airborne ∧ momentum ∧ V_hat_land >= p*}. T ⊆ the maximal safety
# set with the lander as viability WITNESS, so reach-once banking is sound by
# construction (reach ⟹ certified-landable ⟹ success; the user's 2026-07-10
# derivation). Hybrid handover to the frozen lander at l>0 is theory-mandated:
# the witness must actually run post-reach.

_VLAND_CACHE: dict = {}


def _load_vhat_land(device):
  import os
  import torch
  import torch.nn as nn
  path = os.environ.get("VHAT_LAND_PATH", os.path.expanduser("~/SAFE/vhat_land.pt"))
  key = (path, str(device))
  if key not in _VLAND_CACHE:
    st = torch.load(path, map_location=device, weights_only=True)
    mlp = nn.Sequential(nn.Linear(st["in_dim"], 256), nn.ReLU(),
                        nn.Linear(256, 256), nn.ReLU(),
                        nn.Linear(256, 1)).to(device)
    mlp.load_state_dict(st["state_dict"])
    mlp.eval()
    _VLAND_CACHE[key] = (mlp, st["feat_mean"].to(device).float(),
                         st["feat_var"].to(device).float(),
                         float(st["p_star"]))
    print(f"[l_certified_launch] witness certificate {path} "
          f"(p*={st['p_star']:.2f})")
  return _VLAND_CACHE[key]


def l_certified_launch(env):
  """Certified-launch reach margin, min-form:

      l = min(airborne, momentum, V_hat_land - p*)

  The explicit airborne/momentum terms are OOD MASKS, not redundancy: the
  certificate is fit on the LANDER's visited states (airborne + post-flight);
  ground approach states are outside its training distribution, and an
  extrapolating MLP there could bank the trainee without leaving the ground.
  min-form confines l>0 to the certificate's validated domain. State-only
  (features read the scene directly) — no obs-cache lag, unlike l_vhat."""
  import torch
  from ..features import state_features
  from ..margins import ground_reference
  mlp, mean, var, p_star = _load_vhat_land(env.device)
  feats = state_features(env)
  with torch.no_grad():
    nf = torch.clamp((feats - mean) / torch.sqrt(var + 1e-8), -10.0, 10.0)
    p = torch.sigmoid(mlp(nf).squeeze(-1))
  cert = ((p - p_star) / 0.2).clamp(-1.0, 1.0)
  robot = env.scene["robot"]
  base_z, ground_ref = ground_reference(env)
  airborne = ((base_z - ground_ref - 0.42) / 0.10).clamp(-1.0, 1.0)
  vx = robot.data.root_link_lin_vel_w[:, 0]
  momentum = ((vx - 1.5) / 0.5).clamp(-1.0, 1.0)
  return torch.minimum(torch.minimum(airborne, momentum), cert)


def certified_twin_env_cfg(play: bool = False):
  """Approach-distribution env for the certified-launch arms: LONG island
  (3.0 m — the valid-gauntlet/deployment geometry; the 1 m training island
  was the 2026-07-10 train/eval mismatch bug: no x<-1 states existed, V was
  OOD at every filter-engagement state, and momentum BUILDUP had no runway
  to be learned on), 85% stoppable ground spawns across the full approach
  corridor + 15% midair seeds (immediate l>0 -> handover experience anchors
  the banked value), widths 0.05-0.95 (feasibility carried by the certificate
  reading the scan). Short-runway/low-vx slices still exist near the edge —
  the "no runway -> stop" contrast lives INSIDE this distribution."""
  from dataclasses import replace
  cfg = island_twin_env_cfg(play=play)
  cfg.episode_length_s = 10.0     # walk-in from x=-2.8 needs the horizon
  rp = cfg.events["reset_base"].params
  rp["ground_pose_range"]["x"] = (-2.8, -0.1)
  # v3 (flat-latch fix): midair seeds latch at t=0, so the trainee never
  # EXECUTES a launch in seeded episodes — the value bridge across the
  # ground->air action is only learned from trainee-executed launches. Add a
  # committed-ground stratum at the launch arc (warm-start launches ~37%
  # there): banked outcomes become attributable to the trainee's own launch
  # action, and reach value chains backward from the arc into the approach.
  rp["ground_committed_fraction"] = 0.25
  rp["ground_committed_pose_range"] = {"x": (-0.5, -0.05), "y": (-0.15, 0.15),
                                       "yaw": (-0.1, 0.1)}
  rp["ground_committed_velocity_range"] = {"x": (2.0, 3.2), "y": (-0.1, 0.1)}
  rp["midair_fraction"] = 0.10
  rp["midair_vx_range"] = (1.5, 3.2)
  rp["midair_vz_range"] = (-0.5, 1.5)
  gen = cfg.scene.terrain.terrain_generator
  subs = {k: replace(v, island_length=3.0, gap_width_range=(0.05, 0.95))
          for k, v in gen.sub_terrains.items()}
  cfg.scene.terrain.terrain_generator = replace(gen, sub_terrains=subs)
  cfg.scene.terrain.max_init_terrain_level = None
  return cfg


_VGO_CACHE: dict = {}


def _load_vhat_go(device):
  import os
  import torch
  import torch.nn as nn
  path = os.environ.get("VHAT_GO_PATH",
                        os.path.expanduser("~/SAFE/vhat_go_sg.pt"))
  key = (path, str(device))
  if key not in _VGO_CACHE:
    st = torch.load(path, map_location=device, weights_only=True)
    mlp = nn.Sequential(nn.Linear(st["in_dim"], 256), nn.ReLU(),
                        nn.Linear(256, 256), nn.ReLU(),
                        nn.Linear(256, 1)).to(device)
    mlp.load_state_dict(st["state_dict"])
    mlp.eval()
    _VGO_CACHE[key] = (mlp, st["feat_mean"].to(device).float(),
                       st["feat_var"].to(device).float(),
                       float(st["p_star"]))
    print(f"[l_run_up] go-arc certificate {path} (p*={st['p_star']:.2f})")
  return _VGO_CACHE[key]


def l_run_up(env):
  """RUN-UP reach margin (v5): the target is the certified-committed GROUND
  arc — T = {x in the arc band, vx >= v*, V_hat_go >= p*} — witnessed by the
  frozen CROSSING skill (handover at latch, pre-launch). The trainee never
  executes a launch, so the launch death-gradient is absent from its
  objective (the poison that stopped v2-v4.1: a low-success launch bet is
  priced negative on-policy and the gradient converges to stop). Reaching T
  is pure locomotion — near-risk-free — so reach value propagates cleanly.
  The x-band and momentum terms keep certificate queries inside its fitted
  distribution (spawn arc x(-0.7,-0.05), vx(1.2,3.5))."""
  import torch
  from ..features import state_features
  mlp, mean, var, p_star = _load_vhat_go(env.device)
  feats = state_features(env)
  with torch.no_grad():
    nf = torch.clamp((feats - mean) / torch.sqrt(var + 1e-8), -10.0, 10.0)
    p = torch.sigmoid(mlp(nf).squeeze(-1))
  cert = ((p - p_star) / 0.2).clamp(-1.0, 1.0)
  robot = env.scene["robot"]
  x_rel = robot.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
  band = torch.minimum((x_rel + 0.6) / 0.2, (-0.05 - x_rel) / 0.2)
  band = band.clamp(-1.0, 1.0)
  vx = robot.data.root_link_lin_vel_w[:, 0]
  momentum = ((vx - 2.0) / 0.5).clamp(-1.0, 1.0)
  return torch.minimum(torch.minimum(band, momentum), cert)


def l_band_momentum(env):
  """DAgger-collection latch: band AND momentum ONLY (certificate omitted).
  Forces handover to the launcher at every trainee arrival state, so outcomes
  get labeled on the DELIVERY distribution — the incoming-distribution
  requirement (guard rule #2) the v1 certificate violated: fit on launcher
  self-rollouts, it rejected 93% of the trainee's own lip states (median
  p=0.001; diagnostic 2026-07-11)."""
  import torch
  robot = env.scene["robot"]
  x_rel = robot.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
  band = torch.minimum((x_rel + 0.6) / 0.2, (-0.05 - x_rel) / 0.2)
  band = band.clamp(-1.0, 1.0)
  vx = robot.data.root_link_lin_vel_w[:, 0]
  momentum = ((vx - 2.0) / 0.5).clamp(-1.0, 1.0)
  return torch.minimum(band, momentum)


def runup_twin_env_cfg(play: bool = False):
  """v5 run-up twin env: single-gap, walk-in 85% + certified-arc ground seeds
  15% (latch within a few steps -> banked-value anchor with the trainee's own
  locomotion in the loop; no midair spawns — the trainee's world is ground)."""
  cfg = single_gap_twin_env_cfg(play=play)
  cfg.episode_length_s = 10.0
  rp = cfg.events["reset_base"].params
  rp["midair_fraction"] = 0.0
  rp["ground_pose_range"]["x"] = (-1.8, -0.1)
  rp["ground_velocity_range"]["x"] = (0.3, 2.5)
  rp["ground_committed_fraction"] = 0.15
  rp["ground_committed_pose_range"] = {"x": (-0.45, -0.1), "y": (-0.1, 0.1),
                                       "yaw": (-0.1, 0.1)}
  rp["ground_committed_velocity_range"] = {"x": (2.5, 3.5), "y": (-0.1, 0.1)}
  return cfg


def l_band_momentum_tight(env):
  """Oracle latch matched to the v6 launcher's TRAINING envelope: x in
  (-0.35,-0.05) at vx >= 2.5 (bank placement + bank speeds). The v1 oracle
  latched at the EARLIEST band state (x=-0.6, vx=2.0) — outside the
  launcher's envelope — so its 0.7% conflated delivery quality with a
  mis-placed handover surface."""
  import torch
  robot = env.scene["robot"]
  x_rel = robot.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
  band = torch.minimum((x_rel + 0.35) / 0.1, (-0.05 - x_rel) / 0.1)
  band = band.clamp(-1.0, 1.0)
  vx = robot.data.root_link_lin_vel_w[:, 0]
  momentum = ((vx - 2.5) / 0.3).clamp(-1.0, 1.0)
  return torch.minimum(band, momentum)


def launcher_ar_env_cfg(play: bool = False):
  """LAUNCHER v7 — ARRIVAL-BANK roll-in spawns. v6 (lip-translated gait bank)
  measured 4.9% on real delivered arrivals vs 61.6% on its own spawns: lip
  placement preserved teleport mismatch (zeroed action history, contact
  transients, thin 4.5k-row manifold). v7 spawns full sim states harvested
  from the RUN-UP TRAINEE ITSELF at x(-0.8,-0.05) vx>=2.3, placed AT their
  harvested x — deeper states roll in through real contact evolution before
  the edge. 40% midair (landing maintenance) + 60% arrival bank."""
  cfg = launcher_gb_env_cfg(play=play)   # gait-bank reset event (bank_path)
  rp = cfg.events["reset_base"].params
  rp["bank_path"] = "~/SAFE/arrival_bank.pt"
  rp["bank_fraction"] = 0.6
  rp["midair_fraction"] = 1.0
  return cfg


def oracle03_env_cfg(play: bool = False):
  """ORACLE GATE env (w=0.3): run-up walk-in distribution, width pinned to the
  single feasible test width, forced handover at band+momentum to the v6
  gait-bank launcher. One collection = composed gate + delivery-distribution
  certificate data + feasibility-atlas states (revised direction 2026-07-11:
  oracle before any further RA training)."""
  from dataclasses import replace
  cfg = runup_twin_env_cfg(play=play)
  gen = cfg.scene.terrain.terrain_generator
  subs = {k: replace(v, gap_width_range=(0.28, 0.32))
          for k, v in gen.sub_terrains.items()}
  cfg.scene.terrain.terrain_generator = replace(gen, sub_terrains=subs)
  cfg.scene.terrain.max_init_terrain_level = None
  return cfg


def v9_env_cfg(play: bool = False):
  """v9: reverse-curriculum RA over approach+takeoff, certified early-exit at
  the airborne funnel (plan of record 2026-07-11). Single-gap, FIXED width
  0.28-0.32 (single-width-first), 6 curriculum levels from real-state banks,
  latched handover to the lander at l_certified_launch > 0."""
  from dataclasses import replace
  from mjlab.managers.curriculum_manager import CurriculumTermCfg
  from mjlab.managers.event_manager import EventTermCfg
  import safe_mjlab_zoo.envs.parkour.mdp as pmdp
  cfg = single_gap_twin_env_cfg(play=play)
  cfg.episode_length_s = 10.0
  gen = cfg.scene.terrain.terrain_generator
  subs = {k: replace(v, gap_width_range=(0.28, 0.32))
          for k, v in gen.sub_terrains.items()}
  cfg.scene.terrain.terrain_generator = replace(gen, sub_terrains=subs)
  cfg.scene.terrain.max_init_terrain_level = None
  cfg.events["reset_base"] = EventTermCfg(
    func=pmdp.reset_v9_reverse, mode="reset",
    params={
      "arrival_bank_path": "~/SAFE/arrival_bank.pt",
      "retention_fraction": 0.15,
      "ground_pose_range": {"x": (-1.8, -0.1), "y": (-0.15, 0.15),
                            "yaw": (-0.15, 0.15)},
      "ground_velocity_range": {"x": (0.3, 2.5), "y": (-0.2, 0.2)},
    })
  cfg.curriculum = {"v9_levels": CurriculumTermCfg(func=pmdp.v9_reverse_levels)}
  from mjlab.managers.termination_manager import TerminationTermCfg
  cfg.terminations["reached_far_side"] = TerminationTermCfg(
    func=pmdp.reached_far_side, time_out=True)
  return cfg


def _term_certified_entry(env):
  """Success truncation: episode ends the moment the certified launch set is
  entered (time_out=True). The 98%-certified witness executes at EVAL; during
  dense bridge training, certified entry IS task success."""
  return l_certified_launch(env) > 0.0


def _rew_certified_entry(env):
  import torch
  return (l_certified_launch(env) > 0.0).float()


def _rew_alive(env):
  import torch
  return torch.ones(env.num_envs, device=env.device)


def l_geometric_core(env):
  """The PROPOSAL set P (v10.3, non-learned): airborne AND vx>=2.0 AND past
  the face AND sane pitch. Latch source and shaping target — the learned
  certificate NEVER appears in the objective (Goodhart lesson: latch->cross
  conversion collapsed 88%->4% when V-hat became a reward surface)."""
  import torch
  from ..margins import ground_reference
  robot = env.scene["robot"]
  base_z, gref = ground_reference(env)
  airborne = ((base_z - gref) - 0.42) / 0.10
  vx = robot.data.root_link_lin_vel_w[:, 0]
  mom = (vx - 2.0) / 0.3
  x_rel = robot.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
  prog = (x_rel + 0.05) / 0.10
  grav_xy = torch.norm(robot.data.projected_gravity_b[:, :2], dim=1)
  pitch_ok = (0.5 - grav_xy) / 0.2
  out = torch.minimum(torch.minimum(airborne.clamp(-1, 1), mom.clamp(-1, 1)),
                      torch.minimum(prog.clamp(-1, 1), pitch_ok.clamp(-1, 1)))
  return out


def _rew_geometric_entry(env):
  import torch
  return (l_geometric_core(env) > 0.0).float()


def _rew_realized_success(env):
  """Physical composed outcome: far side, grounded, upright — the ONLY large
  bonus (unhackable; requires the latched witness to have actually landed)."""
  import torch
  from ..margins import ground_reference
  robot = env.scene["robot"]
  x_rel = robot.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
  up = -robot.data.projected_gravity_b[:, 2]
  base_z, gref = ground_reference(env)
  grounded = (base_z - gref) < 0.42
  return ((x_rel > 0.9) & (up > 0.7) & grounded).float()


def v103_bridge_env_cfg(play: bool = False):
  """v10.3 (colleague spec): geometric proposal -> witness rollout ->
  realized-success reward. Certificate absent from objective and latch."""
  from mjlab.managers.reward_manager import RewardTermCfg
  cfg = v10_bridge_env_cfg(play=play)
  cfg.terminations.pop("certified_entry", None)
  cfg.rewards.pop("certified_entry_bonus", None)
  cfg.rewards["geometric_entry"] = RewardTermCfg(
    func=_rew_geometric_entry, weight=250.0)     # 5 effective (dt-scaled)
  cfg.rewards["realized_success"] = RewardTermCfg(
    func=_rew_realized_success, weight=1500.0)   # 30 effective
  cfg.events["reset_base"].params["delivery_floor_fraction"] = 0.2
  return cfg


def v10_bridge_env_cfg(play: bool = False):
  """v10 NOMINAL BRIDGE env (colleague directive 2026-07-12): dense-reward
  slow-start conversion. Walker command sweep verdict: the walker cannot
  exceed ~1.5 m/s at any command (ceiling below the 2.3 corridor floor), so
  the bridge must LEARN acceleration+gait-shaping from true delivery states
  (delivery_bank.pt: 128k walker states, vx mean 0.83). Dense stack drives
  the trajectory (command raised to 2.8 so velocity-tracking rewards
  acceleration; certified-entry bonus + truncation); the certified target
  and landing witness still define success."""
  from mjlab.managers.curriculum_manager import CurriculumTermCfg
  from mjlab.managers.event_manager import EventTermCfg
  from mjlab.managers.reward_manager import RewardTermCfg
  from mjlab.managers.termination_manager import TerminationTermCfg
  import safe_mjlab_zoo.envs.parkour.mdp as pmdp
  cfg = v9_env_cfg(play=play)
  cfg.events["reset_base"] = EventTermCfg(
    func=pmdp.reset_v10_bridge, mode="reset",
    params={
      "arrival_bank_path": "~/SAFE/arrival_bank.pt",
      "delivery_bank_path": "~/SAFE/delivery_bank.pt",
      "retention_fraction": 0.15,
      "ground_pose_range": {"x": (-1.8, -0.1), "y": (-0.15, 0.15),
                            "yaw": (-0.15, 0.15)},
      "ground_velocity_range": {"x": (0.3, 1.2), "y": (-0.2, 0.2)},
    })
  cfg.curriculum = {"v10_levels": CurriculumTermCfg(func=pmdp.v10_bridge_levels)}
  cfg.terminations["certified_entry"] = TerminationTermCfg(
    func=_term_certified_entry, time_out=True)
  twist = cfg.commands["twist"]
  twist.ranges.lin_vel_x = (2.8, 2.8)
  # mjlab reward terms are dt-scaled (x0.02): weight 1500 => +30 effective
  # at entry. v10.0 bug: weight 30 paid 0.6 while penalty terms at the
  # untrackable cmd made living net-negative -> PPO learned suicide-by-
  # truncation (ep_len 6.6, levels frozen at 0.75). Alive bonus keeps living
  # net-positive but strictly below the entry bonus (0.05/step * 10 s = 25
  # < 30) so dawdling never beats entering.
  cfg.rewards["certified_entry_bonus"] = RewardTermCfg(
    func=_rew_certified_entry, weight=1500.0)
  # alive at 0.02/step effective (10s standing = 10 << entry 30): living is
  # not punished, idling never competes with entering (v10.1: 25 vs 30 was
  # nearly indifferent and PPO idled).
  cfg.rewards["alive"] = RewardTermCfg(func=_rew_alive, weight=1.0)
  return cfg


def _term_cleared_first_gap(env):
  """Single-patch isolation (colleague Q3): truncate once the first gap is
  decisively cleared (x > 1.5) so the next tiling patch cannot influence
  branch selection or outcomes. time_out=True: clearing is success."""
  return (env.scene["robot"].data.root_link_pos_w[:, 0]
          - env.scene.env_origins[:, 0]) > 1.5


def v103_singlepatch_env_cfg(play: bool = False):
  """v10.3 env with first-gap isolation — the HEADLINE eval geometry."""
  from mjlab.managers.termination_manager import TerminationTermCfg
  cfg = v103_bridge_env_cfg(play=play)
  cfg.terminations["cleared_first_gap"] = TerminationTermCfg(
    func=_term_cleared_first_gap, time_out=True)
  return cfg


def g_gap_rich(env):
  """Enriched SHARED safety margin for actor synthesis (colleague review 2):
  the base terrain-relative g PLUS an impact term — g must grade the safety
  property RA is supposed to improve (soft, controlled touchdowns), while
  staying coarse enough not to prescribe the maneuver morphology. Calibration:
  the threshold must sit ABOVE the spawn-physics floor: L0 midair spawns
  drop from up to 0.45 m and touch down at ~3.0 m/s, so VZ_SAFE=2.2 condemned
  the curriculum's own anchor rung (syn_ra v1: levels pinned 0.4, ep_len 15,
  value collapse at unavoidably-"doomed" landings). VZ_SAFE=3.5 clears every
  physically unavoidable touchdown; genuinely hard slams (>3.5) remain
  condemned, and the graded margin still separates soft (~1.7) from firm
  (~3.0) landings for the min_t g comparison. IDENTICAL for both arms."""
  import torch
  from ..margins import ground_reference
  base = g_terrain_relative(env)
  robot = env.scene["robot"]
  base_z, gref = ground_reference(env)
  near_ground = ((base_z - gref) < 0.30).float()
  vz_down = (-robot.data.root_link_lin_vel_w[:, 2]).clamp(min=0.0)
  g_impact = (3.5 - vz_down * near_ground) / 0.5
  return torch.minimum(base, g_impact.clamp(-3.0, 3.0))


def l_far_stable(env):
  """Reach margin = STABLE FAR-SIDE COMPLETION (colleague-approved target;
  never the airborne detector): past the gap, grounded, upright. Coincides
  with the reached_far_side success truncation, so banking and episode
  success are the same physical event."""
  import torch
  from ..margins import ground_reference
  robot = env.scene["robot"]
  x_rel = robot.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
  prog = (x_rel - 0.9) / 0.15
  up = (-robot.data.projected_gravity_b[:, 2] - 0.7) / 0.1
  base_z, gref = ground_reference(env)
  grounded = (0.42 - (base_z - gref)) / 0.10
  out = torch.minimum(torch.minimum(prog.clamp(-1, 1), up.clamp(-1, 1)),
                      grounded.clamp(-1, 1))
  # MAGNITUDE CALIBRATION (syn_ra v3 lesson): the l/g magnitude ratio is the
  # RA backup's implicit risk-tolerance dial — break-even attempt probability
  # = |g_death| / (|g_death| + l_bank). With l<=1 vs g>=-3 the break-even was
  # 75%: ABOVE every rung's success rate, so "never attempt" was RA-optimal
  # and the policy learned to stop crossing. Scaling l to the same +-3 range
  # as g sets break-even at 50%.
  return out * 3.0


def syn_env_cfg(play: bool = False):
  """Actor-synthesis env: the v10.3 ladder/terminations WITHOUT dense
  bonuses and WITHOUT any witness latch — the synthesized policies own the
  whole maneuver; safety training sees margins only."""
  cfg = v103_bridge_env_cfg(play=play)
  for k in ("geometric_entry", "realized_success", "alive"):
    cfg.rewards.pop(k, None)
  return cfg


_VSAFE_CACHE: dict = {}


def _load_vsafe(device):
  import os
  import torch
  import torch.nn as nn
  path = os.environ.get("VSAFE_PATH", os.path.expanduser("~/SAFE/vsafe_ens.pt"))
  key = (path, str(device))
  if key not in _VSAFE_CACHE:
    st = torch.load(path, map_location=device, weights_only=True)
    mems = []
    for sd in st["state_dicts"]:
      m = nn.Sequential(nn.Linear(223, 256), nn.ReLU(), nn.Linear(256, 256),
                        nn.ReLU(), nn.Linear(256, 1)).to(device)
      m.load_state_dict(sd); m.eval(); mems.append(m)
    _VSAFE_CACHE[key] = (mems, st["feat_mean"].to(device).float(),
                         st["feat_var"].to(device).float())
    print(f"[l_homotopy] V_safe ensemble ({len(mems)} members) from {path}")
  return _VSAFE_CACHE[key]


def l_homotopy(env):
  """Target-homotopy reach margin (dilation form):
     l_hat_r = min( l_prog + r, upright, grounded, (V_safe_LCB - c) )
  r comes from the gated anneal (homotopy_r); this fn also RECORDS target
  entry per env for the anneal's gate. Calibration 2026-07-14: standing/slow
  states are excluded from every target by the V_safe clip (tube hugs the
  fast corridor); the grounded term makes the r=20->16 transition the
  topology jump across the gap that forces the maneuver."""
  import torch
  from ..features import state_features
  from ..margins import ground_reference
  from ..envs.parkour.mdp import safety_events as sev
  robot = env.scene["robot"]
  x_rel = robot.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
  r = sev.homotopy_r(env)
  l_prog = (20.0 * (x_rel - 0.9) + r).clamp(-3.0, 3.0)
  up = (((-robot.data.projected_gravity_b[:, 2]) - 0.7) / 0.1 * 3.0).clamp(-3, 3)
  base_z, gref = ground_reference(env)
  grounded = ((0.42 - (base_z - gref)) / 0.10 * 3.0).clamp(-3, 3)
  mems, fmean, fvar = _load_vsafe(env.device)
  f = state_features(env)
  nf = torch.clamp((f - fmean) / torch.sqrt(fvar + 1e-8), -10, 10)
  with torch.no_grad():
    ps = torch.stack([torch.sigmoid(m(nf).squeeze(-1)) for m in mems])
  lcb = ps.mean(0) - 1.5 * ps.std(0)
  vterm = ((lcb - 0.1) / 0.2 * 3.0).clamp(-3.0, 3.0)
  out = torch.minimum(torch.minimum(l_prog, up),
                      torch.minimum(grounded, vterm))
  # record entry for the anneal gate
  sev._ensure_homotopy(env)
  env._homo_entered |= out >= 0.0
  return out


def homotopy_env_cfg(play: bool = False):
  """Homotopy run env: fixed spawn mixture (the xi-anneal is the only
  curriculum), single 0.3 m gap, far-side truncation retained."""
  from mjlab.managers.curriculum_manager import CurriculumTermCfg
  from mjlab.managers.event_manager import EventTermCfg
  import safe_mjlab_zoo.envs.parkour.mdp as pmdp
  from safe_mjlab_zoo.envs.parkour.mdp import safety_events as sev
  cfg = syn_env_cfg(play=play)
  cfg.events["reset_base"] = EventTermCfg(
    func=sev.reset_homotopy_mix, mode="reset",
    params={
      "arrival_bank_path": "~/SAFE/arrival_bank.pt",
      "delivery_bank_path": "~/SAFE/delivery_bank.pt",
      "ground_pose_range": {"x": (-1.8, -0.1), "y": (-0.15, 0.15),
                            "yaw": (-0.15, 0.15)},
      "ground_velocity_range": {"x": (0.3, 1.2), "y": (-0.2, 0.2)},
    })
  cfg.curriculum = {"homotopy_stage": CurriculumTermCfg(func=sev.homotopy_anneal)}
  return cfg


def register_all() -> None:
  cfgs = _cfgs()
  g = g_terrain_relative
  register(TaskSpec(
    task_id="go2_gap_landing", cfg_builder=cfgs["landing"],
    margin_fn=compose(g, l_zero), default_algo="SafetyPPO",
    description="Mid-air over a gap with clearing velocity; learn soft landing."))
  register(TaskSpec(
    task_id="go2_gap_crossing", cfg_builder=cfgs["crossing"],
    margin_fn=compose(g, l_zero), default_algo="SafetyPPO",
    warmstart_from="go2_gap_landing",
    description="Reverse curriculum from the landed state back to the launch."))
  register(TaskSpec(
    task_id="go2_gap_chain", cfg_builder=cfgs["chain"],
    margin_fn=compose(g, l_rest), default_algo="ReachAvoidPPO",
    warmstart_from="go2_gap_crossing", supports_adversary=False,
    description="Takeover momentum -> reach safe rest (brake / jump when needed)."))
  register(TaskSpec(
    task_id="go2_gap_chain_isaacs", cfg_builder=cfgs["chain_isaacs"],
    margin_fn=compose(g, l_rest), default_algo="IsaacsPPO",
    warmstart_from="go2_gap_chain", supports_adversary=True,
    description="Chain + worst-case base-force adversary (pinned curricula)."))
  # --- R-CBF claim twins: SAME env + spawns + budget; ONLY the backup differs
  # (min(g,V') vs min(g,max(l,V'))). Certified sets: stoppable vs completable.
  register(TaskSpec(
    task_id="go2_gap_chain_avoid", cfg_builder=cfgs["chain"],
    margin_fn=compose(g, l_zero), default_algo="SafetyPPO",
    description="AVOID-ONLY twin on the chain takeover distribution. Filter "
                "prediction: vetoes at the braking boundary, fallback stops -> "
                "never crosses (livelock at gap 1)."))
  register(TaskSpec(
    task_id="go2_gap_chain_ra", cfg_builder=cfgs["chain"],
    margin_fn=compose(g, l_gap_completion), default_algo="ReachAvoidPPO",
    description="REACH-AVOID twin, min-form COMPLETION l (at rest PAST the "
                "cluster; rest-before never satisfies l, unlike l_rest). Filter "
                "prediction: fallback carries the crossing when feasible, "
                "degrades to stop when not. KNOWN RESULT: risk-aversion trap — "
                "converges to standing (0% cross); kept for the ablation table."))
  # --- ISLAND twins (the crossing ckpt's HOME terrain; probe-verified before
  # training). Three l designs on identical env/budget = the l-ablation.
  register(TaskSpec(
    task_id="go2_gap_island_avoid", cfg_builder=island_twin_env_cfg,
    margin_fn=compose(g, l_zero), default_algo="SafetyPPO",
    warmstart_from="go2_gap_crossing",
    description="Island AVOID twin (l=0): certificate-coverage fine-tune."))
  register(TaskSpec(
    task_id="go2_gap_island_launch", cfg_builder=island_twin_env_cfg,
    margin_fn=compose(g, l_launch_island), default_algo="ReachAvoidPPO",
    warmstart_from="go2_gap_crossing",
    description="Island LAUNCH-BASIN RA twin (v_launch 1.5, trap-free but "
                "feasibility-blind)."))
  register(TaskSpec(
    task_id="go2_gap_island_gated", cfg_builder=island_twin_env_cfg,
    margin_fn=compose(g, l_launch_gated), default_algo="ReachAvoidPPO",
    warmstart_from="go2_gap_crossing",
    description="Island GATED-LAUNCH RA twin: basin AND feasible-width "
                "(min-form) — trap-free AND feasibility-aware. The l design "
                "the theory needs."))
  # --- SINGLE-GAP twins: the terrain where the jump exists (96.8% crossing
  # ckpt). Warm-start both from go2_gap_crossing; only the backup differs.
  register(TaskSpec(
    task_id="go2_gap_single_avoid", cfg_builder=single_gap_twin_env_cfg,
    margin_fn=compose(g, l_zero), default_algo="SafetyPPO",
    warmstart_from="go2_gap_crossing",
    description="Single-gap AVOID twin: certificate-coverage fine-tune of the "
                "crossing ckpt on broad spawns (approach + edge + midair)."))
  register(TaskSpec(
    task_id="go2_gap_single_launch", cfg_builder=single_gap_twin_env_cfg,
    margin_fn=compose(g, l_launch_single), default_algo="ReachAvoidPPO",
    warmstart_from="go2_gap_crossing",
    description="Single-gap LAUNCH-BASIN RA twin (gap face at x=0, "
                "v_launch=2.0): trap-free commitment l on the terrain the "
                "jump transfers to."))
  register(TaskSpec(
    task_id="go2_gap_funnel", cfg_builder=funnel_twin_env_cfg,
    margin_fn=compose(g_terrain_relative, l_vhat),
    default_algo="ReachAvoidPPO", warmstart_from="go2_gap_crossing",
    kwargs=dict(hybrid_skill="runs_zoo/go2_gap_crossing"),
    description="FUNNEL-RA twin: l = calibrated MC-outcome certificate of the "
                "FROZEN crossing skill (V_hat - p*), hybrid rollouts hand "
                "over to the frozen skill once l>0 latches. Trap-free (reach "
                "banks at the certified funnel mouth), feasibility-aware "
                "(V_hat is), zero skill erosion (frozen)."))
  register(TaskSpec(
    task_id="go2_gap_lander", cfg_builder=lander_env_cfg,
    margin_fn=compose(g, l_zero), default_algo="SafetyPPO",
    warmstart_from="go2_gap_crossing",
    description="DEDICATED LANDER (viability witness): avoid-only fine-tune "
                "on 100% airborne ballistic spawns (ascent + descent, wide "
                "posture/width). Its certified basin V_hat_land >= p* IS the "
                "RA target set of the certified-launch twin (L2/L3)."))
  # --- L3 certified-launch arms: IDENTICAL env, margin channel, and hybrid
  # handover; the manipulated variable is whether l enters the algo's backup.
  register(TaskSpec(
    task_id="go2_gap_launcher_sg", cfg_builder=launcher_sg_env_cfg,
    margin_fn=compose(g, l_zero), default_algo="SafetyPPO",
    warmstart_from="go2_gap_crossing",
    description="LAUNCHER (v5a): avoid-only forcing on single-gap — midair "
                "(landing) + truly-committed ground (launch). The witness "
                "the run-up target needs; crossing measured at 0.8% here."))
  register(TaskSpec(
    task_id="go2_gap_launcher_nw", cfg_builder=launcher_nw_env_cfg,
    margin_fn=compose(g, l_zero), default_algo="SafetyPPO",
    warmstart_from="go2_gap_crossing",
    description="LAUNCHER v5b at feasible widths (0.1-0.4): the width-ceiling "
                "hypothesis test. Gate: committed-cell success >= 40%."))
  register(TaskSpec(
    task_id="go2_gap_goarc_nw", cfg_builder=goarc_nw_env_cfg,
    margin_fn=compose(g, l_zero), default_algo="SafetyPPO",
    description="Ground-arc certification env, narrow widths (v5b gate)."))
  register(TaskSpec(
    task_id="go2_gap_oracle03b", cfg_builder=oracle03_env_cfg,
    margin_fn=compose(g, l_band_momentum_tight), default_algo="SafetyPPO",
    kwargs=dict(hybrid_skill="runs_zoo/go2_gap_launcher_gb"),
    description="Oracle gate v2 (w=0.3): latch surface matched to the v6 "
                "launcher envelope (x -0.35..-0.05, vx>=2.5)."))
  register(TaskSpec(
    task_id="go2_gap_launcher_ar", cfg_builder=launcher_ar_env_cfg,
    margin_fn=compose(g, l_zero), default_algo="SafetyPPO",
    warmstart_from="go2_gap_launcher_nw",
    description="LAUNCHER v7: roll-in arrival-bank spawns (harvested from "
                "the run-up trainee, spawned at harvested x). Gate: composed "
                "oracle at w=0.3, conditional-on-delivered >= 40%."))
  register(TaskSpec(
    task_id="go2_gap_oracle03c", cfg_builder=oracle03_env_cfg,
    margin_fn=compose(g, l_band_momentum_tight), default_algo="SafetyPPO",
    kwargs=dict(hybrid_skill="runs_zoo/go2_gap_launcher_ar"),
    description="Oracle gate v3 (w=0.3): envelope-matched latch, witness = "
                "v7 arrival-bank launcher. Gate: conditional-on-delivered "
                ">= 40% (v6 measured 4.9%)."))
  register(TaskSpec(
    task_id="go2_gap_oracle03", cfg_builder=oracle03_env_cfg,
    margin_fn=compose(g, l_band_momentum), default_algo="SafetyPPO",
    kwargs=dict(hybrid_skill="runs_zoo/go2_gap_launcher_gb"),
    description="Oracle gate at w=0.3: forced handover to v6 launcher from "
                "real run-up arrivals. Composed success >= ~50% => funnel "
                "exists; dataset doubles as cert-v3 data + atlas."))
  register(TaskSpec(
    task_id="go2_gap_dagger_sg", cfg_builder=runup_twin_env_cfg,
    margin_fn=compose(g, l_band_momentum), default_algo="SafetyPPO",
    kwargs=dict(hybrid_skill="runs_zoo/go2_gap_launcher_nw"),
    description="DAgger collection env: trainee rolls, FORCED handover at "
                "band+momentum (no cert), launcher outcome labels the "
                "delivery distribution -> V_hat_go v2."))
  register(TaskSpec(
    task_id="go2_gap_launcher_gb", cfg_builder=launcher_gb_env_cfg,
    margin_fn=compose(g, l_zero), default_algo="SafetyPPO",
    warmstart_from="go2_gap_crossing",
    description="LAUNCHER v6: gait-bank committed spawns + midair, narrow "
                "widths. The launch-from-gait skill."))
  register(TaskSpec(
    task_id="go2_gap_runup_ra", cfg_builder=runup_twin_env_cfg,
    margin_fn=compose(g, l_run_up), default_algo="ReachAvoidPPO",
    warmstart_from="go2_gap_crossing",
    kwargs=dict(hybrid_skill="runs_zoo/go2_gap_launcher_nw"),
    description="RUN-UP RA arm (v5): reach the certified-committed ground "
                "arc; frozen CROSSING skill takes over at latch (pre-launch). "
                "The trainee is pure locomotion — no launch bet in its "
                "objective."))
  register(TaskSpec(
    task_id="go2_gap_runup_avoid", cfg_builder=runup_twin_env_cfg,
    margin_fn=compose(g, l_run_up), default_algo="SafetyPPO",
    warmstart_from="go2_gap_crossing",
    kwargs=dict(hybrid_skill="runs_zoo/go2_gap_launcher_nw"),
    description="RUN-UP AVOID arm (control, trained only if RA passes): "
                "same env, same latch channel; backup ignores l."))
  register(TaskSpec(
    task_id="go2_gap_goarc_sg", cfg_builder=goarc_sg_env_cfg,
    margin_fn=compose(g, l_zero), default_algo="SafetyPPO",
    description="GO-ARC certification env (no training): crossing skill from "
                "committed ground states -> V_hat_go, the run-up target."))
  register(TaskSpec(
    task_id="go2_gap_lander_sg", cfg_builder=lander_sg_env_cfg,
    margin_fn=compose(g, l_zero), default_algo="SafetyPPO",
    description="Lander CERTIFICATION env on single-gap terrain (no training "
                "— rollout target for collect_vland_data with the frozen "
                "island-trained lander)."))
  register(TaskSpec(
    task_id="go2_gap_v9_ra", cfg_builder=v9_env_cfg,
    margin_fn=compose(g, l_certified_launch), default_algo="ReachAvoidPPO",
    warmstart_from="go2_gap_launcher_nw",
    kwargs=dict(hybrid_skill="runs_zoo/go2_gap_lander"),
    description="v9: reverse-curriculum RA (approach+takeoff in one policy), "
                "reach = certified airborne funnel (vhat_land_sg), latched "
                "lander handover. Levels 0-2 takeoff bank / 3-4 arrival bank "
                "/ 5 walk-in; promote on composed success."))
  register(TaskSpec(
    task_id="go2_gap_stopper2", cfg_builder=v10_bridge_env_cfg,
    margin_fn=compose(g, l_zero), default_algo="SafetyPPO",
    warmstart_from="go2_gap_single_avoid",
    description="Second stopper (colleague Q4 sensitivity): avoid-only "
                "brake-to-stand on the v10 ladder spawns (delivery floor + "
                "walk-in) — commitment-region robustness across the "
                "stopping-controller class."))
  register(TaskSpec(
    task_id="go2_gap_homotopy_ra", cfg_builder=homotopy_env_cfg,
    margin_fn=compose(g_gap_rich, l_homotopy), default_algo="ReachAvoidPPO",
    warmstart_from="go2_gap_v103_bridge",
    description="TARGET-HOMOTOPY RA (professor's xi-anneal, dilation form): "
                "always-on policy, r annealed 34->0 by gated entry rate, "
                "targets clipped to the V_safe ensemble tube. Init = bridge, "
                "no std reset, RA-stable optimizer flags."))
  register(TaskSpec(
    task_id="go2_gap_syn_avoid", cfg_builder=syn_env_cfg,
    margin_fn=compose(g_gap_rich, l_zero), default_algo="SafetyPPO",
    warmstart_from="go2_gap_v103_bridge",
    description="ACTOR SYNTHESIS avoid arm (MAIN RESULT, unregularized): "
                "SafetyPPO from the jump-capable bridge, enriched shared g, "
                "no dense reward, no KL anchor, no witness. Outcome question: "
                "what behavior survives avoid-only optimization?"))
  register(TaskSpec(
    task_id="go2_gap_syn_ra", cfg_builder=syn_env_cfg,
    margin_fn=compose(g_gap_rich, l_far_stable), default_algo="ReachAvoidPPO",
    warmstart_from="go2_gap_v103_bridge",
    description="ACTOR SYNTHESIS reach-avoid arm (MAIN RESULT): identical to "
                "syn_avoid in every respect except the reach term "
                "(l = stable far-side completion). Outcome question: does RA "
                "retain the maneuver and reshape it toward safety (impact "
                "margins, robustness)?"))
  register(TaskSpec(
    task_id="go2_gap_v103_sp", cfg_builder=v103_singlepatch_env_cfg,
    kind="nominal", default_algo="PPO",
    kwargs=dict(hybrid_skill="runs_zoo/go2_gap_lander",
                latch_margin_fn=l_geometric_core),
    description="Single-patch eval variant of v10.3 (truncate at x>1.5): "
                "isolates first-gap attribution for the headline table."))
  register(TaskSpec(
    task_id="go2_gap_v103_bridge", cfg_builder=v103_bridge_env_cfg,
    kind="nominal", default_algo="PPO",
    kwargs=dict(hybrid_skill="runs_zoo/go2_gap_lander",
                latch_margin_fn=l_geometric_core),
    description="v10.3: geometric-proposal latch -> in-rollout landing "
                "witness -> realized-success reward. Pre-registered stop "
                "rule: >=10% held-out composed on the delivery benchmark, "
                "else close the 2m/current-walker config as the measured "
                "ceiling."))
  register(TaskSpec(
    task_id="go2_gap_v10_bridge", cfg_builder=v10_bridge_env_cfg,
    kind="nominal", default_algo="PPO",
    description="v10 NOMINAL BRIDGE: dense slow-start conversion from true "
                "delivery states to the certified launch set (train with "
                "train_nominal.py; gate: >=60% composed on the frozen "
                "delivery benchmark before any RA fine-tune)."))
  register(TaskSpec(
    task_id="go2_gap_v9_probe", cfg_builder=oracle03_env_cfg,
    margin_fn=compose(g, l_certified_launch), default_algo="ReachAvoidPPO",
    kwargs=dict(hybrid_skill="runs_zoo/go2_gap_lander"),
    description="v9 BINDING GATE env (no training): walk-in spawns at w=0.3, "
                "v9 deployment semantics (certified-launch latch -> lander). "
                "The v9 env itself resets at curriculum L0 in a fresh "
                "process — probing it measures the easy rung, not walk-ins."))
  register(TaskSpec(
    task_id="go2_gap_v9w_ra", cfg_builder=v9_env_cfg,
    margin_fn=compose(g, l_certified_launch), default_algo="ReachAvoidPPO",
    warmstart_from="go2_gap_runup_ra",
    kwargs=dict(hybrid_skill="runs_zoo/go2_gap_lander"),
    description="v9 WALKING-PRIOR arm (parallel): identical env/curriculum/"
                "margin/handover to go2_gap_v9_ra; warm-start = runup_ra "
                "(the walk-in-adapted policy, no takeoff bias) to shape "
                "acceleration from a locomotion prior. Comparison axis: "
                "which prior lets the curriculum retreat further."))
  register(TaskSpec(
    task_id="go2_gap_cert_ra", cfg_builder=certified_single_env_cfg,
    margin_fn=compose(g, l_certified_launch), default_algo="ReachAvoidPPO",
    warmstart_from="go2_gap_crossing",
    kwargs=dict(hybrid_skill="runs_zoo/go2_gap_lander"),
    description="CERTIFIED-LAUNCH RA arm: banks at the lander's certified "
                "basin; frozen lander flies post-reach. Prediction: learns "
                "approach->launch on feasible widths, degrades to stop on "
                "infeasible (certificate reads the scan)."))
  register(TaskSpec(
    task_id="go2_gap_cert_avoid", cfg_builder=certified_single_env_cfg,
    margin_fn=compose(g, l_certified_launch), default_algo="SafetyPPO",
    warmstart_from="go2_gap_crossing",
    kwargs=dict(hybrid_skill="runs_zoo/go2_gap_lander"),
    description="CERTIFIED-LAUNCH AVOID arm (control): same env, same l "
                "channel at the bridge (same latch/handover), but SafetyPPO's "
                "backup ignores l. Prediction: never launches — stopping is "
                "strictly optimal under residual maneuver risk."))
  register(TaskSpec(
    task_id="go2_gap_chain_launch", cfg_builder=cfgs["chain"],
    margin_fn=compose(g, l_launch_basin), default_algo="ReachAvoidPPO",
    warmstart_from="go2_gap_crossing",
    description="REACH-AVOID twin, LAUNCH-BASIN l (close-to-gap AND launch "
                "momentum, min-form). Reach banks value AT commitment, so "
                "accelerating to the lip is risk-free under the backup and the "
                "risk-aversion trap cannot open; flight/landing keep training "
                "via g (l<0 post-reach). The trap-free RA twin."))
