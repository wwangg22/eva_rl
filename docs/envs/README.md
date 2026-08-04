# reBot environment reference

One document per environment. Each records what the env *is*, the exact scene geometry,
the MDP (observations, actions, rewards, terminations, events), the measured hardware
constraints that forced the design, how to run it, and its validation evidence.

| doc | gym IDs | skill under test | validation |
|---|---|---|---|
| [precision-slot.md](precision-slot.md) | `Rebot-PrecisionSlot-{Loose-,,Tight-}v0`, `-Play-v0` | tight-tolerance horizontal insertion | smoke test passes (4 negative controls); **insertion probe needs re-running** after the TCP correction |
| [pregrasp.md](pregrasp.md) | `Rebot-PreGrasp-v0`, `-Play-v0` | non-prehensile reconfiguration before grasping | smoke test passes (3 negative controls); **⚠ premise disproven — needs redesign** |
| [clutter-extract.md](clutter-extract.md) | `Rebot-ClutterExtract-v0`, `-Play-v0`, `-Tight-v0`, `-Lenient-v0` | constrained retrieval without disturbing neighbours | smoke test passes (4 negative controls); both constraints proven reachable; **scripted expert 16.4 %** under the 2 mm rule |
| [drawer-order.md](drawer-order.md) | `Rebot-DrawerOrder-v0`, `-Play-v0` | articulated joint + irreversible ordering | smoke test passes (4 negative controls); joint travel under a scripted pull still unverified |

Design rationale and the measured hardware envelope every task respects:
[../CHALLENGE_SUITE.md](../CHALLENGE_SUITE.md).

## How the four differ

They are chosen so that no two can be solved by the same primitive, and so that none of
them is solvable by the existing pick-place policy's "reach, close, carry, open" loop:

| env | what breaks the basket primitive |
|---|---|
| precision-slot | tolerance: ±1.5 mm and ±6.9° instead of ±50 mm and any orientation |
| pre-grasp | the start state has **no** valid grasp; reach-and-close is flat until the object is reoriented |
| clutter-extract | the greedy first action is often wrong; the payoff of pushing a neighbour aside is deferred |
| drawer-order | the carry reward is exactly zero until a *different* subtask is finished |

## Conventions used throughout

- **Frames.** Env-local unless stated: robot base and table top both at the origin, +z up,
  +x away from the robot.
- **Quaternions are (x, y, z, w).** Identity is `(0, 0, 0, 1)`. An Isaac Lab 2.x-style
  `(1, 0, 0, 0)` silently flips 180° about x.
- **Data accessors.** Isaac Lab 3.0 returns `ProxyArray`; use `.torch`.
- **TCP.** The tool centre point is `mdp.TCP_OFFSET = (-0.0419, 0, 0)` from `gripper_end`,
  measured. The `-0.075` inherited from the lift task is 33 mm too far forward and makes any
  scripted grasp close on air (`CHALLENGE_SUITE.md` C10).
- **The gripper opens 89 mm on command and ~120 mm if forced**, not the 45 mm that
  `_GRIPPER_OPEN` suggests — that is a per-finger joint value and both fingers move (C3).
- **The TCP cannot go below ~44 mm above the table** (C9). Any fixture feature the gripper
  must touch has to sit above that; this is what put the drawer cabinet on a plinth.
- **A post-build `xformOp:scale` does not reach the collider** (C8). Vary object size by
  rebuilding the env, or by rotating the object to present a different face.
- **Observation terms must return (N, 1)**, not (N,), or the ObservationManager fails to
  concatenate the group.
- **Finger drive override.** Every challenge env splits the robot's single `.*` actuator
  group into `arm` (gains left `None`, so the USD's validated per-degree-converted revolute
  gains survive) and `fingers` (stiffness 2000 / damping 40). The USD authors the fingers at
  100 N/m, which caps squeeze at ~1.7 N — see `CHALLENGE_SUITE.md` C2.
- **Videos** live in `logs/videos/`, from `scripts/challenge/record_env_video.py`.
