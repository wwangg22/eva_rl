#!/usr/bin/env python
"""Judge probe_vdr_invariance.py's two trajectory dumps. No Isaac needed.

PASS requires all of:
  1. dynamics invariance — the 41-D obs trajectories match across tasks (bit-exact is the
     expectation; anything past 1e-5 fails: a leaked collider shows up as centimetres);
  2. DR determinism — re-seeding torch reproduces the DR task's camera draws exactly;
  3. DR variety — across envs and across resets the DR task's draws actually differ.
"""

import sys

import torch


def main():
    a, b = torch.load(sys.argv[1]), torch.load(sys.argv[2])
    ta, tb = a["traj"], b["traj"]
    assert ta.shape == tb.shape, (ta.shape, tb.shape)
    d = (ta - tb).abs()
    per_dim = d.amax(dim=(0, 1))
    print(f"[1] dynamics: max |Δobs| = {d.max():.3e}  "
          f"(worst dim {int(per_dim.argmax())} = {per_dim.max():.3e}, "
          f"first divergent step "
          f"{int(d.amax(dim=(1, 2)).gt(1e-5).float().argmax()) if d.max() > 1e-5 else -1})")
    ok1 = d.max() <= 1e-5

    dr = b if "VisionDR" in b["task"] else a
    c1, c2, cr = dr["cam_reset1"], dr["cam_reset2"], dr["cam_reseeded"]
    det = (c1 - cr).abs().max()
    print(f"[2] determinism: reseeded reset reproduces draws to {det:.3e}")
    ok2 = det <= 1e-6

    across_env = c1[:, :3].std(dim=0).max()
    across_reset = (c1 - c2).abs().max()
    print(f"[3] variety: across-env pos std {across_env:.4f} m, "
          f"reset1 vs reset2 max Δ {across_reset:.4f}")
    ok3 = across_env > 1e-3 and across_reset > 1e-3

    print("PASS" if (ok1 and ok2 and ok3) else "FAIL")
    sys.exit(0 if (ok1 and ok2 and ok3) else 1)


if __name__ == "__main__":
    main()
