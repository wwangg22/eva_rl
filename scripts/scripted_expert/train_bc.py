"""Behavior-clone a pick-and-place policy from scripted-expert demonstrations.

Loads demonstration shards (``<data_dir>/shard_*.pt``, each a torch-saved list of
episode dicts ``{"obs": (T, 39), "action": (T, 7)}``), splits train/val by episode,
and fits an MLP matching the rl_games actor (units [256, 128, 64], ELU, separate mu
head). Observations are normalized exactly like rl_games RunningMeanStd inference:
``clamp((obs - mean) / sqrt(var + 1e-5), -5, 5)`` with mean/var from the training set,
so the checkpoint can later be transplanted into an rl_games checkpoint unchanged
(see ``bc_to_rlgames.py``).

.. code-block:: bash

    python scripts/scripted_expert/train_bc.py --data_dir data/pick_place_demos/run1

"""

import argparse
import glob
import os
from datetime import datetime

import torch
import torch.nn as nn

parser = argparse.ArgumentParser(description="Behavior cloning on scripted-expert demos.")
parser.add_argument("--data_dir", type=str, action="append", required=True,
                    help="Directory with shard_*.pt files. Repeatable.")
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--batch_size", type=int, default=4096)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--val_frac", type=float, default=0.05, help="Fraction of episodes held out for validation.")
parser.add_argument("--out_dir", type=str, default=None, help="Default: logs/bc/<timestamp>.")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

NORM_EPS = 1e-5  # matches rl_games RunningMeanStd epsilon
NORM_CLIP = 5.0  # matches rl_games RunningMeanStd clamp


def build_actor(obs_dim=39, act_dim=7):
    """Trunk + mu head mirroring rl_games A2CBuilder (units [256,128,64], elu, separate=False)."""
    trunk = nn.Sequential(
        nn.Linear(obs_dim, 256), nn.ELU(),
        nn.Linear(256, 128), nn.ELU(),
        nn.Linear(128, 64), nn.ELU(),
    )
    mu = nn.Linear(64, act_dim)
    return trunk, mu


def load_episodes(data_dirs):
    episodes = []
    for d in data_dirs:
        shards = sorted(glob.glob(os.path.join(d, "shard_*.pt")))
        if not shards:
            raise FileNotFoundError(f"no shard_*.pt files in {d}")
        for path in shards:
            for ep in torch.load(path, map_location="cpu", weights_only=True):
                episodes.append((ep["obs"].float(), ep["action"].float()))
        print(f"{d}: {len(shards)} shards")
    return episodes


def main():
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    out_dir = args.out_dir or os.path.join("logs", "bc", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(out_dir, exist_ok=True)

    episodes = load_episodes(args.data_dir)
    n_eps = len(episodes)
    if n_eps < 2:
        raise ValueError(f"need at least 2 episodes, got {n_eps}")

    # split by episode, not by frame
    perm = torch.randperm(n_eps).tolist()
    n_val = max(1, round(n_eps * args.val_frac))
    val_eps = [episodes[i] for i in perm[:n_val]]
    train_eps = [episodes[i] for i in perm[n_val:]]
    train_obs = torch.cat([o for o, _ in train_eps])
    train_act = torch.cat([a for _, a in train_eps])
    val_obs = torch.cat([o for o, _ in val_eps])
    val_act = torch.cat([a for _, a in val_eps])
    obs_dim, act_dim = train_obs.shape[1], train_act.shape[1]
    print(f"episodes: {len(train_eps)} train / {n_val} val | frames: {len(train_obs)} train / {len(val_obs)} val")

    # obs statistics over the training set; normalization matches rl_games RunningMeanStd inference
    obs_mean = train_obs.mean(0)
    obs_var = train_obs.var(0)

    def normalize(x):
        return ((x - obs_mean) / torch.sqrt(obs_var + NORM_EPS)).clamp(-NORM_CLIP, NORM_CLIP)

    train_obs, train_act = normalize(train_obs).to(device), train_act.to(device)
    val_obs, val_act = normalize(val_obs).to(device), val_act.to(device)

    trunk, mu = build_actor(obs_dim, act_dim)
    trunk.to(device)
    mu.to(device)
    params = list(trunk.parameters()) + list(mu.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    def save(path, meta):
        torch.save({
            "trunk": trunk.state_dict(),
            "mu": mu.state_dict(),
            "obs_mean": obs_mean,
            "obs_var": obs_var,
            "meta": meta,
        }, path)

    meta = {
        "data_dirs": args.data_dir,
        "obs_dim": obs_dim,
        "action_dim": act_dim,
        "train_episodes": len(train_eps),
        "val_episodes": n_val,
        "train_frames": len(train_obs),
        "epochs": args.epochs,
        "lr": args.lr,
        "norm": f"clamp((obs - mean) / sqrt(var + {NORM_EPS}), -{NORM_CLIP}, {NORM_CLIP})",
    }

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        trunk.train()
        mu.train()
        order = torch.randperm(len(train_obs), device=device)
        train_loss, n_batches = 0.0, 0
        for start in range(0, len(order), args.batch_size):
            idx = order[start:start + args.batch_size]
            pred = mu(trunk(train_obs[idx]))
            loss = nn.functional.mse_loss(pred, train_act[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1
        scheduler.step()

        trunk.eval()
        mu.eval()
        with torch.no_grad():
            val_pred = torch.cat([mu(trunk(val_obs[i:i + args.batch_size]))
                                  for i in range(0, len(val_obs), args.batch_size)])
            val_loss = nn.functional.mse_loss(val_pred, val_act).item()
        print(f"epoch {epoch:3d}/{args.epochs} | train {train_loss / n_batches:.6f} | val {val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            save(os.path.join(out_dir, "best.pt"), {**meta, "epoch": epoch, "val_loss": val_loss})

    save(os.path.join(out_dir, "last.pt"), {**meta, "epoch": args.epochs, "val_loss": val_loss})

    # per-dim val MSE with the best checkpoint
    best = torch.load(os.path.join(out_dir, "best.pt"), map_location=device, weights_only=True)
    trunk.load_state_dict(best["trunk"])
    mu.load_state_dict(best["mu"])
    with torch.no_grad():
        val_pred = torch.cat([mu(trunk(val_obs[i:i + args.batch_size]))
                              for i in range(0, len(val_obs), args.batch_size)])
        per_dim = ((val_pred - val_act) ** 2).mean(0)
    print(f"best val loss: {best_val:.6f} (epoch {best['meta']['epoch']})")
    print("per-dim val MSE:", " ".join(f"{v:.5f}" for v in per_dim.tolist()))
    print(f"saved: {os.path.join(out_dir, 'best.pt')} and last.pt")


if __name__ == "__main__":
    main()
