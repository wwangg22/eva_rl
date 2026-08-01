# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Behavior-clone the vision student from collected teacher episodes.

Pure PyTorch -- runs without Isaac Sim, so it can train while the simulator does other work.

.. code-block:: bash

    python scripts/distillation/train_student.py --data_dir data/distillation/<timestamp>

Then evaluate in sim with ``play_student.py``, and optionally close the DAgger loop by
re-running ``collect_episodes.py --student_checkpoint <out_dir>/best.pt`` and training again
on the combined data.
"""

import argparse
import os
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from episode_dataset import EpisodeDataset, split_train_val
from student_model import StudentPolicy, save_student

parser = argparse.ArgumentParser(description="Behavior-clone the vision student.")
parser.add_argument("--data_dir", type=str, required=True, action="append", help="Dataset dir(s); repeat to combine (e.g. BC + DAgger rounds).")
parser.add_argument("--out_dir", type=str, default=None, help="Default: logs/distillation/<timestamp>/")
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--val_frac", type=float, default=0.05)
parser.add_argument("--min_return", type=float, default=None, help="Drop episodes below this return (filters teacher failures).")
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()


def run_epoch(model, loader, device, optimizer=None) -> torch.Tensor:
    """One pass over the loader; returns per-dim mean squared error. Trains if optimizer given."""
    model.train(optimizer is not None)
    sq_err_sum, count = None, 0
    for workspace, wrist, proprio, action in loader:
        workspace, wrist = workspace.to(device, non_blocking=True), wrist.to(device, non_blocking=True)
        proprio, action = proprio.to(device, non_blocking=True), action.to(device, non_blocking=True)
        with torch.set_grad_enabled(optimizer is not None):
            pred = model(workspace, wrist, proprio)
            loss = torch.nn.functional.mse_loss(pred, action)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            batch_sq = ((pred - action) ** 2).sum(dim=0)
            sq_err_sum = batch_sq if sq_err_sum is None else sq_err_sum + batch_sq
            count += action.shape[0]
    return sq_err_sum / count


def main():
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    datasets = [EpisodeDataset(d, min_return=args.min_return) for d in args.data_dir]
    dataset = datasets[0] if len(datasets) == 1 else torch.utils.data.ConcatDataset(datasets)
    if len(datasets) > 1:
        # episode-level split needs the per-frame episode ids; keep it simple for combined
        # datasets by splitting each source and concatenating
        splits = [split_train_val(d, args.val_frac, args.seed) for d in datasets]
        train_set = torch.utils.data.ConcatDataset([s[0] for s in splits])
        val_set = torch.utils.data.ConcatDataset([s[1] for s in splits])
    else:
        train_set, val_set = split_train_val(dataset, args.val_frac, args.seed)

    loader_kwargs = {"batch_size": args.batch_size, "num_workers": args.num_workers, "pin_memory": True}
    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)

    meta = datasets[0].meta
    height, width = meta["image_hw"]
    model = StudentPolicy(
        height=height, width=width, proprio_dim=len(meta["proprio_indices"]), action_dim=meta["action_dim"]
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    out_dir = args.out_dir or os.path.join("logs", "distillation", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(out_dir, exist_ok=True)
    print(f"[train] {len(train_set)} train / {len(val_set)} val frames -> {out_dir}")

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_mse = run_epoch(model, train_loader, device, optimizer).mean().item()
        val_per_dim = run_epoch(model, val_loader, device)
        val_mse = val_per_dim.mean().item()
        marker = ""
        if val_mse < best_val:
            best_val = val_mse
            save_student(model, os.path.join(out_dir, "best.pt"), extra={"epoch": epoch, "val_mse": val_mse})
            marker = "  <- best"
        print(f"[train] epoch {epoch:3d}/{args.epochs}  train mse {train_mse:.5f}  val mse {val_mse:.5f}{marker}")
        save_student(model, os.path.join(out_dir, "last.pt"), extra={"epoch": epoch, "val_mse": val_mse})

    print(f"[train] done. best val mse {best_val:.5f}")
    print("[train] final val per-dim mse:", " ".join(f"{v:.5f}" for v in val_per_dim.tolist()))
    print(f"[train] checkpoints: {os.path.join(out_dir, 'best.pt')} / last.pt")


if __name__ == "__main__":
    main()
