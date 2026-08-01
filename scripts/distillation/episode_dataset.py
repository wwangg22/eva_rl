# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dataset over episode shards written by ``collect_episodes.py``.

Everything is preloaded into RAM as flat tensors (images stay uint8; ~22 MB per 250-step
episode at 160x90, so 500 episodes is ~11 GB). If datasets outgrow RAM, swap this for a
memory-mapped or per-shard lazy loader -- the (workspace, wrist, proprio, action) sample
contract can stay the same.
"""

from __future__ import annotations

import json
import os
from glob import glob

import torch
from torch.utils.data import Dataset, Subset


class EpisodeDataset(Dataset):
    """Flat (per-timestep) view of a collected episode dataset."""

    def __init__(self, data_dir: str, min_return: float | None = None):
        with open(os.path.join(data_dir, "meta.json")) as f:
            self.meta = json.load(f)
        proprio_indices = torch.tensor(self.meta["proprio_indices"], dtype=torch.long)

        shard_paths = sorted(glob(os.path.join(data_dir, "shard_*.pt")))
        if not shard_paths:
            raise FileNotFoundError(f"No shard_*.pt files in {data_dir}")

        workspace, wrist, proprio, action, episode_ids, returns = [], [], [], [], [], []
        episode_id = 0
        skipped = 0
        for path in shard_paths:
            for episode in torch.load(path, weights_only=False):
                ep_return = float(episode["reward"].sum())
                if min_return is not None and ep_return < min_return:
                    skipped += 1
                    continue
                length = episode["action"].shape[0]
                workspace.append(episode["workspace"])
                wrist.append(episode["wrist"])
                proprio.append(episode["state"][:, proprio_indices])
                action.append(episode["action"])
                episode_ids.append(torch.full((length,), episode_id, dtype=torch.long))
                returns.append(ep_return)
                episode_id += 1

        if episode_id == 0:
            raise RuntimeError(f"All episodes filtered out (min_return={min_return}).")
        self.workspace = torch.cat(workspace)
        self.wrist = torch.cat(wrist)
        self.proprio = torch.cat(proprio)
        self.action = torch.cat(action)
        self.episode_ids = torch.cat(episode_ids)
        self.episode_returns = torch.tensor(returns)
        gb = (self.workspace.numel() + self.wrist.numel()) / 1e9
        print(
            f"[dataset] {episode_id} episodes ({skipped} filtered), {len(self)} frames,"
            f" ~{gb:.1f} GB of images, mean return {self.episode_returns.mean():.2f}"
        )

    def __len__(self) -> int:
        return self.action.shape[0]

    def __getitem__(self, idx: int):
        return self.workspace[idx], self.wrist[idx], self.proprio[idx], self.action[idx]


def split_train_val(dataset: EpisodeDataset, val_frac: float = 0.05, seed: int = 0) -> tuple[Subset, Subset]:
    """Split by EPISODE (not frame) so temporally-correlated frames never straddle the split."""
    num_episodes = int(dataset.episode_ids.max()) + 1
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(num_episodes, generator=generator)
    num_val = max(1, int(round(num_episodes * val_frac)))
    val_episodes = set(order[:num_val].tolist())
    val_mask = torch.tensor([int(e) in val_episodes for e in dataset.episode_ids])
    val_indices = torch.nonzero(val_mask).flatten().tolist()
    train_indices = torch.nonzero(~val_mask).flatten().tolist()
    return Subset(dataset, train_indices), Subset(dataset, val_indices)
