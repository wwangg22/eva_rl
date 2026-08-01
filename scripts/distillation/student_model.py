# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Vision student policy for teacher-student distillation.

Pure PyTorch -- no Isaac dependencies -- so the same module serves offline BC training
(``train_student.py``), DAgger collection (``collect_episodes.py --student_checkpoint``) and
in-sim evaluation (``play_student.py``).

Architecture: one small Nature-CNN-style encoder per camera (separate weights -- the two
views have very different statistics), a proprio MLP, and a fused MLP head regressing the
teacher's 7-d action. Inputs are uint8 (N, H, W, 3) frames exactly as stored in the dataset;
normalization lives inside the model so callers never worry about it.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ImageEncoder(nn.Module):
    """Nature-CNN-style encoder for one camera stream."""

    def __init__(self, height: int, width: int, out_dim: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ELU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            flat_dim = self.conv(torch.zeros(1, 3, height, width)).shape[1]
        self.head = nn.Sequential(nn.Linear(flat_dim, out_dim), nn.ELU())

    def forward(self, frames_uint8: torch.Tensor) -> torch.Tensor:
        """frames_uint8: (N, H, W, 3) uint8 -> (N, out_dim) features."""
        x = frames_uint8.permute(0, 3, 1, 2).float() / 255.0 - 0.5
        return self.head(self.conv(x))


class StudentPolicy(nn.Module):
    """Two camera encoders + proprio MLP -> action regression head."""

    def __init__(
        self,
        height: int = 90,
        width: int = 160,
        proprio_dim: int = 30,
        action_dim: int = 7,
        image_feat_dim: int = 256,
        proprio_feat_dim: int = 128,
    ):
        super().__init__()
        # stored so checkpoints are self-describing (see save_student/load_student)
        self.init_kwargs = {
            "height": height,
            "width": width,
            "proprio_dim": proprio_dim,
            "action_dim": action_dim,
            "image_feat_dim": image_feat_dim,
            "proprio_feat_dim": proprio_feat_dim,
        }
        self.workspace_encoder = ImageEncoder(height, width, image_feat_dim)
        self.wrist_encoder = ImageEncoder(height, width, image_feat_dim)
        self.proprio_mlp = nn.Sequential(
            nn.Linear(proprio_dim, proprio_feat_dim),
            nn.ELU(),
            nn.Linear(proprio_feat_dim, proprio_feat_dim),
            nn.ELU(),
        )
        fused = 2 * image_feat_dim + proprio_feat_dim
        self.head = nn.Sequential(
            nn.Linear(fused, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, workspace: torch.Tensor, wrist: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        feats = torch.cat(
            [self.workspace_encoder(workspace), self.wrist_encoder(wrist), self.proprio_mlp(proprio.float())], dim=-1
        )
        return self.head(feats)

    @torch.inference_mode()
    def act(self, workspace: torch.Tensor, wrist: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        """Inference entry point for env rollouts (same signature as forward, eval mode)."""
        was_training = self.training
        self.eval()
        actions = self.forward(workspace, wrist, proprio)
        if was_training:
            self.train()
        return actions


def save_student(model: StudentPolicy, path: str, extra: dict | None = None) -> None:
    payload = {"init_kwargs": model.init_kwargs, "state_dict": model.state_dict(), "extra": extra or {}}
    torch.save(payload, path)


def load_student(path: str, device: str | torch.device = "cuda:0") -> StudentPolicy:
    payload = torch.load(path, map_location=device, weights_only=False)
    model = StudentPolicy(**payload["init_kwargs"]).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model
