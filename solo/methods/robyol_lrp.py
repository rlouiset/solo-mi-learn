# Copyright 2023 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from solo.losses.byol import byol_loss_func, unnormalized_byol_loss_func
from solo.losses.robyol import uniform_loss_func, align_loss_func
from solo.methods.base import BaseMomentumMethod
from solo.utils.momentum import initialize_momentum_params

def ne_predictor(zt, zx):
    zttzx = torch.matmul(zt.T, zx)
    p = 2.0 * zttzx
    p -= torch.matmul(zt.T, torch.matmul(zt, zttzx))
    return p

def lrp(zt, zx, safe_eps=1e-12):
    normfactor = 0.5 * torch.norm(zt, p='fro')
    normfactor += 0.5 * torch.norm(zx, p='fro') + safe_eps

    p = torch.matmul(torch.linalg.pinv(zt / normfactor), zx / normfactor)
    return p

"""def closed_form_linear_predictor(Z, T, ridge=1e-1):

    B, d = Z.shape

    Z_mean = Z.mean(dim=0, keepdim=True)
    T_mean = T.mean(dim=0, keepdim=True)
    Z_centered = Z - Z_mean
    T_centered = T - T_mean

    ZTZ = Z_centered.T @ Z_centered
    ZTT = Z_centered.T @ T_centered / B

    # Regularize ZTZ
    ridge_identity = ridge * torch.eye(d, device=Z.device, dtype=Z.dtype)
    ZTZ_reg = ZTZ + ridge_identity

    # Use pseudo-inverse instead of solve
    W = torch.linalg.pinv(ZTZ_reg) @ ZTT

    return W"""

def closed_form_linear_predictor(z_online, z_teacher):
    """
    Computes the least-squares linear predictor from z_online to z_teacher.

    Args:
        z_online (torch.Tensor): [B, d], requires_grad
        z_teacher (torch.Tensor): [B, d], detached

    Returns:
        torch.Tensor: W — the predictor matrix [d, d]
    """
    B, d = z_online.shape

    # Center the data
    Z = z_online - z_online.mean(dim=0, keepdim=True)
    T = z_teacher - z_teacher.mean(dim=0, keepdim=True)

    # Solve Z @ W ≈ T using least squares
    # torch.linalg.lstsq returns solution to min_W ||Z @ W - T||^2
    # Output shape: [d, d]
    W, *_ = torch.linalg.lstsq(Z, T)
    return W



def apply_predictor(Z, W):
    """
    Apply the linear predictor to z_online.

    Args:
        z_online (torch.Tensor): [B, d]
        W (torch.Tensor): [d, d]

    Returns:
        torch.Tensor: [B, d]
    """
    Z_mean = Z.mean(dim=0, keepdim=True)
    Z_centered = Z - Z_mean
    P = Z_centered @ W
    return P

def refresh_stack(stack, batch, max_batch_size=4096):
    len_batch = len(batch)
    if len(stack) >= max_batch_size:
        stack = torch.cat((stack[len_batch:], batch.detach()), dim=0)
    elif len(stack) > 0:
        stack = torch.cat((stack, batch.detach()), dim=0)
    else:
        stack = batch.detach()
    return stack



class RoBYOLLRP(BaseMomentumMethod):
    def __init__(self, cfg: omegaconf.DictConfig):
        """Implements BYOL (https://arxiv.org/abs/2006.07733).

        Extra cfg settings:
            method_kwargs:
                proj_output_dim (int): number of dimensions of projected features.
                proj_hidden_dim (int): number of neurons of the hidden layers of the projector.
                pred_hidden_dim (int): number of neurons of the hidden layers of the predictor.
        """

        super().__init__(cfg)

        proj_hidden_dim: int = cfg.method_kwargs.proj_hidden_dim
        proj_output_dim: int = cfg.method_kwargs.proj_output_dim

        self.au_scale_loss = cfg.method_kwargs.au_scale_loss

        # projector
        self.projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )

        # momentum projector
        self.momentum_projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )
        initialize_momentum_params(self.projector, self.momentum_projector)

        self.Z_momentum_v2_stack = torch.rand(size=[0, proj_output_dim], device="cuda", requires_grad=False).cuda().float()
        self.Z_v2_stack = torch.rand(size=[0, proj_output_dim], device="cuda", requires_grad=False).cuda().float()
        self.Z_momentum_v1_stack = torch.rand(size=[0, proj_output_dim], device="cuda", requires_grad=False).cuda().float()
        self.Z_v1_stack = torch.rand(size=[0, proj_output_dim], device="cuda", requires_grad=False).cuda().float()

        self.W = None

        # self.predictor = nn.Linear(proj_output_dim, proj_output_dim)

        # predictor
        # self.W = torch.rand(size=[proj_output_dim, proj_output_dim], device="cuda", requires_grad=False).cuda()
        # self.I = torch.eye(n=proj_output_dim, device="cuda", requires_grad=False).cuda()


    @staticmethod
    def add_and_assert_specific_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
        """Adds method specific default values/checks for config.

        Args:
            cfg (omegaconf.DictConfig): DictConfig object.

        Returns:
            omegaconf.DictConfig: same as the argument, used to avoid errors.
        """

        cfg = super(RoBYOLLRP, RoBYOLLRP).add_and_assert_specific_cfg(cfg)

        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_hidden_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_output_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.pred_hidden_dim")

        return cfg


    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector and predictor parameters to the parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        extra_learnable_params = [
            {"name": "projector", "params": self.projector.parameters()},
            # {"name": "predictor", "params": self.predictor.parameters()}
        ]
        return super().learnable_params + extra_learnable_params

    @property
    def momentum_pairs(self) -> List[Tuple[Any, Any]]:
        """Adds (projector, momentum_projector) to the parent's momentum pairs.

        Returns:
            List[Tuple[Any, Any]]: list of momentum pairs.
        """

        extra_momentum_pairs = [(self.projector, self.momentum_projector)]
        return super().momentum_pairs + extra_momentum_pairs

    def forward(self, X: torch.Tensor) -> Dict[str, Any]:
        """Performs forward pass of the online backbone, projector and predictor.

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            Dict[str, Any]: a dict containing the outputs of the parent and the projected features.
        """

        out = super().forward(X)
        z = self.projector(out["feats"])
        out.update({"z": z})
        return out

    def multicrop_forward(self, X: torch.tensor) -> Dict[str, Any]:
        """Performs the forward pass for the multicrop views.

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            Dict[]: a dict containing the outputs of the parent
                and the projected features.
        """

        out = super().multicrop_forward(X)
        z = self.projector(out["feats"])
        out.update({"z": z})
        return out

    @torch.no_grad()
    def momentum_forward(self, X: torch.Tensor) -> Dict:
        """Performs the forward pass of the momentum backbone and projector.

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            Dict[str, Any]: a dict containing the outputs of
                the parent and the momentum projected features.
        """

        out = super().momentum_forward(X)
        z = self.momentum_projector(out["feats"])
        out.update({"z": z})
        return out

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for BYOL reusing BaseMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of BYOL and classification loss.
        """

        out = super().training_step(batch, batch_idx)
        class_loss = out["loss"]
        Z = out["z"]
        Z_momentum = out["momentum_z"]

        with torch.no_grad():
            self.Z_v1_stack = refresh_stack(self.Z_v1_stack, Z[0].float())
            self.Z_v2_stack = refresh_stack(self.Z_v2_stack, Z[1].float())
            self.Z_momentum_v1_stack = refresh_stack(self.Z_momentum_v1_stack, Z_momentum[0].float())
            self.Z_momentum_v2_stack = refresh_stack(self.Z_momentum_v2_stack, Z_momentum[1].float())

        W = (closed_form_linear_predictor(self.Z_v1_stack, self.Z_momentum_v2_stack) +
             closed_form_linear_predictor(self.Z_v2_stack, self.Z_momentum_v1_stack)) / 2
        self.W = 0.8 * self.W + 0.2 * W

        # ------- negative cosine similarity loss -------
        neg_cos_sim = 0
        for v1 in range(self.num_large_crops):
            for v2 in np.delete(range(self.num_crops), v1):
                P = self.momentum_updater.cur_tau * apply_predictor(Z[v2], W) + (1-self.momentum_updater.cur_tau) * Z[v2]

                # P = apply_predictor(Z[v2], W)
                neg_cos_sim += byol_loss_func(P, Z_momentum[v1])

        """# ------- negative cosine similarity loss -------
        au_loss = 0
        for v1 in range(self.num_large_crops):
            for v2 in np.delete(range(self.num_crops), v1):
                au_loss += uniform_loss_func(F.normalize(Z[v1], dim=-1))
                au_loss += align_loss_func(F.normalize(Z[v1], dim=-1), F.normalize(Z[v2], dim=-1))"""

        # calculate std of features
        with torch.no_grad():
            z_std = F.normalize(torch.stack(Z[: self.num_large_crops]), dim=-1).std(dim=1).mean()
            z_std_teacher = F.normalize(torch.stack(Z_momentum[: self.num_large_crops]), dim=-1).std(dim=1).mean()
            student_entropy = (uniform_loss_func(F.normalize(Z[1], dim=-1)) + uniform_loss_func(F.normalize(Z[0], dim=-1))) / 2
            teacher_entropy = (uniform_loss_func(F.normalize(Z_momentum[1], dim=-1)) + uniform_loss_func(F.normalize(Z_momentum[0], dim=-1))) / 2

        metrics = {
            "train_neg_cos_sim": neg_cos_sim,
            "train_z_std": z_std,
            "train_z_std_teacher": z_std_teacher,
            "train_student_entropy": student_entropy,
            "train_teacher_entropy": teacher_entropy
        }
        self.log_dict(metrics, on_epoch=True, sync_dist=True)

        return neg_cos_sim + class_loss # + self.au_scale_loss * au_loss