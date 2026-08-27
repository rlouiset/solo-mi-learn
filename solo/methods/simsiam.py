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

import math
from typing import Any, Dict, List, Sequence

import numpy as np
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import anderson, normaltest
from solo.losses.simsiam import simsiam_loss_func
from solo.losses.robyol import uniform_loss_func, align_loss_func
from solo.methods.base import BaseMethod
from solo.utils.misc import omegaconf_select


class SimSiam(BaseMethod):
    def __init__(self, cfg: omegaconf.DictConfig):
        """Implements SimSiam (https://arxiv.org/abs/2011.10566).

        Extra cfg settings:
            method_kwargs:
                proj_output_dim (int): number of dimensions of projected features.
                proj_hidden_dim (int): number of neurons of the hidden layers of the projector.
                pred_hidden_dim (int): number of neurons of the hidden layers of the predictor.
        """

        super().__init__(cfg)

        proj_hidden_dim: int = cfg.method_kwargs.proj_hidden_dim
        proj_output_dim: int = cfg.method_kwargs.proj_output_dim
        pred_hidden_dim: int = cfg.method_kwargs.pred_hidden_dim

        self.au_scale_loss = cfg.method_kwargs.au_scale_loss
        self.predictor_lr: float = cfg.optimizer.predictor_lr

        # projector
        self.projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim, bias=False),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_hidden_dim, bias=False),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
            nn.BatchNorm1d(proj_output_dim, affine=False),
        )
        self.projector[6].bias.requires_grad = False  # hack: not use bias as it is followed by BN

        # predictor
        self.predictor = nn.Sequential(
            nn.Linear(proj_output_dim, pred_hidden_dim, bias=False),
            nn.BatchNorm1d(pred_hidden_dim),
            nn.ReLU(),
            nn.Linear(pred_hidden_dim, proj_output_dim),
        )

    @staticmethod
    def add_and_assert_specific_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
        """Adds method specific default values/checks for config.

        Args:
            cfg (omegaconf.DictConfig): DictConfig object.

        Returns:
            omegaconf.DictConfig: same as the argument, used to avoid errors.
        """

        cfg = super(SimSiam, SimSiam).add_and_assert_specific_cfg(cfg)

        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_output_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_hidden_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.pred_hidden_dim")

        # constant (non-decaying) predictor lr, matching the SimSiam paper's recipe.
        # defaults to the base lr if unset.
        cfg.optimizer.predictor_lr = omegaconf_select(cfg, "optimizer.predictor_lr", cfg.optimizer.lr)

        return cfg

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector and predictor parameters to the parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        extra_learnable_params: List[dict] = [
            {"name": "projector", "params": self.projector.parameters()},
            {
                "name": "predictor",
                "params": self.predictor.parameters(),
                "lr": self.predictor_lr,
                "static_lr": True,
            },
        ]
        return super().learnable_params + extra_learnable_params

    def forward(self, X: torch.Tensor) -> Dict[str, Any]:
        """Performs the forward pass of the backbone, the projector and the predictor.

        Args:
            X (torch.Tensor): a batch of images in the tensor format.

        Returns:
            Dict[str, Any]:
                a dict containing the outputs of the parent
                and the projected and predicted features.
        """

        out = super().forward(X)
        z = self.projector(out["feats"])
        p = self.predictor(z)
        out.update({"z": z, "p": p})
        return out

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for SimSiam reusing BaseMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of SimSiam loss and classification loss.
        """

        out = super().training_step(batch, batch_idx)
        class_loss = out["loss"]
        z1, z2 = out["z"]
        p1, p2 = out["p"]

        # ------- negative cosine similarity loss -------
        neg_cos_sim = simsiam_loss_func(p1, z2) / 2 + simsiam_loss_func(p2, z1) / 2

        au_loss = 0
        au_loss += uniform_loss_func(F.normalize(z1, dim=-1))
        au_loss += uniform_loss_func(F.normalize(z2, dim=-1))
        au_loss += 2 * align_loss_func(F.normalize(z1, dim=-1), F.normalize(z2, dim=-1))

        # calculate std of features
        z1_std = F.normalize(z1, dim=-1).std(dim=0).mean()
        z2_std = F.normalize(z2, dim=-1).std(dim=0).mean()
        z_std = (z1_std + z2_std) / 2

        # ------- diagnostics (no grad) -------
        with torch.no_grad():

            # L2-normalized representations (on the hypersphere)
            z1n = F.normalize(z1, dim=-1)  # student, view 1
            z2n = F.normalize(z2, dim=-1)  # student, view 2
            p1n = F.normalize(p1, dim=-1)  # predictor, view 1
            p2n = F.normalize(p2, dim=-1)  # predictor, view 2

            # KDE entropy estimator (same as BYOL)
            def kde_entropy(z, sigma=1.0):
                """KDE entropy estimator on normalized representations.
                H(Z) ~ -1/N sum_i log(1/(N-1) sum_{j!=i} exp(-||z_i-z_j||^2 / 2sigma^2))
                """
                dists_sq = torch.cdist(z, z, p=2).pow(2)
                B = z.shape[0]
                mask = ~torch.eye(B, dtype=torch.bool, device=z.device)
                log_density = torch.logsumexp(
                    -dists_sq[mask].view(B, B - 1) / (2 * sigma ** 2), dim=1
                ) - math.log(B - 1)
                return -log_density.mean()

            # Marginal entropies H(Z_phi), H(Z_{phi,psi})
            h_student = (kde_entropy(z1n) + kde_entropy(z2n)) / 2
            h_predictor = (kde_entropy(p1n) + kde_entropy(p2n)) / 2

            # SimSiam has no separate momentum network: the stop-gradient branch
            # IS the online z, so the "teacher" signal equals the student one by
            # construction. Aliased under BYOL's naming for direct wandb comparison.
            h_teacher = h_student

            # Alignment: E[||z(v1) - z(v2)||^2] / 2
            student_alignment = (z1n - z2n).pow(2).sum(dim=1).mean() / 2
            predictor_alignment = (p1n - p2n).pow(2).sum(dim=1).mean() / 2

            # Cross-prediction MSE: predictor(view) vs. stop-grad target (other view)
            cross_prediction_mse = (
                (p1n - z2n).pow(2).sum(dim=1).mean()
                + (p2n - z1n).pow(2).sum(dim=1).mean()
            ) / 2

            # Uniformity
            student_uniformity = (uniform_loss_func(z1n) + uniform_loss_func(z2n)) / 2
            predictor_uniformity = (uniform_loss_func(p1n) + uniform_loss_func(p2n)) / 2

        metrics = {
            "train_neg_cos_sim": neg_cos_sim,
            "train_z_std": z_std,
            "h_student": h_student,
            "h_predictor": h_predictor,
            "student_alignment": student_alignment,
            "predictor_alignment": predictor_alignment,
            "cross_prediction_mse": cross_prediction_mse,
            "train_student_uniformity": student_uniformity,
            "train_predictor_uniformity": predictor_uniformity,
        }

        # =============================================
        # EXPENSIVE DIAGNOSTICS (every N steps): per-coordinate Gaussianity
        # =============================================
        if batch_idx % 100 == 0:
            with torch.no_grad():
                for name, tensor in [("student", z1n), ("predictor", p1n)]:
                    t_np = tensor.detach().cpu().numpy()
                    n_samples, d_dim = t_np.shape

                    ad_stats = []
                    dp_pvals = []

                    for j in range(d_dim):
                        col = t_np[:, j]
                        col_std = (col - col.mean()) / (col.std() + 1e-10)

                        # Anderson-Darling test
                        ad_result = anderson(col_std, dist='norm')
                        ad_stats.append(ad_result.statistic)

                        # D'Agostino-Pearson test
                        if n_samples >= 20:
                            _, dp_pval = normaltest(col)
                            dp_pvals.append(dp_pval)

                    ad_stats = np.array(ad_stats)
                    # AD < 0.752 => cannot reject Gaussianity at 5% level
                    metrics[f"{name}_coord_ad_frac_gaussian"] = float(
                        (ad_stats < 0.752).mean())
                    metrics[f"{name}_coord_ad_avg"] = float(ad_stats.mean())

                    if len(dp_pvals) > 0:
                        dp_pvals = np.array(dp_pvals)
                        # p > 0.05 => cannot reject Gaussianity
                        metrics[f"{name}_coord_dp_frac_gaussian"] = float(
                            (dp_pvals > 0.05).mean())
                        metrics[f"{name}_coord_dp_avg_pval"] = float(
                            dp_pvals.mean())

        self.log_dict(metrics, on_epoch=True, sync_dist=True)

        return neg_cos_sim + class_loss + self.au_scale_loss * au_loss
