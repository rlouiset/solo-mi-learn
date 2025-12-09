# Copyright 2023 solo-learn development team.
import copy
import math
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
import pingouin as pg
import numpy as np
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from solo.losses.byol import byol_loss_func
from solo.losses.robyol import uniform_loss_func, align_loss_func
from solo.methods.base import BaseMomentumMethod
from solo.utils.momentum import initialize_momentum_params


class BYOL(BaseMomentumMethod):
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
        pred_hidden_dim: int = cfg.method_kwargs.pred_hidden_dim

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

        # predictor
        self.predictor = nn.Sequential(
            nn.Linear(proj_output_dim, pred_hidden_dim),
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

        cfg = super(BYOL, BYOL).add_and_assert_specific_cfg(cfg)

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
            {"name": "predictor", "params": self.predictor.parameters()},
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
        p = self.predictor(z)
        out.update({"z": z, "p": p})
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
        p = self.predictor(z)
        out.update({"z": z, "p": p})
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
        P = out["p"]
        Z_momentum = out["momentum_z"]

        # ------- negative cosine similarity loss -------
        neg_cos_sim = 0
        for v1 in range(self.num_large_crops):
            for v2 in np.delete(range(self.num_crops), v1):
                neg_cos_sim += byol_loss_func(P[v2], Z_momentum[v1])

        # calculate std of features
        with torch.no_grad():
            z_std = F.normalize(torch.stack(Z[: self.num_large_crops]), dim=-1).std(dim=1).mean()
            z_std_teacher = F.normalize(torch.stack(Z_momentum[: self.num_large_crops]), dim=-1).std(dim=1).mean()
            z_std_predictor = F.normalize(torch.stack(P[: self.num_large_crops]), dim=-1).std(dim=1).mean()

            student_entropy = (uniform_loss_func(F.normalize(Z[1], dim=-1)) + uniform_loss_func(F.normalize(Z[0], dim=-1))) / 2
            teacher_entropy = (uniform_loss_func(F.normalize(Z_momentum[1], dim=-1)) + uniform_loss_func(F.normalize(Z_momentum[0], dim=-1))) / 2
            predictor_entropy = (uniform_loss_func(F.normalize(P[1], dim=-1)) + uniform_loss_func(F.normalize(P[0], dim=-1))) / 2

            student_alignment = align_loss_func(F.normalize(Z[0], dim=-1), F.normalize(Z[1], dim=-1))
            teacher_alignment = align_loss_func(F.normalize(Z_momentum[0], dim=-1), F.normalize(Z_momentum[1], dim=-1))
            predictor_alignment = align_loss_func(F.normalize(P[0], dim=-1), F.normalize(P[1], dim=-1))

            norm1 = torch.linalg.norm(F.normalize(Z[0], dim=-1) - F.normalize(Z[1], dim=-1), dim=1)
            norm2 = torch.linalg.norm(F.normalize(Z_momentum[0], dim=-1) - F.normalize(Z_momentum[1], dim=-1), dim=1)

            student_teacher_pearson_corr_x = ((norm1 - norm1.mean()) @ (norm2 - norm2.mean())) / (torch.norm(norm1 - norm1.mean()) * torch.norm(norm2 - norm2.mean()))

            student_teacher_pearson_corr = (estimate_rho_isotropic(F.normalize(Z[0], dim=-1), F.normalize(Z_momentum[1], dim=-1)) +
                                            estimate_rho_isotropic(F.normalize(Z[1], dim=-1), F.normalize(Z_momentum[0], dim=-1)))/2

            residual_entropy = (uniform_loss_func(F.normalize(Z_momentum[1], dim=-1) - F.normalize(P[0], dim=-1)) +
                                uniform_loss_func(F.normalize(Z_momentum[0], dim=-1) - F.normalize(P[1], dim=-1))) / 2
            residual_std = ((F.normalize(Z_momentum[1], dim=-1) - F.normalize(P[0], dim=-1)).std(dim=1).mean() +
                            (F.normalize(Z_momentum[0], dim=-1) - F.normalize(P[1], dim=-1))).std(dim=1).mean() / 2

            # computing residuals of normalized representations
            residuals_0 = (F.normalize(Z_momentum[1], dim=-1) - F.normalize(P[0], dim=-1)).cpu().numpy()
            # Gaussianity of residuals
            _, p_value_0_residuals, _ = pg.multivariate_normality(residuals_0, alpha=0.05)
            cov = np.cov(residuals_0, rowvar=False)
            eigvals = np.linalg.eigvalsh(cov)
            isotropy_ratio_0_residuals = eigvals.max() / eigvals.min()

            # Gaussianity of students
            students_0 = Z[0].cpu().numpy()
            _, p_value_0_student, _ = pg.multivariate_normality(students_0, alpha=0.05)
            cov = np.cov(students_0, rowvar=False)
            eigvals = np.linalg.eigvalsh(cov)
            isotropy_ratio_0_student = eigvals.max() / eigvals.min()

            # Gaussianity of teachers
            teachers_0 = Z_momentum[0].cpu().numpy()
            _, p_value_0_teacher, _ = pg.multivariate_normality(teachers_0, alpha=0.05)
            cov = np.cov(teachers_0, rowvar=False)
            eigvals = np.linalg.eigvalsh(cov)
            isotropy_ratio_0_teacher = eigvals.max() / eigvals.min()

            # cross-cov between residuals and students
            cross_cov = cross_covariance_norm(F.normalize(Z[0], dim=-1).cpu().numpy(), residuals_0)

            # TODO: Interpolate backbone and projector
            interpolated_backbone = interpolate_network(self.backbone, self.momentum_backbone, 0.99)
            interpolated_projector = interpolate_network(self.projector, self.momentum_projector, 0.99)

            # Pick a large crop to test
            X_crop = batch[1][0]  # first crop in the batch
            feats_interp = interpolated_backbone(X_crop)  # forward through interpolated backbone
            Z_interp_net = interpolated_projector(feats_interp)  # forward through interpolated projector
            Z_interp_net = F.normalize(Z_interp_net, dim=-1)

            # Weighted sum of original outputs
            # Assume Z_momentum[0] and Z[0] correspond to this crop
            Z_weighted = 0.99 * F.normalize(Z_momentum[0], dim=-1) + (1 - 0.99) * F.normalize(Z[0], dim=-1)

            # Compute difference
            interpolation_diff = torch.norm(Z_interp_net - Z_weighted, dim=1)  # shape: [batch_size]
            interpolation_check = interpolation_diff.mean().item()  # scalar metric

        metrics = {
            "train_neg_cos_sim": neg_cos_sim,
            "train_z_std": z_std,
            "train_z_std_teacher": z_std_teacher,
            "train_z_std_predictor": z_std_predictor,
            "train_student_entropy": student_entropy,
            "train_predictor_entropy": predictor_entropy,
            "train_teacher_entropy": teacher_entropy,
            "train_student_alignment": student_alignment,
            "train_predictor_alignment": predictor_alignment,
            "train_teacher_alignment": teacher_alignment,
            "residual_entropy": residual_entropy,
            "residual_std": residual_std,
            "rho_x": student_teacher_pearson_corr_x,
            "rho": student_teacher_pearson_corr,
            "hz_test_residuals": p_value_0_residuals,
            "isotropy_ratio_residuals": isotropy_ratio_0_residuals,
            "hz_test_student": p_value_0_student,
            "isotropy_ratio_student": isotropy_ratio_0_student,
            "hz_test_teacher": p_value_0_teacher,
            "isotropy_ratio_teacher": isotropy_ratio_0_teacher,
            "interpolation_check": interpolation_check,
            "cross_cov": cross_cov
        }

        self.log_dict(metrics, on_epoch=True, sync_dist=True)

        return neg_cos_sim + class_loss

def estimate_rho_isotropic(X, Y, unbiased=True):
    # X, Y: tensors of shape [B, d]
    B, d = X.shape
    dd = float(B - 1) if unbiased else float(B)

    x_mean = X.mean(dim=0, keepdim=True)    # [1, d]
    y_mean = Y.mean(dim=0, keepdim=True)

    Xc = X - x_mean      # [B, d]
    Yc = Y - y_mean

    # trace of cross-covariance (sum over elementwise products)
    tr_Sxy = (Xc * Yc).sum() / dd   # scalar

    # per-dimension marginal variances (averaged across dims)
    sigma_x2 = (Xc.pow(2).sum() / dd) / d
    sigma_y2 = (Yc.pow(2).sum() / dd) / d

    rho_hat = tr_Sxy / (d * torch.sqrt(sigma_x2 * sigma_y2))
    return rho_hat.item()

from scipy.stats import shapiro, normaltest

def test_gaussianity_random_projections(residuals, num_projections=50):
    """
    Test Gaussianity of high-dimensional residuals via random projections.

    Args:
        residuals: torch.Tensor of shape (n_samples, d)
        num_projections: number of random projections to test
    Returns:
        List of p-values for Shapiro-Wilk and D'Agostino tests
    """
    residuals_np = residuals.detach().cpu().numpy()
    n_samples, d = residuals_np.shape

    shapiro_pvals = []
    dagostino_pvals = []

    for _ in range(num_projections):
        # Random unit vector
        v = np.random.randn(d)
        v /= np.linalg.norm(v)

        # Project residuals
        proj = residuals_np @ v  # shape: (n_samples,)

        # Shapiro-Wilk test
        _, p_sw = shapiro(proj)
        shapiro_pvals.append(p_sw)

        # D'Agostino K2 test
        _, p_k2 = normaltest(proj)
        dagostino_pvals.append(p_k2)

    return np.array(shapiro_pvals), np.array(dagostino_pvals)

@torch.no_grad()
def interpolate_network(online_net: torch.nn.Module, momentum_net: torch.nn.Module, tau: float):
    """
    Returns a new network whose parameters are interpolated:
        tau * momentum_net + (1 - tau) * online_net
    """
    interpolated_net = copy.deepcopy(momentum_net)
    for p_interp, p_mom, p_online in zip(interpolated_net.parameters(),
                                         momentum_net.parameters(),
                                         online_net.parameters()):
        p_interp.data = tau * p_mom.data + (1 - tau) * p_online.data
    return interpolated_net

def cross_covariance_norm(Z, R):
    Zc = Z - Z.mean(0, keepdims=True)
    Rc = R - R.mean(0, keepdims=True)
    C = (Zc.T @ Rc) / (Z.shape[0] - 1)
    return np.linalg.norm(C, ord='fro')
