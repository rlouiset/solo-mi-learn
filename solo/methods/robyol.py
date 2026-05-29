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
from solo.losses.byol import byol_loss_func
from solo.losses.robyol import uniform_loss_func, align_loss_func
from solo.methods.base import BaseMomentumMethod
from solo.utils.momentum import initialize_momentum_params
import math
import copy

class RoBYOL(BaseMomentumMethod):
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

        cfg = super(RoBYOL, RoBYOL).add_and_assert_specific_cfg(cfg)

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
        """Training step for RoBYOL reusing BaseMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of BYOL, AU regularization, and classification loss.
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

        # ------- alignment + uniformity regularization on student -------
        au_loss = 0
        for v1 in range(self.num_large_crops):
            for v2 in np.delete(range(self.num_crops), v1):
                u_Z = uniform_loss_func(F.normalize(Z[v1], dim=-1))
                a_Z = align_loss_func(F.normalize(Z[v1], dim=-1),
                                      F.normalize(Z[v2], dim=-1))
                au_loss += u_Z + a_Z

        # ------- diagnostics (no grad) -------
        with torch.no_grad():

            # L2-normalized representations (on the hypersphere)
            z0 = F.normalize(Z[0], dim=-1)  # student, view 1
            z1 = F.normalize(Z[1], dim=-1)  # student, view 2
            zm0 = F.normalize(Z_momentum[0], dim=-1)  # teacher, view 1
            zm1 = F.normalize(Z_momentum[1], dim=-1)  # teacher, view 2
            p0 = F.normalize(P[0], dim=-1)  # predictor(student), view 1
            p1 = F.normalize(P[1], dim=-1)  # predictor(student), view 2

            # Raw projector outputs (pre-normalization, for reference checks)
            z0_raw = Z[0]
            zm0_raw = Z_momentum[0]
            zm1_raw = Z_momentum[1]

            d = z0.shape[1]

            # =============================================
            # SECTION 3.1: L_bound = TS Cross-Prediction + H(Z_theta)
            # =============================================

            # KDE entropy estimator
            def kde_entropy(z, sigma=1.0):
                """KDE entropy estimator on normalized representations."""
                dists_sq = torch.cdist(z, z, p=2).pow(2)
                B = z.shape[0]
                mask = ~torch.eye(B, dtype=torch.bool, device=z.device)
                log_density = torch.logsumexp(
                    -dists_sq[mask].view(B, B - 1) / (2 * sigma ** 2), dim=1
                ) - math.log(B - 1)
                return -log_density.mean()

            # Marginal entropies
            h_teacher = (kde_entropy(zm0) + kde_entropy(zm1)) / 2
            h_student = (kde_entropy(z0) + kde_entropy(z1)) / 2
            h_predictor = (kde_entropy(p0) + kde_entropy(p1)) / 2

            # Alignment = H(Z|X) proxy
            teacher_alignment = (zm0 - zm1).pow(2).sum(dim=1).mean() / 2
            student_alignment = (z0 - z1).pow(2).sum(dim=1).mean() / 2

            # Cross-prediction MSE (= BYOL loss up to constant)
            cross_prediction_mse = ((zm1 - p0).pow(2).sum(dim=1).mean() +
                                    (zm0 - p1).pow(2).sum(dim=1).mean()) / 2

            # Teacher self-prediction MSE
            teacher_self_prediction_mse = (zm0 - zm1).pow(2).sum(dim=1).mean() / 2

            # Uniformity-based entropy (backward compat)
            student_uniformity = (uniform_loss_func(z0) + uniform_loss_func(z1)) / 2
            teacher_uniformity = (uniform_loss_func(zm0) + uniform_loss_func(zm1)) / 2

            # =============================================
            # SECTION 4.1: Assumption 2 (marginal innovation independence)
            # =============================================
            innovation_v1 = z0 - zm0
            rho_innovation_teacher = abs(estimate_rho_isotropic(
                innovation_v1, zm0))
            rho_innovation_teacher_raw = abs(estimate_rho_isotropic(
                z0_raw - zm0_raw, zm0_raw))

            # =============================================
            # SECTION 5.1: Assumption 3 (conditional innovation independence)
            # =============================================
            rho_innovation_teacher_cond = abs(estimate_rho_isotropic(
                innovation_v1, zm1))
            rho_innovation_teacher_cond_raw = abs(estimate_rho_isotropic(
                z0_raw - zm0_raw, zm1_raw))

        metrics = {
            # Section 3.1: L_bound components
            "train_neg_cos_sim": neg_cos_sim,
            "train_au_loss": au_loss,
            "h_teacher": h_teacher,
            "h_student": h_student,
            "h_predictor": h_predictor,
            "teacher_alignment": teacher_alignment,
            "student_alignment": student_alignment,
            "cross_prediction_mse": cross_prediction_mse,
            "teacher_self_prediction_mse": teacher_self_prediction_mse,
            "train_student_uniformity": student_uniformity,
            "train_teacher_uniformity": teacher_uniformity,
            # Section 4.1: Assumption 2 (marginal innovation independence)
            "rho_innovation_teacher": rho_innovation_teacher,
            "rho_innovation_teacher_raw": rho_innovation_teacher_raw,
            # Section 5.1: Assumption 3 (conditional innovation independence)
            "rho_innovation_teacher_cond": rho_innovation_teacher_cond,
            "rho_innovation_teacher_cond_raw": rho_innovation_teacher_cond_raw,
        }

        # =============================================
        # EXPENSIVE DIAGNOSTICS (every N steps)
        # =============================================
        if batch_idx % 100 == 0:
            with torch.no_grad():

                # =============================================
                # SECTION 4.1: Assumption 1 (linear interpolation)
                # + SECTION 4.2: Lemma 2 (entropy non-decrease)
                # =============================================
                with torch.cuda.amp.autocast(enabled=False):
                    interpolated_backbone = interpolate_network(
                        self.backbone, self.momentum_backbone, 0.99)
                    interpolated_projector = interpolate_network(
                        self.projector, self.momentum_projector, 0.99)

                    X_crop = batch[1][0].float()
                    feats_interp = interpolated_backbone(X_crop)
                    Z_interp_raw = interpolated_projector(feats_interp)
                    z_interp = F.normalize(Z_interp_raw, dim=-1)

                # Interpolation check
                Z_weighted_norm = 0.99 * zm0 + (1 - 0.99) * z0
                interpolation_check_norm = torch.norm(
                    z_interp - Z_weighted_norm, dim=1).mean()

                Z_weighted_raw = 0.99 * zm0_raw + (1 - 0.99) * z0_raw
                interpolation_check_raw = torch.norm(
                    Z_interp_raw - Z_weighted_raw, dim=1).mean()

                # Lemma 2: H(Z_theta^{t+1}) >= H(Z_theta^t)
                h_teacher_next = kde_entropy(z_interp)

                metrics["interpolation_check_norm"] = interpolation_check_norm
                metrics["interpolation_check_raw"] = interpolation_check_raw
                metrics["h_teacher_next"] = h_teacher_next

                # =============================================
                # SECTION 5.2: Lemma (innovation covariance structure)
                # =============================================
                z0_c = (z0 - z0.mean(dim=0, keepdim=True)).float()
                zm0_c = (zm0 - zm0.mean(dim=0, keepdim=True)).float()
                n = z0.shape[0]

                Sigma_phi = (z0_c.T @ z0_c) / (n - 1)
                Sigma_theta = (zm0_c.T @ zm0_c) / (n - 1)
                Sigma_phi_theta = (z0_c.T @ zm0_c) / (n - 1)

                # (i) cross-covariance = teacher covariance
                cross_cov_error = torch.norm(
                    Sigma_phi_theta - Sigma_theta, 'fro'
                ) / (torch.norm(Sigma_theta, 'fro') + 1e-10)
                metrics["cross_cov_equals_teacher_cov"] = cross_cov_error.item()

                # (iii) Sigma_phi - Sigma_theta PSD check
                innovation_cov = Sigma_phi - Sigma_theta
                innovation_cov = 0.5 * (innovation_cov + innovation_cov.T)
                eigvals = torch.linalg.eigvalsh(innovation_cov.float())

                metrics["innovation_cov_min_eigval"] = eigvals.min().item()
                metrics["innovation_cov_psd_frac"] = (
                        eigvals >= -1e-6).float().mean().item()

                # Conditional covariance checks
                diff_teacher = ((zm0 - zm1) / math.sqrt(2)).float()
                diff_teacher_c = diff_teacher - diff_teacher.mean(dim=0, keepdim=True)
                Sigma_theta_X = (diff_teacher_c.T @ diff_teacher_c) / (n - 1)

                diff_student = ((z0 - z1) / math.sqrt(2)).float()
                diff_student_c = diff_student - diff_student.mean(dim=0, keepdim=True)
                Sigma_phi_X = (diff_student_c.T @ diff_student_c) / (n - 1)

                Sigma_phi_theta_X = (diff_student_c.T @ diff_teacher_c) / (n - 1)

                cond_cross_cov_error = torch.norm(
                    Sigma_phi_theta_X - Sigma_theta_X, 'fro'
                ) / (torch.norm(Sigma_theta_X, 'fro') + 1e-10)
                metrics["cond_cross_cov_equals_teacher_cond_cov"] = (
                    cond_cross_cov_error.item())

                cond_innovation_cov = Sigma_phi_X - Sigma_theta_X
                cond_innovation_cov = 0.5 * (
                        cond_innovation_cov + cond_innovation_cov.T)
                cond_eigvals = torch.linalg.eigvalsh(cond_innovation_cov.float())

                metrics["cond_innovation_cov_min_eigval"] = (
                    cond_eigvals.min().item())
                metrics["cond_innovation_cov_psd_frac"] = (
                        cond_eigvals >= -1e-6).float().mean().item()

                # =============================================
                # SECTION 5.1: Assumption 4 (Gaussianity)
                # =============================================
                from scipy.stats import anderson, normaltest

                for name, tensor in [("student", z0), ("teacher", zm0)]:
                    t_np = tensor.detach().cpu().numpy()
                    n_samples, d_dim = t_np.shape

                    ad_stats = []
                    dp_pvals = []

                    for j in range(d_dim):
                        col = t_np[:, j]
                        col_std = (col - col.mean()) / (col.std() + 1e-10)

                        ad_result = anderson(col_std, dist='norm')
                        ad_stats.append(ad_result.statistic)

                        if n_samples >= 20:
                            _, dp_pval = normaltest(col)
                            dp_pvals.append(dp_pval)

                    ad_stats = np.array(ad_stats)
                    metrics[f"{name}_coord_ad_frac_gaussian"] = float(
                        (ad_stats < 0.752).mean())
                    metrics[f"{name}_coord_ad_avg"] = float(ad_stats.mean())

                    if len(dp_pvals) > 0:
                        dp_pvals = np.array(dp_pvals)
                        metrics[f"{name}_coord_dp_frac_gaussian"] = float(
                            (dp_pvals > 0.05).mean())
                        metrics[f"{name}_coord_dp_avg_pval"] = float(
                            dp_pvals.mean())

        self.log_dict(metrics, on_epoch=True, sync_dist=False)

        return neg_cos_sim + self.au_scale_loss * au_loss + class_loss

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

