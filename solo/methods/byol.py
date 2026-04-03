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
from scipy.stats import shapiro

import numpy as np
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from solo.losses.byol import byol_loss_func
from solo.losses.robyol import uniform_loss_func, align_loss_func
from solo.methods.base import BaseMomentumMethod
from solo.utils.momentum import initialize_momentum_params

import torch.nn.init as init

def kaiming_init(module):
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            init.zeros_(module.bias)

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

        #self.momentum_backbone.apply(kaiming_init)
        #self.projector.apply(kaiming_init)

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

        # ------- diagnostics (no grad) -------
        with torch.no_grad():

            # ==========================================================
            # RAW projector outputs (for Section 4: EMA dynamics)
            # The entropy/alignment dynamics operate in pre-normalization
            # space. The connection to the normalized loss is via the
            # Gaussian-vMF equivalence (Sec. 3.1).
            # ==========================================================
            z0_raw = Z[0]  # [B, d] raw projector output
            z1_raw = Z[1]
            zm0_raw = Z_momentum[0]  # [B, d] raw momentum projector output
            zm1_raw = Z_momentum[1]
            p0_raw = P[0]  # [B, d] raw predictor output
            p1_raw = P[1]

            # ==========================================================
            # L2-NORMALIZED representations (for Section 3: InfoMax bound)
            # The BYOL loss, entropy/alignment estimators, and correlation
            # rho^t are computed on the unit sphere.
            # ==========================================================
            z0 = F.normalize(Z[0], dim=-1)
            z1 = F.normalize(Z[1], dim=-1)
            zm0 = F.normalize(Z_momentum[0], dim=-1)
            zm1 = F.normalize(Z_momentum[1], dim=-1)
            p0 = F.normalize(P[0], dim=-1)
            p1 = F.normalize(P[1], dim=-1)

            # =============================================
            # SECTION 3 METRICS: on normalized representations
            # =============================================

            # -- Core training metrics --
            z_std = torch.stack([z0, z1]).std(dim=1).mean()
            z_std_teacher = torch.stack([zm0, zm1]).std(dim=1).mean()

            student_entropy = (uniform_loss_func(z0) + uniform_loss_func(z1)) / 2
            teacher_entropy = (uniform_loss_func(zm0) + uniform_loss_func(zm1)) / 2
            predictor_entropy = (uniform_loss_func(p0) + uniform_loss_func(p1)) / 2

            student_alignment = align_loss_func(z0, z1)
            teacher_alignment = align_loss_func(zm0, zm1)
            predictor_alignment = align_loss_func(p0, p1)

            # -- Corollary 1: rho^t and rho_X^t (on normalized) --
            rho = (estimate_rho_isotropic(z0, zm1) +
                   estimate_rho_isotropic(z1, zm0)) / 2

            norm_s = torch.linalg.norm(z0 - z1, dim=1)
            norm_t = torch.linalg.norm(zm0 - zm1, dim=1)
            ns = norm_s - norm_s.mean()
            nt = norm_t - norm_t.mean()
            rho_x = (ns @ nt) / (torch.norm(ns) * torch.norm(nt) + 1e-10)

            # -- Residual on normalized (what the loss actually minimizes) --
            residual_norm = ((zm1 - p0).norm(dim=1).mean() +
                             (zm0 - p1).norm(dim=1).mean()) / 2

            # =============================================
            # SECTION 4 METRICS: on RAW projector outputs
            # =============================================

            # -- Variance ratios r and r_X --
            sigma_student_raw = z0_raw.var(dim=0).mean().sqrt()
            sigma_teacher_raw = zm0_raw.var(dim=0).mean().sqrt()
            r_marginal = sigma_student_raw / (sigma_teacher_raw + 1e-10)

            cond_var_student_raw = (z0_raw - z1_raw).pow(2).mean(dim=0).mean() / 2
            cond_var_teacher_raw = (zm0_raw - zm1_raw).pow(2).mean(dim=0).mean() / 2
            r_X = (cond_var_student_raw / (cond_var_teacher_raw + 1e-10)).sqrt()

            # -- Lemma 2 check: sigma_{t+1}/sigma_t >= tau --
            if hasattr(self, '_prev_sigma_teacher_raw') and self._prev_sigma_teacher_raw > 0:
                lemma2_ratio = sigma_teacher_raw / self._prev_sigma_teacher_raw
            else:
                lemma2_ratio = torch.tensor(1.0)
            self._prev_sigma_teacher_raw = sigma_teacher_raw.clone()

            # -- Assumption 1: residual diagnostics on raw outputs --
            residual_01_raw = zm1_raw - p0_raw
            residual_10_raw = zm0_raw - p1_raw

            residual_var_raw = residual_01_raw.var(dim=0).mean()

            # Independence check: |corr(G_t, Z_phi)| should be ~0
            # Under Gaussian residual assumption, uncorrelated => independent
            rho_residual_student_raw = abs(estimate_rho_isotropic(
                residual_01_raw, z0_raw))

            # -- Assumption 2 check: linear interpolation of outputs --
            interpolated_backbone = interpolate_network(
                self.backbone, self.momentum_backbone, 0.99)
            interpolated_projector = interpolate_network(
                self.projector, self.momentum_projector, 0.99)

            X_crop = batch[1][0]
            feats_interp = interpolated_backbone(X_crop)
            Z_interp_net = interpolated_projector(feats_interp)  # raw, no normalize
            Z_weighted_raw = 0.99 * zm0_raw + (1 - 0.99) * z0_raw
            interpolation_check = torch.norm(
                Z_interp_net - Z_weighted_raw, dim=1).mean()

        metrics = {
            # Section 3: on normalized
            "train_neg_cos_sim": neg_cos_sim,
            "train_z_std": z_std,
            "train_z_std_teacher": z_std_teacher,
            "train_student_entropy": student_entropy,
            "train_teacher_entropy": teacher_entropy,
            "train_predictor_entropy": predictor_entropy,
            "train_student_alignment": student_alignment,
            "train_teacher_alignment": teacher_alignment,
            "train_predictor_alignment": predictor_alignment,
            "rho": rho,
            "rho_x": rho_x,
            "residual_norm": residual_norm,
            # Section 4: on raw outputs
            "r_marginal": r_marginal,
            "r_X": r_X,
            "sigma_student_raw": sigma_student_raw,
            "sigma_teacher_raw": sigma_teacher_raw,
            "lemma2_sigma_ratio": lemma2_ratio,
            "residual_var_raw": residual_var_raw,
            "rho_residual_student_raw": rho_residual_student_raw,
            "interpolation_check": interpolation_check,
        }

        # =============================================
        # EXPENSIVE DIAGNOSTICS (every N steps)
        # All on RAW outputs (Section 4 assumptions)
        # =============================================
        if batch_idx % 100 == 0:
            with torch.no_grad():

                # -- Per-dimension rho on raw outputs --
                # Needed for: Corollary 1 under diagonal covariance
                z0r_c = z0_raw - z0_raw.mean(dim=0, keepdim=True)
                zm1r_c = zm1_raw - zm1_raw.mean(dim=0, keepdim=True)
                B = z0_raw.shape[0]
                cov_per_dim = (z0r_c * zm1r_c).sum(dim=0) / (B - 1)
                std_z = z0r_c.pow(2).sum(dim=0).div(B - 1).sqrt()
                std_zm = zm1r_c.pow(2).sum(dim=0).div(B - 1).sqrt()
                rho_per_dim = cov_per_dim / (std_z * std_zm + 1e-10)
                rho_np = rho_per_dim.cpu().numpy()

                metrics["rho_diag_std"] = float(rho_np.std())
                metrics["rho_diag_min"] = float(rho_np.min())
                metrics["rho_frac_positive"] = float((rho_np > 0).mean())

                # -- Diagonality and Gaussianity checks on raw outputs --
                residual_01_raw_detached = residual_01_raw  # already in no_grad

                for name, tensor in [("student", z0_raw),
                                     ("teacher", zm0_raw),
                                     ("residual", residual_01_raw_detached)]:
                    t_np = tensor.detach().cpu().numpy()
                    n, d = t_np.shape
                    t_centered = t_np - t_np.mean(axis=0, keepdims=True)
                    cov = (t_centered.T @ t_centered) / (n - 1)

                    # --- Diagonality: off-diagonal Frobenius ratio ---
                    # diagonal => 0, full covariance => large
                    diag_part = np.diag(np.diag(cov))
                    offdiag_frob = np.linalg.norm(cov - diag_part, 'fro')
                    total_frob = np.linalg.norm(cov, 'fro')
                    metrics[f"{name}_offdiag_frob_ratio"] = float(
                        offdiag_frob / (total_frob + 1e-10))

                    # --- Mardia's multivariate kurtosis ratio ---
                    # Gaussian => ratio = 1.0
                    cov_reg = cov + 1e-6 * np.eye(d)
                    cov_inv = np.linalg.inv(cov_reg)
                    mahal_sq = np.sum(
                        (t_centered @ cov_inv) * t_centered, axis=1)
                    mardia_kurtosis = np.mean(mahal_sq ** 2)
                    mardia_expected = d * (d + 2)
                    metrics[f"{name}_mardia_kurtosis_ratio"] = float(
                        mardia_kurtosis / mardia_expected)

                    # --- Henze-Zirkler test ---
                    # Raw outputs are full-rank, so HZ should not fail.
                    # Note: HZ is very powerful in high-d; low p-values
                    # are expected even for "approximately Gaussian" data.
                    try:
                        _, hz_pvalue, _ = pg.multivariate_normality(
                            t_np, alpha=0.05)
                        metrics[f"{name}_hz_pvalue"] = float(hz_pvalue)
                    except Exception:
                        metrics[f"{name}_hz_pvalue"] = float('nan')

        self.log_dict(metrics, on_epoch=True, sync_dist=False)

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
