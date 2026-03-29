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
            # L2-normalize once, reuse everywhere
            z0 = F.normalize(Z[0], dim=-1)
            z1 = F.normalize(Z[1], dim=-1)
            zm0 = F.normalize(Z_momentum[0], dim=-1)
            zm1 = F.normalize(Z_momentum[1], dim=-1)
            p0 = F.normalize(P[0], dim=-1)
            p1 = F.normalize(P[1], dim=-1)

            # =============================================
            # CORE TRAINING METRICS (every step)
            # =============================================
            z_std = torch.stack([z0, z1]).std(dim=1).mean()
            z_std_teacher = torch.stack([zm0, zm1]).std(dim=1).mean()

            student_entropy = (uniform_loss_func(z0) + uniform_loss_func(z1)) / 2
            teacher_entropy = (uniform_loss_func(zm0) + uniform_loss_func(zm1)) / 2
            predictor_entropy = (uniform_loss_func(p0) + uniform_loss_func(p1)) / 2

            student_alignment = align_loss_func(z0, z1)
            teacher_alignment = align_loss_func(zm0, zm1)
            predictor_alignment = align_loss_func(p0, p1)

            # =============================================
            # CORRELATION rho^t and rho_X^t (every step)
            # Needed for: Corollary 1, entropy/alignment dynamics
            # =============================================

            # -- Marginal rho^t (isotropic estimate) --
            rho = (estimate_rho_isotropic(z0, zm1) +
                   estimate_rho_isotropic(z1, zm0)) / 2

            # -- Conditional rho_X^t (proxy via augmentation-pair norms) --
            norm_s = torch.linalg.norm(z0 - z1, dim=1)
            norm_t = torch.linalg.norm(zm0 - zm1, dim=1)
            ns = norm_s - norm_s.mean()
            nt = norm_t - norm_t.mean()
            rho_x = (ns @ nt) / (torch.norm(ns) * torch.norm(nt) + 1e-10)

            # =============================================
            # VARIANCE RATIOS r and r_X (every step)
            # Needed for: entropy/alignment dynamics interpretation
            # =============================================
            sigma_student = z0.var(dim=0).mean().sqrt()
            sigma_teacher = zm0.var(dim=0).mean().sqrt()
            r_marginal = sigma_student / (sigma_teacher + 1e-10)

            # Conditional variance proxy: Var(z(v1) - z(v2)) / 2
            cond_var_student = (z0 - z1).pow(2).mean(dim=0).mean() / 2
            cond_var_teacher = (zm0 - zm1).pow(2).mean(dim=0).mean() / 2
            r_X = (cond_var_student / (cond_var_teacher + 1e-10)).sqrt()

            # =============================================
            # LEMMA 1 CHECK: sigma_{t+1}/sigma_t >= tau (every step)
            # =============================================
            sigma_ratio = sigma_teacher  # will compare with previous step
            # (store self._prev_sigma_teacher to compute ratio across steps)
            if hasattr(self, '_prev_sigma_teacher') and self._prev_sigma_teacher > 0:
                lemma1_ratio = sigma_teacher / self._prev_sigma_teacher
            else:
                lemma1_ratio = torch.tensor(1.0)
            self._prev_sigma_teacher = sigma_teacher.clone()

            # =============================================
            # RESIDUAL DIAGNOSTICS (every step)
            # Needed for: Assumption 1 (predictor-as-additive-map)
            # =============================================
            residual_01 = zm1 - p0  # teacher v2 - predictor(student v1)
            residual_10 = zm0 - p1

            residual_norm = (residual_01.norm(dim=1).mean() +
                             residual_10.norm(dim=1).mean()) / 2
            residual_var = residual_01.var(dim=0).mean()

            # Independence check: |corr(residual, student)| should be ~0
            rho_residual_student = abs(estimate_rho_isotropic(residual_01, z0))

            # =============================================
            # ASSUMPTION 2 CHECK: linear interpolation of outputs
            # =============================================
            interpolated_backbone = interpolate_network(
                self.backbone, self.momentum_backbone, 0.99)
            interpolated_projector = interpolate_network(
                self.projector, self.momentum_projector, 0.99)

            X_crop = batch[1][0]
            feats_interp = interpolated_backbone(X_crop)
            Z_interp_net = F.normalize(interpolated_projector(feats_interp), dim=-1)
            Z_weighted = 0.99 * zm0 + (1 - 0.99) * z0
            interpolation_check = torch.norm(Z_interp_net - Z_weighted, dim=1).mean()

        metrics = {
            # Core training
            "train_neg_cos_sim": neg_cos_sim,
            "train_z_std": z_std,
            "train_z_std_teacher": z_std_teacher,
            "train_student_entropy": student_entropy,
            "train_teacher_entropy": teacher_entropy,
            "train_predictor_entropy": predictor_entropy,
            "train_student_alignment": student_alignment,
            "train_teacher_alignment": teacher_alignment,
            "train_predictor_alignment": predictor_alignment,
            # Correlations (Corollary 1)
            "rho": rho,
            "rho_x": rho_x,
            # Variance ratios (entropy/alignment dynamics)
            "r_marginal": r_marginal,
            "r_X": r_X,
            "sigma_student": sigma_student,
            "sigma_teacher": sigma_teacher,
            # Lemma 1
            "lemma1_sigma_ratio": lemma1_ratio,
            # Assumption 1 (residuals)
            "residual_norm": residual_norm,
            "residual_var": residual_var,
            "rho_residual_student": rho_residual_student,
            # Assumption 2 (interpolation)
            "interpolation_check": interpolation_check,
        }

        # =============================================
        # EXPENSIVE DIAGNOSTICS (every N steps)
        # Needed for: Assumption 4 (Gaussianity + isotropy)
        # =============================================
        if batch_idx % 100 == 0:
            with torch.no_grad():
                # -- Covariance structure (isotropy vs diagonal) --
                for name, tensor in [("student", z0), ("teacher", zm0), ("residual", residual_01)]:
                    t_np = tensor.detach().cpu().numpy()
                    t_centered = t_np - t_np.mean(axis=0, keepdims=True)
                    cov = (t_centered.T @ t_centered) / (t_np.shape[0] - 1)
                    eigvals = np.linalg.eigvalsh(cov)[::-1]

                    # Condition number (isotropic => 1)
                    metrics[f"{name}_condition_number"] = eigvals[0] / (eigvals[-1] + 1e-10)

                    # Eigenvalue CV (isotropic => 0)
                    metrics[f"{name}_eigval_cv"] = eigvals.std() / (eigvals.mean() + 1e-10)

                    # Effective rank ratio (isotropic => 1.0)
                    p = eigvals / (eigvals.sum() + 1e-10)
                    p = p[p > 1e-12]
                    eff_rank = np.exp(-np.sum(p * np.log(p)))
                    metrics[f"{name}_effective_rank_ratio"] = eff_rank / tensor.shape[1]

                    # Off-diagonal Frobenius ratio (diagonal => 0)
                    diag_part = np.diag(np.diag(cov))
                    offdiag_frob = np.linalg.norm(cov - diag_part, 'fro')
                    total_frob = np.linalg.norm(cov, 'fro')
                    metrics[f"{name}_offdiag_frob_ratio"] = offdiag_frob / (total_frob + 1e-10)

                # -- Per-dimension rho (isotropy of cross-covariance) --
                z0_c = z0 - z0.mean(dim=0, keepdim=True)
                zm1_c = zm1 - zm1.mean(dim=0, keepdim=True)
                B = z0.shape[0]
                cov_per_dim = (z0_c * zm1_c).sum(dim=0) / (B - 1)
                std_z = z0_c.pow(2).sum(dim=0).div(B - 1).sqrt()
                std_zm = zm1_c.pow(2).sum(dim=0).div(B - 1).sqrt()
                rho_per_dim = cov_per_dim / (std_z * std_zm + 1e-10)
                rho_np = rho_per_dim.cpu().numpy()

                metrics["rho_diag_mean"] = rho_np.mean()
                metrics["rho_diag_std"] = rho_np.std()
                metrics["rho_diag_min"] = rho_np.min()
                metrics["rho_frac_positive"] = (rho_np > 0).mean()

        # =============================================
        # GAUSSIANITY CHECK (every M steps, expensive)
        # Needed for: Assumption 4 (marginal Gaussianity)
        # Uses Cramer-Wold: project onto random 1D directions, test normality
        # =============================================
        if batch_idx % 500 == 0:
            with torch.no_grad():
                num_projections = 20

                for name, tensor in [("student", z0), ("teacher", zm0), ("residual", residual_01)]:
                    t_np = tensor.detach().cpu().numpy()
                    n, d = t_np.shape
                    t_centered = t_np - t_np.mean(axis=0, keepdims=True)

                    pvals = []
                    kurtoses = []
                    skewnesses = []

                    for _ in range(num_projections):
                        v = np.random.randn(d)
                        v /= np.linalg.norm(v)
                        proj = t_centered @ v
                        proj_std = (proj - proj.mean()) / (proj.std() + 1e-10)

                        # Shapiro-Wilk (subsample if large batch)
                        if n > 5000:
                            idx = np.random.choice(n, 5000, replace=False)
                            _, pval = shapiro(proj_std[idx])
                        else:
                            _, pval = shapiro(proj_std)
                        pvals.append(pval)

                        # Excess kurtosis (Gaussian = 0)
                        kurtoses.append(float(np.mean(proj_std ** 4) - 3.0))
                        # Skewness (Gaussian = 0)
                        skewnesses.append(float(np.mean(proj_std ** 3)))

                    metrics[f"{name}_gauss_shapiro_median_pval"] = np.median(pvals)
                    metrics[f"{name}_gauss_rejection_rate"] = np.mean(np.array(pvals) < 0.05)
                    metrics[f"{name}_gauss_excess_kurtosis"] = np.mean(kurtoses)
                    metrics[f"{name}_gauss_kurtosis_std"] = np.std(kurtoses)
                    metrics[f"{name}_gauss_skewness"] = np.mean(skewnesses)

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
