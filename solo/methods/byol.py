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

from scipy.stats import anderson, normaltest
import math

import penguin as pg

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

            # L2-normalize representations (on the hypersphere)
            z0 = F.normalize(Z[0], dim=-1)
            z1 = F.normalize(Z[1], dim=-1)
            zm0 = F.normalize(Z_momentum[0], dim=-1)
            zm1 = F.normalize(Z_momentum[1], dim=-1)
            p0 = F.normalize(P[0], dim=-1)
            p1 = F.normalize(P[1], dim=-1)

            # Raw projector outputs (pre-normalization)
            z0_raw = Z[0]
            z1_raw = Z[1]
            zm0_raw = Z_momentum[0]
            zm1_raw = Z_momentum[1]

            # =============================================
            # CORE TRAINING METRICS (every step, normalized)
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
            # KDE ENTROPY ESTIMATOR (every step, normalized)
            # H(Z) ≈ -1/N sum_i log (1/(N-1) sum_{j!=i} exp(-||z_i - z_j||^2 / (2*sigma^2)))
            # This is the resubstitution entropy estimator with Gaussian kernel.
            # We use sigma=1 for consistency with the BYOL loss derivation.
            # =============================================
            def kde_entropy(z, sigma=1.0):
                """KDE entropy estimator on normalized representations."""
                # z: [B, d], assumed l2-normalized
                # Pairwise squared distances
                dists_sq = torch.cdist(z, z, p=2).pow(2)  # [B, B]
                B = z.shape[0]
                # Mask diagonal
                mask = ~torch.eye(B, dtype=torch.bool, device=z.device)
                # Log-sum-exp over non-diagonal entries
                log_density = torch.logsumexp(
                    -dists_sq[mask].view(B, B - 1) / (2 * sigma ** 2), dim=1
                ) - math.log(B - 1)
                return -log_density.mean()

            h_teacher_v1 = kde_entropy(zm0)
            h_teacher_v2 = kde_entropy(zm1)
            h_student_v1 = kde_entropy(z0)
            h_student_v2 = kde_entropy(z1)
            h_predictor_v1 = kde_entropy(p0)
            h_predictor_v2 = kde_entropy(p1)

            h_teacher = (h_teacher_v1 + h_teacher_v2) / 2
            h_student = (h_student_v1 + h_student_v2) / 2
            h_predictor = (h_predictor_v1 + h_predictor_v2) / 2

            # =============================================
            # CORRELATIONS rho^t and rho_X^t (every step, normalized)
            # Needed for: Corollary 1, entropy/alignment dynamics
            # =============================================
            rho = (estimate_rho_isotropic(z0, zm1) +
                   estimate_rho_isotropic(z1, zm0)) / 2

            norm_s = torch.linalg.norm(z0 - z1, dim=1)
            norm_t = torch.linalg.norm(zm0 - zm1, dim=1)
            ns = norm_s - norm_s.mean()
            nt = norm_t - norm_t.mean()
            rho_x = (ns @ nt) / (torch.norm(ns) * torch.norm(nt) + 1e-10)

            # =============================================
            # VARIANCE RATIOS r and r_X (every step, normalized)
            # Needed for: entropy/alignment dynamics interpretation
            # =============================================
            sigma_student = z0.var(dim=0).mean().sqrt()
            sigma_teacher = zm0.var(dim=0).mean().sqrt()
            r_marginal = sigma_student / (sigma_teacher + 1e-10)

            cond_var_student = (z0 - z1).pow(2).mean(dim=0).mean() / 2
            cond_var_teacher = (zm0 - zm1).pow(2).mean(dim=0).mean() / 2
            r_X = (cond_var_student / (cond_var_teacher + 1e-10)).sqrt()

            # =============================================
            # LEMMA 2 CHECK (every step)
            # Compare H(Z_theta^{t+1}) with H(Z_theta^t)
            # using interpolated network to get Z_theta^{t+1}
            # =============================================
            d = z0.shape[1]

            # Build the interpolated (next-step) teacher
            interpolated_backbone = interpolate_network(
                self.backbone, self.momentum_backbone, 0.99)
            interpolated_projector = interpolate_network(
                self.projector, self.momentum_projector, 0.99)

            # Forward through interpolated teacher on view 1
            X_crop = batch[1][0]
            feats_interp = interpolated_backbone(X_crop)
            Z_interp_raw = interpolated_projector(feats_interp)
            z_interp = F.normalize(Z_interp_raw, dim=-1)

            # Assumption 2 check: ||z_theta^{t+1} - (tau * z_theta^t + (1-tau) * z_phi^t)||
            Z_weighted = 0.99 * zm0 + (1 - 0.99) * z0
            interpolation_check = torch.norm(z_interp - Z_weighted, dim=1).mean()

            # Lemma 2 entropy check: H(Z_theta^{t+1}) - H(Z_theta^t) >= d * log(tau)
            h_teacher_next = kde_entropy(z_interp)
            lemma2_entropy_diff = h_teacher_next - h_teacher_v1
            lemma2_threshold = d * math.log(0.99)  # d * log(tau)
            lemma2_holds = (lemma2_entropy_diff >= lemma2_threshold).float()

            # Also track sigma ratio as simpler check
            if hasattr(self, '_prev_sigma_teacher') and self._prev_sigma_teacher > 0:
                lemma2_sigma_ratio = sigma_teacher / self._prev_sigma_teacher
            else:
                lemma2_sigma_ratio = torch.tensor(1.0)
            self._prev_sigma_teacher = sigma_teacher.clone()

            # =============================================
            # ASSUMPTION 1: RESIDUAL DIAGNOSTICS (every step, normalized)
            # G_t := Z_theta - h_psi(Z_phi), need: small var, uncorrelated with Z_phi
            # Note: H(tau * G_t) = H(G_t) + d*log(tau) holds for ANY continuous G_t
            # so Gaussianity of G_t is NOT required. Only independence from Z_phi.
            # =============================================
            residual_01 = zm1 - p0
            residual_10 = zm0 - p1

            residual_norm = (residual_01.norm(dim=1).mean() +
                             residual_10.norm(dim=1).mean()) / 2
            residual_var = residual_01.var(dim=0).mean()

            # Under any distributional assumption on G_t:
            # uncorrelated + small variance is sufficient
            rho_residual_student = abs(estimate_rho_isotropic(residual_01, z0))

            # =============================================
            # SECTION 3.2: H(Z_theta) ≈ H(h_psi(Z_phi))  (Lemma 1)
            # and BYOL loss ≈ H(Z_theta | X)
            # =============================================
            # H(h_psi(Z_phi)) should track H(Z_theta) when prediction loss is small
            h_entropy_gap = abs(h_teacher - h_predictor)

            # BYOL loss (cross-view MSE) as proxy for H(Z_theta | Z_phi, X)
            byol_loss_proxy = ((zm1 - p0).pow(2).sum(dim=1).mean() +
                               (zm0 - p1).pow(2).sum(dim=1).mean()) / 2

            # H(Z_theta | X) estimated via alignment (same-input, different augmentations)
            h_teacher_given_x = (zm0 - zm1).pow(2).sum(dim=1).mean() / 2

            # =============================================
            # THIN-SHELL CHECK (every step, raw outputs)
            # CV of norms: small CV => normalization ≈ scaling
            # Justifies that Gaussian analysis transfers to the sphere
            # (Wegner 2021, Thm 1.2; Betser et al. 2026, Assumption 2)
            # =============================================
            norms_raw = torch.norm(zm0_raw, dim=1)
            cv_norms_teacher = norms_raw.std() / (norms_raw.mean() + 1e-10)

            norms_student_raw = torch.norm(z0_raw, dim=1)
            cv_norms_student = norms_student_raw.std() / (norms_student_raw.mean() + 1e-10)

        metrics = {
            # Core training
            "train_neg_cos_sim": neg_cos_sim,
            "train_z_std": z_std,
            "train_z_std_teacher": z_std_teacher,
            # Uniformity-based entropy (for backward compat with existing plots)
            "train_student_entropy": student_entropy,
            "train_teacher_entropy": teacher_entropy,
            "train_predictor_entropy": predictor_entropy,
            "train_student_alignment": student_alignment,
            "train_teacher_alignment": teacher_alignment,
            "train_predictor_alignment": predictor_alignment,
            # KDE entropy estimates (proper estimator)
            "h_teacher": h_teacher,
            "h_student": h_student,
            "h_predictor": h_predictor,
            # Section 3.2: entropy proximity and loss-alignment comparison
            "h_entropy_gap": h_entropy_gap,
            "byol_loss_proxy": byol_loss_proxy,
            "h_teacher_given_x": h_teacher_given_x,
            # Correlations (Corollary 1)
            "rho": rho,
            "rho_x": rho_x,
            # Variance ratios (entropy/alignment dynamics)
            "r_marginal": r_marginal,
            "r_X": r_X,
            "sigma_student": sigma_student,
            "sigma_teacher": sigma_teacher,
            # Lemma 2
            "lemma2_entropy_diff": lemma2_entropy_diff,
            "lemma2_threshold": lemma2_threshold,
            "lemma2_holds": lemma2_holds,
            "lemma2_sigma_ratio": lemma2_sigma_ratio,
            # Assumption 1 (residuals)
            "residual_norm": residual_norm,
            "residual_var": residual_var,
            "rho_residual_student": rho_residual_student,
            # Assumption 2 (interpolation)
            "interpolation_check": interpolation_check,
            # Thin-shell concentration
            "cv_norms_teacher": cv_norms_teacher,
            "cv_norms_student": cv_norms_student,
        }

        # =============================================
        # EXPENSIVE DIAGNOSTICS (every N steps)
        # =============================================
        if batch_idx % 100 == 0:
            with torch.no_grad():

                # -- Per-dimension rho (Corollary 1 under diagonal covariance) --
                z0_c = z0 - z0.mean(dim=0, keepdim=True)
                zm1_c = zm1 - zm1.mean(dim=0, keepdim=True)
                B = z0.shape[0]
                cov_per_dim = (z0_c * zm1_c).sum(dim=0) / (B - 1)
                std_z = z0_c.pow(2).sum(dim=0).div(B - 1).sqrt()
                std_zm = zm1_c.pow(2).sum(dim=0).div(B - 1).sqrt()
                rho_per_dim = cov_per_dim / (std_z * std_zm + 1e-10)
                rho_np = rho_per_dim.cpu().numpy()

                metrics["rho_diag_std"] = float(rho_np.std())
                metrics["rho_diag_min"] = float(rho_np.min())
                metrics["rho_frac_positive"] = float((rho_np > 0).mean())

                # =============================================
                # ASSUMPTION 4: Per-coordinate Gaussianity tests
                # Following Betser et al. (ICLR 2026):
                #   - Anderson-Darling: AD < 0.752 => cannot reject Gaussianity
                #   - D'Agostino-Pearson: p > 0.05 => cannot reject Gaussianity
                # Justified by Maxwell-Poincaré spherical CLT:
                #   uniform-like distributions on S^{d-1} have Gaussian marginals
                # =============================================
                for name, tensor in [("student", z0), ("teacher", zm0)]:
                    t_np = tensor.detach().cpu().numpy()
                    n, d_dim = t_np.shape

                    ad_stats = []
                    dp_pvals = []

                    for j in range(d_dim):
                        col = t_np[:, j]

                        # Anderson-Darling test
                        from scipy.stats import anderson, normaltest
                        ad_result = anderson(col, dist='norm')
                        ad_stats.append(ad_result.statistic)

                        # D'Agostino-Pearson test
                        if n >= 20:  # minimum sample size for normaltest
                            _, dp_pval = normaltest(col)
                            dp_pvals.append(dp_pval)

                    ad_stats = np.array(ad_stats)
                    dp_pvals = np.array(dp_pvals)

                    # AD: fraction of coords where AD < 0.752 (Gaussian acceptance)
                    metrics[f"{name}_ad_avg"] = float(ad_stats.mean())
                    metrics[f"{name}_ad_frac_gaussian"] = float(
                        (ad_stats < 0.752).mean())

                    # DP: fraction of coords where p > 0.05 (cannot reject Gaussian)
                    if len(dp_pvals) > 0:
                        metrics[f"{name}_dp_avg_pval"] = float(dp_pvals.mean())
                        metrics[f"{name}_dp_frac_gaussian"] = float(
                            (dp_pvals > 0.05).mean())

                # =============================================
                # DIAGONAL CHECK: mean absolute correlation vs sphere baseline
                # On S^{d-1}, expected pairwise correlation = -1/(d-1)
                # If observed |corr| ≈ 1/(d-1), dimensions are as uncorrelated
                # as the sphere geometry allows => "diagonal" in spherical sense
                # =============================================
                for name, tensor in [("student", z0), ("teacher", zm0)]:
                    t_np = tensor.detach().cpu().numpy()
                    n, d_dim = t_np.shape
                    t_centered = t_np - t_np.mean(axis=0, keepdims=True)
                    stds = t_centered.std(axis=0, keepdims=True) + 1e-10
                    t_standardized = t_centered / stds

                    # Correlation matrix
                    corr = (t_standardized.T @ t_standardized) / (n - 1)
                    np.fill_diagonal(corr, 0)

                    mean_abs_corr = np.abs(corr).mean()
                    sphere_baseline = 1.0 / (d_dim - 1)

                    metrics[f"{name}_mean_abs_corr"] = float(mean_abs_corr)
                    metrics[f"{name}_sphere_corr_baseline"] = float(sphere_baseline)
                    # Ratio close to 1 => correlations match sphere geometry
                    metrics[f"{name}_corr_ratio_to_baseline"] = float(
                        mean_abs_corr / (sphere_baseline + 1e-10))

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
