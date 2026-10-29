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
import copy

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

    # =====================================================================
    # training_step
    # =====================================================================
    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """Training step for BYOL with information-theoretic diagnostics.

        Every step (cheap):  entropies, alignments, cross-prediction, uniformity,
                             Assumption 2 (RV coefficient of tangential TSD vs teacher).
        Every N steps (expensive): Assumption 1 (interpolation check, needs an extra forward
                             through an interpolated network), Lemma 1 (one-step entropy change),
                             Gaussianity tests.
        """
        out = super().training_step(batch, batch_idx)
        class_loss = out["loss"]
        Z = out["z"]
        P = out["p"]
        Z_momentum = out["momentum_z"]

        # ------- BYOL loss -------
        neg_cos_sim = 0
        for v1 in range(self.num_large_crops):
            for v2 in np.delete(range(self.num_crops), v1):
                neg_cos_sim += byol_loss_func(P[v2], Z_momentum[v1])

        metrics = {"train_neg_cos_sim": neg_cos_sim}

        # =================================================================
        # EVERY STEP: cheap diagnostics
        # =================================================================
        # autocast OFF: otherwise matmuls / cdist silently run in fp16 despite .float()
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=False), full_fp32():
            # l2-normalized representations, cast to fp32 for stable statistics
            z0 = F.normalize(Z[0].float(), dim=-1)  # student, view 1
            z1 = F.normalize(Z[1].float(), dim=-1)  # student, view 2
            zm0 = F.normalize(Z_momentum[0].float(), dim=-1)  # teacher, view 1
            zm1 = F.normalize(Z_momentum[1].float(), dim=-1)  # teacher, view 2
            p0 = F.normalize(P[0].float(), dim=-1)  # predictor, view 1
            p1 = F.normalize(P[1].float(), dim=-1)  # predictor, view 2

            # ---- L_bound components (Sec. 4 / 6) ----
            metrics["h_teacher"] = (kde_entropy(zm0) + kde_entropy(zm1)) / 2
            metrics["h_student"] = (kde_entropy(z0) + kde_entropy(z1)) / 2
            metrics["h_predictor"] = (kde_entropy(p0) + kde_entropy(p1)) / 2

            # alignment = H(Z|X) proxy: E||z(v1) - z(v2)||^2 / 2
            metrics["teacher_alignment"] = (zm0 - zm1).pow(2).sum(1).mean() / 2
            metrics["student_alignment"] = (z0 - z1).pow(2).sum(1).mean() / 2
            metrics["predictor_alignment"] = (p0 - p1).pow(2).sum(1).mean() / 2

            # TS cross-prediction MSE (= BYOL loss up to constants)
            metrics["cross_prediction_mse"] = (
                                                      (zm1 - p0).pow(2).sum(1).mean() + (zm0 - p1).pow(2).sum(1).mean()
                                              ) / 2

            # uniformity (entropy surrogate)
            metrics["train_student_uniformity"] = (uniform_loss_func(z0) + uniform_loss_func(z1)) / 2
            metrics["train_teacher_uniformity"] = (uniform_loss_func(zm0) + uniform_loss_func(zm1)) / 2
            metrics["train_predictor_uniformity"] = (uniform_loss_func(p0) + uniform_loss_func(p1)) / 2

            # ---- Assumption 2: tangential TSD vs teacher (linear signature) ----
            eta = tangential_tsd(z0, zm0)
            perm = torch.randperm(eta.shape[0], device=eta.device)
            metrics["a2_rv"] = rv_coefficient(eta, zm0)
            metrics["a2_rv_null"] = rv_coefficient(eta[perm], zm0)  # independence baseline
            # rolling buffer of (teacher, TSD) pairs for the R^2 diagnostic (needs n >> d)
            if not hasattr(self, "_r2_buffer"):
                self._r2_buffer = []
            self._r2_buffer.append((zm0.detach(), eta.detach()))
            r2_samples = getattr(self, "r2_samples", 10240)
            while sum(b[0].shape[0] for b in self._r2_buffer) > r2_samples:
                self._r2_buffer.pop(0)

        # =================================================================
        # EVERY N STEPS: expensive diagnostics
        # =================================================================
        diag_every = getattr(self, "diag_every", 100)  # use ~10 for small (medical) datasets
        if batch_idx % diag_every == 0:
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=False), full_fp32():
                tau = 0.99  # EMA coefficient (fixed in our runs)
                c = 1.0 - tau
                X_crop = batch[1][0].float()  # same crop as Z[0] / Z_momentum[0]

                # All diagnostic forwards run on COPIES of the networks, so the live student's
                # BatchNorm buffers (saved by autograd for the backward pass) are never modified.
                #   tau = 1 -> copy of the teacher,  tau = 0 -> copy of the student
                def forward_copy(t):
                    bb = interpolate_network(self.backbone, self.momentum_backbone, t)
                    prj = interpolate_network(self.projector, self.momentum_projector, t)
                    z = F.normalize(prj(bb(X_crop)), dim=-1)
                    del bb, prj
                    return z

                # fp32 recompute so that all quantities share the same precision
                zm0_32 = forward_copy(1.0)  # teacher
                z0_32 = forward_copy(0.0)  # student
                eta_32 = tangential_tsd(z0_32, zm0_32)

                # ---- Assumption 1: first-order EMA step on the sphere ----
                z_interp = forward_copy(tau)  # tau * teacher + (1 - tau) * student

                # numerical noise floor: a second teacher copy should give identical outputs;
                # a1_rel_err is only meaningful when a1_step_size >> a1_noise_floor
                metrics["a1_noise_floor"] = torch.norm(forward_copy(1.0) - zm0_32, dim=1).mean()

                z_pred = F.normalize(zm0_32 + c * eta_32, dim=-1)
                step_size = torch.norm(z_interp - zm0_32, dim=1).mean()  # how far the teacher moves
                abs_err = torch.norm(z_interp - z_pred, dim=1).mean()
                metrics["a1_step_size"] = step_size
                metrics["a1_abs_err"] = abs_err
                metrics["a1_rel_err"] = abs_err / (step_size + 1e-12)  # ~0 good, ~1 useless

                # ---- Lemma 1: one-step teacher entropy change (same view, same precision) ----
                metrics["lemma1_h_diff"] = kde_entropy(z_interp) - kde_entropy(zm0_32)

                # ---- Assumption 2: how much of the TSD the teacher explains linearly ----
                # computed on the rolling buffer (~10k samples); only once it is full enough
                buf_z = torch.cat([b[0] for b in self._r2_buffer])
                buf_eta = torch.cat([b[1] for b in self._r2_buffer])
                if buf_z.shape[0] >= 0.9 * getattr(self, "r2_samples", 10240):
                    metrics["a2_r2"] = linear_r2(buf_z, buf_eta)

            with torch.no_grad():
                # ---- Gaussianity (Sec. 8.1), per-coordinate AD / DP tests (Betser et al.) ----
                from scipy.stats import anderson, normaltest

                for name, tensor in [("student", z0), ("teacher", zm0)]:
                    t_np = tensor.cpu().numpy()
                    n_samples, d_dim = t_np.shape
                    ad_stats, dp_pvals = [], []
                    for j in range(d_dim):
                        col = t_np[:, j]
                        col_std = (col - col.mean()) / (col.std() + 1e-10)
                        ad_stats.append(anderson(col_std, dist="norm").statistic)
                        if n_samples >= 20:
                            dp_pvals.append(normaltest(col)[1])
                    ad_stats = np.array(ad_stats)
                    metrics[f"{name}_coord_ad_frac_gaussian"] = float((ad_stats < 0.752).mean())
                    metrics[f"{name}_coord_ad_avg"] = float(ad_stats.mean())
                    if dp_pvals:
                        dp_pvals = np.array(dp_pvals)
                        metrics[f"{name}_coord_dp_frac_gaussian"] = float((dp_pvals > 0.05).mean())
                        metrics[f"{name}_coord_dp_avg_pval"] = float(dp_pvals.mean())

        self.log_dict(metrics, on_epoch=True, sync_dist=False)
        return neg_cos_sim + class_loss


# =====================================================================
# Module-level helpers (define once, outside the class)
# =====================================================================
import math
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F

@contextmanager
def full_fp32():
    """Disable TF32 (on by default for convolutions on A100/H100) during diagnostics only."""
    old = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old

def kde_entropy(z, sigma=1.0):
    """KDE (resubstitution) entropy estimate on l2-normalized representations.
    H(Z) ~ -1/N sum_i log( 1/(N-1) sum_{j != i} exp(-||z_i - z_j||^2 / (2 sigma^2)) )
    """
    z = z.float()
    dists_sq = torch.cdist(z, z, p=2).pow(2)
    B = z.shape[0]
    mask = ~torch.eye(B, dtype=torch.bool, device=z.device)
    log_density = torch.logsumexp(
        -dists_sq[mask].view(B, B - 1) / (2 * sigma ** 2), dim=1
    ) - math.log(B - 1)
    return -log_density.mean()

def tangential_tsd(z_student, z_teacher):
    """Tangential teacher-student difference: part of the student orthogonal to the teacher.
    eta = Z_phi - <Z_phi, Z_theta> Z_theta   (both inputs l2-normalized)
    """
    return z_student - (z_student * z_teacher).sum(-1, keepdim=True) * z_teacher

def linear_r2(x, y, alpha=1e-3):
    """Held-out fraction of y's variance explained by a linear (ridge) map of x.
    ~0 (or slightly negative): no linear dependence;  1: y is exactly linear in x.
    Needs n >> d (use ~10k samples for d = 256): random 50/50 train/test split."""
    x, y = x.float(), y.float()
    idx = torch.randperm(x.shape[0], device=x.device)
    x, y = x[idx], y[idx]
    n = x.shape[0] // 2
    xtr, xte, ytr, yte = x[:n], x[n:], y[:n], y[n:]
    mx, my = xtr.mean(0), ytr.mean(0)
    Xc, Yc = xtr - mx, ytr - my
    G = Xc.T @ Xc
    lam = alpha * torch.trace(G) / x.shape[1]
    eye = torch.eye(x.shape[1], device=x.device, dtype=x.dtype)
    W = torch.linalg.solve(G + lam * eye, Xc.T @ Yc)
    pred = (xte - mx) @ W + my
    return 1 - ((yte - pred) ** 2).sum() / (((yte - yte.mean(0)) ** 2).sum() + 1e-12)

def rv_coefficient(x, y):
    """RV coefficient in [0, 1]; 0 iff all cross-covariances between x and y vanish."""
    x, y = x.float(), y.float()
    xc, yc = x - x.mean(0), y - y.mean(0)
    Cxy, Cxx, Cyy = xc.T @ yc, xc.T @ xc, yc.T @ yc
    return (Cxy * Cxy).sum() / torch.sqrt((Cxx * Cxx).sum() * (Cyy * Cyy).sum() + 1e-12)


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
