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

import torch
import torch.nn.functional as F

def align_loss_func(x, y, alpha=2):
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()

def uniform_loss_func(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean(-1).log().mean()


import torch


def uniform_loss_exclude_knn(x, t=2, k=1):
    """
    Uniformity loss with exclusion of k nearest neighbors (and self).

    Args:
        x: (N, D) tensor of L2-normalized embeddings
        t: temperature
        k: number of nearest neighbors to exclude (in addition to self)

    Returns:
        Scalar uniformity loss
    """
    # Compute pairwise squared Euclidean distances
    dist_sq = torch.cdist(x, x, p=2).pow(2)  # (N, N)

    # Fill diagonal with large value to avoid self
    dist_sq.fill_diagonal_(float('inf'))

    # Find indices of k nearest neighbors for each sample (excluding self)
    knn_indices = dist_sq.topk(k, dim=1, largest=False).indices  # shape: (N, k)

    # Create a mask that excludes self and top-k neighbors
    N = x.size(0)
    mask = torch.ones_like(dist_sq, dtype=torch.bool)
    mask.fill_diagonal_(False)  # exclude self

    # Exclude k nearest neighbors
    row_indices = torch.arange(N).unsqueeze(1).expand(-1, k)  # (N, k)
    mask[row_indices, knn_indices] = False

    # Apply the mask
    filtered_dist_sq = dist_sq[mask].view(N, -1)  # shape: (N, N - k - 1)

    # Compute uniformity loss
    loss = torch.log(torch.exp(-t * filtered_dist_sq).mean())
    return loss


def uniform_loss_per_point(x, t=2):
    # x should be L2-normalized
    dist_sq = torch.cdist(x, x, p=2).pow(2)  # shape: (N, N)
    mask = ~torch.eye(x.size(0), dtype=torch.bool, device=x.device)  # exclude i==j
    dist_sq = dist_sq.masked_select(mask).view(x.size(0), -1)  # shape: (N, N-1)
    loss = torch.log(torch.exp(-t * dist_sq).sum(dim=1)).mean()
    return loss

def robyol_loss_func(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """Computes BYOL's loss given batch of predicted features p and projected momentum features z.

    Args:
        z1 (torch.Tensor): NxD Tensor containing predicted features from view 1
        z2 (torch.Tensor): NxD Tensor containing projected momentum features from view 2
        simplified (bool): faster computation, but with same result. Defaults to True.

    Returns:
        torch.Tensor: Robin BYOL's loss.
    """
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)

    align_loss = align_loss_func(z1, z2)

    uniform_loss = uniform_loss_func(z1) + uniform_loss_func(z2)

    return align_loss + uniform_loss / 2
