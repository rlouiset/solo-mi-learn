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

def entropy_loss_func(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean(-1).mean().log()

def uniform_loss_func(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean(-1).log().mean()

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
