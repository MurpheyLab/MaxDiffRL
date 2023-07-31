#!/usr/bin/env python3

import numpy as np
import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


def jacobian(y, x):
    """Computes the Jacobian of f w.r.t x.

    This is according to the reverse mode autodiff rule,

    sum_i v^b_i dy^b_i / dx^b_j = sum_i x^b_j R_ji v^b_i,

    where:
    - b is the batch index from 0 to B - 1
    - i, j are the vector indices from 0 to N-1
    - v^b_i is a "test vector", which is set to 1 column-wise to obtain the correct
        column vectors out ot the above expression.

    :param f: function R^N -> R^N
    :param x: torch.tensor of shape [B, N]
    :return: Jacobian matrix (torch.tensor) of shape [B, N, N]
    """

    B, N = x.shape
    jacobian = list()
    for i in range(N):
        v = torch.zeros_like(y)
        v[:, i] = 1.
        dy_i_dx = autograd.grad(y,
                                x,
                                grad_outputs=v,
                                retain_graph=True,
                                create_graph=True,
                                allow_unused=True)[0]  # shape [B, N]
        jacobian.append(dy_i_dx)

    jacobian = torch.stack(jacobian, dim=2).requires_grad_()

    return jacobian

class Lambda(nn.Module):
    "An easy way to create a pytorch layer for a simple `func`."
    def __init__(self, func):
        "create a layer that simply calls `func` with `x`"
        super().__init__()
        self.func=func
    def forward(self, x): return self.func(x)

def block_join(A,row_first=True):
    if row_first:
        return torch.cat(torch.cat(A,dim=1).T.chunk(2),dim=1).T
    else:
        return torch.cat(torch.cat(A,dim=0).chunk(2),dim=1)

def quat2angle(quat: torch.Tensor, axis: int):
    proj = quat2mat(quat).select(-1,axis).select(-1,axis) # dot product of local axis with global axis
    theta = torch.arccos(proj).unsqueeze(-1)
    return theta

def quat2mat(quaternions: torch.Tensor):
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).

    Source: https://github.com/facebookresearch/pytorch3d
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


class EmptyBarrier(torch.nn.Module):
    def cost(self,x):
        return 0.
