from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tst.utils import generate_local_map_mask
import math


class MultiHeadAttention(nn.Module):
    """Multi Head Attention block from Attention is All You Need.

    Given 3 inputs of shape (batch_size, K, d_model), that will be used
    to compute query, keys and values, we output a self attention
    tensor of shape (batch_size, K, d_model).

    Parameters
    ----------
    d_model:
        Dimension of the input vector.
    q:
        Dimension of all query matrix.
    v:
        Dimension of all value matrix.
    h:
        Number of heads.
    attention_size:
        Number of backward elements to apply attention.
        Deactivated if ``None``. Default is ``None``.
    """

    def __init__(self,
                 d_model: int,
                 q: int,
                 v: int,
                 h: int,
                 mask: bool = False):
        """Initialize the Multi Head Block."""
        super().__init__()

        self._q = q
        self._h = h

        # Query, keys and value matrices
        self._W_q = nn.Linear(d_model, q * h)
        self._W_k = nn.Linear(d_model, q * h)
        self._W_v = nn.Linear(d_model, v * h)

        # Output linear function
        self._W_o = nn.Linear(v * h, d_model)

        self._mask = mask
        # Score placeholder
        self._scores = None

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[str] = None) -> torch.Tensor:

        Q = torch.cat(self._W_q(query).chunk(self._h, dim=-1), dim=0)
        K = torch.cat(self._W_k(key).chunk(self._h, dim=-1), dim=0)
        V = torch.cat(self._W_v(value).chunk(self._h, dim=-1), dim=0)
        # shape [batchsize * head_num, input, q or k or v]

        # Scaled Dot Product
        self._scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self._q)

        if self._mask:
          mask = torch.ones_like(self._scores).cuda()
          mask = torch.tril(mask).cuda()
          self._scores = torch.where(mask > 0, self._scores, torch.Tensor([-2 ** 32 + 1]).cuda()).cuda()

        # Apply softmax
        self._scores = F.softmax(self._scores, dim=-1)  # shape [batchsize * head_num, input, input]

        # scores * values  结果维度(batchsize*head_nums, input, v)
        attention = torch.matmul(self._scores, V)

        # Concatenat the heads  结果维度(batchsize, input, v*head_nums)
        attention_heads = torch.cat(attention.chunk(self._h, dim=0), dim=-1)

        # Apply linear transformation W^O
        self_attention = self._W_o(attention_heads)  # shape [batchsize, input, d_model]

        return self_attention, self._scores
