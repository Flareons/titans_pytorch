from torch import nn
import torch.nn.functional as F
import math
import torch


class CrossMultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()

        assert dim % num_heads == 0

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.out_proj = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x):
        """
        (B, N, D)
        ->
        (B, H, N, Hd)
        """

        B, N, D = x.shape

        x = x.view(
            B,
            N,
            self.num_heads,
            self.head_dim
        )

        x = x.transpose(1, 2)

        return x

    def merge_heads(self, x):
        """
        (B, H, N, Hd)
        ->
        (B, N, D)
        """

        B, H, N, Hd = x.shape

        x = x.transpose(1, 2).contiguous()

        x = x.view(
            B,
            N,
            H * Hd
        )

        return x

    def forward(self, Q, K, V, mask=None):
        """
        Q : (B, Nq, D)
        K : (B, Nk, D)
        V : (B, Nk, D)
        """

        # split heads
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # attention score
        scores = torch.matmul(
            Q,
            K.transpose(-2, -1)
        )

        scores = scores / math.sqrt(self.head_dim)

        # mask
        if mask is not None:
            scores = scores.masked_fill(
                mask == 0,
                -1e9
            )

        # softmax
        attn = F.softmax(scores, dim=-1)

        attn = self.dropout(attn)

        # weighted value
        out = torch.matmul(attn, V)

        # merge heads
        out = self.merge_heads(out)

        # final projection
        out = self.out_proj(out)

        return out, attn