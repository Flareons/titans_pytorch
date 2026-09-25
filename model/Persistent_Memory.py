# import torch
# import torch.nn as nn


# class PersistentMemory(nn.Module):
#     def __init__(self, dim=768, memory_size=1):
#         super().__init__()

#         self.dim = dim
#         self.memory_size = memory_size

#         self.W_k = nn.Parameter(
#             torch.randn(memory_size, dim) * 0.02
#         )

#         self.W_v = nn.Parameter(
#             torch.randn(memory_size, dim) * 0.02
#         )

#     def forward(self, x):
#         # x: [B, D]

#         # [M, D] × [B, D]
#         # → [B, M]
#         scores = torch.einsum(
#             'md,bd->bm',
#             self.W_k,
#             x
#         )

#         # [B, M]
#         attn = torch.softmax(scores, dim=-1)

#         # [M, D] × [B, M]
#         # → [B, D]
#         out = torch.einsum(
#             'md,bm->bd',
#             self.W_v,
#             attn
#         )

#         return out

import torch
import torch.nn as nn


class PersistentMemory(nn.Module):
    def __init__(self, dim=768, memory_size=1):
        super().__init__()

        self.dim = dim
        self.memory_size = memory_size

        # [M, D]
        self.W_k = nn.Parameter(
            torch.randn(memory_size, dim) * 0.02
        )

        # [M, D]
        self.W_v = nn.Parameter(
            torch.randn(memory_size, dim) * 0.02
        )

    def forward(self, x):
        # x: [N_win, Tok, D]
        # Example: [3, 256, 768]

        # [M,D] × [N_win,Tok,D]
        # -> [N_win,Tok,M]
        scores = torch.einsum(
            'md,ntd->ntm',
            self.W_k,
            x
        )

        # Softmax over memory dimension M
        # [N_win,Tok,M]
        attn = torch.softmax(scores, dim=-1)

        # [N_win,Tok,M] × [M,D]
        # -> [N_win,Tok,D]
        out = torch.einsum(
            'ntm,md->ntd',
            attn,
            self.W_v
        )

        return out