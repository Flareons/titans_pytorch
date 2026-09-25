import torch
import torch.nn as nn
from model.Neural_Memory import NeuralMemory
from model.Persistent_Memory import PersistentMemory
from model.Cross_Attention import CrossMultiHeadAttention

class Core(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        
        self.dim = dim
        
        self.persis_mem = PersistentMemory(dim)
        self.neural_mem = NeuralMemory(dim)
        
        self.Q_attn_prj = nn.Linear(dim, dim)
        self.K_attn_prj = nn.Linear(dim, dim)
        self.V_attn_prj = nn.Linear(dim, dim)

        self.cross_attn = CrossMultiHeadAttention(dim)
        
    def forward(self, x):
        
        retrieval = self.neural_mem.read(x)

        B = x.shape[0]

        p = self.persis_mem(x)
        
        x = torch.cat([p, retrieval, x], dim=1)

        # print("Stacked shape:", x.shape)

        x = x.view(B, -1, self.dim)

        Q_attn = self.Q_attn_prj(x)
        K_attn = self.K_attn_prj(x)
        V_attn = self.V_attn_prj(x)

        # print("Q_attn shape:", Q_attn.shape)
        # print("K_attn shape:", K_attn.shape)
        # print("V_attn shape:", V_attn.shape)

        out, attn = self.cross_attn(Q_attn, K_attn, V_attn)

        loss_mem = self.neural_mem.memory_loss(out)

        neural_out = self.neural_mem.read(out)

        out = out * neural_out
        # print("Final output shape:", out.shape)
        # print("attn shape:", attn.shape)

        return out, loss_mem, out.detach()

    def update_memory(self, y):
        self.neural_mem.update_memory(y)

# Chạy