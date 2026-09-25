import torch
from torch import nn
from torch.nn import functional as F

class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionPooling, self).__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch_size, seq_len, hidden_size)
        attention_weights = F.softmax(self.attention(x), dim=1)  # (batch_size, seq_len, 1)
        pooled_output = torch.sum(attention_weights * x, dim=1)  # (batch_size, hidden_size)
        return pooled_output

class WindowAttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super(WindowAttentionPooling, self).__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (num_windows, hidden_size)
        
        attention_weights = F.softmax(
            self.attention(x), dim=0
        )  # (num_windows, 1)

        pooled_output = torch.sum(
            attention_weights * x, dim=0
        )  # (hidden_size,)

        return pooled_output