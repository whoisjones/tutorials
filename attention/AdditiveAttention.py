import torch
import torch.nn as nn
from torch import tensor
import torch.nn.functional as F

from typing import Tuple

class AdditiveAttention(nn.Module):
    """
     Applies a additive attention (bahdanau) mechanism on the output features from the decoder.
     Additive attention proposed in "Neural Machine Translation by Jointly Learning to Align and Translate" paper.
     Args:
         hidden_dim (int): dimesion of hidden state vector
     Inputs: query, value
         - **query** (batch_size, q_len, hidden_dim): tensor containing the output features from the decoder.
         - **value** (batch_size, v_len, hidden_dim): tensor containing features of the encoded input sequence.
     Returns: context, attn
         - **context**: tensor containing the context vector from attention mechanism.
         - **attn**: tensor containing the alignment from the encoder outputs.
     Reference:
         - **Neural Machine Translation by Jointly Learning to Align and Translate**: https://arxiv.org/abs/1409.0473
    """
    def __init__(self, hidden_dim: int) -> None:
        super(AdditiveAttention, self).__init__()
        self.query_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.bias = nn.Parameter(torch.rand(hidden_dim).uniform_(-0.1, 0.1))
        self.score_proj = nn.Linear(hidden_dim, 1)

    def forward(self, query: tensor, key: tensor, value: tensor) -> Tuple[tensor, tensor]:
        k = self.key_proj(key)
        q = self.query_proj(query)
        kqb = k + q + self.bias
        p = torch.tanh(kqb)
        score = self.score_proj(p).squeeze(-1)
        attn = F.softmax(score, dim=-1)
        context = torch.bmm(attn.unsqueeze(1), value)
        return context, attn

if __name__ == "__main__":
    atn = AdditiveAttention(hidden_dim=256)
    query = torch.rand(1, 1, 256)
    key = torch.rand(1, 10, 256)
    val = torch.rand(1, 10, 256)
    atn(query, key, val)
