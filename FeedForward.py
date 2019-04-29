import torch.nn as nn
import torch
import math
from torch.autograd import Variable
import numpy as np

# Cited work: Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., and Polosukhin, I. (2017). Attention is all you need. In the Annual Conference on Neural Information
# Processing Systems (NIPS).
# in the paper, hyperparameter of these got highest bleu score
# N dmodel dff  h dk dv Pdrop ls
# 6 512   2048  8 64 64 0.1   0.1
# so we set dim_feedforward = 2048
class FeedForward(nn.Module):

    def __init__(self, dim_model, dim_feedforward=2048, dropout = 0.1):

        super().__init__()
        self.linear_1 = nn.Linear(dim_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(dim_feedforward, dim_model)

    def forward(self,x):
        # feed-forward layer in the papaer is described as:
        # FFN(x) = max(0, xW1 + b1)W2 + b2
        # which is linear(x)->relu(X)->linear(x)
        x = self.linear_1(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        return x


# H = 4
# dimm  = 24
# x = torch.rand(7, H, dimm)
# ff = FeedForward(dimm)
# out = ff(x)
# print('x: ', x.size())
# print('out: ', out.size())
# x:  torch.Size([7, 4, 24])
# out:  torch.Size([7, 4, 24])