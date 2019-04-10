import torch.nn as nn
import torch
import math
from torch.autograd import Variable
import numpy as np



class FeedForward(nn.Module):
    def __init__(self, dim_model, dim_feedforward=2048, dropout = 0.1):
        super().__init__()
        self.linear_1 = nn.Linear(dim_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(dim_feedforward, dim_model)
    def forward(self,x):
        x = self.linear_1(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        return x


H = 4
dimm  = 24
x = torch.rand(7, H, dimm)
ff = FeedForward(dimm)
out = ff(x)
# print('x: ', x.size())
# print('out: ', out.size())
# x:  torch.Size([7, 4, 24])
# out:  torch.Size([7, 4, 24])