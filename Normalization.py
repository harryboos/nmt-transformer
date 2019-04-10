
import torch.nn as nn
import torch



class Norm(nn.Module):
    def __init__(self, dim_model, eps=1e-6):
        super().__init__()
        self.size = dim_model
        # assign alpha
        self.a = nn.Parameter(torch.ones(self.size))
        # assign beta
        self.b = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):

        mean = x.mean(dim=-1, keepdim = True)
        diff_x = x - mean
        std = x.std(dim=-1, keepdim=True) + self.eps

        out = self.a * diff_x / std + self.b

        return out

H = 3
dimm  = 12
x = torch.rand(1, H, dimm)

norm = Norm(dimm)
out = norm(x)
# print('x: ', x.size())
# print('out: ', out.size())


