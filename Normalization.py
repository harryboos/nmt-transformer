
import torch.nn as nn
import torch


# normalize input
# reference from:  S. Ioffe and C. Szegedy. Batch normalization: Accelerating deep
# network training by reducing internal covariate shift. In ICML, 2015.
# batch normalization
class Norm(nn.Module):
    def __init__(self, dim_model, eps=1e-6):
        super().__init__()

        # assign alpha
        self.a = nn.Parameter(torch.ones(dim_model))
        # assign beta
        self.b = nn.Parameter(torch.zeros(dim_model))
        self.eps = eps

    def forward(self, x):

        mean = x.mean(dim=-1, keepdim = True)
        diff_x = x - mean
        std = x.std(dim=-1, keepdim=True) + self.eps

        out = self.a * diff_x / std + self.b

        return out

# H = 3
# # dimm  = 12
# # x = torch.rand(1, H, dimm)
# #
# # norm = Norm(dimm)
# # out = norm(x)
# print('x: ', x.size())
# print('out: ', out.size())


