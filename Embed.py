import torch.nn as nn
import torch
import math
from torch.autograd import Variable
import numpy as np

class Embedder(nn.Module):
    def __init__(self, vocab_size, dim_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim_model)

    def forward(self, x):
        return self.embed(x.long())


class PositionalEncoder(nn.Module):
    def __init__(self, dim_model, max_seq_dim = 120):
        super().__init__()


        position_encoder = torch.zeros(max_seq_dim, dim_model)

        for p in range(max_seq_dim):
            for i in range(0, dim_model, 2):
                position_encoder[p, i] = math.sin(p / (1000 ** (2 * i / dim_model)))
                position_encoder[p, i+1] = math.cos(p / (1000 ** (2 * (i+1) / dim_model)))

        position_encoder = position_encoder.unsqueeze(0)
        self.register_buffer('position_encoder', position_encoder)


    def forward(self, x):
        # print('x here; ', x.size())
        # print('position_encoder: ', self.position_encoder.size())
        x = x + Variable(self.position_encoder[:, :x.size(1)], requires_grad=False)
        return x

# vocab_size = 10
# dim_model = 6
# em = Embedder(vocab_size, dim_model)
# pos = PositionalEncoder(dim_model)
#
# x = torch.LongTensor([[3, 4, 6], [3, 4, 6]])
# print("x: ", x.shape)
# emb = em.forward(x)
# print("embedding: ", emb.shape)
# posi = pos(emb)
# print("pe: ", posi.shape)

# print('-------------------')







































