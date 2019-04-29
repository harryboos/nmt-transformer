import torch.nn as nn
import torch
import math
from torch.autograd import Variable
import numpy as np

class GermanEmbedder(nn.Module):
    def __init__(self, vocab_size, dim_model):
        super().__init__()
        ger_embedding = torch.Tensor(np.load('emb/ger.npy'))
        ger_embedding = ger_embedding.cuda()
        self.embed = nn.Embedding.from_pretrained(ger_embedding, freeze=False)

        #
    def forward(self, x):
        return self.embed(x.long())


class EnglishEmbedder(nn.Module):
    def __init__(self, vocab_size, dim_model):
        super().__init__()
        eng_embedding = torch.Tensor(np.load('emb/eng.npy'))
        eng_embedding = eng_embedding.cuda()
        self.embed = nn.Embedding.from_pretrained(eng_embedding, freeze=False)

    def forward(self, x):
        return self.embed(x.long())


class PositionalEncoder(nn.Module):
    def __init__(self, dim_model, max_seq_dim = 700):
        super().__init__()


        position_encoder = torch.zeros(max_seq_dim, dim_model)
        # pe(pos, 2i) = sin(pos / 10000 ^(2i/dim_model)  )
        # pe(pos, 2i+1) = cos(pos / 10000 ^(2i/dim_model)  )

        for p in range(max_seq_dim):
            for i in range(0, dim_model, 2):
                position_encoder[p, i] = math.sin(p / (10000 ** (2 * i / dim_model)))
                position_encoder[p, i+1] = math.cos(p / (10000 ** (2 * i / dim_model)))

        position_encoder = position_encoder.unsqueeze(0)
        self.register_buffer('position_encoder', position_encoder)


    def forward(self, x):
        # print('x here; ', x.size())
        # print('position_encoder: ', self.position_encoder.size())
        pe_value = self.position_encoder[:, :x.size(1)]
        x = x + Variable(pe_value, requires_grad=False)
        return x







































