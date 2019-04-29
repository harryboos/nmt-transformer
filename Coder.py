from Layer import EncoderLayer
from Layer import DecoderLayer
import torch.nn as nn
import torch
import math
from torch.autograd import Variable
import numpy as np
from Normalization import Norm
from FeedForward import FeedForward
from Attention import MultiHeadAttention
from Embed import GermanEmbedder
from Embed import EnglishEmbedder
from Embed import PositionalEncoder
from Layer import EncoderLayer
from Layer import DecoderLayer

import copy


# function for copying model N times
def duplicate_models(model, N):
    return nn.ModuleList([copy.deepcopy(model) for i in range(N)])

class Encoder(nn.Module):
    def __init__(self, vocab_size, dim_model, N, H):
        super().__init__()
        self.N = N
        self.embed = GermanEmbedder(vocab_size, dim_model)
        self.position_encoder = PositionalEncoder(dim_model)
        self.layers = duplicate_models(EncoderLayer(dim_model, H), N)



    def forward(self, source, mask):
        x = self.embed(source)
        x = self.position_encoder(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return x



class Decoder(nn.Module):
    def __init__(self, vocab_size, dim_model, N, H):
        super().__init__()
        self.N = N
        self.embed = EnglishEmbedder(vocab_size, dim_model)
        self.position_encoder = PositionalEncoder(dim_model)
        self.layers = duplicate_models(DecoderLayer(dim_model, H), N)


    def forward(self, target, encoder_output, source_mask, target_mask):
        x = self.embed(target)
        x = self.position_encoder(x)
        for i in range(self.N):
            x = self.layers[i](x, encoder_output, source_mask, target_mask)
            # print(x)
        return x






































