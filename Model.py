from Layer import EncoderLayer
from Layer import DecoderLayer
import torch.nn as nn
import torch
import math
from torch.autograd import Variable
import numpy as np

import copy
from Coder import Encoder
from Coder import Decoder


class Transformer(nn.Module):
    def __init__(self, source_vocab, target_vocab, dim_model, N, H):
        super().__init__()
        self.encoder = Encoder(source_vocab, dim_model, N, H)
        self.decoder = Decoder(target_vocab, dim_model, N, H)
        self.out = nn.Linear(dim_model, target_vocab)
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab

    def forward(self, source, target, source_mask, target_mask):
        encoder_output = self.encoder(source, source_mask)

        decoder_output = self.decoder(target, encoder_output, source_mask, target_mask)

        out = self.out(decoder_output)
        return out



















































