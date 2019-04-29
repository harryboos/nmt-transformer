import torch.nn as nn
import torch
import math
from torch.autograd import Variable
import numpy as np

# mask for source sentence, which set 0 to index where is padding
def padding_mask(source):
    seq_len = source.size(1)

    pad_mask = source.eq(0).float()
    one_mask = torch.ones(pad_mask.size())
    pad_mask = one_mask - pad_mask
    pad_mask = pad_mask.unsqueeze(1).expand(-1, seq_len, -1)  # shape [batch, l_q, l_k]
    return pad_mask


# mask for target sentence to prevent exposing answers to decoder
def single_mask(size):
    mask = torch.ones((size, size))
    mask = torch.tril(mask)
    mask = mask.view(1, size, size)
    return mask

# create masks for source and target
def create_masks(source, target):

    seq_len = target.size(1)
    target_mask = single_mask(seq_len)

    source_mask = padding_mask(source)
    return source_mask, target_mask



# q = torch.Tensor([[[2,3, 4, 0],[1, 5, 0, 0]], [[2,3, 4, 0],[1, 0, 0, 0]]])
# q = torch.Tensor([[2,3, 4, 0],[1, 5, 0, 0]]).long()























