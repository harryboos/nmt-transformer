import torch.nn as nn
import torch
import math
from torch.autograd import Variable
import numpy as np
from Mask import single_mask

def attention_help(q, k, v, dk, mask=None, dropout=None):
    # q, k v: batch_size , seq_len , dim_model.
    # transform k to batch_size , dim_model, seq_len
    # scores dim: batch_size , seq_len , seq_len
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(dk)
    if mask is not None:
        # print(mask.size())
        # print('mask, inside: ', mask.size())
        # add one dimension
        mask = mask.unsqueeze(1)
        # print('mask squeeze inside: ',mask.size())
        # print('scores inside: ', scores.size())
        # print('q inside : ', q.size())
        # print('k inside : ', k.size())
        scores = scores.masked_fill(mask ==0, -1e10)
        # print(scores)
    scores = nn.functional.softmax(scores, dim = -1)

    #add dropout if applicable
    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, H, dim_model, dropout = 0.1):
        super().__init__()

        self.H = H
        self.dim_model = dim_model
        # dk = dim_model / H
        self.dk = dim_model // H

        self.dropout = nn.Dropout(dropout)

        self.q_Linear = nn.Linear(dim_model, dim_model)
        self.k_Linear = nn.Linear(dim_model, dim_model)
        self.v_Linear = nn.Linear(dim_model, dim_model)


        self.out = nn.Linear(dim_model, dim_model)

    def forward(self, q, k, v, mask=None):
        # q, k v: batch_size , H , dim_model.
        batch_size = q.size(0)

        # print('k: ', k.size())
        # transform to batch_size , H , H , dk
        k = self.k_Linear(k).view(batch_size, -1, self.H, self.dk)
        q = self.q_Linear(q).view(batch_size, -1, self.H, self.dk)
        v = self.v_Linear(v).view(batch_size, -1, self.H, self.dk)



        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)




        scaled_dot_attention = attention_help(q, k, v, self.dk, mask, self.dropout)
        # print('attention_scores: ', scaled_dot_attention.size())
        transpose_scores = scaled_dot_attention.transpose(1, 2).contiguous()
        # print('transpose_scores: ', transpose_scores.size())
        concat_scores = transpose_scores.view(batch_size, -1, self.dim_model)
        # print('transform_scores: ', concat_scores.size())
        output = self.out(concat_scores)
        return output





# H = 4
# dimm  = 24
#
# mask = single_mask(H)
# print('mask: ', mask.size())

# mul = MultiHeadAttention(H, dimm)
# q = torch.FloatTensor(torch.rand(1, H, dimm))
# k = torch.FloatTensor(torch.rand(1, H, dimm))
# v = torch.FloatTensor(torch.rand(1, H, dimm))
# print("q: ", q.size())
# score = attention_help(q, k, v, dimm//H, mask )
# print("score : ", score.shape)
# print('mask: ',mask.size())
# out = mul(q, k, v, mask)
# print('out: ', out.shape)







































