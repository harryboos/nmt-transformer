import torch.nn as nn
from Normalization import Norm
from FeedForward import FeedForward
from Attention import MultiHeadAttention
import copy


class EncoderLayer(nn.Module):
    def __init__(self, dim_model, H, dropout=0.1):
        super().__init__()
        # encoder layer is :
        # add & norm
        #     ^
        # feed forward
        #     ^
        # add & norm
        #     ^
        # multi head attention
        self.multi_attention = MultiHeadAttention(H, dim_model)
        self.feedforward = FeedForward(dim_model)
        self.norm_1 = Norm(dim_model)
        self.norm_2 = Norm(dim_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        norm_x_1 = self.norm_1(x)
        attention1 = self.multi_attention(norm_x_1, norm_x_1, norm_x_1, mask)
        drop1 = self.dropout_1(attention1)
        x = x + drop1


        norm_x_2 = self.norm_2(x)
        ff = self.feedforward(norm_x_2)
        drop2 = self.dropout_2(ff)
        x = x + drop2
        return x


class DecoderLayer(nn.Module):
    def __init__(self, dim_model, H, dropout = 0.1):
        super().__init__()
        self.norm_1 = Norm(dim_model)
        self.norm_2 = Norm(dim_model)
        self.norm_3 = Norm(dim_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.multi_attention1 = MultiHeadAttention(H, dim_model)
        self.multi_attention2 = MultiHeadAttention(H, dim_model)
        self.ff = FeedForward(dim_model)


    def forward(self, x, encoder_output, source_mask, target_mask):


        norm_x_1 = self.norm_1(x)
        attention1 = self.multi_attention1(norm_x_1, norm_x_1, norm_x_1, target_mask)
        drop1 = self.dropout_1(attention1)
        x = x + drop1

        norm_x_2 = self.norm_2(x)
        # print('DecoderLayer: ',norm_x_2.size(), encoder_output.size(), encoder_output.size(), source_mask.size())
        attention2 = self.multi_attention2(norm_x_2, encoder_output, encoder_output, source_mask)
        drop2 = self.dropout_2(attention2)
        x = x + drop2


        norm_x_3 = self.norm_3(x)
        feedforward = self.ff(norm_x_3)
        drop3 = self.dropout_3(feedforward)
        x = x + drop3
        return x


def get_clones(model, N):
    return nn.ModuleList([copy.deepcopy(model) for i in range(N)])







































