import torch.nn as nn
from Normalization import Norm
from FeedForward import FeedForward
from Attention import MultiHeadAttention



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
        # frist multi head attention is for target sentence attention
        self.multi_attention1 = MultiHeadAttention(H, dim_model)
        # second multi head attention is for output of encoder and target sentence
        self.multi_attention2 = MultiHeadAttention(H, dim_model)
        # then we send output of them to feed forward layer
        self.ff = FeedForward(dim_model)


    def forward(self, x, encoder_output, source_mask, target_mask):

        # first normalize input
        norm_x_1 = self.norm_1(x)
        # then get attention score of target sentence with target mask
        attention1 = self.multi_attention1(norm_x_1, norm_x_1, norm_x_1, target_mask)
        # then we add dropout
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









































