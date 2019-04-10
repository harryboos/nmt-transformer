from Model import Transformer

import torch.nn as nn
from torch import  optim
import torch
import math
from torch.autograd import Variable
import numpy as np
import torch.functional as F
import copy
import time
import argparse
import time
import torch
import random
#from Translate import translate
import torch.nn.functional as F
from Mask import create_masks


dim_model = 8
batch_size = 400
H = 2
N = 3
src_vocab = 8
trg_vocab = 9
max_seq_dim = 80
model = Transformer(src_vocab, trg_vocab, dim_model, N, H)

source = torch.rand((batch_size , max_seq_dim))
target = source


print('source: ', source.size(), 'target: ', target.size())
#
source_mask, target_mask = create_masks(source, target)
print('source_mask : ', source_mask.size(), 'target_mask: ', target_mask.size())
output = model(source, target, source_mask, target_mask)
out = output.contiguous().view(-1, trg_vocab)
targets = target.contiguous().view(-1)
print(out)
print(targets)
criterion = nn.CrossEntropyLoss()
loss = criterion(out, targets.long())
print('loss: ',loss)


source = torch.rand((1 , max_seq_dim))
target = source
print('test--------------------')
print('source: ', source.size(), 'target: ', target.size())
source_mask, target_mask = create_masks(source, target)
print('source_mask : ', source_mask.size(), 'target_mask: ', target_mask.size())
output = model(source, target, source_mask, target_mask)
print(output.size())




def train_lm(dataset, params, net):
    # since the first index corresponds to the PAD token, we just ignore it
    # when computing the loss
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(net.parameters(), lr=params['learning_rate'])
    num_examples, seq_len = dataset.size()
    batches = [(start, start + params['batch_size']) for start in \
               range(0, num_examples, params['batch_size'])]

    for epoch in range(params['epochs']):
        ep_loss = 0.
        start_time = time.time()
        random.shuffle(batches)

        # for each batch, calculate loss and optimize model parameters
        for b_idx, (start, end) in enumerate(batches):
            batch = dataset[start:end]
            source_mask, target_mask = create_masks(batch, batch)
            preds = net(batch, batch, source_mask, target_mask)


            preds = preds[:, :-1, :].contiguous().view(-1, trg_vocab)
            # q1.1: explain the below line!
            targets = batch[:, 1:].contiguous().view(-1)

            loss = criterion(preds, targets.long())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            ep_loss += loss

        print('epoch: %d, loss: %0.2f, time: %0.2f sec' % \
              (epoch, ep_loss, time.time() - start_time))
params = {}

params['d_emb'] = 50
params['batch_size'] = 16
params['epochs'] = 15
params['learning_rate'] = 0.001

# train_lm(source, params, model)


