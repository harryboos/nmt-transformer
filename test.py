import unicodedata
import string
import re
import random
import time
import math

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

from Model import Transformer

from torch import optim
import numpy as np
import copy
import argparse
from Mask import create_masks
from Text import indexes_from_sentence
from Text import padding_both


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

USE_CUDA = True

PAD_token = 0
SOS_token = 1
EOS_token = 2


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD",
                           SOS_token: "SOS", EOS_token: "EOS"}
        self.n_words = 3  # Count SOS and EOS

    def index_words(self, sentence):
        for word in sentence.split(' '):
            self.index_word(word)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters


def normalize_string(s):
    s = s.lower()  # unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?,'])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?,ÄäÖöÜüẞß']+", r" ", s)
    return s


def read_langs(lang1, lang2, reverse=False):
    print("Reading lines...")

    pairs = []
    line1 = open('data/train.de', encoding='utf-8').read().strip().split('\n')
    # .splitlines()
    line2 = open('data/train.en', encoding='utf-8').read().strip().split('\n')

    for i in range(len(line1)):

        pairs.append([normalize_string(line1[i]), normalize_string(line2[i])])

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


def prepare_data(lang1_name, lang2_name, reverse=False):
    input_lang, output_lang, pairs = read_langs(
        lang1_name, lang2_name, reverse)
    print("Read %s sentence pairs" % len(pairs))

    print("Indexing words...")
    for pair in pairs:
        input_lang.index_words(pair[0])
        output_lang.index_words(pair[1])

    return input_lang, output_lang, pairs


input_lang, output_lang, pairs = prepare_data('ger', 'en')

# Print an example pair
print(random.choice(pairs))

# Return a list of indexes, one for each word in the sentence


def indexes_from_sentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def variable_from_sentence(lang, sentence):
    indexes = indexes_from_sentence(lang, sentence)
    indexes.append(EOS_token)
    var = Variable(torch.LongTensor(indexes).view(-1, 1))
    if USE_CUDA:
        var = var.cuda()
    return var


def variables_from_pair(pair, input_lang, output_lang):
    input_variable = variable_from_sentence(input_lang, pair[0])
    target_variable = variable_from_sentence(output_lang, pair[1])
    return (input_variable, target_variable)


def find_max_len(pair):
    result = 0

    for sents in pair:
        for item in sents:
            if len(item.split()) > result:

                result = len(item.split())
    return result


def find_len(element):
    result = 0
    for item in element:
        if len(item.split()) > result:
            result = len(item.split())
    return result


def paddingSOS(vector, max_len):
    vector = [SOS_token]+vector
    while len(vector) < max_len:
        vector.append(PAD_token)
    return vector

def paddingEOS(vector, max_len):
    vector = vector + [EOS_token]
    while len(vector)< max_len:
        vector.append(PAD_token)
    return vector


def padding_both(vector, max_len):
    vector = [SOS_token] + vector + [EOS_token]
    while len(vector) < max_len:
        vector.append(PAD_token)
    return vector


def padding(vector, max_len):

    while len(vector) < max_len:
        vector.append(PAD_token)
    return vector


max_len = find_max_len(pairs)+2
print('max_len: ', max_len)


def translate_sentence(sent, model, input_lang, output_lang, maxlen):
    sent_as_index = indexes_from_sentence(input_lang, sent)
    input = padding_both(sent_as_index, maxlen)
    input = torch.Tensor(input)
    input = input.cuda()
    source = input.unsqueeze(0)  # add a dimension

    target = torch.zeros((1, maxlen))
    target[0][0] = 1

    source_mask, target_mask = create_masks(source, target)
    output = model(source, target, source_mask, target_mask)

    output = F.softmax(output, -1)
    out = torch.max(output, -1)[1]  # 1 is index, 0 is max malue
    out = out.squeeze(0)
    print('out: ', out)
    result = ''
    for idx in out:
        if idx == 0:
            break
        index = idx.item()
        result += output_lang.index2word[index]+' '
    print(result)
    return result


def pair_to_indexes(pairs, max_len, input_lang, output_lang):
    source = np.zeros((len(pairs), max_len))
    target = np.zeros((len(pairs), max_len))
    for i in range(len(pairs)):
        # add start token for english
        sent2 = padding_both(indexes_from_sentence(
            output_lang, pairs[i][1]), max_len)
        sent2 = torch.Tensor(sent2)
        target[i] = sent2

        # add end token for german
        sent1 = padding(indexes_from_sentence(
            input_lang, pairs[i][0]), max_len)
        sent1 = torch.Tensor(sent1)
        source[i] = sent1

    return source, target





# dataset: pairs
def train_lm(sources, targets, params, net):

    criterion = nn.CrossEntropyLoss(ignore_index=0)

    optimizer = optim.Adam(net.parameters(), lr=params['learning_rate'])
    sources = torch.from_numpy(sources)
    targets = torch.from_numpy(targets)
    sources = sources.cuda()
    targets = targets.cuda()

    num_examples = sources.size(0)
    batches = [(start, start + params['batch_size']) for start in
               range(0, num_examples, params['batch_size'])]

    for epoch in range(params['epochs']):
        ep_loss = 0.
        start_time = time.time()
        random.shuffle(batches)

        # for each batch, calculate loss and optimize model parameters
        for b_idx, (start, end) in enumerate(batches):
            source = sources[start:end]
            target = targets[start:end]

            source_mask, target_mask = create_masks(source.cpu(), target.cpu())
            source_mask = source_mask.cuda()
            target_mask = target_mask.cuda()
            preds = net(source, target, source_mask, target_mask)

            preds = preds.contiguous().view(-1, net.target_vocab)

            labels = target.contiguous().view(-1)

            loss = criterion(preds, labels.long())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            ep_loss += loss.item()
            torch.cuda.empty_cache()

        curr_loss = 'epoch: %d, loss: %0.2f, time: %0.2f sec' % (
            epoch, ep_loss, time.time() - start_time)
        print(curr_loss)
        file.write(curr_loss)
        # save model each .. epoch

        if epoch % save_each == 0:
            substr = 'mytraining'+str(epoch)+'.pt'
            path = 'models/' + substr
            torch.save(net.state_dict(), path)


save_each = 1
file = open("loss.txt", "w")

params = {}


params['batch_size'] = 64
params['epochs'] = 15
params['learning_rate'] = 0.001


dim_model = 128
H = 2
N = 3
src_vocab = input_lang.n_words
trg_vocab = output_lang.n_words

model = Transformer(src_vocab, trg_vocab, dim_model, N, H)
model = model.cuda()
print(10)
#data_1 = [element for element in pairs if find_len(element) < 100]
#data_1 = data_1[:2000]
data_1 = pairs
max_len_1 = find_max_len(data_1) + 2
source, target = pair_to_indexes(data_1, max_len_1, input_lang, output_lang)
train_lm(source, target, params, model)
file.close()
