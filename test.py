from Model import Transformer
import matplotlib.pyplot as plt
from Mask import create_masks
import argparse
import copy
import numpy as np
from torch import optim
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
import torch.nn as nn
import torch
import math
import time
import random
import re
import string
import unicodedata



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

USE_CUDA = True
PAD_token = 0
SOS_token = 1
EOS_token = 2
UNK_token = 3


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS",
            EOS_token: "EOS", UNK_token: "UNK"}
        self.n_words = 4  # Count SOS and EOS

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

def normalize_string(s):
#     s = s.lower()
    s = re.sub(r"([.!?,])", r" \1", s)
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

print(pairs[13])
# Return a list of indexes, one for each word in the sentence


def indexes_from_sentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]





def variable_from_sentence(lang, sentence):
    indexes = indexes_from_sentence(lang, sentence)
    indexes.append(EOS_token)
    var = Variable(torch.LongTensor(indexes).view(-1, 1))
    if USE_CUDA: var = var.cuda()
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
    while len(vector) < max_len:
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


def print_head(lang):
    for i in range(8):
        print('input_lang ', i, ' : ', lang.index2word[i])


print('input: ')
print_head(input_lang)
print('output: ')
print_head(output_lang)

max_len = find_max_len(pairs)+2
print('max_len: ', max_len)
print('input: ', input_lang.name)
print('output: ', output_lang.name)


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





data_1 = [element for element in pairs if find_len(element) < 200]
print(len(data_1) / len(pairs))


def correspond_len(pair, thres):
    le = find_len(pair)
    if le < thres[0]-3:
        return thres[0]
    for i in range(len(thres)):
        if i == len(thres)-1:
            return None
        if le > (thres[i]-3) and le < (thres[i+1]-3):
            return thres[i+1]





def class_data(data_pairs):
    threshold = [20, 40, 60, 80, 100, 200]
    class_pairs = []
    for i in range(len(threshold)):
        class_pairs.append([])
    for pair in data_pairs:
        pair_len = correspond_len(pair, threshold)
        if pair_len is None:
            continue
        class_pairs[threshold.index(pair_len)].append(pair)

    return class_pairs, threshold


class_pairs, thres = class_data(data_1)

for i in range(6):
    print(len(class_pairs[i]))
print(output_lang.name, output_lang.n_words)
print(input_lang.name, input_lang.n_words)
params = {}


params['batch_size'] = 64
params['epochs'] = 50
params['learning_rate'] = 0.001


dim_model = 300
H = 12
N = 6
src_vocab = input_lang.n_words
trg_vocab = output_lang.n_words

model = Transformer(src_vocab, trg_vocab, dim_model, N, H)
model.cuda()




def train_lm(data_pairs, params, net):
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(net.parameters(), lr=params['learning_rate'])

    classed_pairs, thres = class_data(data_pairs)
    sources_set = []
    targets_set = []
    batches_set = []
    for i in range(len(classed_pairs)):
        source, target = pair_to_indexes(
            classed_pairs[i], thres[i], input_lang, output_lang)
        sources = torch.from_numpy(source)
        targets = torch.from_numpy(target)
        sources = sources.cuda()
        targets = targets.cuda()

        num_examples = len(classed_pairs[i])
        batches = [(start, start + params['batch_size']) for start in
               range(0, num_examples, params['batch_size'])]
        sources_set.append(sources)
        targets_set.append(targets)
        batches_set.append(batches)
    file = open('models/loss.txt', 'w')
    for epoch in range(params['epochs']):
        ep_loss = 0.
        start_time = time.time()

        # for each batch, calculate loss and optimize model parameters
        for i in range(len(batches_set)):
            batches = batches_set[i]
            random.shuffle(batches)
            sources = sources_set[i]
            targets = targets_set[i]
            for b_idx, (start, end) in enumerate(batches):
                source = sources[start:end]
                target = targets[start:end]



                source_mask, target_mask = create_masks(
                    source.cpu(), target.cpu())
                source_mask = source_mask.cuda()
                target_mask = target_mask.cuda()
                preds = net(source, target, source_mask, target_mask)


                preds = preds[:, :-1, :].contiguous().view(-1,
                                                    net.target_vocab)
                labels = target[:, 1:].contiguous().view(-1)
                loss = criterion(preds, labels.long())
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                ep_loss += loss.item()
                torch.cuda.empty_cache()

        curr_loss = 'epoch: %d, loss: %0.2f, time: %0.2f sec' % (
            epoch, ep_loss, time.time() - start_time)
        print(curr_loss)
        substr = 'mytraining'+str(epoch)+'.pt'
        path = 'models/' + substr
        torch.save(net.state_dict(), path)

        file.write(curr_loss+'\n')


    file.close()


train_lm(data_1, params, model)


def traslante_sentence(curr_sent, max_len_1, input_lang, output_lang):
    source, target = pair_to_indexes(
        curr_sent, max_len_1, input_lang, output_lang)
    target_fake = np.zeros((1, max_len_1))
    target_fake[0][0] = 1
    target_temp = target_fake

    for i in range(max_len_1-2):
        sou = torch.from_numpy(source)
        tar = torch.from_numpy(target_fake)
        sou = sou.cuda()
        tar = tar.cuda()
        source_mask, target_mask = create_masks(sou.cpu(), tar.cpu())
        source_mask = source_mask.cuda()
        target_mask = target_mask.cuda()
        preds = model(sou, tar, source_mask, target_mask)



        preds = preds[:, :-1, :].contiguous().view(-1, model.target_vocab)
        ss = torch.softmax(preds, dim=-1)
        mm = torch.max(ss, dim=-1)[1]
        target_temp[0][i+1] = mm[i]
        target_fake = target_temp
    result = ''
    for idx in mm:
        if idx == 0:
            break
        index = idx.item()
        if index == 2:
            break
        result += output_lang.index2word[index]+' '
    print(result)
