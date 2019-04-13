

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

# In[2]:


USE_CUDA = False

# In[3]:


PAD_token = 0
SOS_token = 1
EOS_token = 2


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
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


# In[4]:


# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
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
    line1 = open('data/train.de').read().strip().split('\n')
    line2 = open('data/train.en').read().strip().split('\n')  # .splitlines()

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
    input_lang, output_lang, pairs = read_langs(lang1_name, lang2_name, reverse)
    print("Read %s sentence pairs" % len(pairs))

    print("Indexing words...")
    for pair in pairs:
        input_lang.index_words(pair[0])
        output_lang.index_words(pair[1])

    return input_lang, output_lang, pairs


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
    vector = [SOS_token] + vector
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





def pair_to_indexes(pairs, max_len, input_lang, output_lang):
    result = []
    for pair in pairs:
        # add start token for english
        sent2 = padding(indexes_from_sentence(output_lang, pair[1]), max_len)
        # add end token for german
        sent1 = padding_both(indexes_from_sentence(input_lang, pair[0]), max_len)
        result.append([sent1, sent2])

    return result


