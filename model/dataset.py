#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    :   2019/01/22 20:35:53
@Author  :   Aaron Chou
@Desc    :   None
'''

# here put the import lib


import numpy as np
import codecs
import random
import tensorflow as tf
from collections import defaultdict
from .utils import read_file

random.seed(9102)

UNK = "[UNK]"
PAD = "[PAD]"
TAG_O = "o"
TAG_B = "b_"
TAG_I = "i_"


class CustomIOError(Exception):

    def __init__(self, filename):
        message = "ERROR: Unable to locate file {}".format(filename)
        super(CustomIOError, self).__init__(message)


class MsraNer(object):
    def __init__(self, file_name, config, tags2file=0):
        self.file_name = file_name
        self.config = config
        self.word_dict = load_dict(config.bert_vocab_file)
        self.tag_dict = load_dict(config.msraner_tags_file)
        self.dataset = []
        # [word_list, word_id_list, tag_list]
        self.get_dataset()
        if tags2file:
            self.write_tag2file(config.msraner_tags_file)

    pass

    def get_dataset(self):
        self.tags = set()
        for spline in read_file(self.file_name, " "):
            word_list, word_id_list, tag_id_list = [], [], []
            for each in spline:
                tmp_arr = each.split('/')
                if len(tmp_arr) != 2:
                    print("error line element,", each)
                    continue
                words, tag = tmp_arr
                self.tags.add(tag)
                for i, w in enumerate(words):
                    word_list.append(w)
                    word_id_list.append(transfer_str2id(w, self.word_dict, 1))
                    cur_tag = tag if tag == TAG_O else TAG_B + tag
                    if tag != 'o' and i != 0:
                        cur_tag = TAG_I + tag
                    tag_id_list.append(transfer_str2id(cur_tag, self.tag_dict, 0))
            self.dataset.append([word_list, word_id_list, tag_id_list])
            pass
        pass

    # tag写出到文件, bio格式
    def write_tag2file(self, file_name):
        with open(file_name, 'w') as out:
            out.write(TAG_O +'\n')
            for tag in self.tags:
                if tag == TAG_O:
                    continue
                out.write("{}{}\n".format(TAG_B, tag))
                out.write("{}{}\n".format(TAG_I, tag))

    def __iter__(self):
        random.shuffle(self.dataset)
        for each in self.dataset:
            yield each[:8]

    def __len__(self):
        return len(self.dataset)

def load_dict(file_name):
    id_dict = {}
    for i, line in enumerate(read_file(file_name, only_first=1)):
        line = line.strip()
        if len(line) == 0:
            continue
        id_dict[line] = i
    return id_dict

def transfer_str2id(cur_str:str, id_dict:{}, allow_unk=0):
    if cur_str not in id_dict:
        if allow_unk and UNK in id_dict:
            return id_dict[UNK]
        else:
            print("do not find str in id_dict, allow_unk={}, UNK={}".format(allow_unk, UNK))
    return id_dict[cur_str]