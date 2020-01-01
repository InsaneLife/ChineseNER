#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    :   2019/01/22 20:16:14
@Author  :   Aaron Chou
@Desc    :   None
'''

# here put the import lib
from model.utils import read_file


class Config():
    def __init__(self, ):
        self.get_config()
        pass
    
    user_dir = "/Users/alex/"
    msraner_dir = user_dir + "/Project/Python3/ChineseNER/data/msra_ner/"  # D:\Project\Python3\ChineseNER\data\msra_ner
    # 随机分10% 作为验证集。
    msraner_train_file = msraner_dir + "train1.txt"
    msraner_test_file = msraner_dir + "testright1.txt"
    msraner_tags_file = msraner_dir + "tags.txt"

    # bi lstm parameter
    keep_prob = 0.7
    rnn_layer_size = 64
    dim_word = 64
    use_crf = 1

    batch_size = 64
    nepochs = 2
    learning_rate = 1e-3
    lr_decay = 0.9

    # bert config
    bert_dir = user_dir + "/ProjectData/NLP/bert/chinese_wwm_ext_L-12_H-768_A-12/"
    bert_vocab_file = bert_dir + "vocab.txt"
    bert_config_file = bert_dir + "bert_config.json"
    bert_init_checkpoint = bert_dir + "bert_model.ckpt"

    use_onehot_embeddings = False
    warmup_proportion = 0.01
    is_bert_training = 1

    embeddings = None

    def get_config(self):
        self.word_size = len(read_file(self.bert_vocab_file))
        self.tag_size = len(read_file(self.msraner_tags_file))
