#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    :   2019/01/22 20:16:14
@Author  :   Aaron Chou
@Desc    :   None
'''

# here put the import lib



class Config():
    def __init__(self, ):
        pass
    msraner_dir = "D:/Project/Python3/ChineseNER/data/msra_ner/"    # D:\Project\Python3\ChineseNER\data\msra_ner
    # 随机分10% 作为验证集。
    msraner_train_file = msraner_dir + "train1.txt"
    msraner_test_file = msraner_dir + "testright1.txt"
    msraner_tags_file = msraner_dir + "tags.txt"

    # bert config
    bert_dir = "D:/ProjectData/NLP/bert/chinese_wwm_ext_L-12_H-768_A-12/"
    bert_vocab_file = bert_dir + "vocab.txt"
    bert_config_file = bert_dir + "bert_config.json"
    bert_init_checkpoint = bert_dir + "bert_model.ckpt"

    use_onehot_embeddings = False
    warmup_proportion = 0.01
    is_bert_training = 1
    