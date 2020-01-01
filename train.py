#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Time    :   2019/01/22 20:37:11
@Author  :   Aaron Chou
@Desc    :   None
"""

# here put the import lib

from model import dataset
from config import Config
from model.bi_rnn_crf import BiRnn

if __name__ == "__main__":
    conf = Config()
    msra_train = dataset.MsraNer(conf.msraner_train_file, conf, tags2file=0)
    msra_test = dataset.MsraNer(conf.msraner_test_file, conf, tags2file=0, is_test=1)
    # s1 = next(msra_train.get_batch(2))
    model = BiRnn(conf)
    model.build_network()
    model.train(msra_train, msra_test)
    pass
