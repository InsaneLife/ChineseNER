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



if __name__ == "__main__":
    conf = Config()
    msra_train = dataset.MsraNer(conf.msraner_train_file, conf, tags2file=0)
    pass
