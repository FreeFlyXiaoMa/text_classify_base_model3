#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
import sys
import pickle
import logging
import jieba

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tqdm import tqdm
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s -- %(name)s -- %(levelname)s -- %(message)s')
logger = logging.getLogger(__name__)

class fastTextfeature(object):
    def __init__(self, data_path='./data/context'):
        logger.info('fastTextfeature loading corpus ...')
        self.label_list = ['Military', 'Economy', 'Culture', 'Sports', 'Auto', 'Medicine']

        # 枚举所有的文件
        jieba.enable_parallel(8)
        self.context, self.label = [], []
        for file in tqdm(os.listdir(path=data_path)):
            try:
                label = file.split('_')[0]
                filePath = os.path.join(data_path, file)
                with open(filePath, 'r', encoding='utf-8') as fd:
                    context = fd.read().replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
                self.context.append(context)
                self.label.append(self.label_list.index(label))
            except:
                logger.warning('file %s have some problem ...' % file)
        self.context = [' '.join(list(jieba.cut(context))) for context in tqdm(self.context)]
        self.train_context, self.test_context, self.train_label, self.test_label =\
            train_test_split(self.context, self.label, test_size=0.05)

        train_data_fd = open('./data/fastTextData/train_data', 'w+')
        for label, context in zip(self.train_label, self.train_context):
            train_data_fd.write("__label__" + str(label) + '\t' + context + '\n')
        train_data_fd.close()

        valid_data_fd = open('./data/fastTextData/valid_data', 'w+')
        for label, context in zip(self.test_label, self.test_context):
            valid_data_fd.write("__label__" + str(label) + '\t' + context + '\n')
        valid_data_fd.close()

        logger.debug('self.train_context shape: %d' % len(self.train_context))
        logger.debug('self.test_context shape: %d' % len(self.test_context))
        logger.debug('self.train_label shape: %d' % len(self.train_label))
        logger.debug('self.test_label shape: %d' % len(self.test_label))


if __name__ == '__main__':
    # f = NNfeature()
    f = fastTextfeature()
    # f = DynamicRnnfeature()

