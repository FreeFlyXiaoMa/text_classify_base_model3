# -*- coding: utf-8 -*-
# @Time     :2019/11/26 11:24
# @Author   :XiaoMa
# @File     :textrnn_train.py


import TrainModel as textrnn_train

import tensorflow as tf
import json

# tf.app.flags.DEFINE_string("model_type", "transformer", "默认为cnn")
# FLAGS = tf.app.flags.FLAGS
model_type = 'textrnn'

with open("/E/home/mayajun/PycharmProjects/text_classify_base_model/config/config.json", "r", encoding='utf-8') as f:
    data = json.loads(f.read())
    model_parameters = data['model'][model_type]['model_parameters']

# is_cut是否对语句进行分词
# model_type is one of the ["textcnn","charcnn","fasttext","textrnn","birnn_attention","han","leam","transformer"]
train = None

if model_type == 'textrnn':
    train = textrnn_train.TrainModel()
    embedding_dim = model_parameters['embedding_dim']
    dropout_keep_prob = model_parameters['dropout_keep_prob']
    hidden_num = model_parameters['hidden_num']
    hidden_size = model_parameters['hidden_size']
    l2_reg_lambda = model_parameters['l2_reg_lambda']
    train.trainModel(embedding_dim, dropout_keep_prob, hidden_num, hidden_size, l2_reg_lambda)

