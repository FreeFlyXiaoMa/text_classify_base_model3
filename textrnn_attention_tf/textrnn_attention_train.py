# -*- coding: utf-8 -*-
# @Time     :2019/11/26 18:43
# @Author   :XiaoMa
# @File     :textrnn_attention.py

from TrainModel import TrainModel


import json

# tf.app.flags.DEFINE_string("model_type", "transformer", "默认为cnn")
# FLAGS = tf.app.flags.FLAGS
model_type = 'birnn_attention'

with open("/E/home/mayajun/PycharmProjects/text_classify_base_model/config/config.json", "r", encoding='utf-8') as f:
    data = json.loads(f.read())
    model_parameters = data['model'][model_type]['model_parameters']

# is_cut是否对语句进行分词
# model_type is one of the ["textcnn","charcnn","fasttext","textrnn","birnn_attention","han","leam","transformer"]
train = None

if model_type == 'birnn_attention':
    train = TrainModel()
    embedding_dim = model_parameters['embedding_dim']
    dropout_keep_prob = model_parameters['dropout_keep_prob']
    hidden_num = model_parameters['hidden_num']
    attn_size = model_parameters['attn_size']
    train.trainModel(embedding_dim, dropout_keep_prob, hidden_num, attn_size)
