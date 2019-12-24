# -*- coding: utf-8 -*-
# @Time     :2019/11/26 13:13
# @Author   :XiaoMa
# @File     :birnn_attention_train.py

import birnn_attention_tf.TrainModel as birnn_attention_train

import tensorflow as tf
import json

# tf.app.flags.DEFINE_string("model_type", "transformer", "默认为cnn")
# FLAGS = tf.app.flags.FLAGS
model_type = 'birnn_attention'

with open("config/config.json", "r", encoding='utf-8') as f:
    data = json.loads(f.read())
    model_parameters = data['model'][model_type]['model_parameters']

# is_cut是否对语句进行分词
# model_type is one of the ["textcnn","charcnn","fasttext","textrnn","birnn_attention","han","leam","transformer"]
train = None

train = birnn_attention_train.TrainModel()
embedding_dim = model_parameters['embedding_dim']
dropout_keep_prob = model_parameters['dropout_keep_prob']
hidden_num = model_parameters['hidden_num']
attn_size = model_parameters['attn_size']
train.trainModel(embedding_dim, dropout_keep_prob, hidden_num, attn_size)



