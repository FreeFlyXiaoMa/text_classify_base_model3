import TrainModel as textcnn_train

import tensorflow as tf
import json


model_type = 'textcnn'

with open("config.json", "r", encoding='utf-8') as f:
    data = json.loads(f.read())
    model_parameters = data['model'][model_type]['model_parameters']

# is_cut是否对语句进行分词
train = None
if __name__=='__main__':

    train = textcnn_train.TrainModel()
    embedding_dim = model_parameters['embedding_dim']
    filter_sizes = model_parameters['filter_sizes']
    num_filters = model_parameters['num_filters']
    dropout_keep_prob = model_parameters['dropout_keep_prob']
    l2_reg_lambda = model_parameters['l2_reg_lambda']
    train.trainModel(embedding_dim, filter_sizes, num_filters, dropout_keep_prob, l2_reg_lambda)

