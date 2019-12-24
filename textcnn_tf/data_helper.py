# -*- coding: utf-8 -*-
# @Time     :2019/11/20 16:03
# @Author   :XiaoMa
# @File     :data_helper.py

import pandas as pd

#合并训练集和验证集
df_train=pd.read_csv('train.csv')
df_dev=pd.read_csv('val.csv')

print(len(df_train),len(df_dev))
train=pd.concat([df_train,df_dev],axis=0)
# print(train.columns)
train.drop('手机号',axis=1,inplace=True)
train.to_csv('df_train.csv',header=None,index=None,sep='\t')







