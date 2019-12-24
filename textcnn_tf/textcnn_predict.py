import tensorflow as tf
import pandas as pd

sentences = '微信可以登录吗'

infer = None
if __name__=='__main__':
    import Infer as textcnn_infer
    from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

    infer = textcnn_infer.Infer()

    #对模型进行评估
    df_test=pd.read_csv('test.csv')

    label_true_list=[]
    label_pred_list=[]
    for index in df_test.index:
        sent=df_test['用户'][index]
        # print(sent)
        true_label=str(df_test['polar'][index])
        label_true_list.append(true_label)
        # print(type(true_label))

        result=infer.infer([sent])[0][0]
        label_pred_list.append(result)
        # print(type(result))

    acc=round(accuracy_score(label_true_list,label_pred_list),4)
    precision=round(precision_score(label_true_list,label_pred_list,average='macro'),4)
    recall=round(recall_score(label_true_list,label_pred_list,average='macro'),4)
    f1=round(f1_score(label_true_list,label_pred_list,average='macro'),4)

    print('准确率：',acc,'精确率：',precision,'召回率：',recall,'f1：',f1)


# print(infer.infer([sentences]))