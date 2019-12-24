import numpy as np
from fasttext import train_supervised
from tqdm import tqdm
from fasttext import FastText

# from fastText import train_supervised
from sklearn.metrics import classification_report

def print_results(N, p, r):
    print("N\t" + str(N))
    print("P@{}\t{:.3f}".format(1, p))
    print("R@{}\t{:.3f}".format(1, r))


if __name__ == "__main__":
    train_data = './data/train_data'
    valid_data = './data/valid_data'

    test_data = [line.strip().split('\t')[1] for line in open(valid_data, "r")]
    test_label = [line.strip().split('\t')[0] for line in open(valid_data, "r")]
    model = train_supervised(input=train_data,
                             dim=100,
                             lr=0.1,
                             wordNgrams=2,
                             minCount=1,
                             bucket=10000000,
                             epoch=6,
                             thread=4,
                             label='__label__')
    print('test_data.shape:',np.array(test_data).shape)
    print('test_label.shape',np.array(test_label).shape)
    print('test:',model.test("./data/valid_data"))
    # pickle.dump(model,open('fasttext_model.pkl','w'))
    print('模型保存！')
    model.save_model('model.fasttext')

    model__=FastText.load_model('model.fasttext')
    predict_label = []
    for line in tqdm(test_data):
        result, proba = model__.predict(line)
        predict_label.append(result[0])
    print(classification_report(y_pred=predict_label, y_true=test_label))





