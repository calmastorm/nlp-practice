import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# from project1.src.data_process import read_data
# from project1.src.feature_extraction import BagOfWords, NGrams
# from project1.src.softmax_regression import SoftmaxRegression

from data_process import read_data
from feature_extraction import BagOfWords, NGrams
from softmax_regression import SoftmaxRegression

# main
if __name__ == '__main__':
    # read data
    debug = 1
    X_data, y_data = read_data()
    # debug快速调试，如果debug=1，只取前500个数据
    if debug == 1:
        print('Debug mode, only use 500 data.')
        X_data = X_data[:3000]
        y_data = y_data[:3000]
    else:
        print('Normal mode, use all data.')
    # 这里y_data是list，需要转换成numpy array
    y = np.array(y_data).reshape(len(y_data), 1)

    # split data
    print('Spliting data...')
    bow_model = BagOfWords()
    ngrams_model = NGrams(ngram=[1, 2])
    X_bow = bow_model.fit_transform(X_data)
    X_ngrams = ngrams_model.fit_transform(X_data)

    print('Bag of Words:', X_bow.shape)
    print('N-grams:', X_ngrams.shape)

    # 该方法来自sklearn.model_selection，用于将数据集划分为训练集和测试集
    # test_size要么是整数，要么是浮点数，如果是整数，则表示测试集的样本数量，如果是浮点数，则表示测试集的比例
    X_train_bow, X_test_bow, y_train_bow, y_test_bow = train_test_split(X_bow, y, test_size=0.2, random_state=42, stratify=y)
    X_train_ngrams, X_test_ngrams, y_train_ngrams, y_test_ngrams = train_test_split(X_ngrams, y, test_size=0.2, random_state=42, stratify=y)
    print('Spliting data done.')

    # train model
    epochs = 100
    bow_learning_rate = 0.001
    ngrams_learning_rate = 0.1

    print(f'Training model...{epochs} epochs, bow-lr {bow_learning_rate}')
    model1 = SoftmaxRegression()
    history1 = model1.fit(X_train_bow, 
                          y_train_bow, 
                          learning_rate=bow_learning_rate,
                          epochs=epochs,
                          num_classes=5,
                          print_loss_step=epochs//10,
                          update_strategy='stochastic')
    plt.plot(np.arange(len(history1)), np.array(history1))
    plt.title('Bag of Words')
    plt.show()
    print(f'Bow train {model1.score(X_train_bow, y_train_bow)} test {model1.score(X_test_bow, y_test_bow)}')

    print(f'Training model...{epochs} epochs, ngrams-lr {ngrams_learning_rate}')
    model2 = SoftmaxRegression()
    history2 = model2.fit(X_train_ngrams, 
                          y_train_ngrams, 
                          learning_rate=ngrams_learning_rate,
                          epochs=epochs,
                          num_classes=5,
                          print_loss_step=epochs//10,
                          update_strategy='stochastic')
    plt.plot(np.arange(len(history2)), np.array(history2))
    plt.title('N-grams')
    plt.show()
    print(f'N-grams train {model2.score(X_train_ngrams, y_train_ngrams)} test {model2.score(X_test_ngrams, y_test_ngrams)}')