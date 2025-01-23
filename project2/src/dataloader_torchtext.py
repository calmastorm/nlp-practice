'''
这个dataloader_torchtext.py无法使用python3.12运行
torch和torchtext的版本也非常重要 必须匹配才可以
'''

import os
import pandas as pd
import spacy # 这个库是用来做自然语言处理的
from sklearn.model_selection import train_test_split
from torch.nn import init
from torchtext import data

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, '../../data/datasets')

def prepare_data(dataset_path, sent_col_name, label_col_name, debug=False):
    '''read the sentence and labels from the dataset'''
    file_path = os.path.join(dataset_path, 'train.tsv')
    data = pd.read_csv(file_path, sep='\t')
    if debug:
        data = data.sample(n=100)
    # 这里就是用刚才读取的data来分一下数据
    X, y = data[sent_col_name].values, data[label_col_name].values
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    # 这里就是转成dataframe
    train_df, val_df = pd.DataFrame(), pd.DataFrame()
    train_df['sent'], train_df['label'] = X_train, y_train
    val_df['sent'], val_df['label'] = X_val, y_val

    # 这里就是把dataframe转成csv文件 并储存好
    train_file_path = os.path.join(dataset_path, 'train.csv')
    val_file_path = os.path.join(dataset_path, 'val.csv')
    train_df.to_csv(train_file_path, index=False)
    val_df.to_csv(val_file_path, index=False)

    # 返回的是两个文件的路径
    return train_file_path, val_file_path

def dataset2dataloader(dataset_path=file_path, 
                       sent_col_name='Phrase',
                       label_col_name='Sentiment',
                       batch_size=32,
                       vec_file_path='P:/nlp-practice/data/datasets/glove.6B/glove.6B.50d.txt',
                       debug=False):
    train_file_name, val_file_name = prepare_data(dataset_path, 
                                                  sent_col_name=sent_col_name,
                                                  label_col_name=label_col_name,
                                                  debug=debug)
    spacy_en = spacy.load('en_core_web_sm')
    def tokenizer(text):
        '''定义分词操作'''
        return [tok.text for tok in spacy_en.tokenizer(text)]
    TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True)
    LABEL = data.Field(sequential=False, use_vocab=False)
    train, val = data.TabularDataset.splits(
        path='', train=train_file_name, validation=val_file_name,
        format='csv', skip_header=True, fields=[('sent', TEXT), ('label', LABEL)])

    TEXT.build_vocab(train, vectors=vec_file_path, max_size=2000)
    # 当 corpus 中有的 token 在 vectors 中不存在时 的初始化方式.
    TEXT.vocab.vectors.unk_init = init.xavier_uniform

    DEVICE = 'cpu'
    train_iter = data.BucketIterator(train, batch_size=batch_size, sort_key=lambda x: len(x.review), device=DEVICE)
    val_iter = data.BucketIterator(val, batch_size=batch_size, sort_key=lambda x: len(x.review), device=DEVICE, shuffle=True)

    # 在 test_iter , sort一定要设置成 False, 要不然会被 torchtext 搞乱样本顺序
    # test_iter = data.Iterator(dataset=test, batch_size=128, train=False, sort=False, device=DEVICE)

    return train_iter, val_iter, TEXT.vocab.vectors

if __name__ == '__main__':
    train_iter, val_iter, vectors = dataset2dataloader(batch_size=32, debug=True)

    batch = next(iter(train_iter))
    print(batch.sent.shape)
    print(batch.label.shape)