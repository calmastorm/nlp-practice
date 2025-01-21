import pandas as pd
import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split
# pad_sequence is used to pad variable length sequences
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, '../../data/datasets')

def prepare_data(dataset_path, sent_col_name, label_col_name):
    '''read the sentence and labels from the dataset'''
    file_path = os.path.join(dataset_path, 'train.tsv')
    data = pd.read_csv(file_path, sep='\t') # 这里熟悉吧 读成dataframe
    # 使用values转换成numpy array，不用values的话，X和y都是pandas series
    X, y = data[sent_col_name].values, data[label_col_name].values
    return X, y

class Language:
    '''根据句子列表简历词典，并且把单词列表转换为数值型，其实就是fit和transform'''
    def __init__(self):
        # 做id2word，可能是因为还会需要把数值型的单词转换为原始的单词
        self.word2id = {}
        self.id2word ={}
    
    def fit(self, sent_list):
        vocab = set()
        for sent in sent_list:
            vocab.update(sent.split()) # update是把set合并 这里就是不断把新句子的单词加进去
        # 这里把set转换为list，然后排序，然后转换为dict
        word_list = ['<pad>', '<unk>'] + list(vocab)
        self.word2id = {word: idx for idx, word in enumerate(word_list)}
        self.id2word = {idx: word for idx, word in enumerate(word_list)}

    # transform这个方法的目的是，把句子列表转换为数值型的，不再像之前是统计词频
    def transform(self, sent_list, reverse=False): # reverse是不是要把数值型的转换为原始的
        sent_list_id = []
        # 选用word mapper是看要不要reverse，如果不要就是word2id，要就是id2word
        word_mapper = self.word2id if not reverse else self.id2word
        # unk这个变量 不reverse的话 里面存的是unk的id 如果reverse就是None
        unk = self.word2id['<unk>'] if not reverse else None
        for sent in sent_list:
            # 这里是把句子转换为数值型的 具体操作如下
            # sent.split(' ')是把句子分割成单词列表 由于map函数的存在 里面的词会一个接一个通过lambda函数
            # lambda函数里 x就是输入 使用word_mapper.get(x, unk)来获取x的id 如果x不存在就是unk
            # 一整个句子都获取id后 map函数就结束了 但是我们要一个list 所以外面套一个list函数
            # 最后把list函数加入sent_list_id中 这个写法还是挺牛的
            sent_id = list(map(lambda x: word_mapper.get(x, unk), sent.split(' ') if not reverse else sent))
            sent_list_id.append(sent_id)
        return sent_list_id
    

    