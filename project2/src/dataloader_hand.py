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
    
class TextDataset(Dataset):
    '''继承Dataset类，实现__len__和__getitem__方法'''
    def __init__(self, sents, labels):
        self.sents = sents
        self.labels = labels

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, item):
        return self.sents[item], self.labels[item]

def collate_fn(batch_data):
    '''自定义一个batch里面的数据的组织方式'''
    # 这里的batch_data是一个list，里面的元素是一个tuple，tuple里面是一个句子的list和一个label
    # batch_data = [(['sent1', 'sent2], label1), (['sent3', 'sent4'], label2), ...] ？？？
    # data_pair[0]是一个list，data_pair[1]是一个label
    # 这里通过lambda来排序，依据是更长的sents排前面。这样做是为了后面的pad_sequence
    batch_data.sort(key=lambda data_pair: len(data_pair[0]), reverse=True)

    # 这里是把batch_data里面的句子和label分开，然后把句子的长度存储在sents_len里面
    # torch.LongTensor(sent)是把sents中的每个sent拿出来，转换成LongTensor，然后再组合起来。
    # padded_sents使用了pad_sequence函数，这个函数的作用是把不等长的句子填充到相同长度
    # 其中batch_first=True是指第一个维度是batch_size， padding_value=0是指填充的值是0
    sents, labels = zip(*batch_data)
    sents_len = [len(sent) for sent in sents]
    sents = [torch.LongTensor(sent) for sent in sents]
    padded_sents = pad_sequence(sents, batch_first=True, padding_value=0)
    
    # 这里是把label转换为LongTensor，原因是label是整数
    # 把sents_len转换为FloatTensor的原因是，后面要用到的是FloatTensor
    return torch.LongTensor(padded_sents), torch.LongTensor(labels), torch.FloatTensor(sents_len)

def get_wordvec(word2id, vec_file_path, vec_dim=50):
    '''读出txt文件的预训练词向量'''
    # 这里做word_vectors的意义是，把word2id里面的词转化为词向量
    # word_vectors是一个矩阵，行数=词的个数，列数=词向量的维度（默认50）。使用xavier_uniform_是因为这个函数可以初始化权重
    print('Reading word vectors...')
    word_vectors = torch.nn.init.xavier_uniform_(torch.empty(len(word2id), vec_dim))
    word_vectors[0, :] = 0 # <pad>的词向量是0 这里的意思是把第一行的所有元素都变成0
    found = 0 # 找到的词向量的个数
    with open(vec_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split(' ')
            if line[0] in word2id: # 只看line[0]是因为后面的是词向量 只有第一个是单词
                found += 1
                word_vectors[word2id[line[0]]] = torch.tensor(list(map(lambda x: float(x), line[1:])))
                if found == len(word2id) - 1:
                    break
    print(f'Found {found} words with pre-trained word vectors.')
    return word_vectors.float()

def make_dataloader(dataset_path=file_path, 
                    sent_col_name='Phrase', 
                    label_col_name='Sentiment',
                    batch_size=32,
                    vec_file_path='P:/nlp-practice/data/datasets/glove.6B/glove.6B.50d.txt',
                    debug=False):
    X, y = prepare_data(dataset_path, sent_col_name, label_col_name)
    if debug:
        X, y = X[:100], y[:100]
    
    X_language = Language()
    X_language.fit(X)
    X = X_language.transform(X)

    word_vectors = get_wordvec(X_language.word2id, vec_file_path=vec_file_path, vec_dim=50)

    # checking
    # print(f'X但不要最后两个\n{X[:2]}')
    # X_id = X_language.transform(X[:2], reverse=True)
    # print(f'这是什么？\n{X_id}')

    # split the dataset into training and validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    # 这里是把训练集和测试集的句子和标签分别放入TextDataset类中
    # 上面的Language类和TextDataset类的区别在于，Language类是用来处理句子的，TextDataset类是用来处理数据集的
    text_train_dataset, text_test_dataset = TextDataset(X_train, y_train), TextDataset(X_test, y_test)
    text_train_dataloader = DataLoader(text_train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_train_dataloader = DataLoader(text_test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return text_train_dataloader, test_train_dataloader, word_vectors, X_language

if __name__ == '__main__':
    text_train_dataloader, text_test_dataloader, word_vectors, X_language = make_dataloader(debug=True)
    for batch in text_train_dataloader:
        X, y, lens = batch
        print(X.shape, y.shape)
        break