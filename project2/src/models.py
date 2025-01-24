import torch.nn as nn
import torch
import torch.nn.functional as f

class TextRNN(nn.Module):
    def __init__(self, vocab_size,
                 embedding_dim,
                 hidden_size,
                 num_classes,
                 weights=None,
                 rnn_type='RNN'):
        super(TextRNN, self).__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.rnn_type = rnn_type

        if weights is not None:
            # num_embeddings是嵌入词典的大小 就是可以支持的输入索引的范围 如果有1000个词那就设定为1000
            # embedding_dim每个词嵌入的长度
            # _weight是自定义的初始化权重 通常是尺寸为(num_embeddings, embedding_dim)的张量 下划线为私有属性
            self.embed = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=embedding_dim,
                                      _weight=weights)
        else:
            self.embed = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=embedding_dim)
            
        if rnn_type == 'RNN':
            # batch_first参数用于指定输入和输出tensor的维度排列方式 具体来说 它决定了batch size的维度
            # True IOtensor的形状为(batch_size, seq_length, input_size) batch size是第一个维度 适合大多数pytorch数据处理方式
            # False IOtensor的形状为(seq_length, batch_size, input_size) seq_length是第一个维度 传统RNN的常见数据格式
            self.rnn = nn.RNN(input_size=embedding_dim, hidden_size=hidden_size, batch_first=True)
            # self.hidden2label全连接层（线性层），其作用是将输入张量映射到输出类别空间。
            self.hidden2label = nn.Linear(hidden_size, num_classes)
        elif rnn_type == 'LSTM':
            # LSTM也可以直接调用，甚至可以直接选用bidirectional
            self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, batch_first=True, bidirectional=True)
    
    def forward(self, input_sents):
        '''前向传播 对输入的input_sents进行词嵌入处理 然后通过RNN或LSTM处理 生成最后logits
        '''
        # 得到input_sents的尺寸，并把它用之前定义的self.embed转换成嵌入向量，尺寸为(batch_size, seq_len, embedding_dim)
        batch_size, seq_len = input_sents.shape
        embed_out = self.embed(input_sents)

        if self.rnn_type == 'RNN':
            # randn随机初始化隐藏状态h0，形状如下
            # self.rnn那里就是直接输出_(整个序列的隐藏状态)和hn(最后时间步的隐藏状态)
            # hn形状和h0一样
            h0 = torch.randn(1, batch_size, self.hidden_size)
            _, hn = self.rnn(embed_out, h0)
        elif self.rnn_type ==  'LSTM':
            # randn随机初始化隐藏状态h0和细胞状态c0，每个形状如下
            h0, c0 = torch.randn(2, batch_size, self.hidden_size), torch.randn(2, batch_size, self.hidden_size)
            output, (hn, cn) = self.lstm(embed_out, (h0, c0))
        # 把隐藏状态hn通过一个线性层self.hidden2label，生成分类logits
        # self.hidden2label是一个nn.Linear()类型，squeeze(0)是去掉第0维
        # 在代码中，hn的形状是(num_directions, batch_size, hidden_size) 第0维是num_directions 就是说RNN/LSTM是否是单向或双向
        # 如果是单向则为1 如果是双向则为2 
        logits = self.hidden2label(hn).squeeze(0)

        return logits
    
if __name__ == '__main__':
    textrnn = TextRNN(vocab_size=100, embedding_dim=100, hidden_size=100, num_classes=5, weights=None)