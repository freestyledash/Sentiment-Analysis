import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from config import num_labels, num_classes, batch_first


class EncoderRNN(nn.Module):
    '''

    模型类

    '''

    def __init__(self, input_size, hidden_size, n_layers=1, dropout=0):
        '''
        :param input_size: 词库大小
        :param hidden_size: 隐藏层大小，代表了一个词向量的大小
        :param n_layers: rnn层数,一般在5层以内
        :param dropout: 丢包率
        '''
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)

        '''
        Initialize GRU; 
        the input_size and hidden_size params are both set to 'hidden_size'
        because input size is a word embedding with number of features == hidden_size
        '''
        self.gru = nn.GRU(hidden_size,
                          hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout),
                          bidirectional=True)
        # 全连接层，将输入矩阵 * 连接层矩阵 ,把hidden_size 变为n评价结果
        self.fc = nn.Linear(hidden_size, num_labels * num_classes)

    def forward(self, input_seq, input_lengths, hidden=None):
        '''
        每次训练的时候都执行
        :param input_seq: [[词索引序列],[词索引序列],[词索引序列]]
        :param input_lengths: [句子长度，句子长度]
        :param hidden: 隐藏层大小，一个词向量包含的信息
        :return:
        '''

        # 将词索引编码为词向量
        embedded = self.embedding(input_seq)
        # 将无用的占位符删除
        # https://www.cnblogs.com/sbj123456789/p/9834018.html
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        outputs, hidden = self.gru(packed, hidden)
        # 重新填充，返回值是 [[[20*词语状态,20*词语状态],[20*词语状态,20*词语状态]],
        # [[20*词语状态,20*词语状态],[20*词语状态,20*词语状态]]]
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        # 合并双向gru的结果,第三层总长度是40，一般是正向获得的词语状态，一半是反向获得的词语状态
        #   [[[20*词语状态],[20*词语状态]],[[20*词语状态],[20*词语状态]]]
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        # 将每个句子的最后一个词语的输出（状态）作为整个句子的输出
        # Extract the outputs for the last timestep of each example
        # 将每个句子的长度-1然后view为  [[1],[2],[3]]  然后 expand 为
        '''
        [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]]
        '''
        idx = (input_lengths - 1).view(-1, 1).expand(
            len(input_lengths), outputs.size(2))

        time_dimension = 1 if batch_first else 0
        # 增加一个维度
        idx = idx.unsqueeze(time_dimension)
        # time_dimension = 0
        # https://www.cnblogs.com/HongjianChen/p/9451526.html gather 用法
        # 获得每个行（句子）的 idx[行数] 位置上的值，组成一个1*n*20的向量，
        # 最后降维度 最后成为一个n*hidden_size的向量，n为句子的个数
        outputs = outputs.gather(
            time_dimension, Variable(idx)).squeeze(time_dimension)
        # 将所有状态进入全连接层 分类器
        # todo
        outputs = self.fc(outputs)
        outputs = outputs.view((-1, num_classes, num_labels))
        # todo
        # soft_max 让一堆树加起来等于1
        outputs = F.log_softmax(outputs, dim=1)
        return outputs
