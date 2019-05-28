# coding=utf8
import itertools

import jieba
import numpy as np
from torch.utils.data import Dataset

from utils import *


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]


# Meaning	    Positive	Neutral	    Negative	Not mentioned
# Old labels    1	        0	        -1	        -2
# New labels    3           2           1           0
def map_sentimental_type(value):
    '''
    将原始的情感值+2变成非负数
    :param value:
    :return:
    '''
    return value + 2


def parse_user_reviews(user_reviews):
    """
    解析评论数据
    todo
    :param user_reviews:
    :return: samples
    """
    samples = []
    for i in range(len(user_reviews)):
        content = user_reviews['content'][i]
        # 构造一个20*1的矩阵，类型是int
        label_tensor = np.empty((num_labels,), dtype=np.int32)
        for idx, name in enumerate(label_names):
            sentimental_type = user_reviews[name][i]
            y = map_sentimental_type(sentimental_type)
            # label_tensor[:, idx] = to_categorical(y, num_classes)
            # CrossEntropyLoss does not expect a one-hot encoded vector as the target, but class indices.
            label_tensor[idx] = y
        samples.append({'content': content, 'label_tensor': label_tensor})
    return samples


def zeroPadding(indexes_batch, fillvalue=PAD_token):
    '''
    0填充
    :param indexes_batch: [[word indexs],[...]]
    :param fillvalue:  填充值
    :return:
    '''
    # zip_longest('ABCD', 'xy', fillvalue='-') --> Ax By C- D-
    return list(itertools.zip_longest(*indexes_batch, fillvalue=fillvalue))


# Returns padded input sequence tensor and lengths
def inputVar(indexes_batch):
    '''
    :param indexes_batch: 批量的评价文本 []
    :return:padlist 每个句子对其长度之后的结果,lengths  每个句子的长度
    '''
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths


# Returns all items for a given batch of pairs
def batch2TrainData(pair_batch):
    '''
    将输入编码整理成为模型识别的格式
    :param pair_batch: [[[iuput word index] ,[list result]), ...]
    :return: inp 输入的句子集合（长度对其后） lengths 输入的每个句子的长度 output 输出结果
    '''
    pair_batch.sort(key=lambda x: len(x[0]), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = inputVar(input_batch)
    output = torch.LongTensor(output_batch)
    return inp, lengths, output


class SaDataset(Dataset):
    '''
        数据集
        spilt 数据类型
        voc 词库
        samples 样本，格式：
        num_chunks 组数
    '''

    def __init__(self, split, voc):
        self.split = split
        self.voc = voc
        assert self.split in {'train', 'valid'}

        if split == 'train':
            filename = os.path.join(train_folder, train_filename)
        elif split == 'valid':
            filename = os.path.join(valid_folder, valid_filename)
        else:
            filename = os.path.join(test_a_folder, test_a_filename)

        user_reviews = pd.read_csv(filename)
        # 解析数据集 解析成为{'content': content, 'label_tensor': label_tensor}的格式
        self.samples = parse_user_reviews(user_reviews)
        self.num_chunks = len(self.samples) // chunk_size

    def __getitem__(self, i):
        pair_batch = []

        for i_chunk in range(chunk_size):
            idx = i * chunk_size + i_chunk
            content = self.samples[idx]['content']
            content = content.strip()
            seg_list = jieba.cut(content)
            input_indexes = encode_text(self.voc.word2index, list(seg_list))
            label_tensor = self.samples[idx]['label_tensor']
            pair_batch.append((input_indexes, label_tensor))

        return batch2TrainData(pair_batch)

    def __len__(self):
        return self.num_chunks


if __name__ == '__main__':
    tensor = torch.tensor([1, 3, 1])
    print(tensor)
