# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     predict_single_comment
   Description :
   Author :       zhangyanqi
   date：          2018/12/13
-------------------------------------------------
   Change Activity:
                   2018/12/13:
-------------------------------------------------
"""
__author__ = 'zhangyanqi'

import jieba
import torch

from config import device, save_folder
from data_gen import parse_user_reviews, batch2TrainData
from utils import Lang, encode_text


def analyze_single_comment(comment):
    encoder, voc = init()
    return analyze_after_init(comment, encoder, voc)


def init(basePath=""):
    # 加载词库
    voc = Lang(basePath + 'data/WORDMAP.json')
    print("voc.n_words: " + str(voc.n_words))

    # Load model
    checkpoint = torch.load(basePath + '{}/BEST_checkpoint.tar'.format(save_folder), map_location='cpu')
    encoder = checkpoint['encoder']

    # Use appropriate device
    encoder = encoder.to(device)

    # Set dropout layers to eval mode
    encoder.eval()

    return encoder, voc


def analyze_after_init(comment, encoder, voc):
    # 构造分析对象
    sample = {'content': comment, 'label_tensor': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
    pair_batch = []
    result = []
    content = sample['content']
    result.append({'content': content})
    content = content.strip()

    # 分词
    seg_list = jieba.cut(content)
    input_indexes = encode_text(voc.word2index, list(seg_list))
    label_tensor = sample['label_tensor']
    pair_batch.append((input_indexes, label_tensor))

    # 分析
    # todo what?
    input_variable, lengths, _ = batch2TrainData(pair_batch)
    input_variable = input_variable.to(device)
    lengths = lengths.to(device)
    outputs = encoder(input_variable, lengths)
    _, outputs = torch.max(outputs, 1)
    print('outputs.size(): ' + str(outputs.size()))
    outputs = outputs.cpu().numpy()

    # 整理输出
    result[0]['labels'] = (outputs[0] - 2).tolist()
    return result[0]

if __name__ == '__main__':
    a = analyze_single_comment("""
    一直都很喜欢吃蛋糕，在家的附近就有一家，所以经常来光顾，渐渐地就喜欢上了这家蛋糕店，我经常过来买，有时候当早饭，有时候当点心，这家店的面包跟蛋糕我都喜欢吃，刚开始是直接买，后来发现网上有优惠劵就团优惠劵，还比较便宜。前段时间大众点评推出周四半价，我觉得这是活动特别好，又实惠了顾客，又给店家招揽生意，不错，就是希望周四半价最好不要限时间或者限购，因为有时候上班时间上凑不好,这样的话下了班也能买，或者一早上班的时候买。呵呵~当然希望大众给我们多多的谋福利啦！哈哈，说回重点，面包好吃～
    """)
    print(a)
    print(type(a))
