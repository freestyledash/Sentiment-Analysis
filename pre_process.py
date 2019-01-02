import json
from collections import Counter

import jieba
from tqdm import tqdm

from config import *
from utils import parse_user_reviews


def build_wordmap(contents):
    """
    分词并统计词出现的频率,存储出现频率大于min_word_freq
    :param contents:
    :return:
    """
    word_freq = Counter()

    # 分词
    for sentence in tqdm(contents):
        seg_list = jieba.cut(sentence.strip())
        word_freq.update(list(seg_list))

    # Create word map
    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    word_map = {k: v + 4 for v, k in enumerate(words)}
    word_map['<pad>'] = 0
    word_map['<start>'] = 1
    word_map['<end>'] = 2
    word_map['<unk>'] = 3
    print('len(word_map): ' + str(len(word_map)))

    with open('data/WORDMAP.json', 'w') as file:
        json.dump(word_map, file, indent=4)


if __name__ == '__main__':
    user_reviews = parse_user_reviews('train')
    build_wordmap(user_reviews['content'])
