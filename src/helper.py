import numpy as np
from collections import defaultdict
import datetime


def load_code_set(max_length, data_dir,max_vocab_size=2048):
    print("loading dataset...")
    star_time = datetime.datetime.now()
    lines = []
    first = defaultdict(int)
    first_prob = {}
    with open(data_dir, 'r', encoding='utf-8') as f:
        data_size = 0
        for line in f:
            line = line[:-1]
            first[line[0]] += 1
            line = tuple(line)
            lines.append(line + (("⊥",) * (max_length - len(line))))
            data_size += 1
    np.random.shuffle(lines)

    import collections
    counts = collections.Counter(char for line in lines for char in line)

    char_to_idx = {}
    idx_to_char = {}
    inv_charmap = []

    for char, count in counts.most_common(max_vocab_size-1):
        if char not in char_to_idx:
            char_to_idx[char] = len(inv_charmap)
            idx_to_char[len(inv_charmap)] = char
            inv_charmap.append(char)

    _vocab_size = len(inv_charmap)
    for key,value in first.items():
        first_prob[key] = value/data_size
    print('loading code set cost time:{}'.format(datetime.datetime.now()-star_time))
    return data_size, _vocab_size, char_to_idx, idx_to_char,lines, first_prob



if __name__ == '__main__':
    data_size, _vocab_size, char_to_idx, idx_to_char, lines, first_prob = load_code_set(
        max_length=18,
        data_dir='C:\\Users\\zzl\\Desktop\\1.txt'
    )

