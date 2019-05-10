import collections
import os

import jieba

file_dir = 'G:/URECA/chinese_corpus/txt/'
files = os.listdir(file_dir)
vocab_file = 'G:/URECA/textsum/CWD/chinese_data/chinese_vocab'
max_words = 20000
counter = collections.Counter()

for filename in files:
    with open(file_dir + filename, 'r', encoding='utf-8') as f:
        document = f.read()
        '''translate_table = dict((ord(char), None) for char in string.punctuation or string.digits)
        document.translate(translate_table)'''

    tokens = jieba.cut(document, cut_all=False, HMM=True)
    counter.update(tokens)

with open(vocab_file, 'wb') as writer:
    for word, count in counter.most_common(max_words - 2):
        # print(word + ' ' + str(count) + '\n')
        if count >= 16:
            writer.write((word + ' ' + str(count) + '\n').encode('utf-8'))
    writer.write('<s> 0\n'.encode('utf-8'))
    writer.write('</s> 0\n'.encode('utf-8'))
    writer.write('<UNK> 0\n'.encode('utf-8'))
    writer.write('<PAD> 0\n'.encode('utf-8'))
