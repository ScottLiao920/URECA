from os import listdir

import jieba
import re

# data_dir = 'G:/URECA/chinese_corpus/txt/'
file = 'G:/URECA/chinese_corpus/txt/2.txt'
file_no = 0
# for file in listdir(data_dir):
with open(file, 'r', encoding='utf-8') as tmp:
        line_no = 0
        body = ""
        cur_words = 0
        lines = tmp.readlines()
        for line in lines:
            if len(line) > 1:
                if line_no == 0:
                    title = line[:-1]
                    line = ""
                elif line_no == 2:
                    line = line[line.find('】')+1:-1]
                elif line_no == 1:
                    line_no += 1
                    continue
                else:
                    break
                sents = re.split(r'([。？！\s])', line)
                for sent in sents:
                    seg_list = list(jieba.cut(sent, cut_all=False, HMM=True))
                    print("Full Mode: " + "/".join(seg_list), line_no, len(seg_list))
                    # print(seg_list, line_no)
                    if cur_words + len(seg_list) < 120:
                        body += sent
                        cur_words += len(seg_list)
                        print(body)
                    else:
                        break
                line_no += 1
        file_no += 1
        print(body, cur_words)
