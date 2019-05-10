import re
import jieba
# file_new = "G:/URECA/textsum/CWD/chinese_data/chinese_test"
file_old = "G:/URECA/textsum/CWD/chinese_data/chinese_vocab"

file = 'G:/URECA/chinese_corpus/txt/2.txt'
vocab = open(file_old, "rb")
words = vocab.readlines()
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
                line = line[line.find('】') + 1:-1]
            elif line_no == 1:
                line_no += 1
                continue
            else:
                break
            sents = re.split(r'([。？！\s])', line)
            for sent in sents:
                # print(sent)
                seg_list = list(jieba.cut(sent, cut_all=False, HMM=True))
                # print("Full Mode: " + "/".join(seg_list), line_no, len(seg_list))
                for word in seg_list:
                    print(word)
                    if word in words:
                        print(word)
            line_no += 1