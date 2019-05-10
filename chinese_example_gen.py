import os
import random
import re
import struct
import collections
import jieba
from tensorflow.core.example import example_pb2

file_dir = "G:/URECA/textsum/CWD/"
vocab_file = 'G:/URECA/textsum/CWD/chinese_data/chinese_vocab'
files = os.listdir('G:/URECA/chinese_corpus/txt/')
# td = file_dir + "data/corpus_train"
# training = open(td,'w', errors='ignore')
count = 0
train = file_dir + "chinese_data/chinese_train"
train_writer = open(train, 'wb')
train_count = 0
validation = file_dir + "chinese_data/chinese_eval"
eval_writer = open(validation, 'wb')
validation_count = 0
test = file_dir + "chinese_data/chinese_test"
test_writer = open(test, 'wb')
test_count = 0
max_words = 20000
counter = collections.Counter()
for file in files:
    sort = random.randint(0, 9)
    if sort is 8:
        validation_count += 1
        writer = eval_writer
    elif sort is 9:
        test_count += 1
        writer = test_writer
    else:
        train_count += 1
        writer = train_writer
    if writer.closed:
        print("Open ur file!")

    try:
        file = 'G:/URECA/chinese_corpus/txt/' + file
        cur_file = open(file, 'r', encoding='utf-8')
        lines = cur_file.readlines()
        line_no = 0
        tf_example = example_pb2.Example()
        for line in lines:
            if len(line) > 1:
                line = line[:-1]
                if line_no == 0:
                    if len(list(jieba.cut(line, cut_all=False, HMM=True))) < 30 and "=" not in line:
                        # training.write("abstract=b'<d><p><s>" + line.replace("\n", "") + "</s> </p> </d>'")
                        # title = "abstract=b'<d><p><s>" + line.replace("\n", "") + "</s> </p> </d>'"
                        title = '<d><p><s>' + line[:-1].replace("\n", "") + '</s> </p> </d>'
                    else:
                        break
                elif line_no == 2:
                    # training.write('article=b"<d> <p> ')
                    sent_line = re.split(r'[。？！\s]', line)
                    tmp = ""
                    sent_count = 0
                    cur_words = 0
                    for sent in sent_line:
                        sent_count += 1
                        tokens = list(jieba.cut(sent, cut_all=False, HMM=True))
                        # print("Full Mode: " + "/".join(seg_list), line_no, len(seg_list))
                        if cur_words + len(tokens) < 120 and "=" not in sent and sent_count <= 2:
                            tmp += ('<s>' + sent + '</s>')
                            cur_words += len(tokens)
                            counter.update(tokens)
                        else:
                            break
                    print(tmp)
                    if len(tmp) is not 0:
                        body = '<d><p>' + tmp + '</p></d>'
                        tf_example.features.feature['article'].bytes_list.value.extend([body.encode('utf-8')])
                        tf_example.features.feature['abstract'].bytes_list.value.extend([title.encode('utf-8')])
                        tf_example_str = tf_example.SerializeToString()
                        str_len = len(tf_example_str)
                        writer.write(struct.pack('q', str_len))
                        writer.write(struct.pack('%ds' % str_len, tf_example_str))
                elif line_no > 2:
                    break
                line_no += 1
        cur_file.close()
        count += 1
        # print(count)
        print(file)
    except FileNotFoundError:
        print("File not found!")
train_writer.close()
test_writer.close()
eval_writer.close()
with open(vocab_file, 'wb') as writer:
    for word, word_count in counter.most_common(max_words - 2):
        # print(word + ' ' + str(count) + '\n')
        if word_count >= 16:
            writer.write((word + ' ' + str(word_count) + '\n').encode('utf-8'))
    writer.write('<s> 0\n'.encode('utf-8'))
    writer.write('</s> 0\n'.encode('utf-8'))
    writer.write('<UNK> 0\n'.encode('utf-8'))
    writer.write('<PAD> 0\n'.encode('utf-8'))
print("Total count: ", count, "\nTraining_set: ", train_count, "\nvalidation_set: ", validation_count, "\nTest_set: ",
      test_count)
