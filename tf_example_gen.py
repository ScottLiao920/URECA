import os
import nltk
from nltk.tokenize import word_tokenize
import struct
import random
from tensorflow.core.example import example_pb2

nltk.download('punkt')

file_dir = "G:/URECA/textsum/CWD/"
files = os.listdir(file_dir + "corpus/")
# td = file_dir + "data/corpus_train"
# training = open(td,'w', errors='ignore')
count = 0
train = file_dir + "updated_data/corpus_train"
train_writer = open(train, 'wb')
train_count = 0
validation = file_dir + "updated_data/corpus_eval"
eval_writer = open(validation, 'wb')
validation_count = 0
test = file_dir + "updated_data/corpus_test"
test_writer = open(test, 'wb')
test_count = 0
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
        file = file_dir + "corpus/" + file
        cur_file = open(file, 'r', errors='ignore')
        lines = cur_file.readlines()
        line_no = 0
        tf_example = example_pb2.Example()
        for line in lines:
            if line_no is 2:
                if len(nltk.Text(nltk.word_tokenize(line))) < 30 and "=" not in line:
                    # training.write("abstract=b'<d><p><s>" + line.replace("\n", "") + "</s> </p> </d>'")
                    # title = "abstract=b'<d><p><s>" + line.replace("\n", "") + "</s> </p> </d>'"
                    title = '<d><p><s>' + line.replace("\n", "") + '</s> </p> </d>'
                else:
                    break
            if line_no is 3:
                # training.write('article=b"<d> <p> ')
                sent_line = nltk.sent_tokenize(line, 'english')
                tmp = ""
                sent_count = 0
                for sent in sent_line:
                    sent_count += 1
                    tokens = word_tokenize(sent)
                    # print(len(nltk.Text(tokens))+len(nltk.Text(nltk.word_tokenize(tmp))))
                    if sent_count is 3:
                        break
                    if len(nltk.Text(tokens)) + len(nltk.Text(nltk.word_tokenize(tmp))) < 120 and "=" not in sent:
                        tmp += ('<s>' + sent + '</s>')
                    else:
                        break
                # print(len(tmp))
                if len(tmp) is not 0:
                    body = '<d><p>' + tmp + '</p></d>'
                    tf_example.features.feature['article'].bytes_list.value.extend([body.encode('utf-8')])
                    tf_example.features.feature['abstract'].bytes_list.value.extend([title.encode('utf-8')])
                    tf_example_str = tf_example.SerializeToString()
                    str_len = len(tf_example_str)
                    writer.write(struct.pack('q', str_len))
                    writer.write(struct.pack('%ds' % str_len, tf_example_str))
            line_no += 1
        cur_file.close()
        count += 1
        # print(count)
    except FileNotFoundError:
        print("File not found!")
train_writer.close()
test_writer.close()

eval_writer.close()
print("Total count: ", count, "\nTraining_set: ", train_count, "\nvalidation_set: ", validation_count, "\nTest_set: ", test_count)
