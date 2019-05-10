import os
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')

file_dir = "G:/URECA/textsum/CWD/"
files = os.listdir(file_dir + "corpus/")
td = file_dir + "data/corpus_train"
training = open(td, 'w', errors='ignore')
count = 0
for file in files:
    try:
        file = file_dir + "corpus/" + file
        cur_file = open(file, 'r', errors='ignore')
        lines = cur_file.readlines()
        line_no = 0
        for line in lines:
            if line_no is 2:
                if len(nltk.Text(nltk.word_tokenize(line))) < 30 and "=" not in line:
                    # training.write("abstract=b'<d><p><s>" + line.replace("\n", "") + "</s> </p> </d>'")
                    title = "abstract=b'<d><p><s>" + line.replace("\n", "") + "</s> </p> </d>'"
                else:
                    continue
            if line_no is 3:
                # training.write('article=b"<d> <p> ')
                sent_line = nltk.sent_tokenize(line, 'english')
                tmp = ""
                sent_count = 0
                for sent in sent_line:
                    sent_count += 1
                    tokens = word_tokenize(sent)
                    # print(len(nltk.Text(tokens))+len(nltk.Text(nltk.word_tokenize(tmp))))
                    if sent_count is 2:
                        break
                    if len(nltk.Text(tokens)) + len(nltk.Text(nltk.word_tokenize(tmp))) < 120 and "=" not in sent:
                        tmp += ('<s>' + sent + '</s>')
                    else:
                        break
                # print(len(tmp))
                if len(tmp) is not 0:
                    input = title + '\t' + 'article=b"<d> <p> ' + tmp + '</p> </d>' + '\t\n'
                    input = input.encode('utf-8', errors='ignore')
                    training.write(input.decode('utf-8', errors='ignore'))

            line_no += 1
        cur_file.close()
        count += 1
        print(count)
    except FileNotFoundError:
        print("File not found!")
    if count == 31000:
        training.close()
        td = file_dir + "data/corpus_test"
        training = open(td, 'w', errors='ignore')
    if count == 32000:
        training.close()
        td = file_dir + "data/corpus_eval"
        training = open(td, 'w', errors='ignore')
training.close()
