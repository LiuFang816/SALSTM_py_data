## formSentence.py
## Author: Yangfeng Ji
## Date: 08-13-2014
## Time-stamp: <yangfeng 08/17/2014 17:39:54>

import os, nltk, re
from nltk.tokenize import word_tokenize

def process_line(line):
    items = line.strip().split("|")
    try:
        author = items[0]
        line = items[1]
    except IndexError:
        return [], None
    item_list = re.findall(r'\[([^]]*)\]', line)
    for item in item_list:
        line = line.replace(item, "")
    line = line.replace("<p>","").replace("[]","").lower()
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    sents = sent_detector.tokenize(line)
    return sents, author


def main(path, fname_out, fname_index, thresh=4):
    fout = open(fname_out, 'w')
    findex = open(fname_index, 'w')
    filelist = os.listdir(path)
    for fname in filelist:
        counter = 0
        fin = open(os.path.join(path, fname), 'r')
        for line in fin:
            sents, author = process_line(line)
            for sent in sents:
                words = word_tokenize(sent)
                if len(words) >= thresh:
                    fout.write((" ".join(words)) + '\n')
                    findex.write(fname + "\t" + author + "\t" + str(counter) + "\n")
                    counter += 1
    fout.close()
    findex.close()
    print 'Done'


if __name__ == '__main__':
    path = "../Data/"
    fname_out = "../debtates-sent.txt"
    fname_index = "../debtates-sent-index.txt"
    main(path, fname_out, fname_index)
