## huffman.py
## Author: Yangfeng Ji
## Date: 08-10-2014
## Time-stamp: <yangfeng 08/14/2014 13:11:38>

from huffmancoding import *
from datastructure import WordCode

class HuffmanCode(object):
    def __init__(self):
        """
        """
        self.code = {}
        self.max_length = 0

    def load(self, fname):
        """ Load codebook, where word as key

        :type fname: string
        :param fname: codebook file name
        """
        fin = open(fname, 'r')
        for line in fin:
            items = line.strip().split("\t")
            wc = WordCode(int(items[0]), items[1], items[2],
                          float(items[3]))
            self.code[items[1]] = wc
        fin.close()
        return self.code

    def load_idxkey(self, fname):
        """ Load codebook, where index as key

        :type fname: string
        :param fname: codebook file name
        """
        fin = open(fname, 'r')
        for line in fin:
            items = line.strip().split("\t")
            wc = WordCode(int(items[0]), items[1], items[2],
                          float(items[3]))
            self.code[int(items[0])] = wc
        fin.close()
        return self.code

    def save(self, fname):
        """

        Data format
        word \t code \t word-freq
        """
        fout = open(fname, 'w')
        for (key, wc) in self.code.iteritems():
            fout.write(str(wc.index) + "\t" + wc.word + "\t"
                       + wc.code + "\t" + str(wc.freq) + "\n")
        fout.close()

    def coding(self, word_freq):
        """
        word_freq: {word:freq}
        1, Regular word index starts from 1, index 0 for all
           low-frequency words
        """
        word_list = []
        prob_list = []
        for (word, prob) in word_freq.iteritems():
            word_list.append(word)
            prob_list.append(prob)
        code_list = huffman(prob_list)
        for (idx, code) in enumerate(code_list):
            word = word_list[idx]
            prob = prob_list[idx]
            wc = WordCode(idx, word, code, prob)
            if self.max_length < len(code):
                self.max_length = len(code)
            self.code[word] = wc
        return self.code
        

    
