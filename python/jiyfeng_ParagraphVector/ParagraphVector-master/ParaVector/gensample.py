## gensample.py
## Author: Yangfeng Ji
## Date: 08-14-2014
## Time-stamp: <yangfeng 08/17/2014 17:59:18>

"""
Generate sample from word-index file,
  then save sample into a pickle file
"""

from cPickle import load, dump
import gzip
from datastructure import WordCode, Instance
from huffman import *

class GenSample(object):
    def __init__(self, fname_code, n_context=3):
        self.fname_code = fname_code
        self.samples = []
        self.codebook = None
        self.n_context = n_context

    def generate(self, fname, fname_sample):
        # Load Huffman codebook
        coder = HuffmanCode()
        self.codebook = coder.load_idxkey(self.fname_code)
        # Load sentences and generate sample
        fin = open(fname, 'r')
        for line in fin:
            items = line.strip().split("\t")
            try:
                sent_idx = int(items[0])
                words_ids = map(int, items[1].split())
                sent_samples = self.__genonesent(sent_idx,
                                                 words_ids)
                self.samples += sent_samples
            except IndexError:
                print line
        # Save samples
        print "Saving examples ..."
        fout = gzip.open(fname_sample, 'w')
        dump(self.samples, fout)

    def __genonesent(self, sent_idx, words_ids):
        """ Generate sample from one sentence

        :type sent_idx: int
        :param sent_idx: sentence index

        :type words_ids: list of int
        :param words_ids: a list of word indices in sentence
                          sent_idx
        """
        sent_samples = []
        for (idx, w_idx) in enumerate(words_ids):
            context_list = []
            code = self.codebook[w_idx].code
            for inc in range(1, self.n_context+1):
                if (idx-inc) >= 0:
                    context_list.append(words_ids[idx-inc])
            for inc in range(1, self.n_context+1):
                if (idx+inc) < len(words_ids):
                    context_list.append(words_ids[idx+inc])
            instance = Instance(w_idx, sent_idx, context_list,
                                code)
            # print w_idx, sent_idx, code, context_list
            sent_samples.append(instance)
        return sent_samples


def main():
    fname_code = "../Debtates/codebook.txt"
    fname_in = "../Debtates/debtates-word-index.txt"
    fname_sample = "../Debtates/data-sample.pickle.gz"
    gs = GenSample(fname_code, n_context=3)
    gs.generate(fname_in, fname_sample)


if __name__ == '__main__':
    main()
