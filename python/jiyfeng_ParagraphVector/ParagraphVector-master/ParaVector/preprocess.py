## preprocess.py
## Author: Yangfeng Ji
## Date: 08-10-2014
## Time-stamp: <yangfeng 08/18/2014 14:42:16>

import string
from huffman import HuffmanCode
from datastructure import WordCode
from collections import defaultdict

EXCLUDE = set(string.punctuation)

def wordprocess(word):
    clean_word = ''.join(ch for ch in word if ch not in EXCLUDE)
    if len(clean_word) == 0:
        return None
    else:
        try:
            num = float(clean_word)
            return None
        except ValueError:
            return word


class Preprocess(object):
    def __init__(self, thresh=1.0):
        """ Initialize the parameters related to pre-processing
        """
        self.thresh = thresh
        self.word_freq = {}

    def getwordfreq(self, fname):
        """ Create word freqency dictionary
        """
        word_count = defaultdict(int)
        fin = open(fname, "r")
        for line in fin:
            words = line.strip().split()
            for word in words:
                word = wordprocess(word)
                if word is not None:
                    word_count[word] += 1
        # First pass, remove low-freq words and
        # compute the overall counts
        total_count = 0.0
        for (word, count) in word_count.iteritems():
            if (count >= self.thresh):
                self.word_freq[word] = count
                total_count += count
        # Second pass, normalize the probability
        for (word, count) in self.word_freq.iteritems():
            self.word_freq[word] /= total_count

    def generatesample(self):
        """ Generate sample for every word
        """
        pass
        
    def clean(self, fname_in, fname_out, fname_code):
        """ Create a huffman codebook and clean the datafile

        :type fname_in: string
        :param fname_in: input file name

        :type fname_out: string
        :param fname_out: output file name

        :type fname_code: string
        :param fname_code: huffman code file name
        """
        # Word frequency
        print "Get word frequency ..."
        self.getwordfreq(fname_in)
        # Coding and save code
        print "Huffman coding ..."
        coder = HuffmanCode()
        codebook = coder.coding(self.word_freq)
        coder.save(fname_code)
        # Call __clean
        print "Clean file ..."
        self.__clean(fname_in, fname_out, codebook)
        # Print max code length
        print "Max code length = {}".format(coder.max_length)
        
    def cleanwithvocab(self, fname_in, fname_out, fname_code):
        """ Clean the datafile with a pre-computed codebook
        """
        # Load huffman codebook
        codebook = load(fname_code)
        # Call __clean
        self.__clean(fname_in, fname_out, codebook)
        

    def __clean(self, fname_in, fname_out, codebook):
        """ Clean the datafile with a given codebook

        :type fname_in: string
        :param fname_in: input file name

        :type fname_out: string
        :param fname_out: output file name

        :type codebook: dictionary
        :param codebook: code book with word as key, WordCode as value
        """
        # Clean file
        fin = open(fname_in, "r")
        fout = open(fname_out, "w")
        sent_counter = 0
        for line in fin:
            words = line.strip().split()
            ids = []
            for word in words:
                try:
                    wc = codebook[word]
                    ids.append(wc.index)
                except KeyError:
                    pass
            ids = map(str, ids)
            line_ids = str(sent_counter) + "\t" + (" ".join(ids))
            fout.write(line_ids + "\n")
            sent_counter += 1
            # Print out information
            if (sent_counter % 1000 == 0):
                print "Process {} lines".format(sent_counter)
        fin.close()
        fout.close()
        print "DONE"


def main():
    pp = Preprocess(thresh=3.0)
    fname_in = "../Debtates/debtates-sent.txt"
    fname_out = "../Debtates/debtates-word-index.txt"
    fname_code = "../Debtates/codebook.txt"
    pp.clean(fname_in, fname_out, fname_code)


if __name__ == '__main__':
    main()
