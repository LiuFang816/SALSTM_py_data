## datastructure.py
## Author: Yangfeng Ji
## Date: 08-13-2014
## Time-stamp: <yangfeng 08/18/2014 15:10:30>

class WordCode(object):
    def __init__(self, index, word, code, freq):
        """ Data structure for collecting Huffman code for word

        :type index: int
        :param index: index of word

        :type word: string
        :param word: word itself

        :type code: string
        :param code: binary code string for word

        :freq code: float
        :param code: word frequency in data
        """
        self.word = word
        self.index = index
        self.code = code
        self.freq = freq


class Instance(object):
    def __init__(self, windex, sindex, clist, code):
        """ Data structure for training/test instance

        :type windex: int
        :param windex: index of word

        :type sindex: int
        :param sindex: index of sentence

        :type clist: list
        :param clist: a list of word index as context

        :type code: string
        :param code: a binary string as the huffman code of word
        """
        self.windex = windex
        self.sindex = sindex
        self.clist = clist
        self.code = code


class FeatInfo(object):
    def __init__(self, vec, code_idx, label, logprob):
        """ Information about one feature vector related
            to a given word

        :type vec: numpy.array
        :param vec: value of feature vector

        :type code_idx: int
        :param code_idx: transform binary code into an int number indicating
                         the position of this feature vector in the feature
                         matrix

        :type label: string with length 1
        :param label: the label of the given word, distinguishing the path
                      on the Huffman tree

        :type logprob: float
        :param logprob: the logprob of this feature vector given the word
                        and its context
        """
        self.vec = vec
        self.code_idx = code_idx
        self.label = label
        self.logprob = logprob
