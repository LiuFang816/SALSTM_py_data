## sentvector.py
## Author: Yangfeng Ji
## Date: 08-10-2014
## Time-stamp: <yangfeng 08/18/2014 15:11:19>

import numpy, gzip
from huffman import *
from datastructure import WordCode, Instance, FeatInfo
from cPickle import load, dump

rng = numpy.random.RandomState(1234)

# class FeatInfo(object):
#     def __init__(self, vec, code_idx, label, logprob):
#         self.vec = vec
#         self.code_idx = code_idx
#         self.label = label
#         self.logprob = logprob

def sigmoid(x):
    """ Sigmoid function

    Never forget the minus sign again - YJ
    """
    return 1 / (1 + numpy.exp(-x))

def get_code_label(code):
    n_code = len(code)
    label = code[n_code-1]
    if n_code == 1:
        code_idx = 0
    else:
        code_idx = ((2**(n_code-1))-1) + int(code[:n_code-1],2)
    return (code_idx, label)

class SentVector(object):
    def __init__(self, n_word, n_sent, n_feat, n_dim):
        """ Initialize the parameters of the SentVector model

        :type n_word: int
        :param n_word: number of words

        :type n_sent: int
        :param n_sent: number of sentences

        :type n_feat: int
        :param n_feat: number of feature vectors, used for
                       hierarchical softmax
        
        :type n_dim: int
        :param n_dim: number of latent dimension
        """
        self.Word = numpy.asarray(rng.uniform(low=-0.01, high=0.01,
                                            size=(n_dim, n_word)))
        self.Sent = numpy.asarray(rng.uniform(low=-0.01, high=0.01,
                                            size=(n_dim, n_sent)))
        self.Feat = numpy.asarray(rng.uniform(low=-0.01, high=0.01,
                                            size=(n_dim, n_feat)))
        self.b = numpy.zeros((n_feat,))
        U_values = numpy.asarray(rng.uniform(
            low=-numpy.sqrt(6. / (n_dim + n_dim)),
            high=-numpy.sqrt(6. / (n_dim + n_dim)),
            size=(n_dim, n_dim)))
        self.U = U_values * 4
        V_values = numpy.asarray(rng.uniform(
            low=-numpy.sqrt(6. / (n_dim + n_dim)),
            high=-numpy.sqrt(6. / (n_dim + n_dim)),
            size=(n_dim, n_dim)))
        self.V = V_values * 4
        self.params = [self.Word, self.Sent, self.Feat, self.U, self.V, self.b]
        self.n_dim = n_dim
        self.nWords = n_word

    def hierarchical_softmax(self, sample):
        """ Compute the hierarchical softmax for a given word
        Simple average, without involving any parameter - YJ

        Refer to __log_prob_path for parameters explanation
        """
        word_idx, sent_idx = sample.windex, sample.sindex
        cont_list, code = sample.clist, sample.code
        feat_path, _, _ = self.__log_prob_path(word_idx, sent_idx, cont_list, code)
        logprob = 0.0
        for val in feat_path.itervalues():
            logprob += val.logprob
        return numpy.exp(logprob)

    def negative_log_likelihood(self, sample):
        """ Return the mean of the hierarchical softmax for a given word
        """
        word_idx, sent_idx = sample.windex, sample.sindex
        cont_list, code = sample.clist, sample.code
        feat_path, _, _ = self.__log_prob_path(word_idx, sent_idx, cont_list, code)
        logprob = 0.0
        for val in feat_path.itervalues():
            logprob += val.logprob
        return -1.0 * logprob

    def __log_prob_path(self, word_idx, sent_idx, cont_list, code):
        """ Following the huffman tree to compute the log probability
        """
        nWords = self.nWords
        nCode = len(code)
        # Check the length of context list
        if len(cont_list) == 0:
            return None, None, None
        # Average context words and sentence vector
        r_hat = numpy.zeros((self.n_dim,))
        for idx in cont_list:
            r_hat += self.Word[:,idx]
        r_hat = r_hat / len(cont_list) # Average
        # Sentence vector
        sent_vec = self.Sent[:,sent_idx]
        # Log-prob following the path
        feat_path_dct = {}
        for idx in range(nCode):
            # Partial code
            sub_code = code[:idx+1]
            # Get code index and label
            code_idx, label = get_code_label(sub_code)
            # Corresponding feature vector
            feat_vec = self.Feat[:,code_idx]
            b = self.b[code_idx]
            prob_idx_label_1 = sigmoid(r_hat.dot(self.U.dot(feat_vec.T)) +
                                       sent_vec.dot(self.V.dot(feat_vec.T)) + b)
            if label == '1':
                prob_idx = prob_idx_label_1
            elif label == '0':
                prob_idx = 1 - prob_idx_label_1
            # print 'idx = {}, code = {}, label = {}, code_idx={}, prob_{} = {}'.format(idx, sub_code, label, code_idx, idx, prob_idx)
            # log_prob_path[sub_code] = numpy.log(prob_idx)
            feat_info = FeatInfo(feat_vec, code_idx, label, numpy.log(prob_idx))
            feat_path_dct[sub_code] = feat_info
        return feat_path_dct, r_hat, sent_vec,
    

    def gradient(self, sample):
        """ Compute the gradient wrt given training example

        Separate gradient calculating and updating, so we can parallel
          computing in multiple threads

        :type sample: Instance class (see datastructure.py for more detail)
        :param sample: one training sample instance
        """
        word_idx, sent_idx = sample.windex, sample.sindex
        cont_list, code = sample.clist, sample.code
        featpath_dct, rhat, sent_vec = self.__log_prob_path(word_idx, sent_idx,
                                                            cont_list, code)
        if featpath_dct is None:
            return None
        # Gradient of sigmoid function (depends on label)
        # NEED DOUBLE CHECK
        sigmoid_grad = {}
        for (subcode, val) in featpath_dct.iteritems():
            if val.label == '1':
                sigmoid_grad[subcode] = 1 - numpy.exp(val.logprob)
            elif val.label == '0':
                sigmoid_grad[subcode] = numpy.exp(val.logprob) - 1
            else:
                raise ValueError("Unrecognized label in gradient function")
        # All other components for computing gradient
        # Refer to note for more detail
        Uqi, Vqi, Utr, Vts, rqt, sqt = {}, {}, {}, {}, {}, {}
        for (subcode, val) in featpath_dct.iteritems():
            Uqi[subcode] = self.U.dot(val.vec)
            Vqi[subcode] = self.V.dot(val.vec)
            Utr[subcode] = self.U.transpose().dot(rhat)
            Vts[subcode] = self.V.transpose().dot(sent_vec)
            rqt[subcode] = rhat.dot(val.vec.T)
            sqt[subcode] = sent_vec.dot(val.vec.T)
        # Initial gradient
        # This is convenient, but not memory-efficient - YJ
        Word_grad = numpy.zeros(self.Word.shape)
        Sent_grad = numpy.zeros(self.Sent.shape)
        Feat_grad = numpy.zeros(self.Feat.shape)
        b_grad = numpy.zeros(self.b.shape)
        U_grad = numpy.zeros(self.U.shape)
        V_grad = numpy.zeros(self.V.shape)
        w_grad = numpy.zeros((self.Word.shape[0],))
        # Gradient wrt q, b, sent_vec, U, V
        for (subcode, val) in featpath_dct.iteritems():
            Feat_grad[:,val.code_idx] = -1.0*sigmoid_grad[subcode]*(Utr[subcode]+Vts[subcode])
            b_grad[val.code_idx] = -1.0*sigmoid_grad[subcode]
            Sent_grad[:,sent_idx] += -1.0*sigmoid_grad[subcode]*Vqi[subcode]
            U_grad += -1.0*sigmoid_grad[subcode]*rqt[subcode]
            V_grad += -1.0*sigmoid_grad[subcode]*sqt[subcode]
            w_grad += -1.0*sigmoid_grad[subcode]*Uqi[subcode]
        # Gradient wrt w
        for idx in cont_list:
            Word_grad[:,idx] = (w_grad / len(cont_list)) # Average
        # Keep the same order as in self.params
        return [Word_grad, Sent_grad, Feat_grad, U_grad, V_grad, b_grad]


    def grad_update(self, gparams, lr):
        """ Update parameters with gradients

        Keep the correspondence between parameters and their gradient

        :type gparams: list
        :param gparams: a list of gradient of parameters computed by *gradient* function

        :type lr: float
        :param lr: learning rate
        """
        self.Word = self.Word - (gparams[0] * lr)
        self.Sent = self.Sent - (gparams[1] * lr)
        self.Feat = self.Feat - (gparams[2] * lr)
        self.U = self.U - (gparams[3] * lr)
        self.V = self.V - (gparams[4] * lr)
        self.b = self.b - (gparams[5] * lr)
        

    def save_model(self, fname):
        """ Save the shared variables into files

        :type fname: string
        :param fname: file name
        """
        data = {"Word":self.Word,
                "Sent":self.Sent,
                "Feat":self.Feat,
                "U":self.U,
                "V":self.V,
                "b":self.b}
        if ".gz" not in fname:
            fname += ".gz"
        print 'Save model into: {} ...'.format(fname)
        fout = gzip.open(fname, "w")
        dump(data, fout)

    def load_model(self, fname):
        """ Load model parameters from data file fname

        :type fname: string
        :param fname: file name
        """
        fin = gzip.open(fname, "r")
        print "Load model from: {} ...".format(fname)
        data = load(fin)
        self.Word, self.Sent = data["Word"], data["Sent"]
        self.U, self.V = data["U"], data["V"]
        self.Feat, self.b = data["Feat"], data["b"]

    def predict(self, sample):
        """ Prediction on a given sample

        :type sample: Instance
        :param sample: an instance of Instance class
        """
        pass
        


