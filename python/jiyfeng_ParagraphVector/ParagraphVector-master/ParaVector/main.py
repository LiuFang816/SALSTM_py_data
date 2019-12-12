## main.py
## Author: Yangfeng Ji
## Date: 08-13-2014
## Time-stamp: <yangfeng 08/17/2014 18:18:58>

from sentvector import SentVector
from cPickle import load
from datastructure import Instance
from learning import SGDLearn
import gzip

def main():
    n_word, n_sent = 5158, 22431
    n_feat, n_dim = 2**17, 50
    print 'Load data ...'
    trndata = load(gzip.open("../Debtates/data-sample.pickle.gz"))
    print 'Create a SentVector model ...'
    sv = SentVector(n_word, n_sent, n_feat, n_dim)
    print 'Create a SGDLearn instance ...'
    learner = SGDLearn(sv, trndata)
    # print 'Update parameters with one instance ...'
    # learner.sgd_one_word(1)
    nPass = 30
    print 'Update parameters with entire dataset with {} pass ...'.format(nPass)
    for n in range(nPass):
        learner.sgd_perword()
    print 'Done'
    # learner.sgd_minibatch()


if __name__ == '__main__':
    main()
