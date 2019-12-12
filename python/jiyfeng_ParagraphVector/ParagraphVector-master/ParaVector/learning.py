## learning.py
## Author: Yangfeng Ji
## Date: 08-10-2014
## Time-stamp: <yangfeng 08/17/2014 18:16:51>

import numpy, time
import pp
from datastructure import WordCode, Instance
from sentvector import SentVector
from random import shuffle


class SGDLearn(object):
    def __init__(self, model, trndata, learning_rate=1e-4):
        """ Initialize the parameters related to learning

        :type model: instance of class SentVector
        :param model: SentVector model

        :type trndata: list of Instance
        :param trndata: Training data

        :type learning_rate: float
        :param learning_rate: initial learning rate
        """
        self.model = model
        self.trndata = trndata
        self.learning_rate = learning_rate
        self.grads_square_sum = None

    def adagrad(self, grads, eps=1e-30):
        """ Update grads with AdaGrad principle

        :type grads: list
        :param grads: list of gradients of parameters
        """
        if self.grads_square_sum is None:
            self.grads_square_sum = []
            for grad in grads:
                self.grads_square_sum.append(grad**2)
            return grads
        else:
            for (idx, grad) in enumerate(grads):
                self.grads_square_sum[idx] += (grad**2)
                grads[idx] = grads[idx] / (numpy.sqrt(self.grads_square_sum[idx])+eps)
            return grads
        

    def sgd_oneword(self, index):
        """ Update parameters with one training sample

        :type index: int
        :param index: index of training sample
        """
        param_grads = self.model.gradient(self.trndata[index])
        # Update grad with AdaGrad
        param_grads = self.adagrad(param_grads)
        print "Before update:", self.model.hierarchical_softmax(self.trndata[index])
        self.model.grad_update(param_grads, self.learning_rate)
        print "After update:", self.model.hierarchical_softmax(self.trndata[index])

        
    def sgd_perword(self, with_adagrad=False, info_interval=1000):
        """ Read one word, using SGD to update related parameters
        """
        Index = range(len(self.trndata))
        shuffle(Index)
        for (n, index) in enumerate(Index):
            if (n+1) % info_interval == 0:
                print 'Update with {} samples'.format(n+1)
            param_grads = self.model.gradient(self.trndata[index])
            # Update grad with AdaGrad
            if with_adagrad:
                param_grads = self.adagrad(param_grads)
            if (n+1) % info_interval == 0:
                print "Before update:", self.model.hierarchical_softmax(self.trndata[index])
            if param_grads is not None:
                self.model.grad_update(param_grads, self.learning_rate)
            if (n+1) % info_interval == 0:
                print "After update:", self.model.hierarchical_softmax(self.trndata[index])
        self.model.save_model("model.pickle.gz")

            
    def sgd_minibatch(self):
        """ Update parameters with one batch of training examples
        """
        pass
