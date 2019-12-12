import numpy as np
from keras.callbacks import Callback

def make_stepped_schedule(steps):
    """
    Helper to generate a schedule function to perform step-wise changes of a given optimizer hyper-parameter.

    :param steps:   List of tuples (start_epoch, value);
                    start_epochs should be increasing, starting at 0. E.g.
                      momentum_schedule = make_stepped_schedule([(0, 0.5), (5, 0.9)])
    """
    def get_value_for_epoch(epoch):
        return next(x[1][1] for x in reversed(list(enumerate(steps))) if x[1][0] <= epoch)  # search backwards for first step whose epoch <= current epoch, return associated hyper-parameter value
    return get_value_for_epoch

class MomentumScheduler(Callback):
    """
    Simple Callback that changes momentum, e.g. 0.5 for first 5 epochs, then 0.9.

    Example:
       optimizer = SGD(lr=0.001, momentum=0., decay=0.0, nesterov=False)
       momentum_schedule = make_stepped_schedule([(0, 0.5), (5, 0.9)])
       momentum_scheduler = MomentumScheduler(momentum_schdule)
       model = ...
       model.compile(optimizer, loss)
       model.fit(..., callbacks=[momentum_scheduler])

    See: Hinton, "A Practical Guide to Training Restricted Boltzmann Machines", UTML TR 2010-003, 2010, section 9.1.
    """
    def __init__(self, schedule):
        super(MomentumScheduler, self).__init__()
        self.schedule = schedule

    def on_epoch_begin(self, epoch, logs={}):
        cur_momentum = self.model.optimizer.momentum.get_value()
        new_momentum = self.schedule(epoch)
        if not np.isclose(new_momentum, cur_momentum) or epoch == 0:  # cannot use new_momentum != cur_momentum directly, because there may be small numerical differences (Python float vs value from Theano shared scalar)
            self.model.optimizer.momentum.set_value(new_momentum)
            print('Changed momentum to: %.3f' % self.model.optimizer.momentum.get_value())










import numpy as np
import theano
import theano.tensor as T
from keras.models import Model, standardize_X

class UnsupervisedLoss1Logger(Callback):
    def __init__(self, X, loss, verbose=1, label='loss', every_n_epochs=1, display_delta=True):
        super(UnsupervisedLoss1Logger, self).__init__()
        self.X = X
        self.loss = loss
        self.verbose = verbose
        self.label = label
        self.every_n_epochs = every_n_epochs
        self.display_delta = display_delta
        self.prev_loss = None
        self.batch_losses = []

    def compile(self, theano_mode=None):
        # re-use a function from keras.Model
        # it is a class function, but doesn't use anything from its 'self', 
        # so here we just create a dummy instance and get its function
        # alternatively, we could use the model set on Callback._set_model()
        self._dummy_model = Model()
        self._predict_loop = self._dummy_model._predict_loop

        # compile unsupervised loss Theano function
        input = T.matrix()
        loss = self.loss(input)
        ins = [input]
        self._loss = theano.function(ins, loss,
            allow_input_downcast=True, mode=theano_mode)

    def _predict(self, X, batch_size=128, verbose=0):
        X = standardize_X(X)
        return self._predict_loop(self._loss, X, batch_size, verbose)[0]

    def on_batch_end(self, batch, logs={}):
        # loss of training batch after each training batch update
        batch_size = logs['size']
        X_batch = self.X[batch*batch_size:(batch+1)*batch_size]
        batch_loss = np.mean(self._predict(X_batch, verbose=0))
        self.batch_losses.append(batch_loss)

    def on_epoch_end(self, epoch, logs={}):
        if (epoch+1) % self.every_n_epochs == 0:
            loss = np.mean(np.asarray(self.batch_losses))
            self.batch_losses = []
            #loss = np.mean(self._predict(self.X, verbose=self.verbose))
            if self.prev_loss:
                delta_loss = loss - self.prev_loss
            else:
                delta_loss = None
            self.prev_loss = loss
            if self.display_delta and delta_loss:
                print('%s: %f (%+f)' % (self.label, loss, delta_loss))
            else:
                print('%s: %f' % (self.label, loss))

class UnsupervisedLoss2Logger(Callback):
    def __init__(self, X_train, X_test, loss, verbose=1, label='loss', every_n_epochs=1, display_delta=True):
        super(UnsupervisedLoss2Logger, self).__init__()
        self.X_train = X_train
        self.X_test = X_test
        self.loss = loss
        self.verbose = verbose
        self.label = label
        self.every_n_epochs = every_n_epochs
        self.display_delta = display_delta
        self.prev_loss = None

    def compile(self, theano_mode=None):
        # re-use a function from keras.Model
        # it is a class function, but doesn't use anything from its 'self', 
        # so here we just create a dummy instance and get its function
        # alternatively, we could use the model set on Callback._set_model()
        self._dummy_model = Model()
        self._predict_loop = self._dummy_model._predict_loop

        # compile unsupervised loss Theano function
        input_train = T.matrix()
        input_test = T.matrix()
        loss = self.loss(input_train, input_test)
        ins = [input_train, input_test]
        self._loss = theano.function(ins, loss,
            allow_input_downcast=True, mode=theano_mode)

    def _predict(self, X_train, X_test, batch_size=128, verbose=0):
        X = [X_train, X_test]
        return self._predict_loop(self._loss, X, batch_size, verbose)[0]

    def on_epoch_end(self, epoch, logs={}):
        if (epoch+1) % self.every_n_epochs == 0:
            loss = np.mean(self._predict(self.X_train, self.X_test, verbose=self.verbose))
            if self.prev_loss:
                delta_loss = loss - self.prev_loss
            else:
                delta_loss = None
            self.prev_loss = loss
            if self.display_delta and delta_loss:
                print('%s: %f (%+f)' % (self.label, loss, delta_loss))
            else:
                print('%s: %f' % (self.label, loss))

