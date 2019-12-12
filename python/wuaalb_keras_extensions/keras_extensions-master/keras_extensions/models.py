from keras.models import Model, standardize_X
from keras.layers import containers
from keras import optimizers, objectives
from keras import backend as K

class SingleLayerUnsupervised(Model, containers.Sequential):
    """
    Single layer unsupervised learning Model.
    """
    # add Layer, adapted from keras.layers.containers.Sequential
    def add(self, layer):
        if len(self.layers) > 0:
            warnings.warn('Cannot add more than one Layer to SingleLayerUnsupervised!')
            return
        super(SingleLayerUnsupervised, self).add(layer)

    # compile theano graph, adapted from keras.models.Sequential
    def compile(self, optimizer, loss):
        self.optimizer = optimizers.get(optimizer)

        self.loss = objectives.get(loss)

        # input of model
        self.X_train = self.get_input(train=True)
        self.X_test = self.get_input(train=False)

        train_loss = self.loss(self.X_train)
        test_loss = self.loss(self.X_test)

        train_loss.name = 'train_loss'
        test_loss.name = 'test_loss'

        for r in self.regularizers:
            train_loss = r(train_loss)
        updates = self.optimizer.get_updates(self.params, self.constraints, train_loss)
        updates += self.updates

        if type(self.X_train) == list:
            train_ins = self.X_train
            test_ins = self.X_test
        else:
            train_ins = [self.X_train]
            test_ins = [self.X_test]

        self._train = K.function(train_ins, train_loss, updates=updates)
        self._test = K.function(test_ins, test_loss)

    # train model, adapted from keras.models.Sequential
    def fit(self, X, batch_size=128, nb_epoch=100, verbose=1, callbacks=[],
            validation_split=0., validation_data=None, shuffle=True, show_accuracy=False):

        X = standardize_X(X)

        val_f = None
        val_ins = None
        if validation_data or validation_split:
            if show_accuracy:
                val_f = self._test_with_acc
            else:
                val_f = self._test

        if show_accuracy:
            f = self._train_with_acc
            out_labels = ['loss', 'acc']
        else:
            f = self._train
            out_labels = ['loss']

        ins = X# + [y, sample_weight]
        metrics = ['loss', 'acc', 'val_loss', 'val_acc']
        return self._fit(f, ins, out_labels=out_labels, batch_size=batch_size, nb_epoch=nb_epoch,
                         verbose=verbose, callbacks=callbacks,
                         val_f=val_f, val_ins=val_ins,
                         shuffle=shuffle, metrics=metrics)

