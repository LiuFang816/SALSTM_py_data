# coding: utf-8
# pylint: disable=no-member

"""Online evaluation metric module."""
from __future__ import absolute_import

import numpy
from . import ndarray

def check_label_shapes(labels, preds, shape=0):
    """Check to see if the two arrays are the same size."""

    if shape == 0:
        label_shape, pred_shape = len(labels), len(preds)
    else:
        label_shape, pred_shape = labels.shape, preds.shape

    if label_shape != pred_shape:
        raise ValueError("Shape of labels {} does not match shape of "
                         "predictions {}".format(label_shape, pred_shape))

class EvalMetric(object):
    """Base class of all evaluation metrics."""

    def __init__(self, name, num=None):
        self.name = name
        self.num = num
        self.reset()

    def update(self, label, pred):
        """Update the internal evaluation.

        Parameters
        ----------
        labels : list of NDArray
            The labels of the data.

        preds : list of NDArray
            Predicted values.
        """
        raise NotImplementedError()

    def reset(self):
        """Clear the internal statistics to initial state."""
        if self.num is None:
            self.num_inst = 0
            self.sum_metric = 0.0
        else:
            self.num_inst = [0] * self.num
            self.sum_metric = [0.0] * self.num

    def get(self):
        """Get the current evaluation result.

        Returns
        -------
        name : str
           Name of the metric.
        value : float
           Value of the evaluation.
        """
        if self.num is None:
            if self.num_inst == 0:
                return (self.name, float('nan'))
            else:
                return (self.name, self.sum_metric / self.num_inst)
        else:
            names = ['%s_%d'%(self.name, i) for i in range(self.num)]
            values = [x / y if y != 0 else float('nan') \
                for x, y in zip(self.sum_metric, self.num_inst)]
            return (names, values)

    def get_name_value(self):
        """Get zipped name and value pairs"""
        name, value = self.get()
        if not isinstance(name, list):
            name = [name]
        if not isinstance(value, list):
            value = [value]
        return zip(name, value)

class CompositeEvalMetric(EvalMetric):
    """Manage multiple evaluation metrics."""

    def __init__(self, **kwargs):
        super(CompositeEvalMetric, self).__init__('composite')
        try:
            self.metrics = kwargs['metrics']
        except KeyError:
            self.metrics = []

    def add(self, metric):
        """Add a child metric."""
        self.metrics.append(metric)

    def get_metric(self, index):
        """Get a child metric."""
        try:
            return self.metrics[index]
        except IndexError:
            return ValueError("Metric index {} is out of range 0 and {}".format(
                index, len(self.metrics)))

    def update(self, labels, preds):
        for metric in self.metrics:
            metric.update(labels, preds)

    def reset(self):
        try:
            for metric in self.metrics:
                metric.reset()
        except AttributeError:
            pass

    def get(self):
        names = []
        results = []
        for metric in self.metrics:
            result = metric.get()
            names.append(result[0])
            results.append(result[1])
        return (names, results)

########################
# CLASSIFICATION METRICS
########################

class Accuracy(EvalMetric):
    """Calculate accuracy"""

    def __init__(self):
        super(Accuracy, self).__init__('accuracy')

    def update(self, labels, preds):
        check_label_shapes(labels, preds)

        for label, pred_label in zip(labels, preds):
            if pred_label.shape != label.shape:
                pred_label = ndarray.argmax_channel(pred_label)
            pred_label = pred_label.asnumpy().astype('int32')
            label = label.asnumpy().astype('int32')

            check_label_shapes(label, pred_label)

            self.sum_metric += (pred_label.flat == label.flat).sum()
            self.num_inst += len(pred_label.flat)

class TopKAccuracy(EvalMetric):
    """Calculate top k predictions accuracy"""

    def __init__(self, **kwargs):
        super(TopKAccuracy, self).__init__('top_k_accuracy')
        try:
            self.top_k = kwargs['top_k']
        except KeyError:
            self.top_k = 1
        assert(self.top_k > 1), 'Please use Accuracy if top_k is no more than 1'
        self.name += '_%d' % self.top_k

    def update(self, labels, preds):
        check_label_shapes(labels, preds)

        for label, pred_label in zip(labels, preds):
            assert(len(pred_label.shape) <= 2), 'Predictions should be no more than 2 dims'
            pred_label = numpy.argsort(pred_label.asnumpy().astype('float32'), axis=1)
            label = label.asnumpy().astype('int32')
            check_label_shapes(label, pred_label)
            num_samples = pred_label.shape[0]
            num_dims = len(pred_label.shape)
            if num_dims == 1:
                self.sum_metric += (pred_label.flat == label.flat).sum()
            elif num_dims == 2:
                num_classes = pred_label.shape[1]
                top_k = min(num_classes, self.top_k)
                for j in range(top_k):
                    self.sum_metric += (pred_label[:, num_classes - 1 - j].flat == label.flat).sum()
            self.num_inst += num_samples

class F1(EvalMetric):
    """Calculate the F1 score of a binary classification problem."""

    def __init__(self):
        super(F1, self).__init__('f1')

    def update(self, labels, preds):
        check_label_shapes(labels, preds)

        for label, pred in zip(labels, preds):
            pred = pred.asnumpy()
            label = label.asnumpy().astype('int32')
            pred_label = numpy.argmax(pred, axis=1)

            check_label_shapes(label, pred)
            if len(numpy.unique(label)) > 2:
                raise ValueError("F1 currently only supports binary classification.")

            true_positives, false_positives, false_negatives = 0., 0., 0.

            for y_pred, y_true in zip(pred_label, label):
                if y_pred == 1 and y_true == 1:
                    true_positives += 1.
                elif y_pred == 1 and y_true == 0:
                    false_positives += 1.
                elif y_pred == 0 and y_true == 1:
                    false_negatives += 1.

            if true_positives + false_positives > 0:
                precision = true_positives / (true_positives + false_positives)
            else:
                precision = 0.

            if true_positives + false_negatives > 0:
                recall = true_positives / (true_positives + false_negatives)
            else:
                recall = 0.

            if precision + recall > 0:
                f1_score = 2 * precision * recall / (precision + recall)
            else:
                f1_score = 0.

            self.sum_metric += f1_score
            self.num_inst += 1


class Perplexity(EvalMetric):
    """Calculate perplexity

    Parameters
    ----------
    ignore_label : int or None
        index of invalid label to ignore when
        counting. usually should be -1. Include
        all entries if None.
    """
    def __init__(self, ignore_label):
        super(Perplexity, self).__init__('Perplexity')
        self.ignore_label = ignore_label

    def update(self, labels, preds):
        assert len(labels) == len(preds)
        loss = 0.
        num = 0
        probs = []

        for label, pred in zip(labels, preds):
            assert label.size == pred.size/pred.shape[-1], \
                "shape mismatch: %s vs. %s"%(label.shape, pred.shape)
            label = label.as_in_context(pred.context).astype(dtype='int32').reshape((label.size,))
            pred = ndarray.batch_take(pred, label)
            probs.append(pred)

        for label, prob in zip(labels, probs):
            prob = prob.asnumpy()
            if self.ignore_label is not None:
                ignore = label.asnumpy().flatten() == self.ignore_label
                prob = prob*(1-ignore) + ignore
                num += prob.size - ignore.sum()
            else:
                num += prob.size
            loss += -numpy.log(numpy.maximum(1e-10, prob)).sum()

        self.sum_metric += numpy.exp(loss / num)
        self.num_inst += 1


####################
# REGRESSION METRICS
####################

class MAE(EvalMetric):
    """Calculate Mean Absolute Error loss"""

    def __init__(self):
        super(MAE, self).__init__('mae')

    def update(self, labels, preds):
        check_label_shapes(labels, preds)

        for label, pred in zip(labels, preds):
            label = label.asnumpy()
            pred = pred.asnumpy()

            if len(label.shape) == 1:
                label = label.reshape(label.shape[0], 1)

            self.sum_metric += numpy.abs(label - pred).mean()
            self.num_inst += 1 # numpy.prod(label.shape)

class MSE(EvalMetric):
    """Calculate Mean Squared Error loss"""
    def __init__(self):
        super(MSE, self).__init__('mse')

    def update(self, labels, preds):
        check_label_shapes(labels, preds)

        for label, pred in zip(labels, preds):
            label = label.asnumpy()
            pred = pred.asnumpy()

            if len(label.shape) == 1:
                label = label.reshape(label.shape[0], 1)

            self.sum_metric += ((label - pred)**2.0).mean()
            self.num_inst += 1 # numpy.prod(label.shape)

class RMSE(EvalMetric):
    """Calculate Root Mean Squred Error loss"""
    def __init__(self):
        super(RMSE, self).__init__('rmse')

    def update(self, labels, preds):
        check_label_shapes(labels, preds)

        for label, pred in zip(labels, preds):
            label = label.asnumpy()
            pred = pred.asnumpy()

            if len(label.shape) == 1:
                label = label.reshape(label.shape[0], 1)

            self.sum_metric += numpy.sqrt(((label - pred)**2.0).mean())
            self.num_inst += 1

class CrossEntropy(EvalMetric):
    """Calculate Cross Entropy loss"""
    def __init__(self, eps=1e-8):
        super(CrossEntropy, self).__init__('cross-entropy')
        self.eps = eps

    def update(self, labels, preds):
        check_label_shapes(labels, preds)

        for label, pred in zip(labels, preds):
            label = label.asnumpy()
            pred = pred.asnumpy()

            label = label.ravel()
            assert label.shape[0] == pred.shape[0]

            prob = pred[numpy.arange(label.shape[0]), numpy.int64(label)]
            self.sum_metric += (-numpy.log(prob + self.eps)).sum()
            self.num_inst += label.shape[0]

class Torch(EvalMetric):
    """Dummy metric for torch criterions"""
    def __init__(self, name='torch'):
        super(Torch, self).__init__(name)

    def update(self, _, preds):
        for pred in preds:
            self.sum_metric += pred.asnumpy().mean()
        self.num_inst += 1

class Caffe(Torch):
    """Dummy metric for caffe criterions"""
    def __init__(self):
        super(Caffe, self).__init__('caffe')

class CustomMetric(EvalMetric):
    """Custom evaluation metric that takes a NDArray function.

    Parameters
    ----------
    feval : callable(label, pred)
        Customized evaluation function.
    name : str, optional
        The name of the metric
    allow_extra_outputs : bool
        If true, the prediction outputs can have extra outputs.
        This is useful in RNN, where the states are also produced
        in outputs for forwarding.
    """
    def __init__(self, feval, name=None, allow_extra_outputs=False):
        if name is None:
            name = feval.__name__
            if name.find('<') != -1:
                name = 'custom(%s)' % name
        super(CustomMetric, self).__init__(name)
        self._feval = feval
        self._allow_extra_outputs = allow_extra_outputs

    def update(self, labels, preds):
        if not self._allow_extra_outputs:
            check_label_shapes(labels, preds)

        for pred, label in zip(preds, labels):
            label = label.asnumpy()
            pred = pred.asnumpy()

            reval = self._feval(label, pred)
            if isinstance(reval, tuple):
                (sum_metric, num_inst) = reval
                self.sum_metric += sum_metric
                self.num_inst += num_inst
            else:
                self.sum_metric += reval
                self.num_inst += 1

# pylint: disable=invalid-name
def np(numpy_feval, name=None, allow_extra_outputs=False):
    """Create a customized metric from numpy function.

    Parameters
    ----------
    numpy_feval : callable(label, pred)
        Customized evaluation function.
        This will get called with the labels and predictions
        for a minibatch, each as numpy arrays.  This function
        should return a single float.
    name : str, optional
        The name of the metric.
    allow_extra_outputs : bool
        If true, the prediction outputs can have extra outputs.
        This is useful in RNN, where the states are also produced
        in outputs for forwarding.
    """
    def feval(label, pred):
        """Internal eval function."""
        return numpy_feval(label, pred)
    feval.__name__ = numpy_feval.__name__
    return CustomMetric(feval, name, allow_extra_outputs)
# pylint: enable=invalid-name

def create(metric, **kwargs):
    """Create an evaluation metric.

    Parameters
    ----------
    metric : str or callable
        The name of the metric, or a function
        providing statistics given pred, label NDArray.
    """

    if callable(metric):
        return CustomMetric(metric)
    elif isinstance(metric, EvalMetric):
        return metric
    elif isinstance(metric, list):
        composite_metric = CompositeEvalMetric()
        for child_metric in metric:
            composite_metric.add(create(child_metric, **kwargs))
        return composite_metric

    metrics = {
        'acc': Accuracy,
        'accuracy': Accuracy,
        'ce': CrossEntropy,
        'f1': F1,
        'mae': MAE,
        'mse': MSE,
        'rmse': RMSE,
        'top_k_accuracy': TopKAccuracy
    }

    try:
        return metrics[metric.lower()](**kwargs)
    except:
        raise ValueError("Metric must be either callable or in {}".format(
            metrics.keys()))
