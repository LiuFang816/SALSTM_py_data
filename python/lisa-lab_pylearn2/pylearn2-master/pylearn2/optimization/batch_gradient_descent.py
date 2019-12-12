"""
.. todo::

    WRITEME
"""
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "LISA Lab"
__email__ = "pylearn-dev@googlegroups"

import logging
import time

import numpy as np
from theano.compat.six.moves import xrange

from theano import config
from theano.printing import var_descriptor
import theano.tensor as T

from pylearn2.compat import OrderedDict
from pylearn2.utils import function
from pylearn2.utils import grad
from pylearn2.utils import safe_zip
from pylearn2.utils import sharedX


logger = logging.getLogger(__name__)


class BatchGradientDescent(object):
    """
    A class for minimizing a function via the method of steepest descent.

    Parameters
    ----------
    objective : tensor_like
        A theano expression to be minimized should be a function of params and,
        if provided, inputs
    params : list
        A list of theano shared variables. These are the optimization variables
    inputs : list, optional
        A list of theano variables to serve as inputs to the graph.
    param_constrainers : list
        A list of callables to be called on all updates dictionaries to be
        applied to params. This is how you implement constrained
        optimization.
    reset_alpha : bool
        If True, reverts to using init_alpha after each call. If False, the
        final set of alphas is used at the start of the next call to minimize.
    conjugate : bool
        If True, tries to pick conjugate gradient directions. For the
        directions to be truly conjugate, you must use line_search_mode =
        'exhaustive' and the objective function must be quadratic. Using
        line_search_mode = 'exhaustive' on a non-quadratic objective function
        implements nonlinear conjugate gradient descent.
    reset_conjugate : bool
        Has no effect unless conjugate == True. If reset_conjugate ==
        True, reverts to direction of steepest descent for the first
        step in each call to minimize. Otherwise, tries to make the new
        search direction conjugate to the last one (even though the
        objective function might be totally different on each call to
        minimize)
    gradients : WRITEME
        If None, compute the gradients of obj using T.grad otherwise, a
        dictionary mapping from params to expressions for their gradients
        (this allows you to use approximate gradients computed with
        something other than T.grad)
    gradient_updates : dict
        A dictionary of shared variable updates to run each time the
        gradient is computed

    Notes
    -----
    Calling the `minimize` method with values for for `inputs` will
    update `params` to minimize `objective`.
    """
    def __init__(self, objective, params, inputs=None,
                 param_constrainers=None, max_iter=-1,
                 lr_scalers=None, verbose=0, tol=None,
                 init_alpha=None, min_init_alpha=1e-3,
                 reset_alpha=True, conjugate=False,
                 reset_conjugate=True, gradients=None,
                 gradient_updates=None, line_search_mode=None,
                 accumulate=False, theano_function_mode=None):

        self.__dict__.update(locals())
        del self.self

        if line_search_mode is None:
            if init_alpha is None:
                init_alpha = (.001, .005, .01, .05, .1)
        else:
            assert line_search_mode == 'exhaustive'
            if init_alpha is None:
                init_alpha = (.5, 1.)

        self.init_alpha = tuple([float(elem) for elem in init_alpha])

        if inputs is None:
            inputs = []

        if param_constrainers is None:
            param_constrainers = []

        obj = objective

        self.verbose = verbose

        param_to_grad_sym = OrderedDict()
        param_to_grad_shared = OrderedDict()
        updates = OrderedDict()
        if self.gradient_updates is not None:
            updates.update(self.gradient_updates)

        self.params = [param for param in params]

        for param in params:
            if self.gradients is not None and param in self.gradients:
                g = self.gradients[param]
            else:
                g = grad(objective, param)
            param_to_grad_sym[param] = g
            if param.name is not None:
                param_name = param.name
            else:
                param_name = 'anon_param'
            grad_name = 'BatchGradientDescent.grad_' + param_name
            grad_shared = sharedX(param.get_value() * 0., name=grad_name)
            param_to_grad_shared[param] = grad_shared
            updates[grad_shared] = g

        self.param_to_grad_shared = param_to_grad_shared

        if self.verbose:
            logger.info('batch gradient class compiling gradient function')
        t1 = time.time()
        if self.accumulate:
            self._compute_grad = Accumulator(inputs, updates=updates)
        else:
            self._compute_grad = function(
                inputs,
                updates=updates,
                mode=self.theano_function_mode,
                name='BatchGradientDescent._compute_grad')
        if self.verbose:
            t2 = time.time()
            logger.info('done. Took {0}'.format(t2-t1))

        if self.verbose:
            logger.info('batch gradient class compiling objective function')
        if self.accumulate:
            self.obj = Accumulator(inputs, obj)
        else:
            self.obj = function(inputs, obj, mode=self.theano_function_mode,
                                name='BatchGradientDescent.obj')

        if self.verbose:
            logger.info('done')

        self.param_to_cache = OrderedDict()
        alpha = T.scalar(name='alpha')
        alpha.tag.test_value = np.cast[alpha.dtype](.01)
        cache_updates = OrderedDict()
        goto_updates = OrderedDict()
        for param in params:
            if param.name is None:
                param_name = 'anon_param'
            else:
                param_name = param.name
            cache_name = 'BatchGradientDescent.param_to_cache[%s]' % param_name
            self.param_to_cache[param] = sharedX(param.get_value(borrow=False),
                                                 name=cache_name)
            cache_updates[self.param_to_cache[param]] = param
            cached = self.param_to_cache[param]
            g = self.param_to_grad_shared[param]
            if lr_scalers is not None and param in lr_scalers:
                scaled_alpha = alpha * lr_scalers[param]
            else:
                scaled_alpha = alpha
            mul = scaled_alpha * g
            diff = cached - mul
            goto_updates[param] = diff
        self._cache_values = function(
            [],
            updates=cache_updates,
            mode=self.theano_function_mode,
            name='BatchGradientDescent._cache_values')
        assert isinstance(param_constrainers, (list, tuple))
        for param_constrainer in param_constrainers:
            param_constrainer(goto_updates)
        self._goto_alpha = function(
            [alpha],
            updates=goto_updates,
            mode=self.theano_function_mode,
            name='BatchGradientDescent._goto_alpha')

        norm = T.sqrt(sum([T.sqr(elem).sum() for elem in
                           self.param_to_grad_shared.values()]))
        norm.name = 'BatchGradientDescent.norm'
        normalize_grad_updates = OrderedDict()
        for grad_shared in self.param_to_grad_shared.values():
            normalize_grad_updates[grad_shared] = grad_shared / norm

        # useful for monitoring
        self.ave_grad_size = sharedX(0.)
        self.new_weight = sharedX(1.)
        normalize_grad_updates[self.ave_grad_size] = \
            self.new_weight * norm + (1.-self.new_weight) * self.ave_grad_size

        self._normalize_grad = \
            function([],
                     norm,
                     updates=normalize_grad_updates,
                     mode=self.theano_function_mode,
                     name='BatchGradientDescent._normalize_grad')

        if self.conjugate:
            grad_shared = self.param_to_grad_shared.values()

            grad_to_old_grad = OrderedDict()
            for elem in grad_shared:
                grad_to_old_grad[elem] = \
                    sharedX(elem.get_value(), 'old_'+elem.name)

            self._store_old_grad = \
                function([norm],
                         updates=OrderedDict([(grad_to_old_grad[g_], g_ * norm)
                                             for g_ in grad_to_old_grad]),
                         mode=self.theano_function_mode,
                         name='BatchGradientDescent._store_old_grad')

            grad_ordered = list(grad_to_old_grad.keys())
            old_grad_ordered = [grad_to_old_grad[g_] for g_ in grad_ordered]

            def dot_product(x, y):
                return sum([(x_elem * y_elem).sum()
                           for x_elem, y_elem in safe_zip(x, y)])

            beta_pr = (dot_product(grad_ordered, grad_ordered) - dot_product(grad_ordered, old_grad_ordered)) / \
                (1e-7+dot_product(old_grad_ordered, old_grad_ordered))
            assert beta_pr.ndim == 0

            beta = T.maximum(beta_pr, 0.)

            # beta_pr is the Polak-Ribiere formula for beta.
            # According to wikipedia, the beta to use for NCG is "a matter of
            # heuristics or taste" but max(0, beta_pr) is "a popular choice...
            # which provides direction reset automatically." (ie, it is meant
            # to revert to steepest descent when you have traveled far enough
            # that the objective function is behaving non-quadratically enough
            # that the conjugate gradient formulas aren't working anymore)

            # http://en.wikipedia.org/wiki/Nonlinear_conjugate_gradient_method

            assert grad not in grad_to_old_grad

            make_conjugate_updates = \
                [(g_, g_ + beta * grad_to_old_grad[g_]) for g_ in grad_ordered]

            mode = self.theano_function_mode
            if mode is not None and hasattr(mode, 'record'):
                for v, u in make_conjugate_updates:
                    mode.record.handle_line(
                        'BatchGradientDescent._make_conjugate var '
                        + var_descriptor(v) + '\n')
                    mode.record.handle_line(
                        'BatchGradientDescent._make_conjugate update '
                        + var_descriptor(u) + '\n')

            self._make_conjugate = \
                function([], updates=make_conjugate_updates,
                         mode=self.theano_function_mode,
                         name='BatchGradientDescent._make_conjugate')

            if mode is not None and hasattr(mode, 'record'):
                for output in self._make_conjugate.maker.fgraph.outputs:
                    mode.record.handle_line(
                        'BatchGradientDescent._make_conjugate output '
                        + var_descriptor(output) + '\n')

        if tol is None:
            if objective.dtype == "float32":
                self.tol = 1e-6
            else:
                self.tol = 3e-7
        else:
            self.tol = tol

        self.ave_step_size = sharedX(0.)
        self.ave_grad_mult = sharedX(0.)

    def minimize(self, * inputs):
        """
        .. todo::

            WRITEME
        """

        if self.verbose:
            logger.info('minimizing')
        alpha_list = list(self.init_alpha)

        orig_obj = self.obj(*inputs)

        if self.verbose:
            logger.info(orig_obj)

        iters = 0

        # A bit of a hack here: we multiply by norm
        # when calling store_old_grad below. This is mostly
        # so we store the non-normalized version of the gradient,
        # but we can also exploit it to either clear the old grad
        # on the first iteration by setting norm = 0 initially.
        # This makes the algorithm reset to steepest descent on
        # each call to minimize. Or we can set the norm to 1 to
        # save the previous gradient, so we can try to maintain
        # conjugacy across several calls to minimize.
        # If self.conjugate is False none of this matters
        # since store_old_grad is never called anyway.
        if self.reset_conjugate:
            norm = 0.
        else:
            norm = 1.

        while iters != self.max_iter:
            if self.verbose:
                logger.info('batch gradient descent iteration '
                            '{0}'.format(iters))
            iters += 1
            self._cache_values()
            if self.conjugate:
                self._store_old_grad(norm)
            self._compute_grad(*inputs)
            if self.conjugate:
                self._make_conjugate()
            norm = self._normalize_grad()

            if self.line_search_mode is None:
                best_obj, best_alpha, best_alpha_ind = \
                    self.obj(* inputs), 0., -1
                prev_best_obj = best_obj

                for ind, alpha in enumerate(alpha_list):
                    self._goto_alpha(alpha)
                    obj = self.obj(*inputs)
                    if self.verbose:
                        logger.info('\t{0} {1}'.format(alpha, obj))

                    # Use <= rather than = so if there are ties
                    # the bigger step size wins
                    if obj <= best_obj:
                        best_obj = obj
                        best_alpha = alpha
                        best_alpha_ind = ind
                    # end if obj
                # end for ind, alpha

                if self.verbose:
                    logger.info(best_obj)

                assert not np.isnan(best_obj)
                assert best_obj <= prev_best_obj
                self._goto_alpha(best_alpha)

                step_size = best_alpha

                # if best_obj == prev_best_obj and alpha_list[0] < 1e-5:
                #    break
                if best_alpha_ind < 1 and alpha_list[0] > self.tol:
                    alpha_list = [alpha / 3. for alpha in alpha_list]
                    if self.verbose:
                        logger.info('shrinking the step size')
                elif best_alpha_ind > len(alpha_list) - 2:
                    alpha_list = [alpha * 2. for alpha in alpha_list]
                    if self.verbose:
                        logger.info('growing the step size')
                elif best_alpha_ind == -1 and alpha_list[0] <= self.tol:
                    if alpha_list[-1] > 1:
                        if self.verbose:
                            logger.info('converged')
                        break
                    if self.verbose:
                        logger.info('expanding the range of step sizes')
                    for i in xrange(len(alpha_list)):
                        for j in xrange(i, len(alpha_list)):
                            alpha_list[j] *= 1.5
                        # end for j
                    # end for i
                else:
                    # if a step succeeded and didn't result in growing or
                    # shrinking the step size then we can probably benefit
                    # from more fine-grained exploration of the middle
                    # ranges of step size (this is especially necessary if
                    # we've executed the 'expanding the range of step sizes'
                    # case multiple times)
                    a = np.asarray(alpha_list)
                    s = a[1:]/a[:-1]
                    max_gap = 5.
                    if s.max() > max_gap:
                        weight = .99
                        if self.verbose:
                            logger.info('shrinking the range of step sizes')
                        alpha_list = [(alpha ** weight) * (best_alpha
                                      ** (1.-weight)) for alpha in alpha_list]
                        assert all([second > first for first, second in
                                   safe_zip(alpha_list[:-1], alpha_list[1:])])
                        # y^(weight) best^(1-weight) / x^(weight)
                        # best^(1-weight) = (y/x)^weight
                        # so this shrinks the ratio between each successive
                        # pair of alphas by raising it to weight
                        # weight = .99 -> a gap of 5 is shrunk to 4.92

                # end check on alpha_ind
            else:
                assert self.line_search_mode == 'exhaustive'

                # In exhaustive mode, we search until we get very little
                # improvement (or have tried over ten points)
                # and we dynamically pick the search points to try to
                # maximize the improvement.
                # The points we pick are kind of dumb; it's just a binary
                # search. We could probably do better by fitting a function
                # and jumping to its local minima at each step

                if self.verbose > 1:
                    logger.info('Exhaustive line search')

                obj = self.obj(*inputs)
                if np.isnan(obj):
                    logger.warning("Objective is NaN for these parameters.")
                results = [(0., obj)]
                for alpha in alpha_list:
                    if not (alpha > results[-1][0]):
                        logger.error('alpha: {0}'.format(alpha))
                        logger.error('most recent alpha (should be smaller): '
                                     '{0}'.format(results[-1][0]))
                        assert False
                    self._goto_alpha(alpha)
                    obj = self.obj(*inputs)
                    if np.isnan(obj):
                        obj = np.inf
                    results.append((alpha, obj))
                if self.verbose > 1:
                    for alpha, obj in results:
                        logger.info('\t{0} {1}'.format(alpha, obj))

                    logger.info('\t-------')

                prev_improvement = 0.
                while True:
                    alpha_list = [alpha for alpha, obj in results]
                    obj = [obj for alpha, obj in results]
                    mn = min(obj)
                    idx = obj.index(mn)

                    def do_point(x):
                        self._goto_alpha(x)
                        res = self.obj(*inputs)
                        if self.verbose > 1:
                            logger.info('\t{0} {1}'.format(x, res))
                        # Regard NaN results as infinitely bad so they
                        # won't be picked as the min objective
                        if np.isnan(res):
                            res = np.inf
                        for i in xrange(len(results)):
                            elem = results[i]
                            ex = elem[0]
                            if x == ex:
                                raise AssertionError(str(ex) + "is \
                                                     already in the list.")
                            if x > ex:
                                if i + 1 == len(results) \
                                   or x < results[i+1][0]:
                                    results.insert(i+1, (x, res))
                                    return mn - res
                        assert False  # should be unreached

                    if idx == 0:
                        x = (alpha_list[0] + alpha_list[1]) / 2.
                    elif idx == len(alpha_list) - 1:
                        x = 2 * alpha_list[-1]
                    else:
                        if obj[idx+1] < obj[idx-1]:
                            x = (alpha_list[idx] + alpha_list[idx+1])/2.
                        else:
                            x = (alpha_list[idx] + alpha_list[idx-1])/2.

                    if x < 1e-20:
                        break

                    improvement = do_point(x)

                    if (improvement > 0 and
                       improvement < .01 * prev_improvement) or len(obj) > 10:
                        break
                    prev_improvement = improvement

                alpha_list = [alpha for alpha, ignore_obj in results]
                obj = [obj_elem for alpha, obj_elem in results]
                mn = min(obj)
                idx = obj.index(mn)
                x = alpha_list[idx]
                self._goto_alpha(x)
                # used for statistics gathering
                step_size = x
                if self.verbose:
                    logger.info('best objective: {0}'.format(mn))
                assert not np.isnan(mn)

                if idx == 0:
                    x = alpha_list[1]

                if self.min_init_alpha is not None:
                    x = max(x, 2. * self.min_init_alpha)

                alpha_list = [x/2., x]
                best_obj = mn
            # end if branching on type of line search

            new_weight = self.new_weight.get_value()
            old = self.ave_step_size.get_value()
            update = new_weight * step_size + (1-new_weight) * old
            update = np.cast[config.floatX](update)
            if self.ave_step_size.dtype == 'float32':
                assert update.dtype == 'float32'
            self.ave_step_size.set_value(update)

            old = self.ave_grad_mult.get_value()
            update = new_weight * (step_size / norm) + (1. - new_weight) * old
            update = np.cast[config.floatX](update)
            self.ave_grad_mult.set_value(update)
            # it is initialized to 1 to get all the means started at
            # data points, but then we turn it into a running average
            if new_weight == 1.:
                self.new_weight.set_value(.01)

        # end while

        if not self.reset_alpha:
            self.init_alpha = alpha_list

        # The way this optimizer is used (with max_iters set to 3 or 5 so
        # it doesn't go too far on one minibatch) this warning doesn't make
        # a lot of sense, but we might want a switch to turn it on if you
        # really are trying to absolutely minimize the objective for the
        # current inputs.
        # if norm > 1e-2:
        #    warnings.warn(str(norm)+  " seems pretty big for "
        #                  "a gradient at convergence...")

        return best_obj


class Accumulator(object):
    """
    Standin for a theano function with the given inputs, outputs, updates.

    Here in the __init__ method you give the same expression as usual.
    However, instead of passing __call__ the input variables directly, you
    pass it batches, where each batch is a list containing the inputs for
    that batch. It returns the average value of the function, averaged
    across batches, taking batch size into account. The average of all
    updates is also applied.

    One extra change: if any of the inputs is a shared variable, then this
    can assign to that variable, while theano.function would refuse to.
    Those shared variables will be left with the value of the last batch
    when __call__ returns.

    Parameters
    ----------
    inputs : WRITEME
    outputs : WRITEME
    updates : WRITEME
    """

    def __init__(self, inputs, outputs=None, updates=None):
        batch_size = T.cast(inputs[0].shape[0], 'float32')
        total_examples = T.scalar()
        transformed_updates = OrderedDict()
        self.has_updates = updates is not None
        if self.has_updates:
            self._clear = function([],
                                   updates=[(var, 0. * var)
                                   for var in updates])
            for var in updates:
                update = updates[var]
                transformed_updates[var] = var + \
                    (batch_size / total_examples) * update
        self._shared_mask = [hasattr(elem, 'get_value') for elem in inputs]
        true_inputs = self._true_inputs(inputs)
        self._shared = self._shared_inputs(inputs)
        if outputs is not None:
            if not isinstance(outputs, list):
                outputs = [outputs]
            outputs = [output * (batch_size / total_examples)
                       for output in outputs]
        self._func = function(true_inputs + [total_examples], outputs=outputs,
                              updates=transformed_updates)

    def _true_inputs(self, inputs):
        """
        .. todo::

            WRITEME
        """
        return [elem for elem, shared in safe_zip(inputs, self._shared_mask)
                if not shared]

    def _shared_inputs(self, inputs):
        """
        .. todo::

            WRITEME
        """
        return [elem for elem, shared in safe_zip(inputs, self._shared_mask)
                if shared]

    def _set_shared(self, inputs):
        """
        .. todo::

            WRITEME
        """
        for elem, mask, shared in safe_zip(inputs, self._shared_mask,
                                           self._shared):
            if mask:
                shared.set_value(elem)

    def __call__(self, * batches):
        """
        .. todo::

            WRITEME
        """
        for batch in batches:
            if not isinstance(batch, list):
                raise TypeError("Expected each argument to be a list,"
                                " but one argument is " +
                                str(batch) + " of type "+str(type(batch)))
        total_examples = np.cast[config.floatX](
            sum([batch[0].shape[0] for batch in batches]))
        if self.has_updates:
            self._clear()
        augmented = self._true_inputs(batches[0]) + [total_examples]
        self._set_shared(batches[0])
        rval = self._func(*augmented)
        for batch in batches[1:]:
            augmented = self._true_inputs(batch) + [total_examples]
            self._set_shared(batch)
            # This works if there is no output,
            # because the output is an empty list
            cur_out = self._func(*augmented)
            rval = [x + y for x, y in safe_zip(rval, cur_out)]
        if len(rval) == 1:
            return rval[0]
        return rval


def norm_sq(s):
    """
    .. todo::

        WRITEME
    """
    return np.square(s.get_value()).sum()


def scale(s, a):
    """
    .. todo::

        WRITEME
    """
    s.set_value(s.get_value() * np.cast[config.floatX](a))
