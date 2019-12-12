"""Bricks that compose together other bricks in linear sequences."""
import copy
from toolz import interleave, unique
from picklable_itertools.extras import equizip

from ..utils import pack
from .base import Brick, application, lazy
from .interfaces import Feedforward, Initializable
from .simple import Linear


class Sequence(Brick):
    """A sequence of bricks.

    This brick applies a sequence of bricks, assuming that their in- and
    outputs are compatible.

    Parameters
    ----------
    application_methods : list
        List of :class:`.BoundApplication` or :class:`.Brick` to apply.
        For :class:`.Brick`s, the ``.apply`` method is used.

    """
    def __init__(self, application_methods, **kwargs):
        pairs = ((a.apply, a) if isinstance(a, Brick) else (a, a.brick)
                 for a in application_methods)
        self.application_methods, bricks = zip(*pairs)
        kwargs.setdefault('children', []).extend(unique(bricks))
        super(Sequence, self).__init__(**kwargs)

    @application
    def apply(self, *args):
        child_input = args
        for application_method in self.application_methods:
            output = application_method(*pack(child_input))
            child_input = output
        return output

    @apply.property('inputs')
    def apply_inputs(self):
        return self.application_methods[0].inputs

    @apply.property('outputs')
    def apply_outputs(self):
        return self.application_methods[-1].outputs


class FeedforwardSequence(Sequence, Feedforward):
    """A sequence where the first and last bricks are feedforward.

    Parameters
    ----------
    application_methods : list
        List of :class:`.BoundApplication` to apply. The first and last
        application method should belong to a :class:`Feedforward` brick.

    """
    @property
    def input_dim(self):
        return self.children[0].input_dim

    @input_dim.setter
    def input_dim(self, value):
        self.children[0].input_dim = value

    @property
    def output_dim(self):
        return self.children[-1].output_dim

    @output_dim.setter
    def output_dim(self, value):
        self.children[-1].output_dim = value


class MLP(FeedforwardSequence, Initializable):
    """A simple multi-layer perceptron.

    Parameters
    ----------
    activations : list of :class:`.Brick`, :class:`.BoundApplication`,
                  or ``None``
        A list of activations to apply after each linear transformation.
        Give ``None`` to not apply any activation. It is assumed that the
        application method to use is ``apply``. Required for
        :meth:`__init__`.
    dims : list of ints
        A list of input dimensions, as well as the output dimension of the
        last layer. Required for :meth:`~.Brick.allocate`.
    prototype : :class:`.Brick`, optional
        The transformation prototype. A copy will be created for every
        activation. If not provided, an instance of :class:`~simple.Linear`
        will be used.

    Notes
    -----
    See :class:`Initializable` for initialization parameters.

    Note that the ``weights_init``, ``biases_init`` (as well as
    ``use_bias`` if set to a value other than the default of ``None``)
    configurations will overwrite those of the layers each time the
    :class:`MLP` is re-initialized. For more fine-grained control, push the
    configuration to the child layers manually before initialization.

    >>> from blocks.bricks import Tanh
    >>> from blocks.initialization import IsotropicGaussian, Constant
    >>> mlp = MLP(activations=[Tanh(), None], dims=[30, 20, 10],
    ...           weights_init=IsotropicGaussian(),
    ...           biases_init=Constant(1))
    >>> mlp.push_initialization_config()  # Configure children
    >>> mlp.children[0].weights_init = IsotropicGaussian(0.1)
    >>> mlp.initialize()

    """
    @lazy(allocation=['dims'])
    def __init__(self, activations, dims, prototype=None, **kwargs):
        self.activations = activations
        self.prototype = Linear() if prototype is None else prototype
        self.linear_transformations = []
        for i in range(len(activations)):
            linear = copy.deepcopy(self.prototype)
            name = self.prototype.__class__.__name__.lower()
            linear.name = '{}_{}'.format(name, i)
            self.linear_transformations.append(linear)
        if not dims:
            dims = [None] * (len(activations) + 1)
        self.dims = dims
        # Interleave the transformations and activations
        applications = [a for a in interleave([self.linear_transformations,
                                               activations]) if a is not None]
        super(MLP, self).__init__(applications, **kwargs)

    @property
    def input_dim(self):
        return self.dims[0]

    @input_dim.setter
    def input_dim(self, value):
        self.dims[0] = value

    @property
    def output_dim(self):
        return self.dims[-1]

    @output_dim.setter
    def output_dim(self, value):
        self.dims[-1] = value

    def _push_allocation_config(self):
        if not len(self.dims) - 1 == len(self.linear_transformations):
            raise ValueError
        for input_dim, output_dim, layer in \
                equizip(self.dims[:-1], self.dims[1:],
                        self.linear_transformations):
            layer.input_dim = input_dim
            layer.output_dim = output_dim
            if getattr(self, 'use_bias', None) is not None:
                layer.use_bias = self.use_bias
