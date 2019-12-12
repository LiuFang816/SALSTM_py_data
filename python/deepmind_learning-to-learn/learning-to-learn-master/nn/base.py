# Copyright 2016 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Base class for TensorFlow nn.

This file contains the Abstract Base Class for defining Modules in TensorFlow.
A Module is an object which can be connected into the Graph multiple times
using the __call__ method, sharing variables automatically with no need to
explicitly use scopes or specify reuse=True.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import six
from six import string_types
from six.moves import xrange
import tensorflow as tf


class Error(Exception):
  """Base class for all errors from nn.

  This is thrown to indicate a Neural Network specific problem, e.g. wrong
  module arity, module is not connected to the graph when it should be,
  tried to wire together incompatible modules, etc.
  """


class NotConnectedError(Error):
  """Error raised when operating on a module that has not yet been connected.

  Some module properties / methods are valid to access before the module has
  been connected into the graph, but some are not. This Error is raised when
  the user attempts to do anything not valid before connection.
  """


class ParentNotBuiltError(Error):
  """Error raised when the parent of a module has not been built yet.

  For example, when making a transpose of modules which inherit from
  `module.Transposable`, the parent has to be connected to the graph before the
  child transpose to ensure that shape inference has already occurred.
  """


class IncompatibleShapeError(Error):
  """Error raised when the shape of the input at build time is incompatible."""


class UnderspecifiedError(Error):
  """Error raised when too little information is available.

  This does not typically mean the user is trying to do something that doesn't
  work (in which case `IncompatibleShapeError` should be used), just that
  some more information needs to be provided in order to build the Graph.
  """


class NotSupportedError(Error):
  """Error raised when something that cannot be supported is requested.

  For example a Dilated Convolution module cannot be transposed.
  """


@six.add_metaclass(abc.ABCMeta)
class AbstractModule(object):
  """Superclass for nn Modules.

  This class defines the functionality that every module should implement,
  principally the `build` method which is wrapped using `tf.make_template`
  and called from `__call__`. Every time the module is called it will
  be connected into the graph but using the same shared set of variables, thanks
  to the template.

  For this to work correctly, the `build` implementation in the derived class
  must access all variables using `tf.get_variable`, not `tf.Variable`. The same
  set of variables must be created each time, if this is not the case an Error
  will be raised.

  Every subclass must call this class' `__init__` at the start of their
  `__init__`, passing the relevant name. If this step is omitted variable
  sharing will not work.
  """

  # Name of TensorFlow collection containing ops to update every step, such as
  # moving average update ops.
  UPDATE_OPS_COLLECTION = tf.GraphKeys.UPDATE_OPS

  def __init__(self, name):
    """Performs the initialisation necessary for all AbstractModule instances.

    Every subclass of AbstractModule must begin their constructor with a call to
    this constructor, i.e. `super(MySubModule, self).__init__(name=name)`.

    Avoid instantiating sub-modules in __init__ where possible, as they will not
    be defined under the module's scope. Instead, instantiate sub-modules in
    `build`.

    Args:
      name: Name of this module. Used to construct the Templated build function.

    Raises:
      ValueError: If name is not specified.
    """
    if not isinstance(name, string_types):
      raise ValueError("Name must be a string.")
    self._is_connected = False
    self._template = tf.make_template(name, self._build,
                                      create_scope_now_=True)

    # Update __call__ and the object docstrings to enable better introspection
    self.__doc__ = self._build.__doc__
    self.__call__.__func__.__doc__ = self._build.__doc__

  @abc.abstractmethod
  def _build(self, *args, **kwargs):
    """Add elements to the Graph, computing output Tensors from input Tensors.

    Subclasses must implement this method, which will be wrapped in a Template.

    Args:
      *args: Input Tensors.
      **kwargs: Additional Python flags controlling connection.
    """
    pass

  def __call__(self, *args, **kwargs):
    out = self._template(*args, **kwargs)
    # Connect the module only if self._template returns with no errors.
    self._is_connected = True
    return out

  @property
  def variable_scope(self):
    """Returns the variable_scope declared by the module.

    It is valid for library users to access the internal templated
    variable_scope, but only makes sense to do so after connection. Therefore
    we raise an error here if the variable_scope is requested before connection.

    The only case where it does make sense to access the variable_scope before
    connection is to get the post-uniquification name, which we support using
    the separate .name property.

    Returns:
      variable_scope: `tf.VariableScope` instance of the internal `tf.Template`.

    Raises:
      NotConnectedError: If the module is not connected to the Graph.
    """
    self._ensure_is_connected()
    return self._template.variable_scope

  @property
  def name(self):
    """Returns the name of the Module."""
    return self._template.variable_scope.name

  @property
  def is_connected(self):
    """Returns true iff the Module been connected to the Graph at least once."""
    return self._is_connected

  @classmethod
  def get_possible_initializer_keys(cls):
    """Returns the keys the dictionary of variable initializers may contain.

    This provides the user with a way of knowing the initializer keys that are
    available without having to instantiate a nn module. Subclasses may
    override this class method if they need additional arguments to determine
    what initializer keys may be provided.

    Returns:
      Set with strings corresponding to the strings that may be passed to the
          constructor.
    """
    return getattr(cls, "POSSIBLE_INITIALIZER_KEYS", set())

  def _ensure_is_connected(self):
    """Raise an Error if the module has not been connected yet.

    Until the module is connected into the Graph, any variables created do
    not exist yet and cannot be created in advance due to not knowing the size
    of the input Tensor(s). This assertion ensures that any variables contained
    in this module must now exist.

    Raises:
      NotConnectedError: If the module is not connected to the Graph.
    """
    if not self.is_connected:
      raise NotConnectedError(
          "Variables in {} not instantiated yet, __call__ the module "
          "first.".format(self.name))


@six.add_metaclass(abc.ABCMeta)
class Transposable(object):
  """Transposable module interface.

    The Transposable interface requires that transposable modules implement
    a method called `transpose`, returning a module which is the transposed
    version of the one the method is called on.
    Calling the method twice should return a module with the same specifications
    as the original module.

    When implementing a transposable module, special care is required to make
    sure that parameters needed to instantiate the module are provided as
    functions whose invocation is deferred to graph construction time.

    For example, in Linear we might want to call:

    ```python
    linear = nn.Linear(name="linear", output_size=output_size)
    linear_transpose = linear.transpose()
    ```

    where the output_size for linear_transpose is not known yet, as linear is
    not yet connected to the graph: output_size is passed to linear_transpose's
    constructor as a lambda returning linear.input_size. The lambda will return
    the correct value once linear is given an input.
    Notice that linear_transpose's output_size value does not need to be defined
    until the module is connected to the graph.
  """

  @abc.abstractmethod
  def transpose(self, name=None, **kwargs):
    """Builds and returns transposed version of module.

    Args:
      name: Name of the transposed module.
      **kwargs: Additional Python flags controlling transposition.

    Returns:
      Transposed version of the module.
    """
    pass

  @abc.abstractmethod
  def input_shape(self):
    """Returns shape of input `Tensor` passed at last call to `build`."""
    pass


class Module(AbstractModule):
  """Module wrapping a function provided by the user."""

  def __init__(self, build, name="module"):
    """Constructs a module with a given build function.

    The Module class can be used to wrap a function assembling a network into a
    module.

    For example, the following code implements a simple one-hidden-layer MLP
    model by defining a function called make_model and using a Module instance
    to wrap it.

    ```python
    def make_model(inputs):
      lin1 = nn.Linear(name="lin1", output_size=10)(inputs)
      relu1 = tf.nn.relu(lin1, name="relu1")
      lin2 = nn.Linear(name="lin2", output_size=20)(relu1)
      return lin2

    model = nn.Module(name='simple_mlp', build=make_model)
    outputs = model(inputs)
    ```

    The `partial` package from `functools` can be used to bake configuration
    parameters into the function at construction time, as shown in the following
    example.

    ```python
    from functools import partial

    def make_model(inputs, output_sizes):
      lin1 = nn.Linear(name="lin1", output_size=output_sizes[0])(inputs)
      relu1 = tf.nn.relu(lin1, name="relu1")
      lin2 = nn.Linear(name="lin2", output_size=output_sizes[1])(relu1)
      return lin2

    model = nn.Module(name='simple_mlp',
                       build=partial(make_model, output_size=[10, 20])
    outputs = model(inputs)
    ```

    Args:
      build: Callable to be invoked when connecting the module to the graph.
          The `build` function is invoked when the module is called, and its
          role is to specify how to add elements to the Graph, and how to
          compute output Tensors from input Tensors.
          The `build` function signature can include the following parameters:
            *args - Input Tensors.
            **kwargs - Additional Python parameters controlling connection.
      name: Module name.

    Raises:
      TypeError: If build is not callable.
    """
    super(Module, self).__init__(name)

    if not callable(build):
      raise TypeError("Input 'build' must be callable.")
    self._build = build

  def _build(self, *args, **kwargs):
    """Forwards call to the passed-in build function."""
    return self._build(*args, **kwargs)
