"""
:mod:`berry.optimizers` module contains classes for training a neural network.

The training process involves looping over the entire training data multiple
times (epochs) seeing a handful samples at a time (training step/iteration).
Each training step in turn consists mainly of two steps:

* computing gradient of loss function w.r.t. each parameter
* updating each parameter using this gradient weighted by the learning
  rate (this depends on the type of optimizer used)

List of optimizers available:

.. autosummary::
    :nosignatures:

    sgd


It is possible to catch vanishing/exploding gradients problem by monitoring
the gradients every few iterations. The class below prints the mean and
standard deviation of the parameters and it's corresponding gradients in
order to facilitate better debugging in case the neural network is not
learning.

.. autosummary::
    :nosignatures:

    GradientSanity


Helper function
---------------

.. autofunction:: get_optimizer

Examples
--------

You are encouraged to use the helper function :func:`get_optimizer`.
Alternatively, you can use the optimizer functions directly,

>>> from berry.optimizers import sgd
>>> optim = sgd(learning_rate=0.01)

Optimizers
----------

.. autofunction:: sgd

Gradient Monitoring
-------------------

.. autoclass:: GradientSanity
    :members:
    :undoc-members:

"""
import sys
import tensorflow as tf
import numpy as np

__all__ = [
    "get_optimizer",
    "GradientSanity"
]


def sgd(learning_rate):
    """Stochastic Gradient Descent

    Parameters
    ----------
    learning_rate : float
        Rate of update of paramter values based on it's gradients.

    Returns
    -------
    Derived class of ``f.train.Optimizer``
        Class which performs the gradient computation and performs backward
        pass.

    Notes
    -----
    Parameter update step for SGD is

    .. math::
        \\theta_i = \\theta_i + \\alpha \\nabla \\mathcal{J}_{\\theta_i}
        (x^{(i)}, y^{(i)})

    where :math:`\\nabla\\mathcal{J}_i(x^{(i)}, y^{(i)})` is the gradient of
    the loss for :math:`i` -th mini-batch w.r.t. paramter :math:`\\theta_i`.
    """
    return tf.train.GradientDescentOptimizer(learning_rate)


_OPTIMIZERS = {
    'sgd': sgd
}


def get_optimizer(key, loss_op, learning_rate,
                  global_step=None, **kwargs):
    """Helper function to retrieve the appropriate optimizer class.

    Parameters
    ----------
    key : string
        Name of the optimizer class - "sgd", etc.

    Returns
    -------
    ``tf.Tensor``
        Training operation.

    list of ``tf.Variable``
        List of gradients w.r.t. each trainable parameter.

    Examples
    --------
    >>> from berry.optimizers import get_optimizer
    >>> # assume: loss_op is the loss operation returned by
    >>> # berry.objectives.get_objective()
    >>> train_op, grads = get_optimizer("sgd", loss_op, 0.1)
    """
    global _OPTIMIZERS
    if not _OPTIMIZERS.has_key(key):
        raise NotImplementedError(
            "Supported list of optimizers: {}".format(_OPTIMIZERS.keys()))
    # Add a scalar summary for the snapshot loss_op.
    # tf.scalar_summary(loss_op.op.name, loss_op)
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = _OPTIMIZERS[key](learning_rate)
    # train_op = optimizer.minimize(loss_op, global_step=global_step)
    grads_and_vars = optimizer.compute_gradients(loss_op)
    train_op = optimizer.apply_gradients(
        grads_and_vars, global_step=global_step)
    grads = [g for (g, _) in grads_and_vars]

    return train_op, grads


class GradientSanity(object):
    """Montior the value of parameters and it's corresponding gradients.

    Prints the mean and standard deviation of the parameter values an it's
    gradients in a nice formatted table.

    Parameters
    ----------
    session : ``tf.Session``
        The tensorflow session in which the operations are defined.

    param_ops : list of ``tf.Variable``
        List of trainable paramters.

    grad_ops : list of ``tf.Variable``
        List of gradient of loss function w.r.t. the trainable paramters.

    Attributes
    ----------
    ops : list of ``tf.Variable``
        List of parameters and gradients.

    Examples
    --------
    >>> sess = tf.Session()
    >>> _, grads = get_optimizer("sgd", loss_op, 0.1)
    >>> param_ops = tf.trainable_variables()
    >>> sanity = GradientSanity(sess, param_ops, grads)
    >>> # assume: feed_dict = {'x:0': ..., 'y:0': ...}
    >>> sanity.run(feed_dict=feed_dict)
    """

    def __init__(self, session, param_ops, grad_ops):
        self.sess = session
        self.params_name = [p.name for p in param_ops]
        self.num_params = len(param_ops)
        self.ops = param_ops[:]
        self.ops.extend(grad_ops)

    def print_msg(self, str):
        """Format for printing
        """
        print "[INFO] {}: {}".format(self.__class__.__name__, str)

    def run(self, feed_dict={}):
        """Perform the forward pass and print a summary of the parameter and gradient values.

        Parameters
        ----------
        feed_dict : dict
            Dict containing the input and target data for forward pass.
        """
        values = self.sess.run(self.ops, feed_dict=feed_dict)
        params = values[:self.num_params]
        grads = values[self.num_params:]
        self.print_msg("Parameters summary")
        self.print_summary(params, grads, self.params_name)

    def print_summary(self, vals, grads, names):
        """Print a formatted table summary of parameter and gradient values.

        Parameters
        ----------
        vals : list of ``np.ndarray``
            List of parameter values.

        grads : list of ``np.ndarray``
            List of gradient values.

        names : list of string
            List of names of the parameters.
        """
        if len(vals) == 0:
            return
        print_fmt = "|{:^20}|{:^20}|{:^12}|{:^12}|{:^12}|{:^12}|"
        n = len(print_fmt.format('1', '1', '1', '1', '1', '1'))
        line = ''.join(['-' for i in range(n)])
        print line
        print print_fmt.format("Name", "Shape", "Abs Mean",
                               "Abs Std", "Grad Mean", "Grad Std")
        print line
        for name, v, g in zip(names, vals, grads):
            shape = v.shape
            assert shape == g.shape, self.print_msg(
                "while printing gradient shape and parameter shape mis-match")
            mean = np.abs(v).mean()
            std = np.abs(v).std()
            gmean = np.abs(g).mean()
            gstd = np.abs(g).std()
            print print_fmt.format(name, shape, '%.2e' % mean, '%.2e' % std,
                                   '%.2e' % gmean, '%.2e' % gstd)
            print line
        sys.stdout.flush()
