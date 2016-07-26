"""
:mod:`berry.objectives` module contains a list of loss functions and other evaluation methods for training or testing neural networks.

A small list of loss functions currently defined in :mod:`berry` are given below.

.. autosummary::
    :nosignatures:

    categorical_crossentropy
    triplet_loss

The following evaluation methods are provided to montior the train or test performance of the neural network.

.. autosummary::
    :nosignatures:

    accuracy


Helper function
---------------

.. autofunction:: get_objective

Examples
--------

You are encouraged to use the helper function :func:`get_objective`. Alternatively, you can use the objective functions directly,

>>> from berry.objectives import categorical_crossentropy
>>> # assume: `target` - output Tensor to predict, `output` - predicted Tensor
>>> # of the neural network
>>> loss_op = categorical_crossentropy(output, target)

Loss Functions
--------------

.. autofunction:: categorical_crossentropy
.. autofunction:: triplet_loss

Evaluation Methods
------------------

.. autofunction:: accuracy
"""
import tensorflow as tf


__all__ = [
    "get_objective"
]


_EPSILON = 1e-8
_OBJECTIVES = {}


def triplet_loss(Xa, Xp, Xn, margin=1.0):
    """Triplet loss function.

    This function is useful for generating embeddings for images/inputs such
    that similar inputs lie close in the embedding and different objects lie
    further away. It takes 3 inputs - anchor (``Xa``), positive sample
    (``Xp``) and negative sample (``Xn``). ``Xa`` and ``Xp`` are known to be
    similar inputs whereas ``Xa`` and ``Xn`` are different. For more details
    refer to [1]_.

    Parameters
    ----------
    output : ``tf.Tensor``
        Tensor containing the predicted class probabilities of the neural network.

    target : ``tf.Tensor``
        Tensor containing the true classes for the corresponding inputs.

    Returns
    -------
    ``tf.Tensor``
        Symbolic tensorflow Tensor which performs the loss calculation.

    References
    ----------
    .. [1] Florian Schroff, Dmitry Kalenichenko, James Philbin; "FaceNet: A
           Unified Embedding for Face Recognition and Clustering", The IEEE
           Conference on Computer Vision and Pattern Recognition (CVPR),
           2015, pp. 815-823.
    """
    similar_l2_norm = tf.sqrt(tf.reduce_sum(tf.mul(Xa - Xp, Xa - Xp), 1))
    different_l2_norm = tf.sqrt(tf.reduce_sum(tf.mul(Xa - Xn, Xa - Xn), 1))
    s = tf.maximum(0.0, similar_l2_norm - different_l2_norm + self.margin)
    s = tf.reduce_mean(s, name='triplet_loss')
    return s


def categorical_crossentropy(output, target):
    '''Categorical crossentropy between an output tensor
    and a target tensor, where the target is a tensor of the same
    shape as the output.

    Parameters
    ----------
    output : ``tf.Tensor``
        Tensor containing the predicted class probabilities of the neural network.

    target : ``tf.Tensor``
        Tensor containing the true classes for the corresponding inputs.

    Returns
    -------
    ``tf.Tensor``
        Symbolic tensorflow Tensor which performs the loss calculation.

    Notes
    -----
    This function expects ``output`` to contain class probabilities instead of
    class prediction scores. Typically, :func:`activations.softmax` activation
    should be used for the layer producing ``output``.
    '''
    # Note: tf.nn.softmax_cross_entropy_with_logits
    # expects logits, Keras expects probabilities.
    output /= tf.reduce_sum(output,
                            reduction_indices=len(output.get_shape()) - 1,
                            keep_dims=True)
    # manual computation of crossentropy
    output = tf.clip_by_value(output, tf.cast(
        _EPSILON, dtype=tf.float32), tf.cast(1. - _EPSILON, dtype=tf.float32))

    return - tf.reduce_mean(
        tf.reduce_sum(target * tf.log(output),
                      reduction_indices=len(output.get_shape()) - 1),
        name='cross_entropy_loss'
    )


def accuracy(output, target):
    """Multi-class zero-one accuracy

    The fraction of samples in the given mini-batch with the correct
    predicted class.

    Parameters
    ----------
    output : ``tf.Tensor``
        Tensor containing the predicted class probabilities of the neural network.

    target : ``tf.Tensor``
        Tensor containing the true classes for the corresponding inputs.

    Returns
    -------
    ``tf.Tensor``
        Symbolic tensorflow Tensor which performs the accuracy calculation.
    """
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(target, 1))
    accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy_op


_OBJECTIVES = {
    "categorical_crossentropy": categorical_crossentropy,
    "accuracy": accuracy,
    "triplet_loss": triplet_loss,
}


def get_objective(key, output, target):
    """Helper function to retrieve the appropriate objective function.

    Parameters
    ----------
    key : string
        Name of the objective function - "categorical_crossentropy",
        "accuracy", etc.

    Returns
    -------
    function
        The appropriate function given the ``key``.

    Examples
    --------
    >>> from berry.objectives import get_objective
    >>> # assume: `target` - output Tensor to predict, `output` - predicted
    >>> # Tensor of the neural network
    >>> accuracy_op = get_objective("accuracy", output, target)
    """
    global _OBJECTIVES
    if not _OBJECTIVES.has_key(key):
        raise NotImplementedError(
            "Supported list of losses: {}".format(_OBJECTIVES.keys()))
    # Add a scalar summary for the snapshot loss.
    # tf.scalar_summary(loss.op.name, loss)
    # Create the gradient descent loss with the given learning rate.
    loss = _OBJECTIVES[key]
    return loss(output, target)
