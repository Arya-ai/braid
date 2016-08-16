:mod:`berry.layers`
===================

.. currentmodule:: braid.berry.layers

.. toctree::
    :hidden:

    layers/base
    layers/conv
    layers/rnn
    layers/dense
    layers/pool
    layers/shape
    layers/noise


.. rubric:: :doc:`layers/base`

.. autosummary::
    :nosignatures:

    Layer
    get_all_aux_params
    print_layers_summary


.. rubric:: :doc:`layers/conv`

.. autosummary::
    :nosignatures:

    Convolution2D

.. rubric:: :doc:`layers/rnn`

.. autosummary::
    :nosignatures:

    RNN


.. rubric:: :doc:`layers/pool`

.. autosummary::
    :nosignatures:

    MaxPooling2D


.. rubric:: :doc:`layers/dense`

.. autosummary::
    :nosignatures:

    Dense


.. rubric:: :doc:`layers/shape`

.. autosummary::
    :nosignatures:

    Flatten


.. rubric:: :doc:`layers/noise`

.. autosummary::
    :nosignatures:

    Dropout

Helper Functions
----------------

Certain layers like :class:`Dropout` require the definition of additional variables like ``p`` which takes on different values during train and test phase. For running any operation on the tensorflow graph (``tf.Graph``), it is necessary to feed in the value to ``p`` variable as well. In order to handle such situations, a convenient function, :func:`get_all_aux_params` is provided which aggregates such variables along with the appropriate values from all the layers according to the train/test phase.

For additional clarity on the model definition and in order to verify that the intended architecture is being created, one can use the :func:`print_layers_summary` function to print additional information about the layers.

.. autofunction:: get_all_aux_params
.. autofunction:: print_layers_summary
