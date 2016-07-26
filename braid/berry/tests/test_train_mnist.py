""" Sample code for training on mnist data set.
"""
import os
import tensorflow as tf
import numpy as np
from math import ceil
from berry import BerryModel
from berry.optimizers import get_optimizer
from berry.objectives import get_objective
from berry.layers import (Convolution2D, MaxPooling2D,
                          Dense, Flatten)


def model_definition(input_layer, num_classes):
    """Define the LeNet network"""
    nn = BerryModel()
    nn.add(input_layer)
    nn.add(Convolution2D(nn.last, 20, 5, pad='VALID',
                         activation='sigmoid', W_stddev=1e-1))
    nn.add(MaxPooling2D(nn.last, 2, 2))
    nn.add(Convolution2D(nn.last, 50, 5, pad='SAME', W_stddev=1e-1,
                         activation='sigmoid'))
    nn.add(MaxPooling2D(nn.last, 2, 2))
    nn.add(Flatten(nn.last))
    nn.add(Dense(nn.last, 500, activation='relu', W_stddev=5e-3))
    nn.add(Dense(nn.last, num_classes, activation='softmax',
                 W_stddev=5e-3))
    return nn


def next_batch_gen(X, Y, batch_size):
    """Simple data generator from an array"""
    seen_samples = 0
    num_samples = X.shape[0]
    num_batches = int(ceil(num_samples / batch_size))
    while True:
        for i in range(num_batches):
            X_ = X[i * batch_size:(i + 1) * batch_size, ...]
            Y_ = None if Y is None else Y[
                i * batch_size:(i + 1) * batch_size]
            yield (X_, Y_)
            seen_samples += batch_size
        remaining = num_samples - seen_samples
        if remaining > 0:
            X_ = X[-remaining:, ...]
            Y_ = None if Y is None else Y_[-remaining:]
            yield (X_, Y_)


def train():
    from keras.datasets import mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    # Define model input shape and number of output classes
    model_input_shape = [None, 28, 28, 1]
    num_classes = 10
    model_output_shape = [None, num_classes]
    # path to saved weights of the current model - used for initializing
    # weights
    weights_path = '/path/to/model/weights-5'
    # Specify a directory to save the trained models
    model_save_path = './weights'

    # training params
    nb_epochs = 10
    batch_size = 1000
    lr = 0.1
    loss_type = "categorical_crossentropy"
    optim_type = "sgd"

    # Prepare training and testing data
    assert (X_test.shape[1:] == tuple(model_input_shape)[1:] and
            X_train.shape[1:] == tuple(model_input_shape)[1:]), "Input data shape mismatch"
    e = np.eye(num_classes, dtype=np.uint)
    y_train = e[y_train]
    y_test = e[y_test]
    assert (y_train.shape[1:] == tuple(model_output_shape)[1:] and
            y_test.shape[1:] == tuple(model_output_shape)[1:]), (
        "Y data should be one-hot vecotr format")
    num_batches = int(ceil(X_train.shape[0] / batch_size))
    train_gen = next_batch_gen(X_train, y_train, batch_size)

    # Create a new tensorflow graph to run the model
    # Helps run multiple models in separation
    device = '/gpu:0'  # gpu or device to run the graph on
    graph = tf.Graph()

    with graph.as_default() as g, graph.device(device):
        #####################################################
        # Initialization and model setup
        #####################################################
        # input and target placeholders - data will be fed into these
        print "Creating placeholders for taking input data"
        x_placeholder = tf.placeholder(
            tf.float32, shape=model_input_shape, name='X')
        y_placeholder = tf.placeholder(
            tf.float32, shape=model_output_shape, name='Y')
        # weights, biases, input nodes of the model
        print "Loading model..."
        model = model_definition(x_placeholder, num_classes)
        # output node of the model
        output_layer = model.output
        # auxilliary model inputs - train phase (e.g.: 'p' in case of Dropout)
        aux_inputs_train = model.train_aux
        # auxilliary model inputs - test phase (e.g.: 'p' in case of Dropout)
        aux_inputs_test = model.test_aux
        print "Done loading model"
        # loss node for the model
        print "Setting up the loss function"
        loss_op = get_objective(loss_type, output_layer, y_placeholder)
        # node corresponding to the optimizer (SGD, Adagrad, etc)
        with tf.device(None):
            # this is incremented for every step of gradient descent - can be
            # used for learning rate decay
            global_step = tf.Variable(0, name='global_step', trainable=False)
        # Return the train_op and grad_op - train_op performs 1 training step,
        # 'grad_op' can be analysed to check for vanishing or exploding
        # gradients - see berry.optimizers.GradientSanity
        print "Setting up optimizer"
        train_op, grad_op = get_optimizer(
            optim_type, loss_op, lr, global_step=global_step)
        # define the accuracy metric
        accuracy_op = get_objective("accuracy", output_layer, y_placeholder)
        # session for running the graph and training procedure
        sess = tf.Session()
        init = tf.initialize_all_variables()
        print "Initializing model weights randomly"
        sess.run(init)
        # load the saved model weights from file if provided
        if os.path.exists(weights_path):
            with tf.device(None):
                loader = tf.train.Saver()
                loader.restore(sess, weights_path)
            print "Loading parameter values from {}".format(weights_path)

        # Ensure that the X placeholder and model ops are defined in
        # the same `Graph` associated with graph
        layers.print_layers_summary(model.layers)

        if os.path.exists(os.path.dirname(model_save_path)):
            with tf.device(None):
                saver = tf.train.Saver(
                    max_to_keep=10000, keep_checkpoint_every_n_hours=1)
        else:
            saver = None

        #####################################################
        # Train loop
        #####################################################
        with sess.as_default():
            for epoch in range(nb_epochs):
                print "Epoch {}/{}".format(epoch, nb_epochs)
                # epoch begin
                train_loss = 0.
                train_accuracy = 0.
                count = 0.
                loss = {}
                for batch_id in range(num_batches):
                    # iteration/batch begin
                    step = num_batches * epoch + batch_id
                    (X, Y) = next(train_gen)
                    feed_dict = {
                        x_placeholder.name: X,
                        y_placeholder.name: Y
                    }
                    feed_dict.update(aux_inputs_train)

                    train_op.run(feed_dict=feed_dict)

                    if step % 10 == 0:
                        feed_dict.update(aux_inputs_test)
                        loss_value = loss_op.eval(
                            feed_dict=feed_dict)
                        train_acc = accuracy_op.eval(
                            feed_dict=feed_dict)
                        print("epoch {}, iter {}, train acc {}, "
                              "train loss {}".format(
                                  epoch, step, train_acc, loss_value))
                        train_loss += loss_value
                        train_accuracy += train_acc
                        count += 1.
                    # iteration/batch end
                if saver is not None:
                    # save model weights for the given epoch
                    saver.save(sess, model_save_path, global_step=epoch)
                    print "Saving model checkpoint"

if __name__ == '__main__':
    train()
