import sys
sys.path.append('../..')
import lmdb
import math
import numpy as np
import tensorflow as tf
from berry import layers
from kerasflow.proto import Datum
from kerasflow.data_manager import create_fixed_image_shape

_EPSILON = 1e-8


def print_l(l):
    print "{:30} \t {:20}".format(l.type, l.output_shape)


class DataGen:

    def __init__(self, lmdb_path, batch_size):
        self.lmdb_path = lmdb_path
        self.env = lmdb.open(lmdb_path, readonly=True)
        self.num_samples = int(self.env.stat()['entries'])
        self.batch_size = batch_size

    def _parse_data_from_string(self, value):
        datum = Datum()
        datum.ParseFromString(value)
        flat_x = np.fromstring(datum.data, dtype=np.uint8)
        x_ = flat_x.reshape(datum.channels, datum.height, datum.width)
        x = np.transpose(x_, axes=(1, 2, 0))
        x = create_fixed_image_shape(x, (224, 224, 3))
        y = datum.label
        return x, y

    def next_batch(self):
        X = []
        Y = []
        e = np.eye(50)
        while True:
            with self.env.begin() as txn:
                cursor = txn.cursor()
                for key, value in cursor:
                    try:
                        x, y = self._parse_data_from_string(value)
                        y = e[y]
                        if len(X) == self.batch_size:
                            assert (len(X) == self.batch_size and
                                    len(Y) == self.batch_size), (
                                "[ERROR] LMDBGenerator: "
                                "batch size mismatch error")
                            X_ = np.asarray(X)
                            # X_ = np.transpose(X_, (0, 3, 1, 2))
                            yield (X_, Y)
                            X[:] = []
                            Y[:] = []
                        X.append(x)
                        Y.append(y)
                    except Exception as e:
                        print "[ERROR] LMDBGenerator: Error generating data!!",
                        print e
                        raise e


def inference(i, keep_prob):
    print i.get_shape()
    l = layers.Convolution2D(i, 64, 3, stride=1, pad='SAME', activation='relu')
    print l.output_shape
    l = layers.Convolution2D(l, 64, 3, stride=1, pad='SAME', activation='relu')
    print l.output_shape
    l = layers.MaxPooling2D(l, 2, 2)
    print l.output_shape

    l = layers.Convolution2D(l, 128, 3, stride=1, pad='SAME', activation='relu')
    print l.output_shape
    l = layers.Convolution2D(l, 128, 3, stride=1, pad='SAME', activation='relu')
    print l.output_shape
    l = layers.MaxPooling2D(l, 2, 2)
    print l.output_shape

    l = layers.Convolution2D(l, 256, 3, stride=1, pad='SAME', activation='relu')
    print l.output_shape
    l = layers.Convolution2D(l, 256, 3, stride=1, pad='SAME', activation='relu')
    print l.output_shape
    l = layers.Convolution2D(l, 256, 3, stride=1, pad='SAME', activation='relu')
    print l.output_shape
    l = layers.MaxPooling2D(l, 2, 2)
    print l.output_shape

    l = layers.Convolution2D(l, 512, 3, stride=1, pad='SAME', activation='relu')
    print l.output_shape
    l = layers.Convolution2D(l, 512, 3, stride=1, pad='SAME', activation='relu')
    print l.output_shape
    l = layers.Convolution2D(l, 512, 3, stride=1, pad='SAME', activation='relu')
    print l.output_shape
    l = layers.MaxPooling2D(l, 2, 2)
    print l.output_shape
    print l.output_shape
    l = layers.Convolution2D(l, 512, 3, stride=1, pad='SAME', activation='relu')
    print l.output_shape
    l = layers.Convolution2D(l, 512, 3, stride=1, pad='SAME', activation='relu')
    print l.output_shape
    l = layers.Convolution2D(l, 512, 3, stride=1, pad='SAME', activation='relu')
    l = layers.MaxPooling2D(l, 2, 2)
    print l.output_shape
    l = layers.Flatten(l)
    l = layers.Dense(l, 4096, activation='relu')
    l = layers.Dropout(l, keep_prob)
    l = layers.Dense(l, 4096, activation='relu')
    l = layers.Dropout(l, keep_prob)
    l = layers.Dense(l, 50, activation='softmax')
    return l.output


def categorical_crossentropy(output, target):
    '''Categorical crossentropy between an output tensor
    and a target tensor, where the target is a tensor of the same
    shape as the output.
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


def loss(logits, labels):
    # cross_entropy = tf.reduce_mean(-tf.reduce_sum(labels * tf.log(logits),
    #                                               reduction_indices=[1]),
    #                                name='cross_entropy_loss')
    # return cross_entropy
    return categorical_crossentropy(logits, labels)


def training(loss, learning_rate):
    # Add a scalar summary for the snapshot loss.
    tf.scalar_summary(loss.op.name, loss)
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    # Create a variable to track the global step.
    # global_step = tf.Variable(0, name='global_step', trainable=False)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss)
    # train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


def run_train():
    num_epoch = 10
    batch_size = 16
    epoch = 1
    imagnet_train = DataGen(
        '/media/shared/platform_sample/img50/train_db', batch_size)
    imagnet_test = DataGen(
        '/media/shared/platform_sample/img50/test_db', batch_size)

    with tf.Graph().as_default():
        images_ph = tf.placeholder(tf.float32, shape=[batch_size, 224, 224, 3])
        labels_ph = tf.placeholder(tf.float32, shape=[batch_size, 50])
        keep_prob = tf.placeholder(tf.float32)

        output = inference(images_ph, keep_prob)
        loss_op = loss(output, labels_ph)
        train_op = training(loss_op, 5e-2)

        correct_prediction = tf.equal(
            tf.argmax(output, 1), tf.argmax(labels_ph, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        summary_op = tf.merge_all_summaries()

        sess = tf.Session()
        init = tf.initialize_all_variables()
        sess.run(init)

        for step, batch in enumerate(imagnet_train.next_batch()):
            feed_dict = {
                images_ph: batch[0],
                labels_ph: batch[1],
                keep_prob: 0.5
            }
            _, loss_value, out_val = sess.run([train_op, loss_op, output],
                                              feed_dict=feed_dict)
            if step % 10 == 0:
                loss_value, train_acc = sess.run(
                    [loss_op, accuracy], feed_dict={
                        images_ph: batch[0],
                        labels_ph: batch[1],
                        keep_prob: 1.0
                    })
                batch_id = step / (epoch)
                print("epoch %d, iter %d, acc %g, loss %g" %
                      (epoch, batch_id, train_acc, loss_value))
                # print out_val
            if (int(math.ceil(
                    imagnet_train.num_samples / float(batch_size))) == step):
                epoch += 1


if __name__ == '__main__':
    run_train()
