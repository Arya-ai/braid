import os
import numpy as np
import time
import threading
import time
import datetime
import tensorflow as tf


class Callback(object):
    """docstring for Callback"""

    def __init__(self):
        pass

    def on_train_begin(self, logs={}):
        pass

    def on_train_end(self, logs={}):
        pass

    def on_epoch_begin(self, epoch, logs={}):
        pass

    def on_epoch_end(self, epoch, logs={}):
        pass

    def on_batch_begin(self, batch, logs={}):
        pass

    def on_batch_end(self, batch, logs={}):
        pass


class ThreadingCallback(threading.Thread):

    def __init__(self, status_func, callback):
        threading.Thread.__init__(self)
        self._get_status = status_func
        self.callback = callback

    def run(self):
        time.sleep(.2)
        stat = self._get_status()
        self.callback(stat)


class History(Callback):

    def __init__(self):
        self.history = {'epochs': []}

    def on_epoch_end(self, epoch, logs={}):
        self.history['epochs'].append(epoch)
        for k, v in logs.items():
            if not self.history.has_key(k):
                self.history[k] = []
            self.history[k].append(v)


class TrainCallback(Callback):

    def __init__(self, status_func, callback):
        self._get_status = status_func
        self.callback = callback

    def on_epoch_end(self, epoch, logs={}):
        # send callback
        threaded_callback = ThreadingCallback(self._get_status, self.callback)
        threaded_callback.start()


class ModelSaver(Callback):
    """docstring for ModelSaver"""

    def __init__(self, graph, session, path):
        super(ModelSaver, self).__init__()
        self.sess = session
        self.path = path
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        with graph.as_default():
            self.saver = tf.train.Saver(
                max_to_keep=10000,
                keep_checkpoint_every_n_hours=1
            )

    def on_epoch_end(self, epoch, logs={}):
        self.saver.save(self.sess, self.path, global_step=epoch)
        print "[INFO] ModelSaver: Model saved: {}-{}".format(self.path, epoch)


class Timer(Callback):
    '''Callback that records events into a `Timer` object.
    '''

    def on_train_begin(self, logs={}):
        self.epoch = []
        self.history = []

    def on_epoch_begin(self, epoch, logs={}):
        self._t_enter_epoch = time.time()

    def on_epoch_end(self, epoch, logs={}):
        if not hasattr(self, '_t_enter_epoch'):
            self._t_enter_epoch = time.time()
        self._delta_t_epoch = round(time.time() - self._t_enter_epoch, 3)
        self.epoch.append(epoch)
        self.history.append(self._delta_t_epoch)


class LossHistory(Callback):
    '''Callback that records losses of every batch.
    '''

    def on_train_begin(self, logs={}):
        self.losses = {}
        self.key = 0

    def on_batch_end(self, batch, logs={}):
        if not self.losses.has_key(self.key):
            self.losses[self.key] = []
        self.losses[self.key].append(logs.get('loss'))

    def on_epoch_end(self, epoch, logs={}):
        self.key += 1


class TimeEstimator(Callback):

    def __init__(self, number_of_batches):
        self.number_of_batches = number_of_batches
        self.start_time = None
        self.count = 0
        self.progress = 0
        self.time_estm = str(datetime.timedelta(seconds=0))

    def on_train_begin(self, logs={}):
        self.start_time = time.time()

    def on_batch_end(self, batch, logs={}):
        end_time = time.time()
        time_diff = end_time - self.start_time
        self.count += 1
        self.avg_time_per_batch = time_diff / (self.count + 1e-7)

        remaining_batches = self.number_of_batches - self.count
        remain_time = int(round(self.avg_time_per_batch *
                                remaining_batches))
        self.time_estm = str(datetime.timedelta(seconds=remain_time))
        self.progress = round(100.0 * self.count /
                              self.number_of_batches + 1e-7, 2)

    def on_train_end(self, logs={}):
        self.progress = 100.0
        self.time_estm = str(datetime.timedelta(seconds=0))
