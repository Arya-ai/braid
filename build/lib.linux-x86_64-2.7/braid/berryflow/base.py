from __future__ import division, absolute_import
import numpy as np
import os
import time
import pickle
import warnings
import tensorflow as tf
from braid.berry import layers
from math import floor, ceil
from sklearn.metrics import confusion_matrix
from .data_manager import Transformer
from .misc import print_msg
from .misc import FLAGS
try:
    from .misc import custom_formatwarning
    warnings.formatwarning = custom_formatwarning
except:
    pass


class ModelLoader(object):

    def construct_model_graph_from_proto(self, prototxt, input_layer,
                                         num_classes, model_params={}):
        """ a prototxt parser to convert prototxt to berry model.

        Return the standard LeNet, VGG, AlexNet model for now.
        """
        from braid.protoflow import ProtoFlow

        obj = ProtoFlow(prototxt, input_layer, num_classes)
        return obj.model

    def construct_model_graph(self, train_model_name,
                              input_layer, num_classes,
                              model_params={}):
        """ TODO: implement the prototxt parser
        Return the standard LeNet, VGG, AlexNet model for now.
        """
        from .models import (build_berry_lenet,
                             build_berry_vgg16,
                             build_berry_alexnet)
        models = {
            'lenet': build_berry_lenet,
            'alexnet': build_berry_alexnet,
            'vgg': build_berry_vgg16
        }
        func_model = models.get(train_model_name.lower(), None)
        if func_model is None:
            print_msg("Supported models: {}".format(models.keys()),
                      level=NotImplementedError, obj=self)
        return func_model(input_layer, num_classes, model_params)

    def configure_objective(self, loss, output, target):
        from braid.berry.objectives import get_objective

        loss_op = get_objective(loss, output, target)
        return loss_op

    def configure_optimizer(self, optim, loss_op,
                            global_step=None, params={}):
        from braid.berry.optimizers import get_optimizer

        lr = params.get('lr', 0.01)
        train_grad_op = get_optimizer(optim, loss_op, lr,
                                      global_step=global_step, **params)
        return train_grad_op


class BerryFlow(ModelLoader):

    def __init__(self, input_shape=(), num_classes=0,
                 train_model_name=None, batch_size=0,
                 train_lmdb_path=None, test_lmdb_path=None, weights_path=None,
                 device=None, augmentation_params={}, model_params={},
                 metrics=[]):
        #
        # Ensure parameter correctness
        #
        self.bool_train = self._validate_and_parse_params(
            input_shape, num_classes, batch_size, train_model_name,
            train_lmdb_path, test_lmdb_path, weights_path
        )
        loss_type = model_params.get('loss', "categorical_crossentropy")
        optim_type = model_params.get('optim', "sgd")

        if len(self.input_shape) == 2 or (
                len(self.input_shape) == 3 and self.input_shape[2] == 1):
            self.input_shape = tuple(self.input_shape[:2])
        if weights_path is not None and not os.path.exists(weights_path):
            print_msg("'weights_path' file does not exist:"
                      " '%s'; training from scratch." % weights_path,
                      level='warn', obj=self)
            weights_path = None
        #
        # Augmentation params nad Transformer init
        #
        self.image_input_shape = self._configure_transformer(
            self.input_shape, self.num_classes, augmentation_params)
        #
        # Build the tensorflow graph
        #
        # Create a new tf graph for defining all the ops
        self.device = device
        self.graph = tf.Graph()

        self.model_input_shape = [None] + \
            list(self.image_input_shape)
        if len(self.model_input_shape) == 3:
            self.model_input_shape += [1]
        if len(self.model_input_shape) != 4:
            print_msg("invalid model input shape detected: {}"
                      " expected (batch_size, height, width, channels)".format(
                          self.model_input_shape), obj=self)
        self.model_output_shape = [None, self.num_classes]

        # define the computational graph containing the model

        with self.graph.as_default() as g, self.graph.device(self.device):
            # input and target placeholders
            self.x_placeholder = tf.placeholder(
                tf.float32, shape=self.model_input_shape, name='X')
            self.y_placeholder = tf.placeholder(
                tf.float32, shape=self.model_output_shape, name='Y')
            # weights, biases, input nodes of the model
            if os.path.exists(self.train_model_name):
                model = self.construct_model_graph_from_proto(
                    self.train_model_name, self.x_placeholder,
                    self.num_classes, model_params)
            else:
                model = self.construct_model_graph(
                    self.train_model_name, self.x_placeholder,
                    self.num_classes, model_params)
            # output node of the model
            self.output_layer = model.output
            # auxilliary model inputs - train phase
            self.aux_inputs_train = model.train_aux
            # auxilliary model inputs - test phase
            self.aux_inputs_test = model.test_aux
            # loss node for the model
            self.loss_op = self.configure_objective(
                loss_type, self.output_layer, self.y_placeholder)
            # node corresponding to the optimizer (SGD, Adagrad, etc)
            with tf.device(None):
                self.global_step = tf.Variable(0, name='global_step',
                                               trainable=False)
            train_grad_op = self.configure_optimizer(
                optim_type, self.loss_op,
                global_step=self.global_step, params=model_params
            )
            self.grad_op = None
            if isinstance(train_grad_op, tuple):
                self.train_op, self.grad_op = train_grad_op
            else:
                self.train_op = train_grad_op

            correct_prediction = tf.equal(tf.argmax(self.output_layer, 1),
                                          tf.argmax(self.y_placeholder, 1))
            self.accuracy_op = tf.reduce_mean(
                tf.cast(correct_prediction, tf.float32))
            # session for running the graph and training procedure
            self.sess = tf.Session()
            init = tf.initialize_all_variables()
            self.sess.run(init)
            # load the saved model weights from file if provided
            if weights_path is not None:
                saver = tf.train.Saver()
                saver.restore(self.sess, weights_path)
                print_msg("Loading variables from {}".format(weights_path),
                          obj=self, level='info')

            # Ensure that the X placeholder and model ops are defined in
            # the same `Graph` associated with self.graph
            self._verify_all_ops_same_graph(g)
            layers.print_layers_summary(model.layers)
        #
        # Initialize matrices for storing the metrics
        #
        self.metrics = metrics
        self.confusion_matrix = (np.zeros((self.num_classes, self.num_classes),
                                          dtype=np.int64)
                                 if 'confusion_matrix' in metrics else False)
        #
        # Print the params
        #
        print_msg("Model input shape: {}".format(self.image_input_shape),
                  obj=self, level='info')
        print_msg("Model output shape: {}".format(self.num_classes),
                  obj=self, level='info')
        print_msg("Model batch size: {}".format(self.batch_size),
                  obj=self, level='info')
        print_msg("Training mode: {}".format(self.bool_train),
                  obj=self, level='info')
        if model_params:
            print_msg("Model parameters:", obj=self, level='info')
            for k in model_params.keys():
                print("\t{}: {}".format(k, model_params[k]))

        if self.bool_train:
            self._prepare_train(train_lmdb_path, test_lmdb_path)

    def _prepare_train(self, train_lmdb_path, test_lmdb_path):
        self.train_lmdb_path = train_lmdb_path
        self.test_lmdb_path = test_lmdb_path
        self._open()

    def _open(self):
        from .data_manager import LMDBGenerator

        self.generator = LMDBGenerator(
            self.transformer, self.model_input_shape, self.batch_size,
            self.x_placeholder, self.y_placeholder,
            self.train_lmdb_path, self.test_lmdb_path,
        )

    def train(self, nb_epochs, callback=None, output_dir=None,
              analyse_grads=False):
        from .outputs import Timer, LossHistory, TimeEstimator
        from .outputs import TrainCallback, History, ModelSaver

        # Configure callbacks
        if output_dir is None:
            print_msg("output_dir required to "
                      "save the model and model parameters", obj=self)
        self.timer = Timer()
        self.loss_history = LossHistory()
        self.history = History()
        total_num_batches = nb_epochs * \
            ceil(self.generator.num_train_samples / float(self.batch_size))
        print_msg("Total number of batches: %d" %
                  total_num_batches, obj=self, level='info')
        print_msg("Batch size: %d" % self.batch_size, obj=self, level='info')
        self.time_est = TimeEstimator(total_num_batches)
        # output_callbacks = ([EarlyStopping(patience=5)]
        #                     if self.generator.test else [])
        output_callbacks = [self.time_est]
        output_callbacks.append(self.loss_history)
        output_callbacks.append(self.timer)
        output_callbacks.append(self.history)
        if output_dir is not None:
            output_path = os.path.join(output_dir, FLAGS['save_prefix_path'])
            output_callbacks.append(ModelSaver(self.graph, self.sess,
                                               output_path))
        if callback is not None:
            self.traincallback = TrainCallback(self.get_status, callback)
            output_callbacks.append(self.traincallback)

        # Init train and test generators
        if self.generator is None:
            self._open()
        train_gen = self.generator.train_batches_from_lmdb()
        test_gen = None
        if self.generator.test:
            test_gen = self.generator.test_batches_from_lmdb()
        # Save model configuration for predicting/resuming training later
        self.save_class_params(output_dir)
        try:
            self._train_loop(nb_epochs, train_gen, test_gen,
                             output_callbacks, analyse_grads)
        except KeyboardInterrupt:
            print_msg('\nInterrupting training...', obj=self, level='info')
        time.sleep(2)

    def _train_loop(self, nb_epochs, train_gen, test_gen,
                    output_callbacks=[], analyse_grads=False):
        """ Start training.
        """
        from braid.berry.optimizers import GradientSanity

        num_train_samples = self.generator.num_train_samples
        num_train_iters = int(ceil(num_train_samples / self.batch_size))

        num_test_samples = self.generator.num_test_samples
        num_test_iters = int(ceil(num_test_samples / self.batch_size))

        with self.graph.as_default() as g:

            if analyse_grads:
                params = tf.trainable_variables()
                grad_sanity = GradientSanity(self.sess, params, self.grad_op)

            step = 0
            # training begin
            for func in output_callbacks:
                func.on_train_begin()

            with self.sess.as_default():
                for epoch in range(nb_epochs):
                    print_msg("Epoch {}/{}".format(epoch, nb_epochs),
                              obj=self, level='info')
                    # epoch begin
                    for func in output_callbacks:
                        func.on_epoch_begin(epoch)
                    train_loss = 0.
                    train_accuracy = 0.
                    count = 0.
                    loss = {}
                    for batch_id, batch_dict in enumerate(train_gen):
                        # iteration/batch begin
                        step = num_train_iters * epoch + batch_id
                        for func in output_callbacks:
                            func.on_batch_begin(step)
                        feed_dict = batch_dict
                        feed_dict.update(self.aux_inputs_train)

                        self.train_op.run(feed_dict=feed_dict)

                        if step % 10 == 0:
                            feed_dict.update(self.aux_inputs_test)
                            loss_value = self.loss_op.eval(
                                feed_dict=feed_dict)
                            train_acc = self.accuracy_op.eval(
                                feed_dict=feed_dict)
                            train_loss += loss_value
                            train_accuracy += train_acc
                            count += 1.
                            print_msg("epoch {}, iter {}, train acc {}, "
                                      "train loss {}".format(
                                          epoch, step, train_acc, loss_value),
                                      obj=self, level='info')
                        if step % 100 == 0:
                            if analyse_grads:
                                grad_sanity.run(feed_dict=feed_dict)
                        # iteration/batch end
                        for func in output_callbacks:
                            func.on_batch_end(step)

                        if num_train_iters - 1 == batch_id:
                            break

                    # validation evaluation
                    if test_gen is not None:
                        test_acc, test_loss = self._test_loop(
                            test_gen, num_test_iters)
                        print_msg("epoch {}, iter {}, test acc {}, "
                                  "test loss {}".format(
                                      epoch, step, test_acc, test_loss),
                                  obj=self, level='info')

                    # epoch end
                    logs = {
                        'acc': train_accuracy / count,
                        'loss': train_loss / count,
                    }
                    if test_gen is not None:
                        logs['val_acc'] = test_acc
                        logs['val_loss'] = test_loss
                    for func in output_callbacks:
                        func.on_epoch_end(epoch, logs=logs)
                # train end
                for func in output_callbacks:
                    func.on_train_end()

    def predict(self, X, batch_size=None):
        from .data_manager import VanillaGenerator, LMDBGenerator

        batch_size = self.batch_size if batch_size is None else batch_size
        if isinstance(X, str):
            gen = LMDBGenerator(
                self.transformer, self.model_input_shape, batch_size,
                self.x_placeholder, self.y_placeholder,
                X, X
            )
            test_gen = gen.test_batches_from_lmdb()
            num_test_iters = int(ceil(gen.num_test_samples / batch_size))
        else:
            X = np.asarray(X)
            batch_size = (X.shape[0] if batch_size >
                          X.shape[0] else batch_size)
            X = np.asarray(X)
            num_test_iters = int(ceil(X.shape[0] / batch_size))
            test_gen = VanillaGenerator(
                self.num_classes, batch_size, X_test=X,
                transformer=self.transformer,
                input_ph=self.x_placeholder, target_ph=self.y_placeholder
            ).batch_generator()
        return self._predict_loop(test_gen, num_test_iters)

    def _predict_loop(self, test_gen, num_test_iters):
        predictions = []
        with self.sess.as_default():
            for step, batch_dict in enumerate(test_gen):
                feed_dict = batch_dict
                feed_dict.update(self.aux_inputs_test)
                print_msg("batch %d" % (step + 1), obj=self, level='info')
                preds = self.sess.run(self.output_layer, feed_dict=feed_dict)
                predictions.extend(preds)

                if num_test_iters - 1 == step:
                    break
        return np.asarray(predictions)

    def test(self, batch_size=100):
        if self.generator is None:
            self._open()
        if not self.generator.test:
            print_msg("class not configured with test dataset; "
                      "`test_lmdb_path` should be provided", obj=self)

        test_gen = self.generator.test_batches_from_lmdb()
        num_test_samples = self.generator.num_test_samples
        num_test_iters = int(ceil(num_test_samples / self.batch_size))
        test_acc, test_loss = self._test_loop(test_gen, num_test_iters)
        print_msg("test accuracy: {:03f} test loss: {:03f}".format(
            test_acc, test_loss), obj=self, level='info')
        return test_acc, test_loss

    def _test_loop(self, test_gen, num_test_iters):
        test_accuracy = 0.
        test_loss = 0.
        if self.confusion_matrix is not False:
            conf_mat = np.zeros_like(self.confusion_matrix)
        with self.sess.as_default():
            for step, batch_dict in enumerate(test_gen):
                feed_dict = batch_dict
                feed_dict.update(self.aux_inputs_test)

                loss, acc, y_pred = self.sess.run(
                    [self.loss_op, self.accuracy_op, self.output_layer],
                    feed_dict=feed_dict)
                test_loss += loss
                test_accuracy += acc

                if self.confusion_matrix is not False:
                    y_true = feed_dict[self.y_placeholder.name].argmax(1)
                    y_pred = y_pred.argmax(1)
                    conf_mat += confusion_matrix(y_true, y_pred)

                if num_test_iters - 1 == step:
                    break

            test_accuracy /= num_test_iters
            test_loss /= num_test_iters
            if self.confusion_matrix is not False:
                self.confusion_matrix = conf_mat
        return test_accuracy, test_loss

    def get_status(self, batch_losses=False):
        try:
            stat = {
                'time_estm': self.time_est.time_estm,
                'progress': self.time_est.progress
            }
            try:
                stat['prev_epoch_time'] = self.timer.history[-1]
            except Exception:
                stat['prev_epoch_time'] = -1
            for key in self.history.history.keys():
                stat[key] = self.history.history[key]
            if batch_losses and len(self.loss_history.losses) > 1:
                stat['batch_losses'] = self.loss_history.losses
            if self.confusion_matrix is not False:
                stat['confusion_matrix'] = self.confusion_matrix.tolist()
        except Exception as e:
            print e
            stat = {}
        return stat

    def get_activations(self, X, names=[]):
        X = np.asarray(X)
        if X.shape[1:] != tuple(self.model_input_shape[1:]):
            print_msg("`X` shape {} did not match model input shape {}".format(
                X.shape, self.model_input_shape), obj=self)
        feed_dict = {
            self.x_placeholder.name: X,
            self.y_placeholder.name: np.zeros((X.shape[0], self.num_classes),
                                              np.uint8)
        }
        feed_dict.update(self.aux_inputs_test)
        with self.graph.as_default():
            all_act_ops = tf.get_collection(tf.GraphKeys.ACTIVATIONS)
        act_ops = []
        act_op_names = [op.name for op in all_act_ops]
        if names == []:
            act_ops = all_act_ops
            names = act_op_names
        else:
            for n in names:
                for op_name, op in zip(act_op_names, all_act_ops):
                    if n in op_name:
                        act_ops.append(op)
        activations = self.sess.run(act_ops, feed_dict=feed_dict)
        dict_activations = {}
        for (act, name, val) in zip(act_ops, names, activations):
            if tuple(act.get_shape().as_list()[1:]) != val.shape[1:]:
                print_msg(
                    "activation shape {} does not match the layer "
                    "output shape dimesnions {}".format(
                        tuple(act.get_shape().as_list()[1:]), val.shape[1:]),
                    obj=self)
            dict_activations[name] = val
        return dict_activations

    def get_learned_parameters(self):
        with self.graph.as_default():
            all_ops = tf.trainable_variables()
        values = self.sess.run(all_ops)
        dict_params = dict([(k.name, v) for k, v in zip(all_ops, values)])
        return dict_params

    def _verify_all_ops_same_graph(self, graph):
        if self.x_placeholder.graph != graph:
            print_msg("x_placeholder not defined in the class graph context",
                      obj=self)
        if self.y_placeholder.graph != graph:
            print_msg("y_placeholder not defined in the class graph context",
                      obj=self)
        if self.output_layer.graph != graph:
            print_msg("Model output layer not in the class graph context",
                      obj=self)
        if self.loss_op.graph != graph:
            print_msg("Loss op not defined in the class graph context",
                      obj=self)
        if self.train_op.graph != graph:
            print_msg("Train op not defined in the class graph context",
                      obj=self)

    def _configure_transformer(self, input_shape,
                               num_classes, augmentation_params):
        augmentation_params['reshape'] = augmentation_params.get(
            'reshape', input_shape)
        if augmentation_params:
            self.transform_params = augmentation_params
        self.transform_params['num_classes'] = num_classes
        self.transformer = Transformer(**self.transform_params)
        # Input shape determination
        random_crop = augmentation_params.get('random_crop', ())
        if len(random_crop) > 0:
            if len(input_shape) <= len(random_crop):
                input_shape = random_crop
            elif len(random_crop) == 2 and len(input_shape) == 3:
                input_shape = tuple(list(random_crop) + [input_shape[2]])
            if random_crop != input_shape:
                print_msg("random crop shape and"
                          " model input shape not matching.", obj=self)
        return input_shape

    def _validate_and_parse_params(self, input_shape, num_classes, batch_size,
                                   train_model_name, train_lmdb_path,
                                   test_lmdb_path, weights_path):
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.train_model_name = train_model_name
        # check if model should be run in 'train' or 'test' mode
        bool_train = True
        if train_lmdb_path is None:
            bool_train = False
            if weights_path is None or not os.path.exists(weights_path):
                print_msg("both 'weights_path' and 'train_lmdb_path' "
                          "cannot be None/not exist", obj=self)
            self.load_class_params(weights_path)
        elif not os.path.isdir(train_lmdb_path):
            print_msg("'train_lmdb_path' directory does not exist", obj=self)
        if test_lmdb_path is not None and not os.path.isdir(test_lmdb_path):
            print_msg("'test_lmdb_path': does not exist, given {}".format(
                test_lmdb_path), obj=self)
        if self.input_shape and not isinstance(
                self.input_shape, (tuple, list)):
            print_msg("'input_shape': tuple or list expected", obj=self)
        if train_lmdb_path is not None and not os.path.isdir(train_lmdb_path):
            print_msg("'train_lmdb_path': does not exist", obj=self)
        if self.num_classes and not isinstance(self.num_classes, int):
            print_msg("Number of classes `int` expected, given: {}".format(
                self.num_classes), obj=self)
        if self.batch_size is None or self.batch_size <= 0:
            print_msg("`int` batch size > 0 expected, given: {}".format(
                self.batch_size), obj=self)
        if len(self.input_shape) not in {2, 3}:
            print_msg("Incorrect input_shape {}, should be 2 or 3 dims".format(
                self.input_shape), level=ValueError, obj=self)
        return bool_train

    def load_class_params(self, weights_path):
        params_path = os.path.join(os.path.dirname(weights_path), 'params.pkl')
        if params_path is None or not os.path.isfile(params_path):
            print_msg("Unable to load model params; "
                      "No such file %s" % params_path, obj=self)
        try:
            with open(params_path, 'r') as fp:
                saved_params = pickle.load(fp)
        except Exception as e:
            print_msg("Failed to load params_path; "
                      "ensure it is a pickle file.", level=e, obj=self)
        self.input_shape = saved_params['input_shape']
        self.train_model_name = saved_params['train_model_name']
        self.batch_size = saved_params['batch_size']
        self.num_classes = saved_params['num_classes']
        self.transform_params = saved_params['augmentation_params']

    def save_class_params(self, output_dir):
        param_pkl_path = os.path.join(output_dir, 'params.pkl')
        with open(param_pkl_path, 'w') as fp:
            saved_params = {}
            saved_params['input_shape'] = self.image_input_shape
            saved_params['train_model_name'] = self.train_model_name
            saved_params['batch_size'] = self.batch_size
            saved_params['num_classes'] = self.num_classes
            saved_params['augmentation_params'] = self.transform_params
            print_msg("Saving model params to '%s'" % param_pkl_path,
                      obj=self, level="info")
            pickle.dump(saved_params, fp)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def close(self):
        self.sess.close()
        try:
            if self.generator:
                self.generator.close()
            self.generator = None
        except:
            pass
