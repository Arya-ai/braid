import os
import argparse
import numpy as np
import tensorflow as tf
from pprint import pprint
from berryflow import BerryFlow
import lmdb
from skimage.io import imsave
from joblib import Parallel, delayed
from berryflow.data_manager import Transformer
from berryflow.proto import Datum

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '../data/')
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)


def test_callback(stat):
    # print "\nEpoch ended"
    # pprint(stat)
    pass


# Example: ProtoFlow
def test_train_protoflow():
    train_db_path = '/media/shared/platform_sample/img50/train_db'
    test_db_path = '/media/shared/platform_sample/img50/test_db'
    prototxt_path = '/home/arya_03/protoflow/protoflow/models/vgg.pt'
    params = {
        'input_shape': (256, 256, 3),
        'num_classes': 50,
        'train_model_name': prototxt_path,
        'batch_size': 64,
        'train_lmdb_path': train_db_path,
        'test_lmdb_path': test_db_path,
        'augmentation_params': {
            'reshape': (256, 256, 3),
            'random_crop': (224, 224, 3),
            # 'zoom': [0.8, 1.2],
            # 'hflip': True
        },
        'model_params': {
            'lr': 0.1,
            'loss': "categorical_crossentropy"
        }
    }
    with tf.device('/gpu:2'):
        nn = BerryFlow(**params)
        print 'Start training...'
        nn.train(100, output_dir=os.path.join(DATA_DIR, 'tf_vgg_pf'),
                 analyse_grads=False)
        print 'Finished training.'
        pprint(nn.get_status())
        nn.close()


# Example: LeNet
def test_train_lenet():
    train_db_path = '/media/shared/mnist/mnist_train_lmdb/'
    test_db_path = '/media/shared/mnist/mnist_test_lmdb/'
    # train_db_path = '/media/shared/mnist/platform_created/train_db'
    # test_db_path = '/media/shared/mnist/platform_created/valid_db'
    params = {
        'input_shape': (28, 28),
        'num_classes': 10,
        'train_model_name': 'lenet',
        'batch_size': 100,
        'train_lmdb_path': train_db_path,
        'test_lmdb_path': test_db_path,
        # 'weights_path': os.path.join(DATA_DIR, 'tf_lenet/weights-25'),
        'model_params': {
            'lr': 0.1
        },
        'metrics': ['confusion_matrix'],
        'device': '/gpu:0'
    }
    nn = BerryFlow(**params)
    print 'Start training...'
    nn.train(30, output_dir=os.path.join(DATA_DIR, 'tf_lenet'),
             analyse_grads=False)
    print 'Finished training.'
    pprint(nn.get_status())
    nn.close()


def test_activations():
    train_db_path = '/media/shared/mnist/mnist_train_lmdb/'
    test_db_path = '/media/shared/mnist/mnist_test_lmdb/'
    params = {
        'input_shape': (28, 28),
        'num_classes': 10,
        'train_model_name': 'lenet',
        'batch_size': 100,
        'train_lmdb_path': train_db_path,
        'test_lmdb_path': test_db_path,
        'weights_path': os.path.join(DATA_DIR, 'tf_lenet/weights-25'),
        'model_params': {
            'lr': 0.1
        },
        'metrics': ['confusion_matrix']
    }
    with tf.device('/gpu:2'):
        nn = BerryFlow(**params)
        x = np.random.randint(0, 255, tuple([1] + nn.model_input_shape[1:]))
        print 'Printing activations'
        acts = nn.get_activations(x, names=['layer_3'])
        pprint(acts)
        print 'Printing learned features'
        feats = nn.get_learned_parameters()
        pprint(feats)
        nn.close()


def test_predict():
    # from keras.datasets import mnist
    # (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # X_test = X_test[..., np.newaxis]
    # X_test = '/media/shared/mnist/mnist_test_lmdb/'
    X_test = '/media/shared/platform_sample/img50/test_db'

    with tf.device('/gpu:2'):
        nn = BerryFlow(
            # (28, 28), 10, 'lenet', 1000,
            weights_path=os.path.join(DATA_DIR, 'tf_alexnet/weights-29'))
        print 'Start testing...'
        preds = nn.predict(X_test, 64)
        print 'Finished testing.'
        pprint(preds.argmax(1))
        print preds.shape
        nn.close()


# Example: AlexNet
def test_train_alexnet():
    train_db_path = '/media/shared/platform_sample/img50/train_db'
    test_db_path = '/media/shared/platform_sample/img50/test_db'
    # train_db_path = '/media/shared/iksula_ecommarce/POC_set_lmdb/train_db'
    # test_db_path = '/media/shared/iksula_ecommarce/POC_set_lmdb/test_db'
    params = {
        'input_shape': (256, 256, 3),
        'num_classes': 50,
        'train_model_name': 'alexnet',
        'batch_size': 64,
        'train_lmdb_path': train_db_path,
        'test_lmdb_path': test_db_path,
        # 'weights_path': os.path.join(DATA_DIR, 'tf_alexnet/weights-25'),
        'augmentation_params': {
            'reshape': (256, 256, 3),
            'random_crop': (227, 227, 3),
            'zoom': [0.8, 1.2],
            'hflip': True
        },
        'model_params': {
            'lr': 5e-2,
            'loss': "categorical_crossentropy"
        }
    }
    with tf.device('/gpu:2'):
        nn = BerryFlow(**params)
        print 'Start training...'
        nn.train(30, output_dir=os.path.join(DATA_DIR, 'tf_alexnet'),
                 analyse_grads=True)
        print 'Finished training.'
        pprint(nn.get_status())
        nn.close()


# Example: VGG
def test_train_vgg():
    train_db_path = '/media/shared/platform_sample/img50/train_db'
    test_db_path = '/media/shared/platform_sample/img50/test_db'
    # train_db_path = '/media/shared/iksula_ecommarce/POC_set_lmdb/train_db'
    # test_db_path = '/media/shared/iksula_ecommarce/POC_set_lmdb/test_db'
    params = {
        'input_shape': (256, 256, 3),
        'num_classes': 50,
        'train_model_name': 'vgg',
        'batch_size': 128,
        'train_lmdb_path': train_db_path,
        'test_lmdb_path': test_db_path,
        # 'weights_path': os.path.join(DATA_DIR, 'tf_vgg/weights-25'),
        'augmentation_params': {
            'reshape': (256, 256, 3),
            'random_crop': (224, 224, 3),
            'zoom': [0.8, 1.2],
            'hflip': True
        },
        'model_params': {
            'lr': 0.1,
            'loss': "categorical_crossentropy"
        }
    }
    with tf.device('/gpu:2'):
        nn = BerryFlow(**params)
        print 'Start training...'
        nn.train(100, output_dir=os.path.join(DATA_DIR, 'tf_vgg'),
                 analyse_grads=True)
        print 'Finished training.'
        pprint(nn.get_status())
        nn.close()


# Example: data augmentation visualize
def test_transformer():
    train_db_path = '/media/shared/platform_sample/img50/train_db'
    # train_db_path = '/home/arya_01/vision_training_server/jobs/asd_1462527100251/lmdb_data/train_db'
    # train_db_path = '/media/shared/platform_sample/img5/train_db'
    # train_db_path = '/media/shared/platform_sample/imagenet5/lmdb_data/train_db'
    # train_db_path = '/media/shared/platform_sample/iisc_data_old/lmdb_data/train_db'
    # train_db_path = '/media/shared/mnist/mnist_train_lmdb/'
    # train_db_path = '/media/shared/iksula_ecommarce/POC_set_lmdb/train_db'
    output_path = '/media/shared/platform_sample/augmentated'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    else:
        from shutil import rmtree
        rmtree(output_path)
        os.makedirs(output_path)

    transformer = Transformer(**{
        # 'reshape': (256, 256, 3),
        # 'random_crop': (224, 224, 3),
        # 'zoom': [1, 1.8],
        # 'hflip': True
    })

    train_env = lmdb.open(train_db_path, readonly=True)
    with train_env.begin() as txn:
        cursor = txn.cursor()
        Parallel(n_jobs=-1)(delayed(_parallel_lmdb_read)(
            transformer, output_path, key, value, i)
            for i, (key, value) in enumerate(cursor))
    train_env.close()


def _parse_data_from_string(value):
    datum = Datum()
    datum.ParseFromString(value)
    flat_x = np.fromstring(datum.data, dtype=np.uint8)
    x_ = flat_x.reshape(datum.channels, datum.height, datum.width)
    x = np.transpose(x_, axes=(1, 2, 0))
    y = datum.label
    return x, y


def _parallel_lmdb_read(transformer, output_path, key, value, i):
    try:
        x_, y_ = _parse_data_from_string(value)
        x, y = transformer.transform(x_, y_, 'train')
        xmin = x.min()
        xmax = x.max()
        x = float(x - xmin) / (xmax - xmin)
        imsave(os.path.join(output_path, '%s.jpg' % key), x)
    except Exception as e:
        print "[ERROR]:", e
        raise e
    if i % 20 == 0:
        print "Done:", i


def arguments(bool_help=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('-test', dest='func_test', action='store_const',
                        const=test_predict)
    parser.add_argument('-vgg', dest='func_vgg', action='store_const',
                        const=test_train_vgg)
    parser.add_argument('-alex', dest='func_alex', action='store_const',
                        const=test_train_alexnet)
    parser.add_argument('-le', dest='func_le', action='store_const',
                        const=test_train_lenet)
    parser.add_argument('-aug', dest='func_aug', action='store_const',
                        const=test_transformer)
    parser.add_argument('-act', dest='func_act', action='store_const',
                        const=test_activations)
    parser.add_argument('-pf', dest='func_pf', action='store_const',
                        const=test_train_protoflow)
    # parser.add_argument('-test2', dest='func_test2', action='store_const',
    #                     const=test_predict_2models)
    args = parser.parse_args()
    if bool_help:
        parser.print_help()
    return args

if __name__ == '__main__':
    args = arguments()
    try:
        flag = False
        for name, func in vars(args).items():
            if func is None:
                continue
            print '##############################'
            print name
            print '##############################'
            func()
            print '##############################'
            flag = True
        if not flag:
            arguments(bool_help=True)

    except KeyboardInterrupt:
        print "Interrupting..."
