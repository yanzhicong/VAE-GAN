import os
import struct
import gzip
import numpy as np
import matplotlib.pyplot as plt

from keras.utils import to_categorical





class MNIST(object):

    def __init__(self, config):
        
        self._dataset_dir = 'D:\Data\MNIST'
        if not os.path.exists(self._dataset_dir):
            self._dataset_dir = '\mnt\data01\dataset\MNIST'

        self.config = config

        if 'dataset_dir' in config:
            self._dataset_dir = config['dataset_dir']

        if 'shuffle_train' in config:
            self.shuffle_train = config['shuffle_train']
        else:
            self.shuffle_train = True

        if 'shuffle_test' in config:
            self.shuffle_test = config['shuffle_test']
        else:
            self.shuffle_test = False

        if 'input_shape' in config:
            self.input_shape = config['input_shape']
        else:
            self.input_shape = [28, 28, 1]

        if 'batch_size' in config:
            self.batch_size = int(config['batch_size'])
        else:
            self.batch_size = 128

        self.y_train, self.x_train = read_data(
            os.path.join(self._dataset_dir, 'train-labels-idx1-ubyte.gz'),
            os.path.join(self._dataset_dir, 'train-images-idx3-ubyte.gz')
        )

        self.y_test, self.x_test = read_data(
            os.path.join(self._dataset_dir, 't10k-labels-idx1-ubyte.gz'),
            os.path.join(self._dataset_dir, 't10k-images-idx3-ubyte.gz')
        )

        
        # if 'supervised_size' in config:
        #     self.supervised_size = config['supervised_size']
        # else:
        #     self.supervised_size = self.x_train.shape[0]
        # self.


    def iter_trainimages_supervised(self):

        index = np.range()

        pass


    
    def read_data(self.label_url,image_url):
        with gzip.open(label_url) as flbl:
            magic, num = struct.unpack(">II",flbl.read(8))
            label = np.fromstring(flbl.read(),dtype=np.int8)
        with gzip.open(image_url,'rb') as fimg:
            magic, num, rows, cols = struct.unpack(">IIII",fimg.read(16))
            image = np.fromstring(fimg.read(),dtype=np.uint8).reshape(len(label),rows,cols)
        return (label, image)









