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
            self._dataset_dir = '/mnt/data01/dataset/MNIST'
        if not os.path.exists(self._dataset_dir):
            self._dataset_dir = '/mnt/sh_flex_storage/zhicongy/dataset/MNIST'

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

        # if 'input_shape' in config:
        #     self.input_shape = config['input_shape']
        # else:
        #     self.input_shape = [28, 28, 1]

        if 'batch_size' in config:
            self.batch_size = int(config['batch_size'])
        else:
            self.batch_size = int(128)

        self.y_train, self.x_train = self.read_data(
            os.path.join(self._dataset_dir, 'train-labels-idx1-ubyte.gz'),
            os.path.join(self._dataset_dir, 'train-images-idx3-ubyte.gz')
        )

        self.y_test, self.x_test = self.read_data(
            os.path.join(self._dataset_dir, 't10k-labels-idx1-ubyte.gz'),
            os.path.join(self._dataset_dir, 't10k-images-idx3-ubyte.gz')
        )

        self.x_train = self.x_train.astype(np.float32) / 255.0
        self.x_test = self.x_test.astype(np.float32) / 255.0



    def iter_trainimages_supervised(self):

        # index = np.range()

        pass



    def iter_images(self):

        index = np.arange(self.x_train.shape[0])

        if self.shuffle_train:
            np.random.shuffle(index)

        for i in range(int(self.x_train.shape[0] / self.batch_size)):

            batch = self.x_train[index[i*self.batch_size:(i+1)*self.batch_size], :]


            if 'input_shape' in self.config:
                batch = batch.reshape([self.batch_size,] + self.config['input_shape'])
            
            # print(batch.max())
            # print(batch.min())
            

            yield i, batch

        # pass

    
    def read_data(self, label_url, image_url):
        with gzip.open(label_url) as flbl:
            magic, num = struct.unpack(">II",flbl.read(8))
            label = np.fromstring(flbl.read(),dtype=np.int8)
        with gzip.open(image_url,'rb') as fimg:
            magic, num, rows, cols = struct.unpack(">IIII",fimg.read(16))
            image = np.fromstring(fimg.read(),dtype=np.uint8).reshape(len(label),rows,cols)
        return (label, image)









