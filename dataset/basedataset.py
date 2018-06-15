import os
import sys
import time
import numpy as np

from abc import ABCMeta, abstractmethod
from keras.utils import to_categorical


class BaseDataset(object, metaclass=ABCMeta):

    def __init__(self, config):
        
        self.config = config

        self.shuffle_train = self.config.get('shuffle_train', True)
        self.shuffle_test = self.config.get('shuffle_test', False)
        self.batch_size = self.config.get('batch_size', 16)
        # self.output_shape = self.config.get()


    '''
        method for direct iterate image
    '''
    def iter_train_images_supervised(self):
        index = np.arange(self.x_train_l.shape[0])

        if self.shuffle_train:
            np.random.shuffle(index)

        for i in range(int(self.x_train_l.shape[0] / self.batch_size)):
            batch_x = self.x_train_l[index[i*self.batch_size:(i+1)*self.batch_size], :]
            batch_y = self.y_train_l[index[i*self.batch_size:(i+1)*self.batch_size]]

            if 'input_shape' in self.config:
                batch_x = batch_x.reshape([self.batch_size,] + self.config['input_shape'])
            batch_y = to_categorical(batch_y, num_classes=self.nb_classes)
            yield i, batch_x, batch_y


    def iter_train_images_unsupervised(self):
        index = np.arange(self.x_train_u.shape[0])

        if self.shuffle_train:
            np.random.shuffle(index)

        for i in range(int(self.x_train_u.shape[0] / self.batch_size)):
            batch_x = self.x_train_u[index[i*self.batch_size:(i+1)*self.batch_size], :]

            if 'input_shape' in self.config:
                batch_x = batch_x.reshape([self.batch_size,] + self.config['input_shape'])

            yield i, batch_x


    def iter_test_images(self):
        index = np.arange(self.x_test.shape[0])
        if self.shuffle_train:
            np.random.shuffle(index)

        for i in range(int(self.x_test.shape[0] / self.batch_size)):
            batch_x = self.x_test[index[i*self.batch_size:(i+1)*self.batch_size], :]
            batch_y = self.y_test[index[i*self.batch_size:(i+1)*self.batch_size]]

            if 'input_shape' in self.config:
                batch_x = batch_x.reshape([self.batch_size,] + self.config['input_shape'])
            
            batch_y = to_categorical(batch_y, num_classes=self.nb_classes)
            yield i, batch_x, batch_y


    def get_image_indices(self, phase, method):
        '''
        '''
        if phase == 'train':
            if method == 'supervised':
                indices = np.array(range(self.x_train_l.shape[0]))
            elif method == 'unsupervised' : 
                indices = np.array(range(self.x_train_u.shape[0]))
            else:
                raise Exception("None method named " + method)
            
            if self.shuffle_train:
                np.random.shuffle(indices)
        
            return indices

        elif phase == 'test':
            indices = np.array(range(self.x_test.shape[0]))
            if self.shuffle_test:
                np.random.shuffle(indices)
            return indices

        else:
            raise Exception("None phase named " + phase)

    def read_image_by_index_supervised(self, index):
        label = np.zeros((self.nb_classes,))
        label[self.y_train_l[index]] = 1.0
        return self.x_train_l[index].reshape(self.input_shape), label

    def read_image_by_index_unsupervised(self, index):
        return self.x_train_u[index].reshape(self.input_shape)
