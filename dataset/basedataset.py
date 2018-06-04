import os
import sys
import time
import numpy as np

from abc import ABCMeta, abstractmethod

class BaseDataset(object, metaclass=ABCMeta):

    def __init__(self, config):
        
        self.config = config

        self.shuffle_train = self.config.get('shuffle_train', True)
        self.shuffle_test = self.config.get('shuffle_test', False)
        self.batch_size = self.config.get('batch_size', 16)


    # def iter_train_images(self):
    #     index = np.arange(self.x_train.shape[0])

    #     if self.shuffle_train:
    #         np.random.shuffle(index)

    #     for i in range(int(self.x_train.shape[0] / self.batch_size)):
    #         batch_x = self.x_train[index[i*self.batch_size:(i+1)*self.batch_size], :]
    #         batch_y = self.y_train[index[i*self.batch_size:(i+1)*self.batch_size]]

    #         if 'input_shape' in self.config:
    #             batch_x = batch_x.reshape([self.batch_size,] + self.config['input_shape'])
            
    #         # print(batch.max())
    #         # print(batch.min())
            
    #         yield i, batch_x, batch_y

    def iter_test_images(self):

        index = np.arange(self.x_test.shape[0])

        if self.shuffle_train:
            np.random.shuffle(index)

        for i in range(int(self.x_test.shape[0] / self.batch_size)):
            batch_x = self.x_test[index[i*self.batch_size:(i+1)*self.batch_size], :]
            batch_y = self.y_test[index[i*self.batch_size:(i+1)*self.batch_size]]

            if 'input_shape' in self.config:
                batch_x = batch_x.reshape([self.batch_size,] + self.config['input_shape'])
            
            yield i, batch_x, batch_y
