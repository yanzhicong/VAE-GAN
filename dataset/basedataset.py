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
        self.batch_size = self.config.get('batch_size', 128)

