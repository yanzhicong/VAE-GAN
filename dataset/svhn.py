# -*- coding: utf-8 -*-
# MIT License
# 
# Copyright (c) 2018 ZhicongYan
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================


import os
import numpy as np
import pickle

from scipy.io import loadmat

from .base_dataset import BaseDataset
from .base_simple_dataset import BaseSimpleDataset

class SVHN(BaseSimpleDataset):

	def __init__(self, config):
		super(SVHN, self).__init__(config)

		self._dataset_dir = "C:\\Data\\SVHN"
		if not os.path.exists(self._dataset_dir):
			self._dataset_dir = self.config.get("dataset dir", "")
		if not os.path.exists(self._dataset_dir):
			raise Exception("SVHN : the dataset dir is not exist")

		self.name = "SVHN"
		self.train_mat_fp = os.path.join(self._dataset_dir, "train_32x32.mat")
		self.test_mat_fp = os.path.join(self._dataset_dir, "test_32x32.mat")

		train_data = loadmat(self.train_mat_fp)
		test_data = loadmat(self.test_mat_fp)

		self.x_train = np.array(train_data["X"]).transpose((3, 0, 1, 2)).astype(np.float32) / 255.0
		self.y_train = np.array(train_data["y"]).reshape([-1,])

		self.x_test = np.array(test_data["X"]).transpose((3, 0, 1, 2)).astype(np.float32) / 255.0
		self.y_test = np.array(test_data["y"]).reshape([-1,])

		indices = np.where(self.y_train == 10)[0]
		self.y_train[indices] = 0
		indices = np.where(self.y_test == 10)[0]
		self.y_test[indices] = 0

		self.output_shape = config.get('output shape', [32, 32, 3])
		self.nb_classes = 10
 
 
		self.build_dataset()

