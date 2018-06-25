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

from .basedataset import BaseDataset

class Cifar10(BaseDataset):

	def __init__(self, config):

		super(Cifar10, self).__init__(config)
		
		self._dataset_dir = 'D:\\Data\\Cifar10'
		if not os.path.exists(self._dataset_dir):
			self._dataset_dir = '/mnt/data01/dataset/Cifar10'
		if not os.path.exists(self._dataset_dir):
			self._dataset_dir = '/mnt/sh_flex_storage/zhicongy/dataset/Cifar10'

		self._dataset_dir = self.config.get('dataset_dir', self._dataset_dir)
		self.nb_classes = 10
 
		train_batch_list = [
			'data_batch_1',
			'data_batch_2',
			'data_batch_3',
			'data_batch_4',
			'data_batch_5',
		]
		test_batch_file = 'test_batch'

		train_data = []
		train_label = []
		for train_file in train_batch_list:
			image_data, image_label = self.read_data(train_file, self._dataset_dir)
			train_data.append(image_data)
			train_label.append(image_label)

		self.x_train = np.vstack(train_data).reshape([-1, 3, 32, 32]).transpose([0, 2, 3, 1]).astype(np.float32) / 255.0
		self.y_train = np.hstack(train_label)

		test_data, test_label = self.read_data(test_batch_file, self._dataset_dir)
		self.x_test = np.reshape(test_data, [-1, 3, 32, 32]).transpose([0, 2, 3, 1]).astype(np.float32) / 255.0
		self.y_test = test_label

	def read_data(self, filename, data_path):
		with open(os.path.join(data_path, filename), 'rb') as datafile:
			data_dict = pickle.load(datafile, encoding='bytes')
		image_data = np.array(data_dict[b'data'])
		image_label = np.array(data_dict[b'labels'])
		return image_data, image_label

