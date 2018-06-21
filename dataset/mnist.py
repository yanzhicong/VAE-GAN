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
import struct
import gzip
import numpy as np

from .base_dataset import BaseDataset
from .base_simple_dataset import BaseSimpleDataset



class MNIST(BaseSimpleDataset):

	def __init__(self, config):
		
		super(MNIST, self).__init__(config)
		self.config = config

		self._dataset_dir = 'D:/Data/MNIST'
		if not os.path.exists(self._dataset_dir):
			self._dataset_dir = 'D:/dataset/MNIST'
		if not os.path.exists(self._dataset_dir):
			self._dataset_dir = 'C:/Data/MNIST'
		if not os.path.exists(self._dataset_dir):
			self._dataset_dir = 'G:/dataset/MNIST'
		if not os.path.exists(self._dataset_dir):
			self._dataset_dir = '/mnt/data01/dataset/MNIST'
		if not os.path.exists(self._dataset_dir):
			self._dataset_dir = '/mnt/sh_flex_storage/zhicongy/tmpdataset/MNIST'
		if not os.path.exists(self._dataset_dir):
			self._dataset_dir = config.get('dataset dir', '')
		if not os.path.exists(self._dataset_dir):
			raise Exception("MNIST : the dataset dir " + self._dataset_dir + " is not exist")

		self.name = 'mnist'
		self.output_shape = config.get('output shape', [28, 28, 1])
		self.nb_classes = 10

		self.y_train, self.x_train = self._read_data(
			os.path.join(self._dataset_dir, 'train-labels-idx1-ubyte.gz'),
			os.path.join(self._dataset_dir, 'train-images-idx3-ubyte.gz')
		)

		self.y_test, self.x_test = self._read_data(
			os.path.join(self._dataset_dir, 't10k-labels-idx1-ubyte.gz'),
			os.path.join(self._dataset_dir, 't10k-images-idx3-ubyte.gz')
		)

		self.x_train = self.x_train.astype(np.float32) / 255.0
		self.x_test = self.x_test.astype(np.float32) / 255.0

		self.build_dataset()

	def _read_data(self, label_url, image_url):
		with gzip.open(label_url) as flbl:
			magic, num = struct.unpack(">II",flbl.read(8))
			label = np.fromstring(flbl.read(),dtype=np.int8)
		with gzip.open(image_url,'rb') as fimg:
			magic, num, rows, cols = struct.unpack(">IIII",fimg.read(16))
			image = np.fromstring(fimg.read(),dtype=np.uint8).reshape(len(label),rows,cols)
		return (label, image)


