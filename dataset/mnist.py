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
import matplotlib.pyplot as plt
import pickle

from keras.utils import to_categorical

from .basedataset import BaseDataset



class MNIST(BaseDataset):

	def __init__(self, config):
		
		super(MNIST, self).__init__(config)

		self._dataset_dir = 'D:\Data\MNIST'
		if not os.path.exists(self._dataset_dir):
			self._dataset_dir = '/mnt/data01/dataset/MNIST'
		if not os.path.exists(self._dataset_dir):
			self._dataset_dir = '/mnt/sh_flex_storage/zhicongy/dataset/MNIST'

		self.config = config
		self.name = 'mnist'

		self._dataset_dir = config.get('dataset_dir', self._dataset_dir)
		self.input_shape = config.get('input_shape', [28, 28, 1])
		self.batch_size = int(config.get('batch_size', 128))
		self.nb_classes = 10

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

		if self.config.get('semi-supervised', False):
			self.extra_file_path = './dataset/extra_files'
			if not os.path.exists(self.extra_file_path):
				os.mkdir(self.extra_file_path)
			self.extra_file_path = os.path.join(self.extra_file_path, self.name)
			if not os.path.exists(self.extra_file_path):
				os.mkdir(self.extra_file_path)

			self.nb_labelled_images_per_classes = self.config.get('nb_labelled_images_per_classes', 100)
			self.labelled_image_indices = self.get_labelled_image_indices(self.nb_labelled_images_per_classes)

			self.x_train_u = self.x_train
			
			self.x_train_l = self.x_train[self.labelled_image_indices]
			self.y_train_l = self.y_train[self.labelled_image_indices]
		else:
			self.x_train_l = self.x_train
			self.y_train_l = self.y_train


	def get_labelled_image_indices(self, nb_images_per_classes):
		pickle_filepath = os.path.join(self.extra_file_path, 'labelled_image_indices_%d.pkl'%nb_images_per_classes)
		if os.path.exists(pickle_filepath):
			return pickle.load(open(pickle_filepath, 'rb'))
		else:
			train_indices = []
			for i in range(self.nb_classes):

				print(self.y_train.shape)
				print(np.where(self.y_train == i)[0].shape)
				indices = np.random.choice(np.where(self.y_train == i)[0], size=nb_images_per_classes).tolist()
				print(len(indices))
				train_indices += indices
			train_indices = np.array(train_indices)
			pickle.dump(train_indices, open(pickle_filepath, 'wb'))
			return train_indices



	
	def read_data(self, label_url, image_url):
		with gzip.open(label_url) as flbl:
			magic, num = struct.unpack(">II",flbl.read(8))
			label = np.fromstring(flbl.read(),dtype=np.int8)
		with gzip.open(image_url,'rb') as fimg:
			magic, num, rows, cols = struct.unpack(">IIII",fimg.read(16))
			image = np.fromstring(fimg.read(),dtype=np.uint8).reshape(len(label),rows,cols)
		return (label, image)









