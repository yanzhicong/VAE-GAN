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
		if not os.path.exists(self._dataset_dir):
			self._dataset_dir = self.config.get('dataset_dir', self._dataset_dir)


		self.name = 'cifar10'
		self.output_shape = config.get('output_shape', [32, 32, 3])
		self.batch_size = int(config.get('batch_size', 128))
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


		# whether perpare semi-supervised datset or not
		if self.config.get('semi-supervised', False):

			self.extra_file_path = os.path.join('./dataset/extra_files', self.name)
			if not os.path.exists(self.extra_file_path):
				os.makedirs(self.extra_file_path)

			# if semisupervised training, prepare labelled train set indices,
			self.nb_labelled_images_per_class = self.config.get('nb_labelled_images_per_class', 100)
			self.labelled_image_indices = self._get_labelled_image_indices(self.nb_labelled_images_per_class)

			# unlabelled train set
			self.x_train_u = self.x_train
			
			# labelled train set
			self.x_train_l = self.x_train[self.labelled_image_indices]
			self.y_train_l = self.y_train[self.labelled_image_indices]
		else:
			# else training in supervised manner
			self.x_train_l = self.x_train
			self.y_train_l = self.y_train

			self.x_train_u = self.x_train


	def _get_labelled_image_indices(self, nb_images_per_class):
		pickle_filepath = os.path.join(self.extra_file_path, 'labelled_image_indices_%d.pkl'%nb_images_per_class)
		if os.path.exists(pickle_filepath):
			return pickle.load(open(pickle_filepath, 'rb'))
		else:
			train_indices = []
			for i in range(self.nb_classes):
				indices = np.random.choice(np.where(self.y_train == i)[0], size=nb_images_per_class).tolist()
				train_indices += indices
			train_indices = np.array(train_indices)
			pickle.dump(train_indices, open(pickle_filepath, 'wb'))
			return train_indices
	

	def read_data(self, filename, data_path):
		with open(os.path.join(data_path, filename), 'rb') as datafile:
			data_dict = pickle.load(datafile, encoding='bytes')
		image_data = np.array(data_dict[b'data'])
		image_label = np.array(data_dict[b'labels'])
		return image_data, image_label

