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

import matplotlib
matplotlib.use('Agg')


import matplotlib.pyplot as plt
import pickle

from .base_dataset import BaseDataset


class BaseImageListDataset(BaseDataset):

	def __init__(self, config):
		
		super(BaseImageListDataset, self).__init__(config)
		self.config = config
		self.batch_size = int(config.get('batch_size', 128))

		# please fill in the following field in the drived dataset class
		self._dataset_dir = None
		self.train_imagelist_fp = None      # the txt file which contains the list of images and labels
		self.val_imagelist_fp = None        # for example:
		self.test_imagelist_fp = None       #   train.txt:
											# line 1 : image_filepath,classname1,classname2
											# line 2 : image1.jpg,1,0
											# line 3 : image2.jpg,0,0
											# ...

	def read_dataset(self):
		raise NotImplementedError



	# def _get_labelled_image_indices(self, nb_images_per_class):
	# 	pickle_filepath = os.path.join(self.extra_file_path, 'labelled_image_indices_%d.pkl'%nb_images_per_class)
	# 	if os.path.exists(pickle_filepath):
	# 		return pickle.load(open(pickle_filepath, 'rb'))
	# 	else:
	# 		train_indices = []
	# 		for i in range(self.nb_classes):
	# 			indices = np.random.choice(np.where(self.y_train == i)[0], size=nb_images_per_class).tolist()
	# 			train_indices += indices
	# 		train_indices = np.array(train_indices)
	# 		pickle.dump(train_indices, open(pickle_filepath, 'wb'))
	# 		return train_indices

	# '''
	# 	method for direct access images
	# 	E.g.
	# 	for index, batch_x, batch_y in dataset.iter_train_images_supervised():
	# 		(training...)
	# '''
	# def iter_train_images_supervised(self):
	# 	index = np.arange(self.x_train_l.shape[0])
	# 	if self.shuffle_train:
	# 		np.random.shuffle(index)
	# 	for i in range(int(self.x_train_l.shape[0] / self.batch_size)):
	# 		batch_x = self.x_train_l[index[i*self.batch_size:(i+1)*self.batch_size], :]
	# 		batch_y = self.y_train_l[index[i*self.batch_size:(i+1)*self.batch_size]]
	# 		if 'output shape' in self.config:
	# 			batch_x = batch_x.reshape([self.batch_size,] + self.config['output shape'])
	# 		batch_y = self.to_categorical(batch_y, num_classes=self.nb_classes)
	# 		yield i, batch_x, batch_y

	# def iter_train_images_unsupervised(self):
	# 	index = np.arange(self.x_train_u.shape[0])
	# 	if self.shuffle_train:
	# 		np.random.shuffle(index)
	# 	for i in range(int(self.x_train_u.shape[0] / self.batch_size)):
	# 		batch_x = self.x_train_u[index[i*self.batch_size:(i+1)*self.batch_size], :]
	# 		if 'output shape' in self.config:
	# 			batch_x = batch_x.reshape([self.batch_size,] + self.config['output shape'])
	# 		yield i, batch_x

	# def iter_test_images(self):
	# 	index = np.arange(self.x_test.shape[0])
	# 	if self.shuffle_test:
	# 		np.random.shuffle(index)

	# 	for i in range(int(self.x_test.shape[0] / self.batch_size)):
	# 		batch_x = self.x_test[index[i*self.batch_size:(i+1)*self.batch_size], :]
	# 		batch_y = self.y_test[index[i*self.batch_size:(i+1)*self.batch_size]]

	# 		if 'output shape' in self.config:
	# 			batch_x = batch_x.reshape([self.batch_size,] + self.config['output shape'])
			
	# 		batch_y = self.to_categorical(batch_y, num_classes=self.nb_classes)
	# 		yield i, batch_x, batch_y
	# '''

	# '''
	# def get_image_indices(self, phase, method='supervised'):
	# 	'''

	# 	'''
	# 	if phase == 'train':
	# 		if method == 'supervised':
	# 			indices = np.array(range(self.x_train_l.shape[0]))
	# 		elif method == 'unsupervised' : 
	# 			indices = np.array(range(self.x_train_u.shape[0]))
	# 		else:
	# 			raise Exception("None method named " + str(method))
	# 		if self.shuffle_train:
	# 			np.random.shuffle(indices)
	# 		return indices

	# 	elif phase == 'val':
	# 		indices = np.array(range(self.x_test.shape[0]))
	# 		if self.shuffle_test:
	# 			np.random.shuffle(indices)
	# 		return indices

	# 	elif phase == 'test':
	# 		indices = np.array(range(self.x_test.shape[0]))
	# 		if self.shuffle_test:
	# 			np.random.shuffle(indices)
	# 		return indices

	# 	else:
	# 		raise Exception("None phase named " + str(phase))

	# def read_image_by_index(self, index, phase='train', method='supervised'):
	# 	assert(method in ['supervised', 'unsupervised'])
	# 	assert(phase in ['train', 'val', 'test'])

	# 	if method == 'supervised':
	# 		if phase == 'train':
	# 			label = np.zeros((self.nb_classes,))
	# 			label[self.y_train_l[index]] = 1.0
	# 			return self.x_train_l[index].reshape(self.output_shape), label
	# 		elif phase == 'val' or phase == 'test':
	# 			label = np.zeros((self.nb_classes, ))
	# 			label[self.y_test[index]] = 1.0
	# 			return self.x_test[index].reshape(self.output_shape), label

	# 	elif method == 'unsupervised':
	# 		if phase == 'train':
	# 			return self.x_train_u[index].reshape(self.output_shape)
	# 		elif phase == 'val' or phase == 'test':
	# 			return self.x_test[index].reshape(self.output_shape)



	# @property
	# def nb_labelled_images(self):
	# 	return self.x_train_l.shape[0]

	# @property
	# def nb_unlabelled_images(self):
	# 	return self.x_train_u.shape[0]

	# @property
	# def nb_test_images(self):
	# 	return self.x_test.shape[0]

