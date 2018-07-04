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

	'''
		method for direct iterate image
		E.g.
		for index, batch_x, batch_y in dataset.iter_train_images_supervised():
			(training...)
	'''
	def iter_train_images_supervised(self):
		index = np.arange(self.x_train_l.shape[0])

		if self.shuffle_train:
			np.random.shuffle(index)

		for i in range(int(self.x_train_l.shape[0] / self.batch_size)):
			batch_x = self.x_train_l[index[i*self.batch_size:(i+1)*self.batch_size], :]
			batch_y = self.y_train_l[index[i*self.batch_size:(i+1)*self.batch_size]]

			if 'output shape' in self.config:
				batch_x = batch_x.reshape([self.batch_size,] + self.config['output shape'])
			batch_y = self.to_categorical(batch_y, num_classes=self.nb_classes)
			yield i, batch_x, batch_y


	def iter_train_images_unsupervised(self):
		index = np.arange(self.x_train_u.shape[0])

		if self.shuffle_train:
			np.random.shuffle(index)

		for i in range(int(self.x_train_u.shape[0] / self.batch_size)):
			batch_x = self.x_train_u[index[i*self.batch_size:(i+1)*self.batch_size], :]

			if 'output shape' in self.config:
				batch_x = batch_x.reshape([self.batch_size,] + self.config['output shape'])

			yield i, batch_x


	def iter_test_images(self):
		index = np.arange(self.x_test.shape[0])
		if self.shuffle_train:
			np.random.shuffle(index)

		for i in range(int(self.x_test.shape[0] / self.batch_size)):
			batch_x = self.x_test[index[i*self.batch_size:(i+1)*self.batch_size], :]
			batch_y = self.y_test[index[i*self.batch_size:(i+1)*self.batch_size]]

			if 'output shape' in self.config:
				batch_x = batch_x.reshape([self.batch_size,] + self.config['output shape'])
			
			batch_y = self.to_categorical(batch_y, num_classes=self.nb_classes)
			yield i, batch_x, batch_y


	'''

	'''
	def get_image_indices(self, phase, method='supervised'):
		'''
		'''
		if phase == 'train':
			if method == 'supervised':
				indices = np.array(range(self.x_train_l.shape[0]))
			elif method == 'unsupervised' : 
				indices = np.array(range(self.x_train_u.shape[0]))
			else:
				raise Exception("None method named " + str(method))
			
			if self.shuffle_train:
				np.random.shuffle(indices)
		
			return indices

		elif phase == 'test':
			indices = np.array(range(self.x_test.shape[0]))
			if self.shuffle_test:
				np.random.shuffle(indices)
			return indices

		else:
			raise Exception("None phase named " + str(phase))

	def read_image_by_index_supervised(self, index):
		label = np.zeros((self.nb_classes,))
		label[self.y_train_l[index]] = 1.0
		return self.x_train_l[index].reshape(self.output_shape), label

	def read_image_by_index_unsupervised(self, index):
		return self.x_train_u[index].reshape(self.output_shape)

	def read_test_image_by_index(self, index):
		label = np.zeros((self.nb_classes,))
		label[self.y_test[index]] = 1.0
		return self.x_test[index].reshape(self.output_shape), label


	@property
	def nb_labelled_images(self):
		return self.x_train_l.shape[0]

	@property
	def nb_unlabelled_images(self):
		return self.x_train_u.shape[0]

	@property
	def nb_test_images(self):
		return self.x_test.shape[0]


	def to_categorical(self, y, num_classes=None):
		"""
			Copyed from keras
			Converts a class vector (integers) to binary class matrix.

		E.g. for use with categorical_crossentropy.

		# Arguments
			y: class vector to be converted into a matrix
				(integers from 0 to num_classes).
			num_classes: total number of classes.

		# Returns
			A binary matrix representation of the input. The classes axis
			is placed last.
		"""
		y = np.array(y, dtype='int')
		input_shape = y.shape
		if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
			input_shape = tuple(input_shape[:-1])
		y = y.ravel()
		if not num_classes:
			num_classes = np.max(y) + 1
		n = y.shape[0]
		categorical = np.zeros((n, num_classes), dtype=np.float32)
		categorical[np.arange(n), y] = 1
		output_shape = input_shape + (num_classes,)
		categorical = np.reshape(categorical, output_shape)
		return categorical

