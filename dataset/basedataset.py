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
# import time
import numpy as np
import cv2

from abc import ABCMeta, abstractmethod


class BaseDataset(object, metaclass=ABCMeta):

	def __init__(self, config):
		
		self.config = config

		self.shuffle_train = self.config.get('shuffle train', True)
		self.shuffle_test = self.config.get('shuffle test', False)
		self.batch_size = self.config.get('batch_size', 16)

	'''
		method for direct access images
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


	'''
		util functions
	'''
	def to_categorical(self, y, num_classes):
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
		if isinstance(y, int):
			ret = np.zeros([num_classes,], dtype=np.float32)
			ret[y] = 1.0
			return ret
		elif isinstance(y, np.ndarray):
			input_shape = y.shape
			y = y.ravel()
			n = y.shape[0]
			ret = np.zeros((n, num_classes), dtype=np.float32)
			indices = np.where(y >= 0)[0]
			ret[np.arange(n)[indices], y[indices]] = 1.0
			ret = ret.reshape(list(input_shape) + [num_classes,])
			return ret


	def from_categorical(self, cat):
		"""
		"""
		return np.argmax(cat, axis=-1)


	def mask_colormap_encode(self, colored_mask, color_map, default_value=-1):
		'''
			from colored mask to 1-channel mask
			Inputs : 
				colored_mask : [h, w, c]
			output : 
				mask : [h, w]
		'''
		if len(colored_mask.shape) != 3 or colored_mask.shape[2] != 3:
			raise ValueError('Unsupported color mask shape : ', colored_mask.shape)

		h, w, c = colored_mask.shape
		out_c = len(color_map)
		colored_mask = colored_mask.reshape([h*w, c])

		mask = np.ones((h * w,), dtype=np.uint8) * default_value

		def color_match(mask, color):
			return np.logical_and(
					np.logical_and(
						mask[:, 0] == color[0],
						mask[:, 1] == color[1],
					),
						mask[:, 2] == color[2]
				)

		for ind, color in enumerate(color_map):
			mask[np.where(color_match(colored_mask, color))[0]] = ind
		mask = mask.reshape([h, w])
		return mask


	def mask_colormap_decode(self, mask, color_map, default_color=[224, 224, 192]):
		'''
			from colored mask to 1-channel mask
			Inputs : 
				colored_mask : [h, w, c]
			output : 
				mask : [h, w]
		'''
		mask_shape = mask.shape
		mask = mask.reshape([-1])
		colored_mask = np.ones([int(np.product(mask_shape)), 3,], dtype=np.uint8) * (np.array(default_color).reshape([1, 3]))
		for ind, color in enumerate(color_map):
			colored_mask[np.where(mask == ind)[0], :] = np.array(color)
		colored_mask = colored_mask.reshape(list(mask_shape) + [3, ])
		return colored_mask


	def random_scaling(self, img, minval=0.5, maxval=1.5, mask=None):
		scale = np.random.uniform(minval, maxval)
		h = int(img.shape[0])
		w = int(img.shape[1])
		c = int(img.shape[2])
		h_new = int(img.shape[0] * scale)
		w_new = int(img.shape[1] * scale)

		if mask is not None:
			img = cv2.resize(img,(w_new, h_new))
			mask = cv2.resize(mask, (w_new, h_new), interpolation=cv2.INTER_NEAREST)
			return img, mask
		else:
			img = cv2.resize(img,(w_new, h_new))
			return img


	def random_mirroring(self, img, mask=None):
		eps = np.random.uniform(0.0, 1.0)
		if eps < 0.5:
			img = img[:, ::-1]
			if mask is not None:
				mask = mask[:, ::-1]

		eps = np.random.uniform(0.0, 1.0)
		if eps < 0.5:
			img = img[::-1, :]
			if mask is not None:
				mask = mask[::-1, :]
		if mask is not None:
			return img, mask
		else:
			return img


	def random_crop_and_pad(self, img, size, mask=None, center_range=[0.2, 0.8]):
		'''
			randomly crop and pad image to the given size
			Arguments : 
				img : array of shape(h, w, c)
				size : [crop_image_width, crop_image_height]

		'''
		h, w, c = img.shape
		crop_w, crop_h = size[0:2]

		def pad_img_to_fit_bbox(img, x1, x2, y1, y2):
			img = np.pad(img, (
								(	np.abs(np.minimum(0, y1)), 
								 	np.maximum(y2 - img.shape[0], 0)),
					   			(	np.abs(np.minimum(0, x1)), 
					   		 	 	np.maximum(x2 - img.shape[1], 0)), 
					   			(0,0)), mode="constant")
			_y1 = y1 + np.abs(np.minimum(0, y1))
			_y2 = y2 + np.abs(np.minimum(0, y1))
			_x1 = x1 + np.abs(np.minimum(0, x1))
			_x2 = x2 + np.abs(np.minimum(0, x1))
			return img, _x1, _x2, _y1, _y2

		def imcrop(img, bbox): 
			x1,y1,x2,y2 = bbox
			if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
				img, x1, x2, y1, y2 = pad_img_to_fit_bbox(img, x1, x2, y1, y2)
			return img[y1:y2, x1:x2, :]

		if mask is not None and (mask.shape[0] != h or mask.shape[1] != w):
			raise ValueError('mask shape error : ', mask.shape)

		if mask is not None:
			mask_crop_shape = list(mask.shape)
			mask_crop_shape[0] = size[0]
			mask_crop_shape[1] = size[1]
			if len(mask.shape) == 2:
				mask = mask.reshape(list(mask.shape) + [1,])
			combined = np.concatenate([img, mask], axis=-1)
		else:
			combined = img

		center_h = int(np.random.uniform(center_range[0], center_range[1]) * h)
		center_w = int(np.random.uniform(center_range[0], center_range[1]) * w)

		x1 = int(center_w - crop_w / 2.0)
		y1 = int(center_h - crop_h / 2.0)
		x2 = int(x1 + crop_w)
		y2 = int(y1 + crop_h)
		bbox = (x1, y1, x2, y2)

		combined_crop = imcrop(combined, bbox)
		img_crop = combined_crop[:, :, :c]

		if mask is not None:
			mask_crop = combined_crop[:, :, c:]
			mask_crop = mask_crop.reshape(mask_crop_shape)
			return img_crop, mask_crop
		else:
			return img_crop

