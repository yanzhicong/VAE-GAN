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



class BaseDataset(object):
	""" Base dataset class

	defines the dataset class interface and implements some util function

	the util functions instruction:
		1. label transform: 
			to_categorical, from_categorical
		2. mask to colormap and colormap to mask:
			mask_colormap_decode, mask_colormap_encode
		3. image resize:
			flexible_scaling, random_scaling,
		4. image crop:
			crop_and_pad_image, random_crop_and_pad_image,
		5. output scalar rescale:
			scale_output, unscale_output
	"""

	def __init__(self, config):
		
		self.config = config
		self.shuffle_train = self.config.get('shuffle train', True)
		self.shuffle_val = self.config.get('shuffle val', False)
		self.shuffle_test = self.config.get('shuffle test', False)
		self.scalar_range = self.config.get('scalar range', [0.0, 1.0])


	#
	#	interface, please implement them in the derived class
	#
	def get_image_indices(self, phase, method):
		"""	return the image indices list 

		Arguments:
			phase : common input are "train", "val", "trainval", "test"
			method : common input are "supervised", "unsupervised"
		
		1. the phase "test" is always with method "unsupervised".
		2. 
		"""
		raise NotImplementedError

	def read_image_by_index(self, ind, phase, method):
		"""	Read image by its ind

		Arguments:
			phase : common input are "train", "val", "trainval", "test"
			method : common input are "supervised", "unsupervised"

		1. in "supervised" method, this function returns the pre-processed image and its label
		2. if the dataset provides multiple-instance interface, this function returns a bag of images and the bag's label
		3. in "unsupervised" method, this function returns the pre-processed image
		"""
		raise NotImplementedError


	def iter_train_images(self, method='supervised'):
		"""
		"""
		raise NotImplementedError
	def iter_val_images(self):
		raise NotImplementedError


	#
	#	util functions
	#
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
		""" from colored mask to 1-channel mask
			Inputs : 
				colored_mask : [h, w, c]
			output : 
				mask : [h, w]
		"""
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
		""" from colored mask to 1-channel mask
			Inputs : 
				colored_mask : [h, w, c]
			output : 
				mask : [h, w]
		"""
		mask_shape = mask.shape
		mask = mask.reshape([-1])
		colored_mask = np.ones([int(np.product(mask_shape)), 3,], dtype=np.uint8) * (np.array(default_color).reshape([1, 3]))
		for ind, color in enumerate(color_map):
			colored_mask[np.where(mask == ind)[0], :] = np.array(color)
		colored_mask = colored_mask.reshape(list(mask_shape) + [3, ])
		return colored_mask

	def flexible_scaling(self, img, min_h, min_w, mask=None):
		img_h = img.shape[0]
		img_w = img.shape[1]

		scale_h = min_h / img_h
		scale_w = min_w / img_w

		scale = np.maximum(scale_h, scale_w)

		h_new = int(img.shape[0] * scale)
		w_new = int(img.shape[1] * scale)

		if mask is not None:
			img = cv2.resize(img,(w_new, h_new))
			mask = cv2.resize(mask, (w_new, h_new), interpolation=cv2.INTER_NEAREST)
			return img, mask
		else:
			img = cv2.resize(img,(w_new, h_new))
			return img


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


	def crop_and_pad_image(self, img, bbox):
		""" crop and pad image
		"""
		def pad_img_to_fit_bbox(img, x0, x1, y0, y1):
			img = np.pad(img, (
								(	np.abs(np.minimum(0, y0)), 
								 	np.maximum(y1 - img.shape[0], 0)),
					   			(	np.abs(np.minimum(0, x0)), 
					   		 	 	np.maximum(x1 - img.shape[1], 0)), 
					   			(0,0)), mode="constant")
			_y0 = y0 + np.abs(np.minimum(0, y0))
			_y1 = y1 + np.abs(np.minimum(0, y0))
			_x0 = x0 + np.abs(np.minimum(0, x0))
			_x1 = x1 + np.abs(np.minimum(0, x0))
			return img, _x0, _x1, _y0, _y1
		def imcrop(img, bbox): 
			x0,y0,x1,y1 = bbox
			if x0 < 0 or y0 < 0 or x1 > img.shape[1] or y1 > img.shape[0]:
				img, x0, x1, y0, y1 = pad_img_to_fit_bbox(img, x0, x1, y0, y1)
			return img[y0:y1, x0:x1, :]
		return imcrop(img, bbox)


	def random_crop_and_pad_image(self, img, size, mask=None, center_range=[0.2, 0.8]):
		""" randomly crop and pad image to the given size
			Arguments : 
				img : array of shape(h, w, c)
				size : [crop_image_width, crop_image_height]

		"""
		h, w, c = img.shape
		crop_w, crop_h = size[0:2]



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

		combined_crop = self.crop_and_pad_image(combined, bbox)
		img_crop = combined_crop[:, :, :c]

		if mask is not None:
			mask_crop = combined_crop[:, :, c:]
			mask_crop = mask_crop.reshape(mask_crop_shape)
			return img_crop, mask_crop
		else:
			return img_crop


	def scale_output(self, data):
		""" input data is in range of [0.0, 1.0]
			this function rescale the data to the range of config parameters "scalar range"
		"""
		if self.scalar_range[0] == 0.0 and self.scalar_range[1] == 1.0:
			return data
		else:
			return data * (self.scalar_range[1] - self.scalar_range[0]) + self.scalar_range[0]

	def unscale_output(self, data):
		""" the reverse function of scale_output
		"""
		if self.scalar_range[0] == 0.0 and self.scalar_range[1] == 1.0:
			return data
		else:
			return (data - self.scalar_range[0]) / (self.scalar_range[1] - self.scalar_range[0]) 


