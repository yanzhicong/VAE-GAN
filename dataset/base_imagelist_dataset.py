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
from skimage import io
import cv2

from .base_dataset import BaseDataset


class BaseImageListDataset(BaseDataset):
	""" The base dataset class for a number of images and labels.
	if you have a number of images and labels, you can prepare a imagelist.txt file.
	this dataset class is just for classification

	Optional params in @params.config:
		'flexible scaling' : if set to True, the image will be resize to just fit the output shape, while keeping w/h ratio the same
		'random scaling' : if set to True, the image will be randomly resized after flexible scaling
		'random mirroring' : 
		'random cropping' : 
		''
	"""

	def __init__(self, config):
		
		super(BaseImageListDataset, self).__init__(config)
		self.config = config

		assert('output shape' in config)

		self.is_flexible_scaling = self.config.get('flexible scaling', True)
		self.is_random_scaling = self.config.get('random scaling', True)
		self.is_random_mirroring = self.config.get('random mirroring', True)
		self.is_random_cropping = self.config.get('random cropping', True)
		self.scaling_range = self.config.get('scaling range', [1.0, 1.5])
		self.crop_range = self.config.get('crop range', [0.4, 0.6])
		self.crop_range_hor = self.config.get('horizontal crop range', self.crop_range)
		self.crop_range_ver = self.config.get('vertical crop range', self.crop_range)

		self.output_shape = self.config['output shape']
		self.output_size = self.output_shape[0:2]
		self.output_h = self.output_shape[0]
		self.output_w = self.output_shape[1]
		self.output_c = self.output_shape[2]

		# self.multiple_classes = self.config.get('multiple classes', False)

		self.show_warning = self.config.get('show warning', False)
		assert(self.output_c in [1, 3])

		# please fill in the following field in the drived dataset class
		self._dataset_dir = None
		self.train_imagelist_fp = None      # the txt file which contains the list of images and labels
		self.val_imagelist_fp = None        # for example:(if self.multiple_categories)
		self.test_imagelist_fp = None       #   train.txt:
											# line 1 : image_filepath,classname1,classname2
											# line 2 : image1.jpg,1,0
											# line 3 : image2.jpg,0,0
											# 
											# for example:(if not self.multiple_categories)
										    #   train.txt:
											# line 1 : image_filepath,classname1,classname2,...
											# line 2 : image1.jpg,1
											# line 3 : image2.jpg,3  ->  the class_index
											# line 4 : image3.jpg,0  ->  the class_index
		self.multiple_categories = False


	def build_dataset(self):
		assert(self.train_imagelist_fp != None)
		assert(self.val_imagelist_fp != None)

		self.train_images, self.train_labels = self.read_imagelist(self.train_imagelist_fp)
		self.val_images, self.val_labels = self.read_imagelist(self.val_imagelist_fp)


		print('nb train images : ', len(self.train_images))
		print('nb val images : ', len(self.val_images))
		if self.test_imagelist_fp != None:
			self.test_images = self.read_imagelist(self.test_imagelist_fp, has_label=False)
			print('nb test_images : ', len(self.test_images))



	
	def read_imagelist(self, filename, has_label=True):
		image_list = []
		label_list = []
		class_head_list = []

		with open(filename, 'r', encoding='utf-8') as infile:
			if self.multiple_categories:
				for ind, line in enumerate(infile):
					if ind == 0:
						head_split = line[:-1].split(',')
						class_head_list = head_split[1:],
						# self.nb_classes = len(class_head_list)
					else:
						split = line[:-1].split(',')
						image_fp = split[0]
						image_list.append(image_fp)

						if has_label:
							label = np.array([1 if int(i) else 0 for i in split[1:]])
							label_list.append(label)
			else:
				for ind, line in enumerate(infile):
					if ind == 0:
						head_split = line[:-1].split(',')
						class_head_list = head_split[1:]
					else:
						split = line[:-1].split(',')
						image_fp = split[0]
						image_list.append(image_fp)

						if has_label:
							label = np.array([int(i) for i in split[1:]])
							label_list.append(label)
				# self.nb_classes = int(np.max(label_list) + 1)

		if has_label:
			return image_list, np.array(label_list)
		else:
			return image_list

	def get_image_indices(self, phase='train', method='supervised'):
		assert(phase in ['train', 'test', 'val', 'trainval'])

		if phase == 'train':
			indices = np.arange(len(self.train_images))
			if self.shuffle_train:
				np.random.shuffle(indices)
			return indices
		elif phase == 'val':
			indices = np.arange(len(self.val_images))
			if self.shuffle_val:
				np.random.shuffle(indices)
			return indices
		elif phase == 'test':
			indices = np.arange(len(self.test_images))
			if self.shuffle_test:
				np.random.shuffle(indices)
			return indices
		elif phase == 'trainval':
			indices = np.arange(len(self.train_images) + len(self.val_images))
			if self.shuffle_train:
				np.random.shuffle(indices)
			return indices

	
	def _get_image_path_and_label(self, ind, phase='train', method='supervised'):
		# get image path and label
		if phase == 'train':
			image_fp = os.path.join(self._dataset_dir, self.train_images[ind])
			if method == 'supervised':
				image_label = self.train_labels[ind]
		elif phase == 'val':
			image_fp = os.path.join(self._dataset_dir, self.val_images[ind])
			if method == 'supervised':
				image_label = self.val_labels[ind]
		elif phase == 'trainval':
			if ind < len(self.train_images):
				image_fp = os.path.join(self._dataset_dir, self.train_images[ind])
				if method == 'supervised':
					image_label = self.train_labels[ind]
			else:
				image_fp = os.path.join(self._dataset_dir, self.val_images[ind - len(self.train_images)])
				if method == 'supervised':
					image_label = self.val_labels[ind - len(self.train_images)]
		elif phase == 'test':
			assert method == 'unsupervised'
			assert self.test_images != None
			image_fp = os.path.join(self._dataset_dir, self.test_images[ind])

		if method=='supervised':
			return image_fp, image_label  
		else:
			return image_fp


	def _image_correct(self, img, image_fp):
		"""	correct the image shape to fixed shape [height, width, channel]
		
		1. the argument image_fp is just for debugging.
		2. for some image file has multiple images and with shape of [num, height, width, channel],
		this function will return the first image and discard others.

		"""
		if img is None:
			if self.show_warning:
				print('Warning : read image ' + image_fp + ' failed!')
			return None

		if img.ndim == 4:
			img = img[0]

		if img.ndim != 2 and img.ndim != 3:
			if self.show_warning:
				print('Warning : wrong image shape ' + image_fp + ' : ' + str(img.shape))
			return None

		if self.output_c == 3:
			if img.ndim == 2:   						# in case of single channel image
				img = cv2.merge([img, img, img])
			elif img.ndim == 3 and img.shape[2] == 1:
				img = cv2.merge([img[:,:,0], img[:,:,0], img[:,:,0]])
			elif img.ndim == 3 and img.shape[2] == 4:
				img = img[:, :, 0:3]

		if img.ndim != 3 or img.shape[2] != 3:
			if self.show_warning:
				print('Warning : wrong image shape ' + image_fp + ' : ' + str(img.shape))
			return None

		return img


	def read_image_by_index(self, ind, phase='train', method='supervised'):
		assert(phase in ['train', 'test', 'val', 'trainval'])
		assert(method in ['supervised', 'unsupervised'])

		if method == 'supervised':
			image_fp, image_label = self._get_image_path_and_label(ind, phase, method)
		else:
			image_fp = self._get_image_path_and_label(ind, phase, method)
		
		try:
			img = io.imread(image_fp)
			img = self._image_correct(img, image_fp)
		except Exception as e:
			if self.show_warning:
				print('Warning : read image error : ' + str(e))
			return None, None if method == 'supervised' else None

		if img is None:
			return None, None if method == 'supervised' else None

		# preeprocess image and label
		if phase in ['train', 'trainval']:
			if self.is_flexible_scaling:
				img = self.flexible_scaling(img, min_h=self.output_h, min_w=self.output_w)
			if self.is_random_scaling:
				img = self.random_scaling(img, minval=self.scaling_range[0], maxval=self.scaling_range[1])
			if self.is_random_mirroring:
				img = self.random_mirroring(img)

		elif phase in ['val']:
			if self.is_flexible_scaling:
				img = self.flexible_scaling(img, min_h=self.output_h, min_w=self.output_w)
			if self.is_random_scaling:
				scale = (self.scaling_range[0] + self.scaling_range[1]) / 2
				img = self.random_scaling(img, minval=scale, maxval=scale)

		if not self.multiple_categories:
			image_label = self.to_categorical(int(image_label), self.nb_classes)

		if phase in ['train', 'trainval']:
			img = self.random_crop_and_pad_image(img, size=self.output_shape, center_range=self.crop_range)
		elif phase in ['val']:
			img = self.random_crop_and_pad_image(img, size=self.output_shape, center_range=[0.5,0.5])

		img = self.scale_output(img)

		return img, image_label if method == 'supervised' else img
