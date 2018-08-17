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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
from skimage import io
import cv2


from .base_dataset import BaseDataset


class BaseImageListDataset(BaseDataset):

	def __init__(self, config):
		
		super(BaseImageListDataset, self).__init__(config)
		self.config = config
		self.batch_size = int(config.get('batch_size', 128))


		assert('output shape' in config)


		self.is_random_scaling = config.get('random scaling', True)
		self.is_random_mirroring = config.get('random mirroring', True)
		self.is_random_cropping = config.get('random cropping', True)
		self.scaling_range = config.get('scaling range', [0.5, 1.5])
		self.crop_range = self.config.get('crop range', [0.1, 0.9])
		self.crop_range_hor = self.config.get('horizontal crop range', self.crop_range)
		self.crop_range_ver = self.config.get('vertical crop range', self.crop_range)
		self.output_shape = config.get('output shape', [256, 256, 3])
		self.output_size = self.output_shape[0:2]
		self.output_h = self.output_shape[0]
		self.output_w = self.output_shape[1]
		self.output_c = self.output_shape[2]


		# please fill in the following field in the drived dataset class
		self._dataset_dir = None
		self.train_imagelist_fp = None      # the txt file which contains the list of images and labels
		self.val_imagelist_fp = None        # for example:
		self.test_imagelist_fp = None       #   train.txt:
											# line 1 : image_filepath,classname1,classname2
											# line 2 : image1.jpg,1,0
											# line 3 : image2.jpg,0,0
											# ...

	

	def build_dataset(self):
		assert(self.train_imagelist_fp != None)
		assert(self.val_imagelist_fp != None)

		self.train_images, self.train_labels = self.read_imagelist(self.train_imagelist_fp)
		self.val_images, self.val_labels = self.read_imagelist(self.val_imagelist_fp)

		if self.test_imagelist_fp != None:
			self.test_images = self.read_imagelist(self.test_imagelist_fp)


	
	def read_imagelist(self, filename, has_label=True):
		image_list = []
		label_list = []
		class_head_list = []
		nb_classes = 0

		with open(filename, 'r', encoding='utf-8') as infile:

			for ind, line in enumerate(infile):
				if ind == 0:
					head_split = line[:-1].split(',')
					class_head_list = head_split[1:]
					nb_classes = len(class_head_list)
				else:
					split = line[:-1].split(',')
					image_fp = split[0]
					image_list.append(image_fp)

					if has_label:
						label = np.array([int(i) for i in split[1:]])
						label_list.append(label)

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
				np.random.suffle(indices)
			return indices

	def read_image_by_index(self, ind, phase='train', method='supervised'):
		assert(phase in ['train', 'test', 'val', 'trainval'])
		assert(method in ['supervised', 'unsupervised'])
		
		if phase == 'train':
			image_fp = os.path.join(self._dataset_dir, self.train_images[ind])
			if method == 'superivsed':
				image_label = self.train_labels[ind]
		elif phase == 'val':
			image_fp = os.path.join(self._dataset_dir, self.val_images[ind])
			if method == 'superivsed':
				image_label = self.val_labels[ind]
		elif phase == 'trainval':
			if ind < len(self.train_images):
				image_fp = os.path.join(self._dataset_dir, self.train_images[ind])
				if method == 'superivsed':
					image_label = self.train_labels[ind]
			else:
				image_fp = os.path.join(self._dataset_dir, self.val_images[ind - len(self.train_images)])
				if method == 'superivsed':
					image_label = self.val_labels[ind - len(self.train_images)]
		elif phase == 'test':
			assert method == 'unsupervised'
			assert self.test_images != None
			image_fp = os.path.join(self._dataset_dir, self.test_images[ind])
		
		img = io.imread(image_fp)

		# in case of single channel image
		if img.ndims == 2:
			img = cv2.merge([img, img, img])


		if phase in ['train', 'trainval']:
			if self.is_random_scaling:
				img = self.random_scaling(img, minval=self.scaling_range[0], maxval=self.scaling_range[1])
			if self.is_random_mirroring:
				img = self.random_mirroring(img)
		elif phase in ['val']:
			scale = (self.scaling_range[0] + self.scaling_range[1]) / 2
			img = self.random_scaling(img, minval=scale, maxval=scale)

		if phase in ['train', 'trainval']:
			img = self.random_crop_and_pad_image(img, size=self.output_shape, center_range=self.crop_range)
		elif phase in ['val']:
			img = self.random_crop_and_pad_image(img, size=self.output_shape, center_range=[0.5,0.5])

		if method == 'supervised':
			return img, image_label
		else:
			return img


