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
sys.path.append('./')
sys.path.append('../')
sys.path.append('./lib')

import numpy as np
import matplotlib.pyplot as plt

from skimage import io
# import pickle

from .basedataset import BaseDataset



class PASCAL_VOC(BaseDataset):

	def __init__(self, config):
		
		super(PASCAL_VOC, self).__init__(config)
		self.config = config
		self.year = str(config.get('year', 2012))
		if self.year not in ['2012', '2007']:
			raise Exception('Pascal voc config error')

		# colour map
		self.label_colours = [(0,0,0)
		                # 0=background
		                ,(128,0,0),(0,128,0),(128,128,0),(0,0,128),(128,0,128)
		                # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
		                ,(0,128,128),(128,128,128),(64,0,0),(192,0,0),(64,128,0)
		                # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
		                ,(192,128,0),(64,0,128),(192,0,128),(64,128,128),(192,128,128)
		                # 11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person
		                ,(0,64,0),(128,64,0),(0,192,0),(128,192,0),(0,64,128)]
		                # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor



		if self.year == '2012':
			self._dataset_dir = '/home/zhicongy/ws/dataset/PASCAL_VOC/2012/VOCdevkit/VOC2012'
			if not os.path.exists(self._dataset_dir):
				self._dataset_dir = '/mnt/data02/dataset/PASCAL_VOC/VOCdevkit/VOC2012'
			if not os.path.exists(self._dataset_dir):
				self._dataset_dir = config.get('dataset dir', '')
			if not os.path.exists(self._dataset_dir):
				raise Exception("PASCAL_VOC : the dataset dir " + self._dataset_dir + " is not exist")

			self.name = 'pascal_voc'

			self.task = config.get('task', 'segmentation_class')

			self.train_image_list, self.train_mask_list = self.read_labelled_image_list(self.task, phase='train')
			self.test_image_list = self.read_labelled_image_list(self.task, phase='val')



		elif self.year == '2007':
			self._dataset_dir = '/home/zhicongy/ws/dataset/PASCAL_VOC/2007/VOCdevkit/VOC2007'

			if not os.path.exists(self._dataset_dir):
				self._dataset_dir = config.get('dataset dir', '')
			if not os.path.exists(self._dataset_dir):
				raise Exception("MNIST : the dataset dir " + self._dataset_dir + " is not exist")


		self.output_shape = config.get('output shape', [256, 256, 3])
		self.output_size = self.output_shape[0:2]
		self.output_h = self.output_shape[0]
		self.output_w = self.output_shape[1]
		self.output_c = self.output_shape[2]
		self.batch_size = int(config.get('batch_size', 128))
		self.nb_classes = 10


	def read_labelled_image_list(self, task='segmentation_class', phase='train'):

		if task == 'segmentation_class' or task == 'segmentation' or task == 'segmentation_object':
			if phase not in ['train', 'val', 'test', 'trainval']:
				raise Exception('pascal voc erro : no phase named ' + str(phase))
			
			image_list_file = os.path.join(self._dataset_dir, 'ImageSets', 'Segmentation', phase + '.txt')

			if phase in ['train', 'val', 'trainval']:

				input_image_filepath_list = []
				output_mask_filepath_list = []

				if task == 'segmentation_class' or task == 'segmentation':
					with open(image_list_file, 'r') as infile:
						for line in infile:
							line = line[:-1]
							input_image_filepath_list.append('JPEGImages/' + line + '.jpg')
							output_mask_filepath_list.append('SegmentationClass/' + line + '.png')
				elif task == 'segmentation_object':
					with open(image_list_file, 'r') as infile:
						for line in infile:
							line = line[:-1]
							input_image_filepath_list.append('JPEGImages/' + line + '.jpg')
							output_mask_filepath_list.append('SegmentationObject/' + line + '.png')
				return input_image_filepath_list, output_mask_filepath_list

			elif phase == 'test':
				input_image_filepath_list = []
				with open(image_list_file, 'r') as infile:
					for line in infile:
						line = line[:-1]
						input_image_filepath_list.append('JPEGImages/' + line + '.jpg')
				return input_image_filepath_list


	'''

	'''
	def get_image_indices(self, phase, method='supervised'):
		'''
		'''
		if self.task in ['segmentation_class', 'segmentation', 'segmentation_object']:
			if phase == 'train':
				indices = np.array(range(len(self.train_image_list)))
				if self.shuffle_train:
					np.random.shuffle(indices)		
				return indices

			elif phase == 'test':
				indices = np.array(range(len(self.test_image_list)))
				if self.shuffle_test:
					np.random.shuffle(indices)
				return indices

			else:
				raise Exception("None phase named " + str(phase))
		else:
			raise NotImplementedError


	def read_image_by_index_supervised(self, ind):
		if self.task in ['segmentation_class', 'segmentation', 'segmentation_object']:
			image_filepath = os.path.join(self._dataset_dir, self.train_image_list[ind])
			mask_filepath = os.path.join(self._dataset_dir, self.train_mask_list[ind])
			img = io.imread(image_filepath)
			mask = io.imread(mask_filepath)



			return img, mask
		else:
			raise NotImplementedError


	def read_image_by_index_unsupervised(self, ind):
		if self.task in ['segmentation_class', 'segmentation', 'segmentation_object']:		
			image_filepath = os.path.join(self._dataset_dir, self.test_image_list[ind])
			img = io.imread(image_filepath)
			return img
		else:
			raise NotImplementedError


	# def read_test_image_by_index(self, index):

	# 	image_filepath = os.path.join(self._dataset_dir, self.test_image_list[index])
	# 	mask_filepath = os.path.join(self._dataset_dir, self.mask_image_list[index])

	# 	img = io.imread(image_filepath)
	# 	mask = io.imread(mask_filepath)

	# 	print(img.max())
	# 	print(img.min())
	# 	print(mask.max())
	# 	print(mask.min())

	# 	return img, mask


