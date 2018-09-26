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
import pickle

from .base_dataset import BaseDataset


class PASCAL_VOC(BaseDataset):

	def __init__(self, config):
		
		super(PASCAL_VOC, self).__init__(config)
		self.config = config
		self.year = str(config.get('year', 2012))
		self.task = config.get('task', 'segmentation_class')

		assert(self.year in ['2012', '2007'])
		assert(self.task in [
			'segmentation_class_aug',
			'segmentation_class', 'segmentation',
			'segmentation_object',
			'classification'
		])



		if self.task == 'segmentation_class_aug':
			self.color_map = [

			]

			self.nb_classes = len(self.color_map)

		elif self.task in ['segmentation', 'segmentation_class', 'segmentation_object']:
			# colour map
			self.color_map = [(0,0,0)
							# 0=background
							,(128,0,0),(0,128,0),(128,128,0),(0,0,128),(128,0,128)
							# 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
							,(0,128,128),(128,128,128),(64,0,0),(192,0,0),(64,128,0)
							# 6=bus, 7=car, 8=cat, 9=chair, 10=cow
							,(192,128,0),(64,0,128),(192,0,128),(64,128,128),(192,128,128)
							# 11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person
							,(0,64,0),(128,64,0),(0,192,0),(128,192,0),(0,64,128)]
							# 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
			self.nb_classes = len(self.color_map)
		elif self.task in ['classification']:
			self.class_name_list = ['aeroplane','bicycle','bird','boat','bottle','bus',
						'car','cat','chair','cow','diningtable','dog','horse','motorbike',
						'person','pottedplant','sheep','sofa','train','tvmonitor',]
			self.nb_classes = len(self.class_name_list)
		else:
			raise NotImplementedError

		if self.year == '2012':
			self._dataset_dir = '/home/zhicongy/ws/dataset/PASCAL_VOC/2012/VOCdevkit/VOC2012'
			if not os.path.exists(self._dataset_dir):
				self._dataset_dir = '/mnt/data02/dataset/PASCAL_VOC/VOCdevkit/VOC2012'
			if not os.path.exists(self._dataset_dir):
				self._dataset_dir = 'E:\\dataset\\PASCAL_VOC\\VOCdevkit\\VOC2012'
			if not os.path.exists(self._dataset_dir):
				self._dataset_dir = config.get('dataset dir', '')
			if not os.path.exists(self._dataset_dir):
				raise Exception("PASCAL_VOC : the dataset dir " + self._dataset_dir + " is not exist")

			self.name = 'pascal_voc'


			if self.task == 'segmentation_class_aug':
				
				self.nb_val_samples = self.config.get('nb validate samples', 2000)
				self.train_image_list, self.train_mask_list = self.read_augmented_image_list(phase='train', nb_val_samples=self.nb_val_samples)
				self.val_image_list, self.val_mask_list = self.read_augmented_image_list(phase='val', nb_val_samples=self.nb_val_samples)




			elif self.task in ['segmentation', 'segmentation_class', 'segmentation_object']:
				self.train_image_list, self.train_mask_list = self.read_image_list(self.task, phase='train')
				self.val_image_list, self.val_mask_list = self.read_image_list(self.task, phase='val')
				self.test_image_list = self.read_image_list(self.task, phase='val')
			elif self.task in ['classification']:
				self.train_image_list, self.train_label_list = self.read_image_list(self.task, phase='train')
				self.val_image_list, self.val_label_list = self.read_image_list(self.task, phase='val')


		elif self.year == '2007':
			self._dataset_dir = '/home/zhicongy/ws/dataset/PASCAL_VOC/2007/VOCdevkit/VOC2007'
			if not os.path.exists(self._dataset_dir):
				self._dataset_dir = config.get('dataset dir', '')
			if not os.path.exists(self._dataset_dir):
				raise Exception("MNIST : the dataset dir " + self._dataset_dir + " is not exist")


		self.batch_size = int(config.get('batch_size', 128))

		self.is_random_scaling = config.get("random scale", True)
		self.is_random_mirroring = config.get("random mirror", True)
		self.is_random_cropping = config.get("random crop", True)
		self.scaling_range = config.get('scaling range', [0.5, 1.5])
		self.crop_range = self.config.get('crop range', [0.1, 0.9])
		self.crop_range_hor = self.config.get('horizontal crop range', self.crop_range)
		self.crop_range_ver = self.config.get('vertical crop range', self.crop_range)
		self.output_shape = config.get('output shape', [256, 256, 3])
		self.output_size = self.output_shape[0:2]
		self.output_h = self.output_shape[0]
		self.output_w = self.output_shape[1]
		self.output_c = self.output_shape[2]

	def read_image_list(self, task='segmentation_class', phase='train'):
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

		elif task == 'classification':
			if phase not in ['train', 'val', 'trainval']:
				raise Exception('pascal voc error : no phase named ' + str(phase))

			input_image_filepath_list = []
			image_list_file = os.path.join(self._dataset_dir, 'ImageSets', 'Main', phase + '.txt')

			with open(image_list_file, 'r') as infile:
				for line in infile:
					line = line[:-1]
					input_image_filepath_list.append('JPEGImages/' + line + '.jpg')
			nb_images = len(input_image_filepath_list)
			image_class_array = np.zeros([nb_images, self.nb_classes])

			for class_ind, class_name in enumerate(self.class_name_list):
				image_list_file = os.path.join(self._dataset_dir, 'ImageSets', 'Main', class_name + '_' + phase + '.txt')
				with open(image_list_file, 'r') as infile:
					for line_ind, line in enumerate(infile):
						is_object = int(line[:-1].split()[-1])
						if is_object == 1:
							image_class_array[line_ind, class_ind] = 1
			return input_image_filepath_list, image_class_array


	def read_augmented_image_list(self, phase='train', nb_val_samples=2000):
		aug_masks_dir = os.path.join(self._dataset_dir, 'SegmentationClassAug')
		if not os.path.exists(aug_masks_dir):
			raise Exception('The SegmentationClassAug folder is not exists in the dataset, please download it')
		if phase not in ['train', 'val']:
			raise ValueError('phase error')

		mask_list = os.listdir(aug_masks_dir)

		nb_val_samples = np.minimum(len(mask_list), nb_val_samples)
		self.extra_file_path = os.path.join('./dataset/extra_files', self.name)
		if not os.path.exists(self.extra_file_path):
			os.makedirs(self.extra_file_path)

		if phase == 'train':
			indices_filepath = os.path.join(self.extra_file_path, 'segmentation_class_aug_val_indices_%d.pkl'%nb_val_samples)
		else:
			indices_filepath = os.path.join(self.extra_file_path, 'segmentation_class_aug_train_indices_%d.pkl'%nb_val_samples)

		if not os.path.exists(indices_filepath):
			image_indices = list(range(len(mask_list)))
			val_indices = np.random.choice(image_indices, size=nb_val_samples, replace=False)
			train_indices = image_indices.copy()
			for ind in val_indices:
				train_indices.remove(ind)
			pickle.dump(val_indices, open(os.path.join(self.extra_file_path, 'segmentation_class_aug_val_indices_%d.pkl'%nb_val_samples), 'wb'))
			pickle.dump(train_indices, open(os.path.join(self.extra_file_path, 'segmentation_class_aug_train_indices_%d.pkl'%nb_val_samples), 'wb'))

		indices = pickle.load(open(indices_filepath, 'rb'))
		mask_list = [mask_list[ind] for ind in indices]

		mask_filepath_list = ['SegmentationClassAug/' + fp for fp in mask_list]
		image_filepath_list = ['JPEGImages/' + fp.split('.')[0] + '.jpg' for fp in mask_list]

		return image_filepath_list, mask_filepath_list

	'''
	
	'''
	def get_image_indices(self, phase, method='supervised'):
		'''
		'''
		assert(phase in ['train', 'val', 'trainval', 'test'])
		assert(method in ['supervised', 'unsupervised'])
		
		if self.task == 'segmentation_class_aug':
			if phase == 'train':
				indices = np.array(range(len(self.train_image_list)))
				if self.shuffle_train:
					np.random.shuffle(indices)
				return indices
			elif phase == 'val':
				indices = np.array(range(len(self.val_image_list)))
				if self.shuffle_val:
					np.random.shuffle(indices)
				return indices
			else:
				raise ValueError('None phase named ' + str(phase))

		elif self.task in ['segmentation_class', 'segmentation', 'segmentation_object']:
			if phase == 'train':
				indices = np.array(range(len(self.train_image_list)))
				if self.shuffle_train:
					np.random.shuffle(indices)		
				return indices

			elif phase == 'val':
				indices = np.array(range(len(self.val_image_list)))
				return indices

			elif phase == 'test':
				indices = np.array(range(len(self.test_image_list)))
				if self.shuffle_test:
					np.random.shuffle(indices)
				return indices
			else:
				raise Exception("None phase named " + str(phase))

		elif self.task in ['classification']:
			if phase == 'train':
				indices = np.array(range(len(self.train_image_list)))
				if self.shuffle_train:
					np.random.shuffle(indices)
			elif phase == 'val' or phase == 'test':
				indices = np.array(range(len(self.val_image_list)))
				if self.shuffle_test:
					np.random.shuffle(indices)
			return indices

		else:
			raise NotImplementedError


	def read_image_by_index(self, ind, phase='train', method='supervised'):
		assert(phase in ['train', 'val', 'test'])
		assert(method in ['supervised', 'unsupervised'])

		if method == 'supervised':

			if self.task == 'segmentation_class_aug':
				if phase == 'train':
					image_filepath = os.path.join(self._dataset_dir, self.train_image_list[ind])
					mask_filepath = os.path.join(self._dataset_dir, self.train_mask_list[ind])
				elif phase == 'val':
					image_filepath = os.path.join(self._dataset_dir, self.val_image_list[ind])
					mask_filepath = os.path.join(self._dataset_dir, self.val_mask_list[ind])

				img = io.imread(image_filepath)
				mask_c = io.imread(mask_filepath)

				return img, mask_c

			elif self.task in ['segmentation_class', 'segmentation', 'segmentation_object']:

				if phase == 'train':
					image_filepath = os.path.join(self._dataset_dir, self.train_image_list[ind])
					mask_filepath = os.path.join(self._dataset_dir, self.train_mask_list[ind])
				elif phase == 'val':
					image_filepath = os.path.join(self._dataset_dir, self.val_image_list[ind])
					mask_filepath = os.path.join(self._dataset_dir, self.val_mask_list[ind])

				img = io.imread(image_filepath)
				mask_c = io.imread(mask_filepath)
				mask = self.mask_colormap_encode(mask_c, self.color_map)

				if phase == 'train':
					if self.is_random_scaling:
						img, mask = self.random_scale(img, mask=mask, minval=self.scaling_range[0], maxval=self.scaling_range[1])
					if self.is_random_mirroring:
						img, mask = self.random_mirroring(img, mask=mask)
					if self.is_random_cropping:
						img, mask = self.random_crop_and_pad_image(img, mask=mask, size=self.output_shape, center_range=self.crop_range)
				elif phase == 'val':
					if self.is_random_scaling:
						scale = (self.scaling_range[0] + self.scaling_range[1]) / 2
						img, mask = self.random_scale(img, mask=mask, minval=scale, maxval=scale)

				img = img.astype(np.float32) / 255.0
				mask_onehot = self.to_categorical(mask, self.nb_classes)

				return img, mask_onehot

			elif self.task in ['classification']:
				if phase == 'train':
					image_filepath = os.path.join(self._dataset_dir, self.train_image_list[ind])
					label = self.train_label_list[ind]
				elif phase == 'val' or phase == 'test':
					image_filepath = os.path.join(self._dataset_dir, self.val_image_list[ind])
					label = self.val_label_list[ind]
					
				img = io.imread(image_filepath)

				if phase == 'train':
					if self.is_random_scaling:
						img = self.random_scale(img, minval=self.scaling_range[0], maxval=self.scaling_range[1])
					if self.is_random_mirroring:
						img = self.random_mirroring(img)
					if self.is_random_cropping:
						img = self.random_crop_and_pad_image(img, size=self.output_shape, center_range=self.crop_range)
				elif phase == 'val' or phase == 'test':
					if self.is_random_scaling:
						scale = (self.scaling_range[0] + self.scaling_range[1]) / 2
						img = self.random_scale(img, minval=scale, maxval=scale)
				img = img.astype(np.float32) / 255.0
				return img, label
			else:	
				raise NotImplementedError

		elif method == 'unsupervised':

			if self.task == 'segmentation_class_aug':
				if phase == 'train':
					image_filepath = os.path.join(self._dataset_dir, self.train_image_list[ind])
				elif phase == 'val':
					image_filepath = os.path.join(self._dataset_dir, self.val_image_list[ind])
				img = io.imread(image_filepath)
				return img

			elif self.task in ['segmentation_class', 'segmentation', 'segmentation_object']:
				if phase == 'train':
					image_filepath = os.path.join(self._dataset_dir, self.train_image_list[ind])
				elif phase == 'val':
					image_filepath = os.path.join(self._dataset_dir, self.val_image_list[ind])
				elif phase == 'test':
					image_filepath = os.path.join(self._dataset_dir, self.test_image_list[ind])
				img = io.imread(image_filepath)
				return img

			elif self.task in ['classification']:
				if phase == 'train':
					image_filepath = os.path.join(self._dataset_dir, self.train_image_list[ind])
				elif phase == 'val' or phase == 'test':
					image_filepath = os.path.join(self._dataset_dir, self.val_image_list[ind])
				img = io.imread(image_filepath)
				if phase == 'train':
					if self.is_random_scaling:
						img = self.random_scale(img, minval=self.scaling_range[0], maxval=self.scaling_range[1])
					if self.is_random_mirroring:
						img = self.random_mirroring(img)
					if self.is_random_cropping:
						img = self.random_crop_and_pad_image(img, size=self.output_shape, center_range=self.crop_range)
				elif phase == 'val' or phase == 'test':
					if self.is_random_scaling:
						scale = (self.scaling_range[0] + self.scaling_range[1]) / 2
						img = self.random_scale(img, minval=scale, maxval=scale)
				img = img.astype(np.float32) / 255.0
				return img
			else:
				raise NotImplementedError

