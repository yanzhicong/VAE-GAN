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

from skimage import io
import cv2

from .basedataset import BaseDataset



class CelebA(BaseDataset):

	def __init__(self, config):

		super(CelebA, self).__init__(config)
		self.config = config

		self.name = 'CelebA'
		self.output_shape = config.get('output shape', [64, 64, 3])
		self.output_size = self.output_shape[0:2]
		self.unsupervised = config.get('unsupervised', False)
		self.crop_bbox = config.get('crop bbox', [9, 29, 169, 189]) # x0, y0, x1, y1

		self._dataset_dir = '/mnt/data01/dataset/CelebA'
		if not os.path.exists(self._dataset_dir):
			self._dataset_dir = 'F:/Data/CelebA'
		if not os.path.exists(self._dataset_dir):
			self._dataset_dir = 'E:/dataset/CelebA'
		if not os.path.exists(self._dataset_dir):
			self._dataset_dir = config.get('dataset_dir', self._dataset_dir)
		if not os.path.exists(self._dataset_dir):
			raise Exception("CelebA : the dataset dir " + self._dataset_dir + " is not exist")

		if self.unsupervised:
			self.image_list = self.read_image_list(read_label=False)
		else:
			self.image_list, self.label_list = self.read_image_list(read_label=True)


	def read_image_list(self, read_label=True):
		list_attr_celeba_filepath = os.path.join(self._dataset_dir, 'list_attr_celeba.txt')
		image_list = []
		attr_list = []
		self.attr_name_list = []
		self.attr_id2name_dict = {}
		self.attr_name2id_dict = {}

		if read_label:
			with open(list_attr_celeba_filepath, 'r') as infile:
				for line_ind, line in enumerate(infile):
					if line_ind == 0:
						total_images = int(line[:-1])
					elif line_ind == 1:
						line = line[:-1]
						for attr_name in line.split():
							attr_id = len(self.attr_name_list)
							self.attr_name_list.append(attr_name)
							self.attr_id2name_dict[attr_id] = attr_name
							self.attr_name2id_dict[attr_name] = attr_id
					else:
						line_split = line[:-1].split()
						image_filename = line_split[0][:-4] + '.png'
						image_attr = np.array([1 if int(attr) == 1 else 0 for attr in line_split[1:]])

						image_list.append(image_filename)
						attr_list.append(image_attr)
			return image_list, attr_list
		else:
			with open(list_attr_celeba_filepath, 'r') as infile:
				for line_ind, line in enumerate(infile):
					if line_ind == 0:
						total_images = int(line[:-1])
					elif line_ind == 1:
						pass
					else:
						line_split = line[:-1].split()
						image_filename = line_split[0][:-4] + '.png'
						image_list.append(image_filename)
			return image_list

	def get_image_indices(self, phase='train', method='supervised'):
		assert(phase in ['train'])
		assert(method in ['supervised', 'unsupervised'])
		indices = np.arange(len(self.image_list))
		if self.shuffle_train:
			np.random.shuffle(indices)
		return indices
	
	def read_image_by_index_supervised(self, ind, phase='train'):
		assert(phase in ['train'])
		assert(not self.unsupervised)

		image_filepath = os.path.join(self._dataset_dir, 'images', self.image_list[ind])
		image_attr = np.array(self.label_list[ind])

		img = io.imread(image_filepath)
		if self.crop_bbox is not None:
			img = self.crop_and_pad_image(img, self.crop_bbox)
		img = img.astype('float32') / 255.0
		img = cv2.resize(img, (self.output_size[1], self.output_size[0]))
		img = self.scale_output(img)
		return img, image_attr

	def read_image_by_index_unsupervised(self, ind, phase='train'):
		assert(phase in ['train'])
		image_filepath = os.path.join(self._dataset_dir, 'images', self.image_list[ind])
		img = io.imread(image_filepath)
		if self.crop_bbox is not None:
			img = self.crop_and_pad_image(img, self.crop_bbox)
		img = img.astype('float32') / 255.0
		img = cv2.resize(img, (self.output_size[1], self.output_size[0]))
		img = self.scale_output(img)
		return img

		
	# 	if not os.path.exists(self._dataset_dir):
	# 		raise Exception("CelebA : the dataset dir " + self._dataset_dir + " is not exist")

	# 	self.name = 'mnist'
	# 	self.input_shape = config.get('output shape', [28, 28, 1])
	# 	self.batch_size = int(config.get('batch_size', 128))
	# 	self.nb_classes = 10

	# 	self.y_train, self.x_train = self._read_data(
	# 		os.path.join(self._dataset_dir, 'train-labels-idx1-ubyte.gz'),
	# 		os.path.join(self._dataset_dir, 'train-images-idx3-ubyte.gz')
	# 	)

	# 	self.y_test, self.x_test = self._read_data(
	# 		os.path.join(self._dataset_dir, 't10k-labels-idx1-ubyte.gz'),
	# 		os.path.join(self._dataset_dir, 't10k-images-idx3-ubyte.gz')
	# 	)

	# 	self.x_train = self.x_train.astype(np.float32) / 255.0
	# 	self.x_test = self.x_test.astype(np.float32) / 255.0

	# 	if self.config.get('semi-supervised', False):
	# 		self.extra_file_path = './dataset/extra_files'
	# 		if not os.path.exists(self.extra_file_path):
	# 			os.mkdir(self.extra_file_path)
	# 		self.extra_file_path = os.path.join(self.extra_file_path, self.name)
	# 		if not os.path.exists(self.extra_file_path):
	# 			os.mkdir(self.extra_file_path)

	# 		self.nb_labelled_images_per_class = self.config.get('nb_labelled_images_per_class', 100)
	# 		self.labelled_image_indices = self._get_labelled_image_indices(self.nb_labelled_images_per_class)

	# 		self.x_train_u = self.x_train
			
	# 		self.x_train_l = self.x_train[self.labelled_image_indices]
	# 		self.y_train_l = self.y_train[self.labelled_image_indices]
	# 	else:
	# 		self.x_train_l = self.x_train
	# 		self.y_train_l = self.y_train

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

	# def _read_data(self, label_url, image_url):
	# 	with gzip.open(label_url) as flbl:
	# 		magic, num = struct.unpack(">II",flbl.read(8))
	# 		label = np.fromstring(flbl.read(),dtype=np.int8)
	# 	with gzip.open(image_url,'rb') as fimg:
	# 		magic, num, rows, cols = struct.unpack(">IIII",fimg.read(16))
	# 		image = np.fromstring(fimg.read(),dtype=np.uint8).reshape(len(label),rows,cols)
	# 	return (label, image)


