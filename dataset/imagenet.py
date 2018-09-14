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
import xml.etree.ElementTree as ET
from skimage import io
import cv2

from .base_dataset import BaseDataset

class ImageNet(BaseDataset):

	def __init__(self, config):

		super(ImageNet, self).__init__(config)
		
		self.name = 'imagenet2012'
		self._dataset_dir = 'F:\\Data\\Imagenet\\imagenet_object_localization\\ILSVRC'
		if not os.path.exists(self._dataset_dir):
			self._dataset_dir = '/mnt/data01/dataset/ImageNet/ILSVRC'
		self._dataset_dir = self.config.get('dataset_dir', self._dataset_dir)
		self.nb_classes = 1000
 
		self.annotations_dir = os.path.join(self._dataset_dir, 'Annotations', 'CLS-LOC')
		self.data_dir = os.path.join(self._dataset_dir, 'Data', 'CLS-LOC')
		self.imagesets_dir = os.path.join(self._dataset_dir, 'ImageSets', 'CLS-LOC')

		self.task = self.config.get('task', 'classification')  # classification or localization

		self.input_shape = self.config.get('output shape', [224, 224, 3])

		self.extra_file_path = './dataset/extra_files'
		if not os.path.exists(self.extra_file_path):
			os.mkdir(self.extra_file_path)
		self.extra_file_path = os.path.join(self.extra_file_path, self.name)
		if not os.path.exists(self.extra_file_path):
			os.mkdir(self.extra_file_path)

		# convertion between classid and index 
		self.temp_path = os.path.join(self.extra_file_path, 'classid2index_dict.pkl')
		if not os.path.exists(self.temp_path):
			self.make_class_id_to_index_dict()
		else:
			self.classid2index_dict = pickle.load(open(os.path.join(self.extra_file_path, 'classid2index_dict.pkl'), 'rb'))
			self.index2classid_dict = pickle.load(open(os.path.join(self.extra_file_path, 'index2classid_dict.pkl'), 'rb'))

		# load train file list
		if self.task == 'classification':
			self.x_train, self.y_train = self.load_classification_train_set('train', 'train_cls')


	def make_class_id_to_index_dict(self):
		classid_list = os.listdir(os.path.join(self.data_dir, 'train'))
		self.classid2index_dict = {}
		self.index2classid_dict = {}
		for ind, classid in enumerate(classid_list):
			self.classid2index_dict[classid] = ind
			self.index2classid_dict[ind] = classid
		pickle.dump(self.classid2index_dict, open(os.path.join(self.extra_file_path, 'classid2index_dict.pkl'), 'wb'))
		pickle.dump(self.index2classid_dict, open(os.path.join(self.extra_file_path, 'index2classid_dict.pkl'), 'wb'))


	def load_classification_train_set(self, phase, imageset_file_name):
		x_pickle_filepath = os.path.join(self.extra_file_path, self.task + '_' + imageset_file_name + '_x.pkl')
		y_pickle_filepath = os.path.join(self.extra_file_path, self.task + '_' + imageset_file_name + '_y.pkl')
		
		if (os.path.exists(x_pickle_filepath) 
			and os.path.exists(y_pickle_filepath)):
			x = pickle.load(open(x_pickle_filepath, 'rb'))
			y = pickle.load(open(y_pickle_filepath, 'rb'))
			return x, y

		x_list = []
		y_list = []

		if self.task == 'classification':
			with open(os.path.join(self.imagesets_dir, imageset_file_name + '.txt'), 'r') as imagesets_file:
				for line in imagesets_file:
					image_filename = line.split()[0]

					object_classid = image_filename.split('/')[0]
					object_class_index = self.classid2index_dict[object_classid]

					x_list.append(os.path.join(phase, image_filename+'.JPEG'))
					y_list.append(object_class_index)

			pickle.dump(x_list, open(x_pickle_filepath, 'wb'))
			pickle.dump(y_list, open(y_pickle_filepath, 'wb'))

			return x_list, y_list

		elif self.task == 'localization':
			with open(os.path.join(self.imagesets_dir, imageset_file_name + '.txt'), 'r') as imagesets_file:
				for line in imagesets_file:
					image_filename = line.split()[0]

					annotation_filename = os.path.join(self.annotations_dir, phase, image_filename + '.xml')
					annotation_file = ET.parse(annotation_filename)
					root = annotation_file.getroot()
					for object_anno in root.findall('object'):
						object_id = object_anno.find('name').text
						object_bndbox = object_anno.find('bndbox')
						x1 = int(object_bndbox.find('xmin').text)
						y1 = int(object_bndbox.find('ymin').text)
						x2 = int(object_bndbox.find('xmax').text)
						y2 = int(object_bndbox.find('ymax').text)
						object_class = self.classid2index_dict[object_id]
						x_list.append(
							(os.path.join(phase, image_filename+'.JPEG'),x1,y1,x2,y2)
						)
						y_list.append(
							object_class
						)

			pickle.dump(x_list, open(x_pickle_filepath, 'wb'))
			pickle.dump(y_list, open(y_pickle_filepath, 'wb'))

			return x_list, y_list            


	def iter_train_images(self):
		file_index = np.arange(len(self.x_train))
		if self.shuffle_train:
			np.random.shuffle(file_index)

		batch_x = []
		batch_y = []

		for i, ind in enumerate(file_index):
			train_image_filepath = os.path.join(self.data_dir, self.x_train[ind])
			train_image_label = np.zeros((self.nb_classes,))
			train_image_label[self.y_train[ind]] = 1
			train_image = io.imread(train_image_filepath)

			# in case of single channel image
			if len(train_image.shape) == 2:
				train_image = cv2.merge([train_image, train_image, train_image])

			train_image = cv2.resize(train_image, (self.input_shape[1], self.input_shape[0]))
			batch_x.append(train_image)
			batch_y.append(train_image_label)

			if len(batch_x) == self.batch_size:
				yield i, np.array(batch_x), np.array(batch_y)
				batch_x = []
				batch_y = []


	def get_train_indices(self):
		file_index = np.arange(len(self.x_train))
		if self.shuffle_train:
			np.random.shuffle(file_index)
		return file_index


	def read_train_image_by_index(self, index):
			train_image_filepath = os.path.join(self.data_dir, self.x_train[index])
			train_image_label = np.zeros((self.nb_classes,))
			train_image_label[self.y_train[index]] = 1
			train_image = io.imread(train_image_filepath)

			# in case of single channel image
			if len(train_image.shape) == 2:
				train_image = cv2.merge([train_image, train_image, train_image])
				
			# in case of RGBA image
			if train_image.shape[2] == 4:
				train_image = train_image[:, :, 0:3]

			# other cases
			if len(train_image.shape) != 3 or train_image.shape[2] != 3:
				return None, None

			train_image = cv2.resize(train_image, (self.input_shape[1], self.input_shape[0])).astype(np.float32) / 255.0
			return train_image, train_image_label
