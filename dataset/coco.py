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

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

from .base_dataset import BaseDataset


class MSCOCO(BaseDataset):

	def __init__(self, config):

		super(MSCOCO, self).__init__(config)
		self.config = config

		self._dataset_dir = 'E:\\Kaggle\\Data\\MS_COCO'
		if not os.path.exists(self._dataset_dir):
			self._dataset_dir = '/mnt/data01/dataset/MS_COCO'
		if not os.path.exists(self._dataset_dir):
			self._dataset_dir = 'D:\\Data\\MS_COCO'
		if not os.path.exists(self._dataset_dir):
			self._dataset_dir = self.config.get('dataset dir', '')
		if not os.path.exists(self._dataset_dir):
			raise Exception("MS_COCO : the dataset dir " + self._dataset_dir + " is not exists")


		self.year = str(self.config.get('year', 2017))
		self.task = str(self.config.get('task', 'instances'))
		assert self.year in ['2017', '2014']
		assert self.task in ['instances']

		train_json_file = os.path.join(self._dataset_dir, 'annotations', 'instances_train' + self.year + '.json')
		val_json_file = os.path.join(self._dataset_dir, 'annotations', 'instances_val' + self.year + '.json')
		self.train_image_dir = os.path.join(self._dataset_dir, 'train'+self.year)
		self.val_image_dir = os.path.join(self._dataset_dir, 'val'+self.year)

		if not os.path.exists(train_json_file):
			raise Exception("MS_COCO : the train json file " + train_json_file + " is not exists")
		if not os.path.exists(val_json_file):
			raise Exception("MS_COCO : the val json file " + val_json_file + " is not exists")
		if not os.path.exists(self.train_image_dir):
			raise Exception("MS_COCO : the train image dir " + self.train_image_dir + " is not exists")
		if not os.path.exists(self.val_image_dir):
			raise Exception("MS_COCO : the val image dir " + self.val_image_dir + " is not exists")

		self.show_warning = self.config.get('show warning', False)

		self.train_coco = COCO(train_json_file)
		self.val_coco = COCO(val_json_file)

		class_ids = self.train_coco.getCatIds()
		self.class_id_list = class_ids

		image_id_list = []
		for i in class_ids:
			image_id_list += self.train_coco.getImgIds(catIds=[i])
		image_id_list = list(set(image_id_list))
		self.train_image_id_list = image_id_list

		image_id_list = []
		for i in class_ids:
			image_id_list += self.val_coco.getImgIds(catIds=[i])
		image_id_list = list(set(image_id_list))
		self.val_image_id_list = image_id_list


	def get_image_indices(self, phase, method='supervised'):
		assert phase in ['train', 'val', 'trainval']
		assert method in ['supervised']

		if phase == 'train':
			indices = np.array(self.train_image_id_list.copy())
			if self.shuffle_train:
				np.random.shuffle(indices)
		elif phase == 'val':
			indices = np.array(self.val_image_id_list.copy())
			if self.shuffle_val:
				np.random.shuffle(indices)
		elif phase == 'trainval':
			indices = np.array(range(len(self.train_image_id_list) + len(self.val_image_id_list)))
			if self.shuffle_train:
				np.random.shuffle(indices)

		return indices

	def read_image_by_index(self, ind, phase, method):
		assert phase in ['train', 'val', 'trainval']
		assert method in ['supervised']

		if phase == 'train':
			path = self.train_image_dir
			coco = self.train_coco
			i = ind
		elif phase == 'val':
			path = self.val_image_dir
			coco = self.val_coco
			i = ind
		elif phase == 'trainval':
			if ind < len (self.train_image_id_list):
				path = self.train_image_dir
				coco = self.train_coco
				i = self.train_image_id_list[ind]
			else:
				path = self.val_image_dir
				coco = self.val_coco
				i = self.val_image_id_list[ind - len(self.train_image_id_list)]

		image_path = os.path.join(path, coco.imgs[i]['file_name'])
		anno = coco.loadAnns(coco.getAnnIds(imgIds=[i], catIds=self.class_id_list, iscrowd=None))
		
		try:
			img = io.imread(image_path)
		except Exception:
			if self.show_warning:
				print('Warning : read image ' + image_path + ' failed')
			return None, None
		return img, anno

	def show_image_and_anno(self, plt, img, anns):
		plt.imshow(img)
		self.train_coco.showAnns(anns)


