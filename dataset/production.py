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
# import struct
# import gzip
import numpy as np
# import matplotlib.pyplot as plt
# import pickle
# import gzip
import json
import cv2
from .base_dataset import BaseDataset


																						
class ChipProduction(BaseDataset):

	def __init__(self, config):
		
		super(ChipProduction, self).__init__(config)
		self.config = config

		self._total_production_dataset_dir = '/home/zhicongy/ws/dataset/production'
		self._dataset_defects_mask_dir = '/home/zhicongy/ws/dataset/defects_mask'

		self.lot_number_list = os.listdir(self._total_production_dataset_dir)
		

		self.name = 'chip_production'
		self.output_shape = self.config.get('output shape', [64, 64, 2])
		self.output_size = [self.output_shape[1], self.output_shape[0]]
		self.batch_size = int(self.config.get('batch_size', 128))

		self.subapp_list = [u'SI-XGA-B-BLand2-WX163L2CxxV02', u'SI-RT-DUMMY', u'SI-LT-DUMMY']
		self.output_subapp_list = ['composed']

		self.scalar_range = config.get('scalar range', [0.0, 1.0])

		self.crop_range = self.config.get('crop range', [0.1, 0.9])
		self.crop_range_hor = self.config.get('horizontal crop range', self.crop_range)
		self.crop_range_ver = self.config.get('vertical crop range', self.crop_range)


	def get_image_indices(self, phase, method='unsupervised'):
		return np.arange(len(self.lot_number_list))

	def read_image_by_index(self, ind, phase=None, method='unsupervised'):
		assert(phase in ['train'])
		assert(method in ['unsupervised'])

		if method == 'supervised':
			lot_number = self.lot_number_list[ind]
			image_dict = self.get_aligned_dataset_in_lot2(lot_number)

			if image_dict is not None:
				subapp_list = [u'SI-XGA-B-BLand2-WX163L2CxxV02', u'SI-RT-DUMMY', u'SI-LT-DUMMY']
				unit_id_list = [key for key in image_dict.keys()]
				unit_id_list = np.random.choice(unit_id_list, size=4)
				output_image_list = []

				for unit_id in unit_id_list:
					image_data = image_dict[unit_id]
					images_path = [os.path.join(self._total_production_dataset_dir, lot_number, image_data[subapp]) for subapp in subapp_list]

					if not os.path.exists(images_path[0]):
						print('info : images do not exist ', images_path[0])
					else:
						images = [cv2.imread(image_path)[:, :, 0].astype(np.float32) for image_path in images_path]
						image0 = self.compose_fake_3d_image(images[1], images[2])
						# image1 = images[0]
						# mixed_image = np.array([image0, image1])
						mixed_image = image0.reshape(list(image0.shape) + [1,])
						# mixed_image = mixed_image.transpose([1, 2, 0])
						for i in range(self.batch_size):
							img = self.random_crop_and_pad_image(mixed_image, size=self.output_size, center_range=self.crop_range)
							img = img / 255.0 
							img = img * (self.scalar_range[1] - self.scalar_range[0]) + self.scalar_range[0]
							output_image_list.append(img)
				return output_image_list
			else:
				return None


	def get_aligned_dataset_in_lot(self, lot, subapp_list, dataset_dir=None):
		if dataset_dir is None:
			dataset_dir = self._total_production_dataset_dir
		'''
			Return:
				images_dict = dict({
									subapp_name_1 : {
										unit_id_1 : image_path_str,
										unit_id_2 : image_path_str,
										...
									},
									subapp_name_2 : {
										unit_id_1 : image_path_str,
										unit_id_2 : image_path_str,
										...
									}
									...
								})
		'''
		lot_dir = os.path.join(dataset_dir, lot)
		lot_json_file = os.path.join(lot_dir, 'aligned.json')
		if os.path.exists(lot_json_file):
			images_list_dict = dict()
			for subapp in subapp_list:
				images_list_dict[subapp] = dict()

			with open(lot_json_file, 'r') as infile:
				json_data = json.loads(infile.read())

			# print(json_data['lot_number'])
			units = json_data['units']

			for unit in units:
				unit_id = unit['unit_id']
				unit_images = unit['images']
				for image in unit_images:
					images_list_dict[image['sub_app']][unit_id] = os.path.join(lot_dir, image['aligned_image'])
			return images_list_dict
		else:
			return None


	def get_aligned_dataset_in_lot2(self, lot, dataset_dir=None):
		if dataset_dir is None:
			dataset_dir = self._total_production_dataset_dir
		'''
			Return:
				images_dict = dict({
									unit_id_1 : {
										subapp_name_1 : image_path_str,
										subapp_name_2 : image_path_str,
										...
									},
									unit_id_2 : {
										subapp_name_1 : image_path_str,
										subapp_name_2 : image_path_str,
										...
									}
									...
								})
		'''
		lot_dir = os.path.join(dataset_dir, lot)
		lot_json_file = os.path.join(lot_dir, 'aligned.json')
		if os.path.exists(lot_json_file):
			images_dict = dict()

			with open(lot_json_file, 'r') as infile:
				json_data = json.loads(infile.read())

			# print(json_data['lot_number'])
			units = json_data['units']

			for unit in units:
				unit_id = unit['unit_id']
				unit_images = unit['images']
				unit_id_dict = dict()
				discard = False

				for image in unit_images:
					# for some units we dont have aligned image file, just discard this unit
					if 'aligned_image' not in image:
						# print('warning : aligned_image field not in image ' + str(lot) + ' ' + str(unit_id))
						discard = True
					else:
						# in some repository the image path is wrong absolute path, 
						# e.g. 'aligned_image' : '/mnt/cephfs/yjiang2/data/production/L750D205_1x/M83682DA02601_SI-XGA-B-BLand2-WX163L2CxxV02.png'
						# we just use the filename string
						aligned_image_path = image['aligned_image'].split('/')[-1]
						unit_id_dict[image['sub_app']] = os.path.join(lot_dir, aligned_image_path)
				if not discard:
					images_dict[unit_id] = unit_id_dict

			return images_dict
		else:
			return None


	def compose_fake_3d_image(self, sub0, sub1):
		'''
		create new subapp = normal(log(sub0 + 1) - log(sub1+1), 0, 255.99)
		'''
		# from utils.utils import normal
		composed_image = self.normal(np.log(sub0.astype(np.float32) + 1) - np.log(sub1.astype(np.float32) + 1),
								0, 255.99)
		return composed_image



	def normal(self, img, min=0, max=1, default=0):
		s = img.copy().astype(np.float32)
		smin = s.min()
		sdiff = s.max() - smin
		diff = max-min
		if sdiff == 0:
			s.fill(default*diff+min)
			return s
		else:
			diffrate = diff / sdiff
			return (s - smin) * diffrate+min


