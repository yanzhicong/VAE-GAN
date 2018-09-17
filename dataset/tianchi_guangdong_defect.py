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
from .base_imagelist_dataset import BaseImageListDataset
from .base_mil_dataset import BaseMILDataset

class TianChiGuangdongDefect(BaseImageListDataset, BaseMILDataset):

	def __init__(self, config):
		super(TianChiGuangdongDefect, self).__init__(config)

		self.config = config
		self.name = 'GuangdongDefect_for_TianChi'
	
		self.class_id2name = {
			0: "正常样本",
			1: "不导电",
			2: "擦花",
			3: "横条压凹",
			4: "桔皮",
			5: "漏底",
			6: "碰伤",
			7: "起坑",
			8: "凸粉",
			9: "涂层开裂",
			10: "脏点",
			11: "其他",
		}

		self.nb_classes = 11

		assert('output shape' in config)

		self._dataset_dir = "F:\\Data\\GuangDongIndustrialBigData"
		if not os.path.exists(self._dataset_dir):
			self.dataset_dir = "/mnt/data03/tianchi/GuangDongIndustrialBigData	"
		if not os.path.exists(self._dataset_dir):
			self._dataset_dir = self.config.get('dataset dir', '')

		if not os.path.exists(self._dataset_dir):
			raise Exception("TianChiGuangdongDectect Dataset : the dataset dir is not exists")

		self.stage = self.config.get('stage', 'stage1')
		assert self.stage in ['stage1']

		self.multiple_categories = True

		self.extra_fp = os.path.join('./dataset/extra_files', self.name)
		if not os.path.exists(self.extra_fp):
			os.makedirs(self.extra_fp)

		self.train_imagelist_fp = os.path.join(self.extra_fp, 'train_' + self.stage + '.csv')
		self.val_imagelist_fp = os.path.join(self.extra_fp, 'val_' + self.stage + '.csv')
		# self.test_imagelist_fp = os.path.join(self.extra_fp, 'test_' + self.stage + '.csv')

		if not os.path.exists(self.train_imagelist_fp) or not os.path.exists(self.val_imagelist_fp):
			self.__build_csv_files(os.path.join(self.extra_fp, 'trainval_' + self.stage + '.csv'), self.stage)

		# useing the method from BaseImageListDataset
		self.build_dataset()


	def read_image_by_index(self, ind, phase='train', method='supervised'):
		"""	this function is overrided to support multiple instance learning
		"""
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


		# # preeprocess image and label
		# if phase in ['train', 'trainval']:
		# 	if self.is_flexible_scaling:
		# 		img = self.flexible_scaling(img, min_h=self.output_h, min_w=self.output_w)
		# 	if self.is_random_scaling:
		# 		img = self.random_scaling(img, minval=self.scaling_range[0], maxval=self.scaling_range[1])
		# 	if self.is_random_mirroring:
		# 		img = self.random_mirroring(img)

		# elif phase in ['val']:
		# 	if self.is_flexible_scaling:
		# 		img = self.flexible_scaling(img, min_h=self.output_h, min_w=self.output_w)
		# 	if self.is_random_scaling:
		# 		scale = (self.scaling_range[0] + self.scaling_range[1]) / 2
		# 		img = self.random_scaling(img, minval=scale, maxval=scale)

		# if not self.multiple_categories:
		# 	image_label = self.to_categorical(int(image_label), self.nb_classes)

		# if phase in ['train', 'trainval']:
		# 	img = self.random_crop_and_pad_image(img, size=self.output_shape, center_range=self.crop_range)
		# elif phase in ['val']:
		# 	img = self.random_crop_and_pad_image(img, size=self.output_shape, center_range=[0.5,0.5])

		# img = self.scale_output(img)

		return img, image_label if method == 'supervised' else img

	
	@classmethod
	def find_most_possible_material_bound(cls, img):
		"""
		"""
		img = cv2.resize(img, dsize=(int(img.shape[1]*0.3), int(img.shape[0]*0.3)))
		img = cv2.GaussianBlur(img,(7,7),0)
		edges = cv2.Canny(img, 50, 150, apertureSize = 3)

		lines = cv2.HoughLines(edges,1,np.pi/180, 100) #这里对最后一个参数使用了验型的值

		if lines is not None:
			result = img.copy()

			theta_list = [l[1] for l in lines[:, 0]]
			theta_list = [int(t*180/np.pi) for t in theta_list]
			counter = np.zeros(shape=[180,], dtype=np.int32)
			for t in theta_list:
				counter[t] += 1
			
			max_count_theta = np.argmax(counter)

			# print(max_count_theta)

			lines = lines[:, 0, :]
			lines = [l for i, l in enumerate(lines) if theta_list[i] == max_count_theta]

			for line in lines:
				rho = line[0] #第一个元素是距离rho
				theta = line[1] #第二个元素是角度theta
				print(theta)
				if  (theta < (np.pi/4. )) or (theta > (3.*np.pi/4.0)): #垂直直线
							#该直线与第一行的交点
					pt1 = (int(rho/np.cos(theta)),0)
					#该直线与最后一行的焦点
					pt2 = (int((rho - result.shape[0] * np.sin(theta)) / np.cos(theta)),result.shape[0])
					#绘制一条白线
					cv2.line( result, pt1, pt2, (255), 2)
				else: #水平直线
					# 该直线与第一列的交点
					pt1 = (0,int(rho/np.sin(theta)))
					#该直线与最后一列的交点
					pt2 = (result.shape[1], int((rho-result.shape[1]*np.cos(theta))/np.sin(theta)))
					#绘制一条直线
					cv2.line(result, pt1, pt2, (255), 2)


	def __build_csv_files(self, filepath, stage, train_val_split=0.2):
		""" Build csv files for reading the dataset, 
		
		"""
		if stage == 'stage1':

			data = {
				'train_image_list':[],
				'train_label_list':[],
				'val_image_list':[],
				'val_label_list':[],
			}
			
			def split(img_list, lbl_list, data, k=0.2):
				indices = np.arange(len(img_list))
				nb_train = int(len(img_list) * (1.0-k))
				np.random.shuffle(indices)
				train_indices = indices[:nb_train]
				val_indices = indices[nb_train:]

				data['train_image_list'] += [img_list[i] for i in train_indices]
				data['train_label_list'] += [lbl_list[i] for i in train_indices]
				data['val_image_list'] += [img_list[i] for i in val_indices]
				data['val_label_list'] += [lbl_list[i] for i in val_indices]

			normal_image_list = os.listdir(os.path.join(self._dataset_dir, 'guangdong_round1_train2_20180916', '无瑕疵样本'))
			normal_image_list = [os.path.join('guangdong_round1_train2_20180916', '无瑕疵样本', fn) for fn in normal_image_list if '.jpg' in fn]
			normal_label_list = [0 for img in normal_image_list]
			split(normal_image_list, normal_label_list, data, train_val_split)

			for i in range(1, 11):
				image_path = os.path.join(self._dataset_dir, 'guangdong_round1_train2_20180916', '瑕疵样本', self.class_id2name[i])
				abnormal_image_list = os.listdir(image_path)
				abnormal_image_list = [os.path.join('guangdong_round1_train2_20180916', '瑕疵样本', self.class_id2name[i], fn) for fn in abnormal_image_list if '.jpg' in fn]
				abnormal_label_list = [i for img in abnormal_image_list]
				split(abnormal_image_list, abnormal_label_list, data, train_val_split)

	
			others_image_path = os.path.join(self._dataset_dir, 'guangdong_round1_train2_20180916', '瑕疵样本', self.class_id2name[11])
			others_image_list = []
			others_label_list = []
			for other_cls in os.listdir(others_image_path):
				if os.path.isdir(os.path.join(others_image_path, other_cls)):
					image_path = os.path.join(self._dataset_dir, 'guangdong_round1_train2_20180916', '瑕疵样本', self.class_id2name[11], other_cls)
					image_list = os.listdir(image_path)
					image_list = [os.path.join('guangdong_round1_train2_20180916', '瑕疵样本', self.class_id2name[11], other_cls, fn) for fn in image_list if '.jpg' in fn]
					others_image_list += image_list

			others_label_list = [11 for img in others_image_list]
			split(others_image_list, others_label_list, data, train_val_split)

			with open(self.train_imagelist_fp, 'w', encoding='utf-8') as outfile:
				outfile.write('images,' + ','.join([self.class_id2name[i] for i in range(1, 11)]) + '\n')
				for img, label in zip(data['train_image_list'], data['train_label_list']):
					label_str = ['0' for i in range(11)]
					if label != 0:
						label_str[label - 1] = '1'
					outfile.write(img + ',' + ','.join(label_str) + '\n')

			with open(self.val_imagelist_fp, 'w', encoding='utf-8') as outfile:
				outfile.write('images,' + ','.join([self.class_id2name[i] for i in range(1, 11)]) + '\n')
				for img, label in zip(data['val_image_list'], data['val_label_list']):
					label_str = ['0' for i in range(11)]
					if label != 0:
						label_str[label-1] = '1'
					outfile.write(img + ',' + ','.join(label_str) + '\n')


