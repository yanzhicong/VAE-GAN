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
import math

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
			self._dataset_dir = "/mnt/data03/tianchi/GuangDongIndustrialBigData"
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
		self.test_imagelist_fp = os.path.join(self.extra_fp, 'test_' + self.stage + '.csv')

		if not os.path.exists(self.train_imagelist_fp) or not os.path.exists(self.val_imagelist_fp):
			self.__build_train_csv_files(self.stage)
		if not os.path.exists(self.test_imagelist_fp):
			self.__build_test_csv_files(self.stage)

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
		
		print(image_fp)

		try:
			img = io.imread(image_fp)
			img = self._image_correct(img, image_fp)
		except Exception as e:
			if self.show_warning:
				print('Warning : read image error : ' + str(e))
			return None, None if method == 'supervised' else None

		if img is None:
			return None, None if method == 'supervised' else None 

		area = self.find_most_possible_metal_area(img, show_warning=self.show_warning)
		area_img = self.crop_and_reshape_image_area(img, area)

		area_img = area_img.astype(np.float32) / 255.0
		self.scale_output(area_img)

		image_bag, image_bbox, row, col = self.crop_image_to_bag(area_img, self.output_shape)

		if method == 'supervised':
			return image_bag, image_label 
		else:
			return image_bag

	
	@classmethod
	def find_most_possible_metal_area(cls, img, show_warning=True, minimal_width=500, detect_shrink=0.2, blur_ksize=(19,9)):
		""" Find the most possible area of metal, the area is convex polygon with four corners,
		may be not a parallelogram, 

		Arguments:
			minimal_width
			detect_shrink
		Output :
			[[x1,y1],[x2,y2],[x3,y3],[x4,y4]] : the four coordinate of area corners, at the position of 
												left-top, right-top, right-bottom, left-bottom, they all in the first or last row of image
		""" 
		h = img.shape[0]
		w = img.shape[1]
		default_area = [(0,0), (w,0), (w,h), (0,h)]
		minimal_width = minimal_width * detect_shrink

		img = cv2.resize(img, dsize=(int(img.shape[1]*detect_shrink), int(img.shape[0]*detect_shrink)))
		img = cv2.GaussianBlur(img,blur_ksize,0)
		edges = cv2.Canny(img, 50, 150, apertureSize = 3)
		lines = cv2.HoughLines(edges, 1, np.pi/180, 100)

		if lines is not None:
			result = img.copy()

			lines = lines[:, 0, :]

			lines = [l for l in lines if l[1] > (30.0 / 180.0 * np.pi) and l[1] < (150.0 / 180.0 * np.pi)]

			theta_list = [l[1] for l in lines]
			theta_list = [int(t*180/np.pi) for t in theta_list]
			counter = np.zeros(shape=[180,], dtype=np.int32)
			for t in theta_list:
				counter[t] += 1
			max_count_theta = np.argmax(counter)
			lines = [l for i, l in enumerate(lines) if theta_list[i] == max_count_theta]
			lines.sort(key=lambda x:x[0])

			remain_lines = []
			for i, l in enumerate(lines):
				if i == 0:
					last_line = l
					remain_lines.append(l)
				else:
					if abs(l[0] - last_line[0]) > minimal_width:
						remain_lines.append(l)
						last_line = l
			lines = remain_lines

			if len(lines) == 0:
				if show_warning:
					print("Guangdong Defect warning : nb lines is 0")
				return default_area
			elif len(lines) == 1:
				if show_warning:
					print("Guangdong Defect warning : only find 1 line  (angle : %f degree)"%(lines[0][1] * 180.0 / np.pi) )

				rho = lines[0][0] / detect_shrink
				theta = lines[0][1]
	
				# this line separate the img to two areas, return the largest area
				
				pt1 = (0, int(rho/np.sin(theta)))
				pt2 = (w, int((rho-w*np.cos(theta))/np.sin(theta)))
				cpt = ((pt1[0]+pt2[0])/2.0, (pt1[1]+pt2[1])/2.0)

				if cpt[1] > (h/2):
					# return the top area
					return [(0,0), (w,0), pt2, pt1]
				else:
					# return the bottom area
					return [pt1, pt2, (w,h), (0,h)]

			else:
				top_line = lines[0]
				bottom_line = lines[-1]

				top_line[0] = top_line[0] / detect_shrink
				bottom_line[0] = bottom_line[0] / detect_shrink

				pt1 = (0, int(top_line[0]/np.sin(top_line[1])))
				pt2 = (w, int((top_line[0]-w*np.cos(top_line[1]))/np.sin(top_line[1])))

				pt3 = (0, int(bottom_line[0]/np.sin(bottom_line[1])))
				pt4 = (w, int((bottom_line[0]-w*np.cos(bottom_line[1]))/np.sin(bottom_line[1])))

				return [pt1, pt2, pt4, pt3]

		else :
			if show_warning:
				print("Guangdong Defect warning : no line found")

			return default_area
	
	@classmethod
	def crop_and_reshape_image_area(cls, img, area, fixed_height=256, margin_ratio=0.2):
		""" for a quadrangle area in the image, crop and transform it to a new image
		"""
		point0_, point1_, point2_, point3_ = area

		point0_ = np.array(point0_)
		point1_ = np.array(point1_)
		point2_ = np.array(point2_)
		point3_ = np.array(point3_)

		# make a margin to the top and bottom of the area
		point0 = point0_ - (point3_ - point0_) * (margin_ratio / 2.0)
		point3 = point3_ + (point3_ - point0_) * (margin_ratio / 2.0)

		point1 = point1_ - (point2_ - point1_) * (margin_ratio / 2.0)
		point2 = point2_ + (point2_ - point1_) * (margin_ratio / 2.0)

		area_h = abs(point3[1] - point0[1])
		area_w = abs(point1[0] - point0[0])

		output_h = int(fixed_height)
		output_w = int(fixed_height * area_w / area_h)

		dst_point0 = [0,0]
		dst_point1 = [output_w, 0]
		dst_point3 = [0, output_h]
		src_tri = np.array([point0, point1, point3]).astype(np.float32)
		dst_tri = np.array([dst_point0, dst_point1, dst_point3]).astype(np.float32)

		trans_affine_matrix = cv2.getAffineTransform(src_tri, dst_tri)

		affine_img = cv2.warpAffine(img, trans_affine_matrix, (output_w, output_h))

		return affine_img


	def __build_train_csv_files(self, stage, train_val_split=0.2):
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


	def __build_test_csv_files(self, stage):
		if stage == 'stage1':
			image_list = os.listdir(os.path.join(self._dataset_dir, 'guangdong_round1_test_a_20180916'))
			image_list = [fn for fn in image_list if '.jpg' in fn]
			ind_list = np.array([int(fn.split('.')[0]) for fn in image_list])
			sort_ind = np.argsort(ind_list)
			image_list = [image_list[i] for i in sort_ind]
			
			image_list = [os.path.join('guangdong_round1_test_a_20180916', fn) for fn in image_list if '.jpg' in fn]
			
			# ind_list = []

			with open(self.test_imagelist_fp, 'w', encoding='utf-8') as outfile:
				outfile.write('images\n')
				for img in image_list:
					outfile.write(img + '\n')


	def write_submission_file(self, filepath, probs):
		test_indices = self.get_image_indices('test', 'unsupervised')
		img_fp_list = ['%d.jpg'%i for i in test_indices]
		
		with open(filepath, 'w') as outfile:
			for img_fp, pred in zip(img_fp_list, probs):

				max_class = np.argmax(pred)
				max_class_id = max_class + 1
				# max_class_name = self.class_id2name[max_class_id]
				max_class_prob = pred[max_class]

				if max_class_prob > 0.5:
					outfile.write(img_fp + ',defect%d\n'%max_class_id)
				else:
					outfile.write(img_fp + ',norm\n')
