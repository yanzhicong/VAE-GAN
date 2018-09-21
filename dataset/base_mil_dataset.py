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

class BaseMILDataset(BaseDataset):
	""" The base dataset class for supporting multiple-instance learning.

	"""
	def __init__(self, config):
		
		super(BaseMILDataset, self).__init__(config)
		self.config = config

	def crop_image_to_bag(self, img, output_shape, *, 
			max_nb_crops=None, nb_crops=None, nb_crop_col=None, nb_crop_row=None):

		output_h, output_w = output_shape[0:2]
		image_h, image_w = img.shape[0:2]

		nb_col = int(np.floor(image_w / output_w))
		nb_row = int(np.floor(image_h / output_h))

		step_h = float(image_h) / float(nb_row)
		step_w = float(image_w) / float(nb_col)

		img_bag = []
		img_bbox = []

		for i in range(nb_row):
			for j in range(nb_col):

				x1 = int(j * step_w)
				y1 = int(i * step_h)
				x2 = int(j * step_w) + output_w
				y2 = int(i * step_h) + output_h

				crop_image = self.crop_and_pad_image(img, [x1,y1,x2,y2])

				if crop_image.shape[0] != output_h or crop_image.shape[1] != output_w or crop_image.shape[2] != output_shape[2]:
					print("Warning : Crop out image shape " + str(crop_image.shape) )

				img_bbox.append([x1,y1,x2,y2])
				img_bag.append(crop_image)
	
		return img_bag, img_bbox, nb_col, nb_row
