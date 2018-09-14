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
from .base_imagelist_dataset import BaseImageListDataset

class Violence(BaseImageListDataset):
	def __init__(self, config):
		super(Violence, self).__init__(config)
		self.name = 'Violence'
		self.time_string = config.get('time string', '201808142212')
		self.nb_classes = 2

		self._dataset_dir = 'F:\\Documents\\new BK\\已整理好--与BK有关的复杂场景-肉眼不好区分的等等情形都作为负面样本'
		if not os.path.exists(self._dataset_dir):
			self._dataset_dir = '/mnt/data03/dataset/new BK/已整理好--与BK有关的复杂场景-肉眼不好区分的等等情形都作为负面样本'
		if not os.path.exists(self._dataset_dir):
			raise Exception("Violence : the dataset dir " + self._dataset_dir + " is not exist")

		self._imagelist_dir = 'F:\\Documents\\new BK\\Proj'
		if not os.path.exists(self._imagelist_dir):
			self._imagelist_dir = '/mnt/data03/dataset/new BK/Proj'
		if not os.path.exists(self._imagelist_dir):
			raise Exception("Violence : the imagelist dir " + self._imagelist_dir + " is not exist")

		self.train_imagelist_fp = os.path.join(self._imagelist_dir, 'train_' + self.time_string + '.txt')
		self.val_imagelist_fp = os.path.join(self._imagelist_dir, 'val_' + self.time_string + '.txt')

		assert(os.path.exists(self.train_imagelist_fp))
		assert(os.path.exists(self.val_imagelist_fp))

		self.build_dataset()

