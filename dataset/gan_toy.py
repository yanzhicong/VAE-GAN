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
# import matplotlib.pyplot as plt
import pickle

from .base_dataset import BaseDataset



class GanToy(BaseDataset):

	def __init__(self, config):
		
		super(GanToy, self).__init__(config)
		self.config = config

		self.name = 'gan_toy'
		self.batch_size = int(config.get('batch_size', 128))
		self.variance = float(config.get('variance', 0.02))
		self.dataset = config.get('dataset', '8gaussians')

		if self.dataset == '8gaussians':
			scale = 2.0
			centers = [
				(1, 0),
				(-1, 0),
				(0, 1),
				(0, -1),
				(  1.0/np.sqrt(2),  1.0/np.sqrt(2)),
				( -1.0/np.sqrt(2),  1.0/np.sqrt(2)),
				(  1.0/np.sqrt(2), -1.0/np.sqrt(2)),
				( -1.0/np.sqrt(2), -1.0/np.sqrt(2))
			]
			self.centers = np.array([ (scale*x, scale*y) for x, y in centers])
			self.centers = np.tile(self.centers, (1000, 1))
			self.centers = self.centers + np.random.randn(*self.centers.shape) * self.variance

		elif self.dataset == '25gaussians':
			# for i in range(25):
			centers = [(x, y) for x in range(-2, 3) 
							  for y in range(-2, 3)]
			self.centers = np.tile(centers, (1000, 1))
			self.centers = self.centers + np.random.randn(*self.centers.shape) * self.variance


	def get_image_indices(self, phase=None, method=None):
		indices = np.arange(self.centers.shape[0])
		if self.shuffle_train:
			np.random.shuffle(indices)
		return indices

	def read_image_by_index(self, ind, phase=None, method='unsupervised'):
		assert(method in ['unsupervised'])
		ind = ind % self.centers.shape[0]
		return self.centers[ind]	



	def iter_train_images(self, method):
		assert method == 'unsupervised'

		indices = np.array(self.get_image_indices())
		centers = np.array(self.centers)
		for i in range(int(len(indices) // self.batch_size)):
			batch_ind = indices[i*self.batch_size : (i+1)*self.batch_size]
			batch_x = centers[batch_ind, :]
			yield i, batch_x
