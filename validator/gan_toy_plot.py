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
import matplotlib.pyplot as plt
from scipy.stats import norm

from .basevalidator import BaseValidator

class GanToyPlot(BaseValidator):
	
	def __init__(self, config):
	
		super(GanToyPlot, self).__init__(config)
		self.assets_dir = config['assets dir']
		self.log_dir = config.get('log dir', 'gan_toy')
		self.log_dir = os.path.join(self.assets_dir, self.log_dir)

		self.z_shape = [int(i) for i in config.get('z shape', [2])]
		self.nb_points = int(config.get('nb points', 100))
		self.plot_range = int(config.get('plot range', 3))
		if not os.path.exists(self.log_dir):
			os.mkdir(self.log_dir)


	def validate(self, model, dataset, sess, step):

		points = np.zeros([self.nb_points, self.nb_points,] + self.z_shape, dtype=np.float32)
		points[:, :, 0] = np.linspace(-self.plot_range, self.plot_range, self.nb_points)[:, None]
		points[:, :, 1] = np.linspace(-self.plot_range, self.plot_range, self.nb_points)[None, :]
		points = points.reshape((-1, 2))

		disc_map = model.discriminate(sess, points)
		disc_map = disc_map.reshape([self.nb_points, self.nb_points]).transpose()


		dataset_indices = dataset.get_image_indices()
		dataset_indices = np.random.choice(dataset_indices, size=100)
		dataset_x = np.array([dataset.read_image_by_index_unsupervised(ind) for ind in dataset_indices])
		indices = np.where(np.logical_and(
				np.logical_and(dataset_x[:, 0] >= -self.plot_range, dataset_x[:, 0] <= self.plot_range),
				np.logical_and(dataset_x[:, 1] >= -self.plot_range, dataset_x[:, 1] <= self.plot_range)
			))[0]
		dataset_x = dataset_x[indices, :]

		random_z = np.random.randn(100, *self.z_shape)
		generated_x = model.generate(sess, random_z)
		indices = np.where(np.logical_and(
				np.logical_and(generated_x[:, 0] >= -self.plot_range, generated_x[:, 0] <= self.plot_range),
				np.logical_and(generated_x[:, 1] >= -self.plot_range, generated_x[:, 1] <= self.plot_range)
			))[0]
		generated_x = generated_x[indices, :]

		plt.clf()
		x = np.linspace(-self.plot_range, self.plot_range, self.nb_points)
		y = np.linspace(-self.plot_range, self.plot_range, self.nb_points)

		plt.contour(x, y, disc_map)
		plt.scatter(dataset_x[:, 0], dataset_x[:, 1], c='orange', marker='+')
		plt.scatter(generated_x[:, 0], generated_x[:, 1], c='green', marker='*')

		plt.savefig(os.path.join(self.log_dir, '%07d.jpg'%step))




