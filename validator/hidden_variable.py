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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy.stats import norm

from .base_validator import BaseValidator


class HiddenVariable(BaseValidator):
	"""
	"""
	
	def __init__(self, config):
		"""plot the function x = f(z) to a grid
		"""
		super(HiddenVariable, self).__init__(config)

		self.z_dim = int(config['z_dim'])	 # the dim of z
		self.x_shape = config['x shape']	 # the shape of x
		self.assets_dir = config['assets dir']
		self.dim_x = config.get('dim x', 0)  # the x axis of plot grid is z[0]
		self.dim_y = config.get('dim y', 1)	 # the y axis of plot grid is z[1]
		self.scalar_range = config.get('scalar range', [0, 1])

		# if nb_classes == 0, then we plot the function x=f(z)
		# else if nb_classes != 0, then we plot the function x=f(z,y), where nb_classes is the dim of y
		self.nb_classes = config.get('nb classes', 0)

		assert (len(self.x_shape) == 2 or (len(self.x_shape) == 3 and self.x_shape[2] in [1,3]))
		assert(self.dim_x < self.z_dim)
		assert(self.dim_y < self.z_dim)

		self.log_dir = config.get('log dir', 'hidden')
		self.log_dir = os.path.join(self.assets_dir, self.log_dir)
		if not os.path.exists(self.log_dir):
			os.mkdir(self.log_dir)

		self.generate_method = config.get('generate_method', 'normppf')
		if self.generate_method == 'normppf':
			self.nb_samples = config.get('num_samples', 15)
		else:
			raise NotImplementedError

	def validate(self, model, dataset, sess, step):

		n = self.nb_samples  # figure with 15x15 digits

		figure_list = []

		if self.generate_method == 'normppf':

			# 用正态分布的分位数来构建隐变量对
			grid_x = norm.ppf(np.linspace(0.01, 0.99, n))
			grid_y = norm.ppf(np.linspace(0.01, 0.99, n))

		if self.nb_classes == 0:
			rounds = 1						# plot x=f(z)
			plt.figure(figsize=(10, 10))

		else:
			rounds = self.nb_classes		# for each class y, plot x=f(z,y)
			plt.figure(figsize=(10 * self.nb_classes, 10))

		# for each class
		for r in range(rounds):
			# for each x position
			for i, xi in enumerate(grid_x):
				z_sample = np.zeros((n, self.z_dim))
				z_sample[:, self.dim_x] = xi
				z_sample[:, self.dim_y] = grid_y

				if self.nb_classes == 0:
					x_generated = model.generate(sess, z_sample)  	# x = f(z)
				else:
					cond = np.zeros(self.nb_classes)
					cond[r] = 1.0
					x_generated = model.generate(sess, z_sample, condition=cond) # x = f(z,y)

				figure_list.append(x_generated)

			figure = np.array(figure_list)  # [nx, ny] + self.x_shape
			figure_list = []

			# reshape to the correct image array shape
			if len(self.x_shape) == 2:
				figure = figure.transpose([1, 2, 0, 3]).reshape((n*self.x_shape[0], n*self.x_shape[1]))
			else:
				figure = figure.transpose([1, 2, 0, 3, 4]).reshape((n*self.x_shape[0], n*self.x_shape[1], self.x_shape[2]))

			# normalize to [0, 1]
			if self.scalar_range != [0, 1]:
				figure = (figure - self.scalar_range[0]) / (self.scalar_range[1] - self.scalar_range[0])

			if self.nb_classes != 0:
				plt.subplot(1, self.nb_classes, r+1)

			if len(self.x_shape) == 2:
				plt.imshow(figure, cmap='Greys_r')
			elif self.x_shape[2] == 1:
				plt.imshow(figure[:, :, 0], cmap='Greys_r')
			else:
				plt.imshow(figure)


		plt.savefig(os.path.join(self.log_dir, '%07d.png' % step))

		return None
