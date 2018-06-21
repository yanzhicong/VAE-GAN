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



class ScatterPlotValidator(object):
	
	def __init__(self, config):
	
		self.assets_dir = config['assets dir']
		self.log_dir = config.get('log dir', 'test')
		self.log_dir = os.path.join(self.assets_dir, self.log_dir)

		self.x_dim = int(config.get('x_dim', 0))
		self.y_dim = int(config.get('y_dim', 1))

		if not os.path.exists(self.log_dir):
			os.mkdir(self.log_dir)

		self.watch_variable = config.get('watch_variable', 'pred')


	def validate(self, model, dataset, sess, step):

		x_pos_array = []
		y_pos_array = []
		label_array = []

		for ind, batch_x, batch_y in dataset.iter_test_images():

			if self.watch_variable == 'pred':
				y_pred = model.predict(sess, batch_y)

				x_pos_array.append(y_pred[:, self.x_dim])
				y_pos_array.append(y_pred[:, self.y_dim])
				label_array.append(batch_y)

			elif self.watch_variable == 'hidden_dist':
				z_mean, z_log_var = model.hidden_distribution(sess, batch_x)

				x_pos_array.append(
					np.concatenate([
											z_mean[:, self.x_dim:self.x_dim+1], 
									np.exp(z_log_var[:, self.x_dim:self.x_dim+1])
									], axis=1)
				)
				y_pos_array.append(
					np.concatenate([
											z_mean[:, self.y_dim:self.y_dim+1], 
										np.exp(z_log_var[:, self.y_dim:self.y_dim+1])
									], axis=1)
				)
				label_array.append(batch_y)
			else:
				raise Exception("None watch variable named " + self.watch_variable)

		x_pos_array = np.concatenate(x_pos_array, axis=0)
		y_pos_array = np.concatenate(y_pos_array, axis=0)
		label_array = np.concatenate(label_array, axis=0)


		if len(x_pos_array.shape) == 2:
			for i in range(x_pos_array.shape[1]):
				plt.figure(figsize=(6, 6))
				plt.clf()
				plt.scatter(x_pos_array[:, i], y_pos_array[:, i], c=label_array)
				plt.colorbar()
				plt.savefig(os.path.join(self.log_dir, '%07d_%d.png'%(step, i)))

		else:
			plt.figure(figsize=(6, 6))
			plt.clf()
			plt.scatter(x_pos_array, y_pos_array, c=label_array)
			plt.colorbar()
			plt.savefig(os.path.join(self.log_dir, '%07d.png'%step))
		pass


