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

class RandomGenerate(BaseValidator):
	
	def __init__(self, config):
	
		super(RandomGenerate, self).__init__(config)
		self.assets_dir = config['assets dir']
		self.log_dir = config.get('log dir', 'generated')
		self.log_dir = os.path.join(self.assets_dir, self.log_dir)


		self.z_shape = list(config['z shape'])
		self.x_shape = list(config['x_shape'])

		self.nb_col_images = int(config.get('nb col', 8))
		self.nb_row_images = int(config.get('nb row', 8))

		self.config = config
		if not os.path.exists(self.log_dir):
			os.mkdir(self.log_dir)

		# self.watch_variable = config.get('watch variable', 'pred')

	def validate(self, model, dataset, sess, step):

		batch_size = self.nb_col_images * self.nb_row_images
		batch_z = np.random.randn([batch_size, ] + self.z_shape)
		batch_x = model.generate(sess, batch_z)


		fig, axes = plt.subplots(nrows=self.nb_row_images, ncols=self.nb_col_images, figsize=(4, 4))
		fig.subplots_adjust(hspace=0.1, wspace=0.1)
		for ind, ax in enumerate(axes.flat):
			ax.imshow(batch_x[ind])

		plt.tight_layout()
		plt.savefig(os.path.join(self.log_dir, '%07d.png'%step))

		return None


		# if len(self.x_shape) == 3:
		# 	batch_x = np.permute(batch_x, (0, 2, 1, 3, 4))
		# 	batch_x = batch_x.reshape([self.nb_row_images * self.x_shape[0], self.nb_col_images * self.x_shape[1], self.x_shape[2]])
		# elif len(self.x_shape) == 2:
		# 	batch_x = np.permute(batch_x, (0, 2, 1, 3))
		# 	batch_x = batch_x.reshape([self.nb_row_images * self.])





		# for ind, batch_x, batch_y in dataset.iter_test_images():
		# 	if self.watch_variable == 'pred':
		# 		y_pred = model.predict(sess, batch_y)

		# 		x_pos_array.append(y_pred[:, self.x_dim])
		# 		y_pos_array.append(y_pred[:, self.y_dim])
		# 		label_array.append(np.argmax(batch_y, axis=1))

		# 	elif self.watch_variable == 'hidden dist':
		# 		z_mean, z_log_var = model.hidden_variable_distribution(sess, batch_x)

		# 		x_pos_array.append(
		# 			np.concatenate([	z_mean[:, self.x_dim:self.x_dim+1], 
		# 								np.exp(z_log_var[:, self.x_dim:self.x_dim+1]) ], axis=1)
		# 		)
		# 		y_pos_array.append(
		# 			np.concatenate([	z_mean[:, self.y_dim:self.y_dim+1], 
		# 								np.exp(z_log_var[:, self.y_dim:self.y_dim+1]) ], axis=1)
		# 		)
		# 		label_array.append(np.argmax(batch_y, axis=1))
		# 	else:
		# 		raise Exception("None watch variable named " + self.watch_variable)

		# x_pos_array = np.concatenate(x_pos_array, axis=0)
		# y_pos_array = np.concatenate(y_pos_array, axis=0)
		# label_array = np.concatenate(label_array, axis=0)

		# if len(x_pos_array.shape) == 2:
		# 	for i in range(x_pos_array.shape[1]):
		# 		plt.figure(figsize=(6, 6))
		# 		plt.clf()
		# 		plt.scatter(x_pos_array[:, i], y_pos_array[:, i], c=label_array)
		# 		plt.colorbar()
		# 		plt.savefig(os.path.join(self.log_dir, '%07d_%d.png'%(step, i)))
		# else:
		# 	plt.figure(figsize=(6, 6))
		# 	plt.clf()
		# 	plt.scatter(x_pos_array, y_pos_array, c=label_array)
		# 	plt.colorbar()
		# 	plt.savefig(os.path.join(self.log_dir, '%07d.png'%step))

		# return None


