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
import cv2
from scipy.stats import norm

from .basevalidator import BaseValidator

class RandomGenerate(BaseValidator):
	
	def __init__(self, config):
	
		super(RandomGenerate, self).__init__(config)

		self.assets_dir = config['assets dir']
		self.log_dir = config.get('log dir', 'generated')
		self.log_dir = os.path.join(self.assets_dir, self.log_dir)

		self.z_shape = list(config['z shape'])
		self.x_shape = list(config['x shape'])

		self.nb_col_images = int(config.get('nb col', 8))
		self.nb_row_images = int(config.get('nb row', 8))

		self.config = config
		if not os.path.exists(self.log_dir):
			os.mkdir(self.log_dir)

	def validate(self, model, dataset, sess, step):

		batch_size = self.nb_col_images * self.nb_row_images
		batch_z = np.random.randn(*([batch_size, ] + self.z_shape))
		batch_x = model.generate(sess, batch_z)
		fig, axes = plt.subplots(nrows=self.nb_row_images, ncols=self.nb_col_images, figsize=(8, 8),
								subplot_kw={'xticks': [], 'yticks': []})
		fig.subplots_adjust(hspace=0.01, wspace=0.01)
		for ind, ax in enumerate(axes.flat):
			img = batch_x[ind]
			if len(img.shape) == 3 and img.shape[2] == 1:
				img = cv2.merge([img, img, img])
			elif len(img.shape) == 2:
				img = img.reshape(list(img.shape) + [1,])
				img = cv2.merge([img, img, img])
			elif len(img.shape) == 3 and img.shape[2] == 3:
				img = img
			else:
				raise ValueError('Unsupport Shape : ' + str(img.shape))
			ax.imshow(img, vmin=0.0, vmax=1.0)

		plt.tight_layout()
		plt.savefig(os.path.join(self.log_dir, '%07d.png'%step))
		return None
