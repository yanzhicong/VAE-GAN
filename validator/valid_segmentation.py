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
import sys
import numpy as np

sys.path.append('.')
sys.path.append('../')

import tensorflow as tf
import tensorflow.contrib.layers as tcl
import matplotlib.pyplot as plt

from utils.metric import get_metric

from .base_validator import BaseValidator

class ValidSegmentation(BaseValidator):

	def __init__(self, config):
		super(ValidSegmentation, self).__init__(config)
		self.config = config
		self.log_dir = config.get('log dir', 'val_seg')
		self.log_dir = os.path.join(self.assets_dir, self.log_dir)
		if not os.path.exists(self.log_dir):
			os.makedirs(self.log_dir)
		self.has_summary = True

	def build_summary(self, model):

		self.img_ph = tf.placeholder(tf.float32, shape=[None, None, None, 3])
		self.colored_mask_ph = tf.placeholder(tf.float32, shape=[None, None, None, 3])
		self.colored_pred_ph = tf.placeholder(tf.float32, shape=[None, None, None, 3])

		sum_list = []
		sum_list.append(tf.summary.image('image', self.img_ph))
		sum_list.append(tf.summary.image('mask', self.colored_mask_ph))
		sum_list.append(tf.summary.image('pred', self.colored_pred_ph))
		self.image_summary = tf.summary.merge(sum_list)


	def validate(self, model, dataset, sess, step):

		nb_samples = 5

		indices = dataset.get_image_indices(phase='val', method='superviesd')
		indices = np.random.choice(indices, size=nb_samples)
		summary_list = []

		img_list = []
		mask_list = []
		pred_list = []

		for ind in indices:
			img, mask = dataset.read_image_by_index(ind, phase='val', method='supervised')

			if img is not None:
				h = img.shape[0]
				w = img.shape[1]
				h = int(h // 16 * 16)
				w = int(w // 16 * 16)
				img = img[0:h, 0:w]
				mask = mask[0:h, 0:w]
				batch_x = img.reshape([1,] + list(img.shape))
				mask = mask.reshape([1, ] + list(mask.shape))
				pred = model.predict(sess, batch_x)

				colored_mask = dataset.mask_colormap_decode(
										dataset.from_categorical(mask), dataset.color_map)
				colored_pred = dataset.mask_colormap_decode(
										dataset.from_categorical(pred), dataset.color_map)
				feed_dict = {
					self.img_ph : batch_x,
					self.colored_mask_ph : colored_mask,
					self.colored_pred_ph : colored_pred,
				}

				img_list.append(batch_x[0])
				mask_list.append(colored_mask[0])
				pred_list.append(colored_pred[0])
				summary = sess.run(self.image_summary, feed_dict=feed_dict)
				summary_list.append([step, summary])

		# plt.subplot(5, 3)

		# plt.figure(figsize=(6, 6))

		fig, axes = plt.subplots(nrows=nb_samples,
								 ncols=3, figsize=(6, 10),
						subplot_kw={'xticks': [], 'yticks': []})
		fig.subplots_adjust(hspace=0.01, wspace=0.01)
		for ind, ax in enumerate(axes.flat):

			if ind % 3 == 0:
				ax.imshow(img_list[ind // 3])
			elif ind % 3 == 1:
				ax.imshow(mask_list[ind // 3])
			elif ind % 3 == 2:
				ax.imshow(pred_list[ind // 3])


		plt.tight_layout()
		plt.savefig(os.path.join(self.log_dir, '%07d.png'%step))

		return summary_list


