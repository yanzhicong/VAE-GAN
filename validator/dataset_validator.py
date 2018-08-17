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

from utils.metric import get_metric

from .basevalidator import BaseValidator

class DatasetValidator(BaseValidator):

	def __init__(self, config):


		super(DatasetValidator, self).__init__(config)

		self.config = config
		self.nb_samples = config.get('num_samples', 5000)
		self.batch_size = config.get('batch_size', 128)
		self.metric = config.get('metric', 'accuracy')
		self.metric_type = config.get('metric type', 'top1')
		self.assets_dir = config['assets dir']

		if self.metric == 'accuracy':
			self.has_summary = True


	def build_summary(self, model):

		if self.metric == 'accuracy':
			self.label = tf.placeholder(tf.float32, shape=[None, model.nb_classes],
							name='test_label')
			self.predict = tf.placeholder(tf.float32, shape=[None, model.nb_classes],
							name='test_predict')
			self.accuracy = get_metric(self.metric, self.metric_type, 
						{'probs' : self.predict, 'labels' : self.label, 'decay' : 1})			

			self.summary_list = []
			self.summary_list.append(tf.summary.scalar('test acc ' + self.metric_type, self.accuracy))

			self.log_filepath = os.path.join(self.assets_dir, 'test_dataset_' + self.metric + "_" + self.metric_type + '.csv')

			if not self.config.get('continue train', False):
				with open(self.log_filepath, 'w') as logfile:
					logfile.write('step,' + self.metric_type + '\n')
		
		self.summary = tf.summary.merge(self.summary_list)


	def validate(self, model, dataset, sess, step):
		label_list = []
		pred_list = []

		indices = dataset.get_image_indices(phase='val', method='supervised')
		nb_samples = np.minimum(len(indices), self.nb_samples)
		indices = np.random.choice(indices, size=nb_samples, replace=False)

		batch_x = []
		batch_y = []
		for i, ind in enumerate(indices):
			img, label = dataset.read_image_by_index(ind, phase='val', method='supervised')
			batch_x.append(img)
			batch_y.append(label)

			if (i+1) % self.batch_size == 0:
				batch_p = model.predict(sess, np.array(batch_x))
				label_list.append(np.array(batch_y))
				pred_list.append(np.array(batch_p))
				batch_x = []
				batch_y = []

		label_list = np.concatenate(label_list, axis=0)
		pred_list = np.concatenate(pred_list, axis=0)

		if self.metric == 'accuracy' : 
			feed_dict = {
				self.label : label_list,
				self.predict : pred_list,
			}
			acc, summary = sess.run([self.accuracy, self.summary], feed_dict=feed_dict)
			with open(self.log_filepath, 'a') as logfile:
				logfile.write('%d,%f\n'%(step, acc))
			print('test acc %f'%acc)
			summary = sess.run([self.summary], feed_dict=feed_dict)[0]

		return summary


