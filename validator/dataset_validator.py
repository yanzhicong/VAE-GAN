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
import queue
import threading
import time

sys.path.append('.')
sys.path.append('../')

import tensorflow as tf
import tensorflow.contrib.layers as tcl

from utils.metric import get_metric

from .base_validator import BaseValidator


class DatasetValidator(BaseValidator):
	""" measure the model performance with val dataset.
	the metric and metric type can be configured

	the model must implement the [predict] function

	Optional parameters in @params.config:
		'nb samples' : 
		'batch_size' : 
		'metric', 'metric type' : 
	"""

	def __init__(self, config):
		super(DatasetValidator, self).__init__(config)

		self.config = config
		self.nb_samples = self.config.get('nb samples', 5000)
		self.batch_size = self.config.get('batch_size', 128)

		self.metric = self.config.get('metric', 'accuracy')
		self.metric_type = self.config.get('metric type', 'top1')

		self.multiple_instance_learning = self.config.get('mil', False)
		self.assets_dir = self.config['assets dir']

		self.buffer_depth = self.config.get('buffer depth', 50)

		if self.metric == 'accuracy':
			self.has_summary = True


	def build_summary(self, model):

		if self.metric == 'accuracy':
			self.label = tf.placeholder(tf.float32, shape=[None, model.nb_classes],
							name='test_label')
			self.predict = tf.placeholder(tf.float32, shape=[None, model.nb_classes],
							name='test_predict')
			self.accuracy = get_metric(self.metric, self.metric_type, {'probs' : self.predict, 'labels' : self.label, 'decay' : 1})

			self.summary_list = []
			self.summary_list.append(tf.summary.scalar('test_acc_' + self.metric_type, self.accuracy))

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

		self.t_should_stop = False
		t, data_queue = self.parallel_data_reading(dataset, indices, 'val', 'supervised', self.batch_size*self.buffer_depth)

		if self.multiple_instance_learning:

			while not self.t_should_stop or not data_queue.empty():
				if not data_queue.empty():
					img_bag, label = data_queue.get()
					img_bag_p = model.predict(sess, np.array(img_bag))
					label_list.append(label)
					pred_list.append(img_bag_p)

		else:
				
			batch_x = []
			batch_y = []

			while not self.t_should_stop or not data_queue.empty():
				if not data_queue.empty():
					img, label = data_queue.get()
					batch_x.append(img)
					batch_y.append(label)

				if len(batch_x) == self.batch_size:
					batch_p = model.predict(sess, np.array(batch_x))
					label_list.append(np.array(batch_y))
					pred_list.append(np.array(batch_p))
					batch_x = []
					batch_y = []

			if len(batch_x) > 0:
				batch_p = model.predict(sess, np.array(batch_x))
				label_list.append(np.array(batch_y))
				pred_list.append(np.array(batch_p))

		t.join()


		if self.multiple_instance_learning:
			label_list = np.array(label_list)
			pred_list = np.array(pred_list)

			# print(pred_list)
			# print(label_list)
			# print(pred_list[0])
			# print(label_list[0])
			# print(label_list.shape)
			# print(pred_list.shape)
		else:
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


