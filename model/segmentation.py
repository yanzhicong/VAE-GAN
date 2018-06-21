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
sys.path.append('.')
sys.path.append("../")

import tensorflow as tf
import tensorflow.contrib.layers as tcl
import numpy as np


from classifier.classifier import get_classifier


from netutils.optimizer import get_optimizer
from netutils.optimizer import get_optimizer_by_config
from netutils.loss import get_loss
from netutils.metric import get_metric

from .base_model import BaseModel


class Segmentation(BaseModel):

	def __init__(self, config):
		super(Segmentation, self).__init__(config)

		self.input_shape = config['input shape']
		self.mask_shape = config['mask shape']
		self.nb_classes = config['nb classes']
		self.config = config
		self.build_model()
		self.build_summary()

	def build_model(self):

		self.config['classifier params']['name'] = 'classifier'
		self.config['classifier params']["output dims"] = self.nb_classes

		self.classifier = get_classifier(self.config['classifier'], self.config['classifier params'], self.is_training)

		# for training
		self.x = tf.placeholder(tf.float32, shape=[None,]  + self.input_shape, name='x_input')
		self.mask = tf.placeholder(tf.float32, shape=[None, ] + self.mask_shape, name='mask')

		self.logits, self.end_points = self.classifier.features(self.x)

		self.loss = get_loss('segmentation', self.config['segmentation loss'], 
						{'logits' : self.logits, 'mask' : self.mask})

		self.train_miou = get_metric('segmentation', 'miou', 
						{'logits': self.logits, 'mask':self.mask, 'nb_classes':self.nb_classes})

		# for testing
		self.test_x = tf.placeholder(tf.float32, shape=[None, None, None, self.input_shape[-1]], name='test_x_input')
		self.test_logits = self.classifier(self.test_x)
		self.test_y = tf.nn.softmax(self.test_logits)
		
		(self.train_op, 
			self.learning_rate, 
				self.global_step) = get_optimizer_by_config(self.config['optimizer'], self.config['optimizer params'],
														target=self.loss, variables=self.classifier.vars)

		# model saver
		self.saver = tf.train.Saver(self.classifier.store_vars + [self.global_step,])


	def build_summary(self):
		# summary scalars are logged per step
		if self.has_summary:

			sum_list = []
			sum_list.append(tf.summary.scalar('lr', self.learning_rate))
			sum_list.append(tf.summary.scalar('train loss', self.loss))
			sum_list.append(tf.summary.scalar('train miou', self.train_miou))
			self.sum_scalar = tf.summary.merge(sum_list)

			for key, var in self.end_points.items():
				sum_list.append(tf.summary.histogram('netout/' + key, var))
			self.sum_scalar2 = tf.summary.merge(sum_list)

			# summary hists are logged by calling self.summary()
			sum_list = [tf.summary.histogram(var.name, var) for var in self.classifier.store_vars]
			self.histogram_summary = tf.summary.merge(sum_list)

		else:
			self.sum_scalar = None
			self.sum_scalar2 = None
			self.histogram_summary = None

	'''
		train operations
	'''
	def train_on_batch_supervised(self, sess, x_batch, y_batch):
		feed_dict = {
			self.x : x_batch,
			self.mask : y_batch,
			self.is_training : True
		}

		step = sess.run([self.global_step])[0]

		if step % 100 == 0:
			return self.train(sess, feed_dict, 
							update_op=self.train_op,
							learning_rate=self.learning_rate,
							loss=self.loss,
							summary=self.sum_scalar2)
		else:
			return self.train(sess, feed_dict, 
							update_op=self.train_op,
							learning_rate=self.learning_rate,
							loss=self.loss,
							summary=self.sum_scalar)


	def train_on_batch_unsupervised(self, sess, x_batch):
		raise NotImplementedError

	'''
		test operations
	'''
	def predict(self, sess, x_batch):
		feed_dict = {
			self.test_x : x_batch,
			self.is_training : False
		}
		y = sess.run([self.test_y], feed_dict = feed_dict)[0]
		return y

	'''
		summary operations

	'''
	def summary(self, sess):
		if self.has_summary:
			summ = sess.run(self.histogram_summary)
			return summ
		else:
			return None
