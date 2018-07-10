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


from utils.optimizer import get_optimizer
from utils.optimizer import get_optimizer_by_config
from utils.loss import get_loss
from utils.metric import get_metric

from .basemodel import BaseModel


class Classification(BaseModel):

	def __init__(self, config,
		**kwargs
	):
		super(Classification, self).__init__(config, **kwargs)

		self.input_shape = config['input shape']
		# self.z_dim = config['z_dim']
		self.nb_classes = config['nb_classes']
		self.config = config
		self.build_model()
		if self.is_summary:
			self.build_summary()

	def build_model(self):

		self.config['classifier params']['name'] = 'classifier'
		self.classifier = get_classifier(self.config['classifier'], self.config['classifier params'], self.is_training)

		# for training
		self.x = tf.placeholder(tf.float32, shape=[None,]  + self.input_shape, name='x_input')
		self.label = tf.placeholder(tf.float32, shape=[None, self.nb_classes], name='label')

		self.logits, self.end_points = self.classifier.features(self.x)

		# print(self.logits.get_shape())
		self.loss = get_loss('classification', self.config['classification loss'], 
						{'logits' : self.logits, 'labels' : self.label})
		self.train_acc = get_metric('accuracy', 'top1', 
						{'logits': self.logits, 'labels':self.label})

		# for testing
		self.test_x = tf.placeholder(tf.float32, shape=[None,]  + self.input_shape, name='test_x_input')
		self.test_logits = self.classifier(self.test_x)
		self.test_y = tf.nn.softmax(self.test_logits)
		
		print('vars')
		for var in self.classifier.vars:
			print(var.name, ' --> ', var.get_shape())

		print('vars_to_save_and_restore')
		for var in self.classifier.vars_to_save_and_restore:
			print(var.name, ' --> ', var.get_shape())

		(self.train_op, 
			self.learning_rate, 
				self.global_step) = get_optimizer_by_config(self.config['optimizer'], self.config['optimizer params'],
														target=self.loss, variables=self.classifier.vars)

		# model saver
		self.saver = tf.train.Saver(self.classifier.vars_to_save_and_restore + [self.global_step,])


	def build_summary(self):
		# summary scalars are logged per step
		sum_list = []
		sum_list.append(tf.summary.scalar('lr', self.learning_rate))
		sum_list.append(tf.summary.scalar('train loss', self.loss))
		sum_list.append(tf.summary.scalar('train acc', self.train_acc))
		self.sum_scalar = tf.summary.merge(sum_list)

		for key, var in self.end_points.items():
			sum_list.append(tf.summary.histogram('netout/' + key, var))

		self.sum_scalar2 = tf.summary.merge(sum_list)

		# summary hists are logged by calling self.summary()
		sum_list = [tf.summary.histogram(var.name, var) for var in self.classifier.vars_to_save_and_restore]
		self.sum_hist = tf.summary.merge(sum_list)

	'''
		train operations
	'''
	def train_on_batch_supervised(self, sess, x_batch, y_batch):
		feed_dict = {
			self.x : x_batch,
			self.label : y_batch,
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
		if self.is_summary:
			sum = sess.run(self.sum_hist)
			return sum
		else:
			return None
