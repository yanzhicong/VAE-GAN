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

from utils.loss import get_loss
from utils.metric import get_metric

from .base_model import BaseModel


class Classification(BaseModel):

	def __init__(self, config):
		super(Classification, self).__init__(config)

		self.config = config
		self.input_shape = self.config['input shape']
		self.nb_classes = self.config['nb classes']

		self.finetune_steps = int(self.config.get('finetune steps', 0))
		self.has_endpoint_summary = self.config.get("has endpoint summary", True)
		self.endpoint_summary_steps = self.config.get("endpoint summary steps", 1000)
		
		self.build_model()
		self.build_summary()

	def build_model(self):

		self.config['classifier params']['name'] = 'classifier'
		self.classifier = self._build_classifier('classifier')

		# for training
		self.x = tf.placeholder(tf.float32, shape=[None,]  + self.input_shape, name='x_input')
		self.label = tf.placeholder(tf.float32, shape=[None, self.nb_classes], name='label')

		self.logits, self.end_points = self.classifier.features(self.x)

		self.loss = get_loss('classification', self.config['classification loss'], 
						{'logits' : self.logits, 'labels' : self.label})
		self.train_acc = tf.summary.scalar('train acc', get_metric('accuracy', 'top1', 
						{'logits': self.logits, 'labels':self.label}))

		# for testing
		self.probs = tf.nn.softmax(self.logits)
		
		# print('vars')
		# for var in self.classifier.vars:
		# 	print(var.name, ' --> ', var.get_shape())

		# print('store_vars')
		# for var in self.classifier.store_vars:
		# 	print(var.name, ' --> ', var.get_shape())

		self.global_step, self.global_step_update = self._build_step_var('global_step')
	
		if self.finetune_steps > 0:
			self.finetune_classifier, _ = self._build_train_function('finetune', self.loss, self.classifier.top_vars, 
						step=self.global_step, step_update=self.global_step_update, 
						build_summary=self.has_summary, sum_list=[self.train_acc,])
	
		self.train_classifier, _ = self._build_train_function('optimizer', self.loss, self.classifier.vars, 
						step=self.global_step, step_update=self.global_step_update, 
						build_summary=self.has_summary, sum_list=[self.train_acc,])
		# model saver
		self.saver = tf.train.Saver(self.classifier.store_vars + [self.global_step,])


	def build_summary(self):
		if self.has_summary:
			sum_list = [tf.summary.histogram(var.name, var) for var in self.classifier.store_vars]
			self.histogram_summary = tf.summary.merge(sum_list)
		else:
			self.histogram_summary = None

		if self.has_endpoint_summary:
			sum_list = []
			for key, var in self.end_points.items():
				sum_list.append(tf.summary.histogram('netout/' + key, var))
			self.endpoint_summary = tf.summary.merge(sum_list)
		else:
			self.endpoint_summary = None

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

		if step < self.finetune_steps:
			if self.has_endpoint_summary and step % self.endpoint_summary_steps == 0:
				return self.finetune_classifier(sess, feed_dict, summary=self.endpoint_summary)
			else:
				return self.finetune_classifier(sess, feed_dict)
		else:
			if self.has_endpoint_summary and step % self.endpoint_summary_steps == 0:
				return self.train_classifier(sess, feed_dict, summary=self.endpoint_summary)
			else:
				return self.train_classifier(sess, feed_dict)


	def train_on_batch_unsupervised(self, sess, x_batch):
		raise NotImplementedError

	'''
		test operations
	'''
	def predict(self, sess, x_batch):
		feed_dict = {
			self.x : x_batch,
			self.is_training : False
		}
		y = sess.run([self.probs], feed_dict = feed_dict)[0]
		return y

	def reduce_features(self, sess, x_batch, feature_name):
		assert(isinstance(feature_name, str) or isinstance(feature_name, list))
		if isinstance(feature_name, str):
			feature_name_list = [feature_name,]
		else:
			feature_name_list = feature_name
		feature_list = [self.end_points[f] for f in feature_name_list]
		feed_dict = {
			self.x : x_batch,
			self.is_training : False
		}
		features = sess.run(feature_list, feed_dict=feed_dict)

		if isinstance(feature_name, str):
			features = features[0]
		return features
