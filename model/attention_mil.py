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

from network.base_network import BaseNetwork
from .base_model import BaseModel


class AttentionNet(BaseNetwork):
	def __init__(self, config, is_training):
		super(AttentionNet, self).__init__(config, is_training)
		self.config = config
		self.name = self.config.get('name', 'attention_net')
	
	def __call__(self, x):
		return self.features(x)[0]

	def features(self, x):
		self.end_points = {}
		with tf.variable_scope(self.name):
			if self.reuse:
				tf.get_variable_scope().reuse_variables()
			else:
				assert tf.get_variable_scope().reuse is False
				self.reuse = True

			x = self.fc('fc1', x, 128, winit_fn='xavier', has_bias=False, disp=False)
			x = tf.tanh(x)
			x = self.fc('fc2', x, 1, winit_fn='xavier', has_bias=False, disp=False)
			return x, self.end_points


class AttentionMIL(BaseModel):
	""" Implementation of "Attention-based Deep Multiple Instance Learning"
		Maximilian Ilse, Jakub M. Tomczak, Max Welling

		@article{DBLP:journals/corr/abs-1802-04712,
			author    = {Maximilian Ilse and
						Jakub M. Tomczak and
						Max Welling},
			title     = {Attention-based Deep Multiple Instance Learning},
			journal   = {CoRR},
			volume    = {abs/1802.04712},
			year      = {2018},
			url       = {http://arxiv.org/abs/1802.04712},
			archivePrefix = {arXiv},
			eprint    = {1802.04712},
			timestamp = {Mon, 13 Aug 2018 16:48:13 +0200},
			biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1802-04712},
			bibsource = {dblp computer science bibliography, https://dblp.org}
		}
	"""

	def __init__(self, config,**kwargs):

		super(AttentionMIL, self).__init__(config)
		self.config = config

		assert 'input shape' in self.config
		assert 'z dims' in self.config
		assert 'nb classes' in self.config

		self.input_shape = self.config['input shape']
		self.z_dims = int(self.config['z dims'])
		self.nb_classes = int(self.config['nb classes'])
		self.mil_pooling = self.config.get('MIL pooling', 'attention')

		assert self.mil_pooling in ['maxpooling', 'avgpooling', 'attention']
		
		self.build_model()
		self.build_summary()

	def build_model(self):

		self.feature_ext_net = self.build_classifier('feature_ext', params={
			'name' : 'feature_ext',
			'output_dims' : self.z_dims
		})

		if self.mil_pooling == 'attention':
			# self.attention_net = self.build_classifier('attention_net', params={
			# 	'name' : 'attention_net',
			# 	'output_dims' : 1
			# })
			self.attention_net_params = self.config.get('attention_net params')
			self.attention_net_params['name'] = 'attention_net'
			self.attention_net = AttentionNet(self.attention_net_params, self.is_training)

		self.classifier = self.build_classifier('classifier', params={
			'name' : 'classifier',
			'output_dims' : self.nb_classes
		})

		#
		# Build model
		#
		# 1. inputs
		self.x_bag = tf.placeholder(tf.float32, shape=[None,]  + self.input_shape, name='x_bag')
		self.label = tf.placeholder(tf.float32, shape=[self.nb_classes], name='label')

		# 2.  feature extraction
		self.features, self.fea_ext_net_endpoints = self.feature_ext_net.features(self.x_bag)

		print(self.features.get_shape())

		# 3. mil pooling
		if self.mil_pooling == 'maxpooling':
			self.bag_feature = tf.reduce_max(self.features, axis=0)
			self.bag_feature = tf.reshape(self.bag_feature, [1, -1])

		elif self.mil_pooling == 'avgpooling':
			self.bag_feature = tf.reduce_mean(self.features, axis=0)
			self.bag_feature = tf.reshape(self.bag_feature, [1, -1])

		elif self.mil_pooling == 'attention':
			self.instance_weight, self.attention_net_endpoints = self.attention_net.features(self.features)
			shape = tf.shape(self.instance_weight)
			self.instance_weight = tf.reshape(self.instance_weight, [1, -1])
			# self.instance_weight = tf.tanh(self.instance_weight)
			self.instance_weight = tf.nn.softmax(self.instance_weight)
			self.instance_weight = tf.reshape(self.instance_weight, shape)
			self.bag_feature = tf.reduce_sum(self.features * self.instance_weight, axis=0)
			self.bag_feature = tf.reshape(self.bag_feature, [1, -1])

		# 4. classify
		self.logits, self.classifier_endpoints = self.classifier.features(self.bag_feature)
		self.probs = tf.sigmoid(self.logits)
		self.bag_label = tf.reshape(self.label, [1, -1])

		# 5. loss and metric
		self.loss = get_loss('classification', 'binary entropy', 
						{'logits' : self.logits, 'labels' : self.bag_label})

		if self.mil_pooling == 'attention':
			self.loss += get_loss('regularization', 'l2', {'var_list' : self.attention_net.trainable_vars}) * 0.0005

		self.train_acc = get_metric('accuracy', 'multi-class acc', 
						{'probs': self.probs, 'labels':self.bag_label})

		# build optimizer
		self.global_step, self.global_step_update = self.build_step_var('global_step')

		if self.mil_pooling == 'attention':
			self.train_classifier, self.learning_rate = self.build_train_function('optimizer', 
						self.loss, self.feature_ext_net.vars + self.attention_net.vars + self.classifier.vars, 
						step=self.global_step, step_update=self.global_step_update)
		else:
			self.train_classifier, self.learning_rate = self.b	uild_train_function('optimizer', 
						self.loss, self.feature_ext_net.vars + self.classifier.vars, 
						step=self.global_step, step_update=self.global_step_update)

		# model saver
		self.saver = tf.train.Saver(self.classifier.store_vars + [self.global_step,])

	def build_summary(self):
		if self.has_summary:

			sum_list = []
			# sum_list.append(tf.summary.scalar('lr', self.learning_rate))
			sum_list.append(tf.summary.scalar('train loss', self.loss))
			sum_list.append(tf.summary.scalar('train acc', self.train_acc))
			self.sum_scalar = tf.summary.merge(sum_list)

			if self.mil_pooling == 'attention':
				for key, var in self.attention_net_endpoints.items():
					sum_list.append(tf.summary.histogram('netout_attention/' + key, var))
				sum_list.append(tf.summary.histogram('netout_attention/instance_weight', self.instance_weight))

			for key, var in self.fea_ext_net_endpoints.items():
				sum_list.append(tf.summary.histogram('netout_feature_ext/' + key, var))
			for key, var in self.classifier_endpoints.items():
				sum_list.append(tf.summary.histogram('netout_classifier/' + key, var))
			sum_list.append(tf.summary.histogram('netout_classifier/bag_feature', self.bag_feature))
			sum_list.append(tf.summary.histogram('netout_classifier/logits', self.logits))
			sum_list.append(tf.summary.histogram('netout_classifier/probs', self.probs))
			sum_list.append(tf.summary.histogram('netout_classifier/labels', self.label))
			self.sum_scalar2 = tf.summary.merge(sum_list)

			# summary hists are logged by calling self.summary()
			sum_list = self.feature_ext_net.histogram_summary_list
			if self.mil_pooling == 'attention':
				sum_list += self.attention_net.histogram_summary_list
			sum_list += self.classifier.histogram_summary_list
			self.sum_hist = tf.summary.merge(sum_list)
		else:
			self.sum_scalar = None
			self.sum_scalar2 = None
			self.sum_hist = None

	'''
		train operations
	'''
	def train_on_batch_supervised(self, sess, x_batch, y_label):
		feed_dict = {
			self.x_bag : x_batch,
			self.label : y_label,
			self.is_training : True
		}

		step = int(sess.run([self.global_step])[0])
		if step % 100 == 0:
			return self.train_classifier(sess, feed_dict, summary=self.sum_scalar2)
		else:
			return self.train_classifier(sess, feed_dict, summary=self.sum_scalar)

	def train_on_batch_unsupervised(self, sess, x_batch):
		raise NotImplementedError

	'''
		test operations
	'''
	def predict(self, sess, x_batch):
		feed_dict = {
			self.x_bag : x_batch,
			self.is_training : False
		}

		y = sess.run([self.probs], feed_dict=feed_dict)[0][0, :]
		return y

	def attention(self, sess, x_batch):
		assert self.mil_pooling == 'attention'
		feed_dict = {
			self.x_bag : x_batch,
			self.is_training : False
		}
		instance_weight = sess.run([self.instance_weight], feed_dict=feed_dict)[0]
		return instance_weight
