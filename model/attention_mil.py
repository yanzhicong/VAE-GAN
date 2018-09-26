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

			shape = tf.shape(x)
			x = tf.reshape(x, [1, -1])
			x = tf.nn.softmax(x)
			x = tf.reshape(x, shape)

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
		self.mil_pooling = self.config.get('mil pooling', 'attention')
		self.finetune_steps = int(self.config.get('finetune steps', 0))

		assert self.mil_pooling in ['maxpooling', 'avgpooling', 'attention']
		
		self.build_model()
		self.build_summary()

	def build_model(self):

		self.feature_ext_net = self._build_classifier('feature_ext', params={
			'name' : 'feature_ext',
			"output dims" : self.z_dims
		})

		if self.mil_pooling == 'attention':
			self.attention_net_params = self.config['attention_net params'].copy()
			self.attention_net_params.update({
				'name' : 'attention_net',
				'output dims' : 1
			})
			self.attention_net = AttentionNet(self.attention_net_params, self.is_training)

		self.classifier = self._build_classifier('classifier', params={
			'name' : 'classifier',
			"output dims" : self.nb_classes
		})

		#
		# Build model
		#
		# 1. inputs
		self.x_bag = tf.placeholder(tf.float32, shape=[None,]  + self.input_shape, name='x_bag')
		self.label = tf.placeholder(tf.float32, shape=[self.nb_classes], name='label')

		# 2.  feature extraction
		self.features, self.fea_ext_net_endpoints = self.feature_ext_net.features(self.x_bag)

		# 3. mil pooling
		if self.mil_pooling == 'maxpooling':
			self.bag_feature = tf.reduce_max(self.features, axis=0)
			self.bag_feature = tf.reshape(self.bag_feature, [1, -1])

		elif self.mil_pooling == 'avgpooling':
			self.bag_feature = tf.reduce_mean(self.features, axis=0)
			self.bag_feature = tf.reshape(self.bag_feature, [1, -1])

		elif self.mil_pooling == 'attention':
			self.instance_weight, self.attention_net_endpoints = self.attention_net.features(self.features)
			self.bag_feature = tf.reduce_sum(self.features * self.instance_weight, axis=0)
			self.bag_feature = tf.reshape(self.bag_feature, [1, -1])

		# 4. classify
		self.logits, self.classifier_endpoints = self.classifier.features(self.bag_feature)
		# self.probs = tf.nn.softmax(self.logits)
		self.probs = tf.nn.sigmoid(self.logits)
		self.bag_label = tf.reshape(self.label, [1, -1])

		# 5. loss and metric
		self.entropy_loss = get_loss('classification', 'binary entropy', {'logits' : self.logits, 'labels' : self.bag_label})

		# self.regulation_loss = get_loss('regularization', 'l2', {'var_list' : self.classifier.trainable_vars}) * 0.005
		# self.regulation_loss += get_loss('regularization', 'l2', {'var_list' : self.feature_ext_net.trainable_vars}) * 0.005

		# if self.mil_pooling == 'attention':
		# 	self.regulation_loss += get_loss('regularization', 'l2', {'var_list' : self.attention_net.trainable_vars}) * 0.005

		self.loss = self.entropy_loss
		#  + self.regulation_loss

		self.train_acc = get_metric('accuracy', 'multi-class acc2', {'probs': self.probs, 'labels':self.bag_label})

		# build optimizer
		self.global_step, self.global_step_update = self._build_step_var('global_step')

		if self.has_summary:
			sum_list = []
			# sum_list.append(tf.summary.scalar('train entropy loss', self.entropy_loss))
			# sum_list.append(tf.summary.scalar('train regulation loss', self.regulation_loss))
			sum_list.append(tf.summary.scalar('train acc', self.train_acc))
		else:
			sum_list = []

		train_function_args = {
			'step' : self.global_step, 
			'step_update' : self.global_step_update,
			'build_summary' : True, 
			'sum_list' : sum_list
		}

		if self.finetune_steps > 0:
			self.finetune_classifier, _ =  self._build_train_function('finetune', self.loss, self.finetune_vars, **train_function_args)

		self.train_classifier,  _, = self._build_train_function('optimizer', self.loss, self.vars, **train_function_args)

		self.saver = tf.train.Saver(self.store_vars + [self.global_step,])


	def build_summary(self):
		if self.has_summary:

			endpoints_sum_list = []
			if self.mil_pooling == 'attention':
				for key, var in self.attention_net_endpoints.items():
					endpoints_sum_list.append(tf.summary.histogram('netout_attention/' + key, var))
				endpoints_sum_list.append(tf.summary.histogram('netout_attention/instance_weight', self.instance_weight))
			for key, var in self.fea_ext_net_endpoints.items():
				endpoints_sum_list.append(tf.summary.histogram('netout_feature_ext/' + key, var))
			for key, var in self.classifier_endpoints.items():
				endpoints_sum_list.append(tf.summary.histogram('netout_classifier/' + key, var))
			endpoints_sum_list.append(tf.summary.histogram('netout_classifier/bag_feature', self.bag_feature))
			endpoints_sum_list.append(tf.summary.histogram('netout_classifier/logits', self.logits))
			endpoints_sum_list.append(tf.summary.histogram('netout_classifier/probs', self.probs))
			endpoints_sum_list.append(tf.summary.histogram('netout_classifier/labels', self.label))
			self.endpoints_sum = tf.summary.merge(endpoints_sum_list)


			sum_list = self.feature_ext_net.histogram_summary_list
			if self.mil_pooling == 'attention':
				sum_list += self.attention_net.histogram_summary_list
			sum_list += self.classifier.histogram_summary_list
			self.sum_hist = tf.summary.merge(sum_list)
		else:
			self.endpoints_sum = None
			self.sum_hist = None


	def load_pretrained_weights(self, sess):
		return self.feature_ext_net.load_pretrained_weights(sess)


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

		if step < self.finetune_steps:
			if step % 1000 == 0:
				return self.finetune_classifier(sess, feed_dict, summary=self.endpoints_sum)
			else:
				return self.finetune_classifier(sess, feed_dict)
		else:
			if step % 1000 == 0:
				return self.train_classifier(sess, feed_dict, summary=self.endpoints_sum)
			else:
				return self.train_classifier(sess, feed_dict)


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


	@property
	def vars(self):
		if self.mil_pooling == 'attention':
			return self.feature_ext_net.vars + self.attention_net.vars + self.classifier.vars
		else:
			return self.feature_ext_net.vars + self.classifier.vars
	
	@property
	def finetune_vars(self):
		if self.mil_pooling == 'attention':
			return self.feature_ext_net.top_vars + self.attention_net.vars + self.classifier.vars
		else:
			return self.feature_ext_net.top_vars + self.classifier.vars

	@property
	def store_vars(self):
		if self.mil_pooling == 'attention':
			return self.feature_ext_net.store_vars + self.attention_net.store_vars + self.classifier.store_vars
		else:
			return self.feature_ext_net.store_vars + self.classifier.store_vars

