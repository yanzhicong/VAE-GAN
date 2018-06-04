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

from encoder.encoder import get_encoder
from decoder.decoder import get_decoder
from classifier.classifier import get_classifier
from discriminator.discriminator import get_discriminator


from utils.learning_rate import get_learning_rate
from utils.learning_rate import get_global_step
from utils.optimizer import get_optimizer
from utils.sample import get_sample
from utils.loss import get_loss
from utils.metric import get_metric

from .basemodel import BaseModel


class Classification(BaseModel):

	def __init__(self, config,
		**kwargs
	):

		super(Classification, self).__init__(config, **kwargs)

		self.input_shape = config['input_shape']
		# self.z_dim = config['z_dim']
		self.nb_classes = config['nb_classes']
		self.config = config
		self.build_model()
		if self.is_summary:
			self.build_summary()



	def build_model(self):
		self.x = tf.placeholder(tf.float32, shape=[None,]  + self.input_shape, name='x_input')
		self.label = tf.placeholder(tf.float32, shape=[None, self.nb_classes], name='label')
		
		self.classifier = get_classifier(self.config['classifier'], self.config['classifier params'], 
					 self.config, self.is_training)

		# build encoder
		self.logits, end_points = self.classifier(self.x)
		self.y = tf.nn.softmax(self.logits)
		self.loss = get_loss('classification', self.config['classification loss'], {'pred' : self.logits, 'label' : self.label})

		self.acc = get_metric('accuracy', 'top1', {'logits': self.logits, 'labels':self.label})

		self.global_step, self.global_step_update = get_global_step()
		if 'lr' in self.config:
			self.learning_rate = get_learning_rate(self.config['lr_scheme'], float(self.config['lr']), self.global_step, self.config['lr_params'])
			self.optimizer = get_optimizer(self.config['optimizer'], {'learning_rate' : self.learning_rate}, self.loss, self.classifier.vars)
		else:
			self.optimizer = get_optimizer(self.config['optimizer'], {}, self.loss, self.classifier.vars)

		self.train_update = tf.group([self.optimizer, self.global_step_update])

		# model saver
		self.saver = tf.train.Saver(self.classifier.vars + [self.global_step,])
		
		
	def train_on_batch_supervised(self, sess, x_batch, y_batch):
		feed_dict = {
			self.x : x_batch,
			self.label : y_batch,
			self.is_training : True
		}
		return self.train(sess, feed_dict)


	def train_on_batch_unsupervised(self, sess, x_batch):
		raise NotImplementedError

	def predict(self, z_sample):
		raise NotImplementedError

	def summary(self, sess):
		if self.is_summary:
			sum = sess.run(self.sum_hist)
			return sum
		else:
			return None

	def help(self):
		pass

	def build_summary(self):
		# summary scalars are logged per step
		sum_list = []
		sum_list.append(tf.summary.scalar('lr', self.learning_rate))
		sum_list.append(tf.summary.scalar('train loss', self.loss))
		sum_list.append(tf.summary.scalar('train acc', self.acc))
		self.sum_scalar = tf.summary.merge(sum_list)

		# summary hists are logged by calling self.summary()
		hist_sum_list = [tf.summary.histogram(var.name, var) for var in self.classifier.vars]
		self.sum_hist = tf.summary.merge(hist_sum_list)
