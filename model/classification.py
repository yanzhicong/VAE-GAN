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

from .basemodel import BaseModel


class Classification(BaseModel):

	def __init__(self, config,
		**kwargs
	):

		super(Classification, self).__init__(input_shape=config['input_shape'], **kwargs)

		self.input_shape = config['input_shape']
		self.z_dim = config['z_dim']
		self.config = config

		self.build_model()


	def build_model(self):

		if self.config.get('flatten', False):
			self.x_real = tf.placeholder(tf.float32, shape=[None, np.product(self.input_shape)], name='x_input')
			self.encoder_input_shape = int(np.product(self.input_shape))
		else:
			self.x_real = tf.placeholder(tf.float32, shape=[None, ] + list(self.input_shape), name='x_input')
			self.encoder_input_shape = list(self.input_shape)


		self.config['encoder params']['output_dim'] = self.z_dim
		self.config['decoder params']['output_dim'] = self.encoder_input_shape
		
		self.encoder = get_encoder(self.config['encoder'], self.config['encoder params'], self.config)
		self.decoder = get_decoder(self.config['decoder'], self.config['decoder params'], self.config)


		# build encoder
		self.z_mean, self.z_log_var = self.encoder(self.x_real)

		# sample z from z_mean and z_log_var
		self.eps = tf.placeholder(tf.float32, shape=[None,self.z_dim], name='eps')
		self.z_sample = self.z_mean + tf.exp(self.z_log_var / 2) * self.eps

		# build decoder
		self.x_decode = self.decoder(self.z_sample)

		# build test decoder
		self.z_test = tf.placeholder(tf.float32, shape=[None, self.z_dim], name='z_test')
		self.x_test = self.decoder(self.z_test, reuse=True)


		self.kl_loss = get_loss('kl', self.config['kl loss'], {'z_mean' : self.z_mean, 'z_log_var' : self.z_log_var})
		self.xent_loss = get_loss('reconstruction', self.config['reconstruction loss'], {'x' : self.x_real, 'y' : self.x_decode })


		self.kl_loss = self.kl_loss * self.config.get('kl loss prod', 1.0)
		self.xent_loss = self.xent_loss * self.config.get('reconstruction loss prod', 1.0)

		self.loss = self.kl_loss + self.xent_loss

		self.global_step, self.global_step_update = get_global_step()
		if 'lr' in self.config:
			self.learning_rate = get_learning_rate(self.config['lr_scheme'], float(self.config['lr']), self.global_step, self.config['lr_params'])
			self.optimizer = get_optimizer(self.config['optimizer'], {'learning_rate' : self.learning_rate}, self.loss, self.decoder.vars + self.encoder.vars)
		else:
			self.optimizer = get_optimizer(self.config['optimizer'], {}, self.loss, self.decoder.vars + self.encoder.vars)

		self.train_update = tf.group([self.optimizer, self.global_step_update])
		
	def train_on_batch_supervised(self, sess, x_batch, y_batch):
		raise NotImplementedError


	def train_on_batch_unsupervised(self, sess, x_batch):
		if 'flatten' in self.config and self.config['flatten']:
			x_batch = x_batch.reshape([x_batch.shape[0], -1])

		feed_dict = {
			self.x_real : x_batch,
			self.eps : np.random.random([x_batch.shape[0], self.z_dim])
		}
		_, step, lr, loss, kl_loss, xent_loss = sess.run([
				self.train_update, self.global_step, self.learning_rate, self.loss, self.kl_loss, self.xent_loss
			],
			feed_dict = feed_dict
			)

		return step, lr, loss

	def predict(self, z_sample):
		raise NotImplementedError

	def summary(self):
		pass

	def help():