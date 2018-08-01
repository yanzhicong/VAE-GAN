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
from utils.optimizer import get_optimizer_by_config
from utils.sample import get_sample
from utils.loss import get_loss

from .basemodel import BaseModel

class VAE(BaseModel):

	'''
		Implementation of "Auto-Encoding Variational Bayes"
		Diederik P Kingma, Max Welling

	'''

	def __init__(self, config, **kwargs):

		super(VAE, self).__init__(config, **kwargs)

		self.input_shape = config['input shape']
		self.z_dim = config['z_dim']
		self.config = config

		self.build_model()
		self.build_summary()


	def build_model(self):

		self.x_real = tf.placeholder(tf.float32, shape=[None, ] + list(self.input_shape), name='x_input')

		self.config['encoder params']['name'] = 'Encoder'
		self.config['encoder params']['output_dims'] = self.z_dim
		self.config['decoder params']['name'] = 'Decoder'
		self.config['decoder params']['output_dims'] = list(self.input_shape)
		self.encoder = self.build_encoder('encoder')
		self.decoder = self.build_decoder('decoder')

		# build encoder
		self.mean_z, self.log_var_z = self.encoder(self.x_real)

		# sample z from mean_z and log_var_z
		sample_z = self.draw_sample(self.mean_z, self.log_var_z)

		# build decoder
		self.x_decode = self.decoder(sample_z)

		# build test decoder
		self.z_test = tf.placeholder(tf.float32, shape=[None, self.z_dim], name='z_test')
		self.x_test = self.decoder(self.z_test)

		# loss function
		self.kl_loss = (get_loss('kl', self.config['kl loss'], {'mean' : self.mean_z, 'log_var' : self.log_var_z})
							* self.config.get('kl loss prod', 1.0))

		self.recon_loss = (get_loss('reconstruction', self.config['reconstruction loss'], {'x' : self.x_real, 'y' : self.x_decode })
							 * self.config.get('reconstruction loss prod', 1.0))

		self.loss = self.kl_loss + self.recon_loss

		# optimizer configure
		self.train_op, self.learning_rate, self.global_step = get_optimizer_by_config(
																	self.config['optimizer'], self.config['optimizer params'],
																	self.loss, self.vars)


		# model saver
		self.saver = tf.train.Saver(self.store_vars + [self.global_step,])


	def build_summary(self):
		# summary scalars are logged per step
		if self.is_summary:
			sum_list = []
			sum_list.append(tf.summary.scalar('encoder/kl_loss', self.kl_loss))
			sum_list.append(tf.summary.scalar('lr', self.learning_rate))
			sum_list.append(tf.summary.scalar('decoder/reconstruction_loss', self.recon_loss))
			sum_list.append(tf.summary.scalar('loss', self.loss))
			self.sum_scalar = tf.summary.merge(sum_list)

			# summary hists are logged by calling self.summary()
			sum_list = []
			sum_list += [tf.summary.histogram('encoder/'+var.name, var) for var in self.encoder.vars]
			sum_list += [tf.summary.histogram('decoder/'+var.name, var) for var in self.decoder.vars]
			self.sum_hist = tf.summary.merge(sum_list)
		else:
			self.sum_scalar = None
			self.sum_hist = None

	@property
	def vars(self):
		return self.encoder.vars + self.decoder.vars

	'''
		train operations
	'''
	def train_on_batch_supervised(self, sess, x_batch, y_batch):
		raise NotImplementedError

	def train_on_batch_unsupervised(self, sess, x_batch):
		feed_dict = {
			self.x_real : x_batch,
			# self.eps : np.random.randn(x_batch.shape[0], self.z_dim),
			self.is_training : True
		}
		return self.train(sess, feed_dict)

	'''
		test operation
	'''
	def predict(self, sess, z_batch):
		feed_dict = {
			self.z_test : z_batch,
			self.is_training : False
		}
		x_batch = sess.run([self.x_test], feed_dict = feed_dict)
		return x_batch

	def hidden_variable_distribution(self, sess, x_batch):
		if self.config.get('flatten', False):
			x_batch = x_batch.reshape([x_batch.shape[0], -1])
		feed_dict = {
			self.x_real : x_batch,
			self.is_training : False
		}
		mean_z, log_var_z = sess.run([self.mean_z, self.log_var_z], feed_dict=feed_dict)
		return mean_z, log_var_z

	'''
		summary operation
	'''

	def summary(self, sess):
		if self.is_summary:
			sum = sess.run(self.sum_hist)
			return sum
		else:
			return None
