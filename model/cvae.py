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



class CVAE(BaseModel):

	def __init__(self, config,
		**kwargs
	):

		super(CVAE, self).__init__(config, **kwargs)

		self.input_shape = config['input_shape']
		self.z_dim = config['z_dim']
		self.nb_classes = config['nb_classes']
		self.config = config

		self.is_training = tf.placeholder(tf.bool, name='is_training')

		self.build_model()

		if self.config.get('summary', False):
			self.is_summary = True
			self.get_summary()
		else:
			self.is_summary = False


	def build_model(self):

		self.x_real = tf.placeholder(tf.float32, shape=[None, np.product(self.input_shape)], name='x_input')
		self.y_real = tf.placeholder(tf.float32, shape=[None, self.nb_classes], name='y_input')		


		# if self.config.get('flatten', False):

		self.encoder_input_shape = int(np.product(self.input_shape))
		# else:
		# 	self.x_real = tf.placeholder(tf.float32, shape=[None, ] + list(self.input_shape), name='x_input')
		# 	self.y_real = tf.placeholder(tf.float32, shape=[None, self.nb_classes], name='y_input')
		# 	self.encoder_input_shape = list(self.input_shape)

		self.config['x encoder params']['output_dims'] = self.z_dim
		self.config['y encoder params']['output_dims'] = self.z_dim
		self.config['decoder params']['output_dims'] = self.encoder_input_shape

		self.x_encoder = get_encoder(self.config['x encoder'], self.config['x encoder params'], self.config, self.is_training,
					net_name='EncoderSimpleX')
		self.y_encoder = get_encoder(self.config['y encoder'], self.config['y encoder params'], self.config, self.is_training,
					net_name='EncoderSimpleY')
		self.decoder = get_decoder(self.config['decoder'], self.config['decoder params'], self.config, self.is_training)


		# build encoder
		self.z_mean, self.z_log_var = self.x_encoder(self.x_real)
		self.z_mean_y = self.y_encoder(self.y_real)


		# sample z from z_mean and z_log_var
		self.eps = tf.placeholder(tf.float32, shape=[None,self.z_dim], name='eps')
		self.z_sample = self.z_mean + tf.exp(self.z_log_var / 2) * self.eps

		# build decoder
		self.x_decode = self.decoder(self.z_sample)

		# build test decoder
		self.z_test = tf.placeholder(tf.float32, shape=[None, self.z_dim], name='z_test')
		self.x_test = self.decoder(self.z_test, reuse=True)

		# loss function
		self.kl_loss = get_loss('kl', self.config['kl loss'], {'z_mean' : (self.z_mean - self.z_mean_y), 'z_log_var' : self.z_log_var})
		self.xent_loss = get_loss('reconstruction', self.config['reconstruction loss'], {'x' : self.x_real, 'y' : self.x_decode })
		self.kl_loss = tf.reduce_mean(self.kl_loss * self.config.get('kl loss prod', 1.0))
		self.xent_loss = tf.reduce_mean(self.xent_loss * self.config.get('reconstruction loss prod', 1.0))
		self.loss = self.kl_loss + self.xent_loss


		# optimizer configure
		self.global_step, self.global_step_update = get_global_step()
		if 'lr' in self.config:
			self.learning_rate = get_learning_rate(self.config['lr_scheme'], float(self.config['lr']), self.global_step, self.config['lr_params'])
			self.optimizer = get_optimizer(self.config['optimizer'], {'learning_rate' : self.learning_rate}, self.loss, 
							self.decoder.vars + self.x_encoder.vars + self.y_encoder.vars)
		else:
			self.optimizer = get_optimizer(self.config['optimizer'], {}, self.loss, self.decoder.vars + self.x_encoder.vars + self.y_encoder.vars)

		self.train_update = tf.group([self.optimizer, self.global_step_update])

		# model saver
		self.saver = tf.train.Saver(self.x_encoder.vars + self.y_encoder.vars, self.decoder.vars + [self.global_step,])
		

	def train_on_batch_supervised(self, sess, x_batch, y_batch):
		if self.config.get('flatten', False):
			x_batch = x_batch.reshape([x_batch.shape[0], -1])
			
		feed_dict = {
			self.x_real : x_batch,
			self.y_real : y_batch,
			self.eps : np.random.randn(x_batch.shape[0], self.z_dim),
			self.is_training : True
		}
		return self.train(sess, feed_dict)


	def train_on_batch_unsupervised(self, sess, x_batch):
		return NotImplementedError


	def predict(self, sess, z_batch):
		feed_dict = {
			self.z_test : z_batch,
			self.is_training : False
		}
		x_batch = sess.run([self.x_test], feed_dict = feed_dict)
		return x_batch


	def hidden_distribution(self, sess, x_batch):
		if self.config.get('flatten', False):
			x_batch = x_batch.reshape([x_batch.shape[0], -1])

		feed_dict = {
			self.x_real : x_batch,
			self.is_training : False
		}

		z_mean, z_log_var = sess.run([self.z_mean, self.z_log_var], feed_dict=feed_dict)
		return z_mean, z_log_var


	def summary(self, sess):
		if self.is_summary:
			sum = sess.run(self.sum_hist)
			return sum
		else:
			return None


	def get_summary(self):
		# summary scalars are logged per step
		sum_1 = tf.summary.scalar('encoder/kl_loss', self.kl_loss)
		sum_2 = tf.summary.scalar('lr', self.learning_rate)
		sum_3 = tf.summary.scalar('decoder/reconstruction_loss', self.xent_loss)
		sum_4 = tf.summary.scalar('loss', self.loss)

		
		self.sum_scalar = tf.summary.merge([sum_1, sum_2, sum_3, sum_4])

		# summary hists are logged by calling self.summary()
		hist_sum_d_list = [tf.summary.histogram('encoder/'+var.name, var) for var in self.x_encoder.vars + self.y_encoder.vars]
		hist_sum_g_list = [tf.summary.histogram('decoder/'+var.name, var) for var in self.decoder.vars]
		self.sum_hist = tf.summary.merge(hist_sum_g_list + hist_sum_d_list)
