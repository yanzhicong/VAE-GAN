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
from generator.generator import get_generator

from utils.learning_rate import get_learning_rate
from utils.learning_rate import get_global_step
from utils.optimizer import get_optimizer
from utils.optimizer import get_optimizer_by_config
from utils.sample import get_sample
from utils.loss import get_loss

from .basemodel import BaseModel


class WGAN_GP(BaseModel):

	def __init__(self, config,
		**kwargs
	):

		super(WGAN_GP, self).__init__(config, **kwargs)

		self.input_shape = config['input shape']
		self.z_dim = config['z_dim']
		self.config = config

		self.discriminator_warm_up_steps = int(config.get('discriminator warm up steps', 40))
		self.discriminator_training_steps = int(config.get('discriminator training steps', 5))

		self.build_model()
		self.build_summary()

	def build_model(self):
		# network config
		self.config['discriminator params']['name'] = 'Discriminator'
		self.config['generator params']['name'] = 'Generator'

		self.discriminator = get_discriminator(self.config['discriminator'], self.config['discriminator params'], self.is_training)
		self.generator = get_generator(self.config['generator'], self.config['generator params'], self.is_training)

		# build model
		self.x_real = tf.placeholder(tf.float32, shape=[None, ] + list(self.input_shape), name='x_input')
		self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim], name='z')

		self.x_fake = self.generator(self.z)
		
		dis_real = self.discriminator(self.x_real)
		dis_fake = self.discriminator(self.x_fake)

		# loss config
		eplison = tf.random_uniform(shape=[tf.shape(self.x_real)[0], 1, 1, 1], minval=0.0, maxval=1.0)
		x_hat = (eplison * self.x_real) + ((1 - eplison) * self.x_fake)
		dis_hat = self.discriminator(x_hat)

		self.d_loss_adv = (get_loss('adversarial down', 
									'wassterstein', 
									{'dis_real' : dis_real, 'dis_fake' : dis_fake})
							* self.config.get('adversarial loss weight', 1.0))

		self.d_loss_gp = (get_loss('gradient penalty',
									'l2',
									{'x' : x_hat, 'y' : dis_hat})
							* self.config.get('gradient penalty loss weight', 1.0))
		self.d_loss = self.d_loss_gp + self.d_loss_adv
		# self.d_loss = self.d_loss_adv

		self.g_loss = get_loss('adversarial up', 'wassterstein', {'dis_fake' : dis_fake})


		for var in self.generator.vars:
			print(var.name, ' --> ', var.get_shape())
		for var in self.discriminator.vars:
			print(var.name, ' --> ', var.get_shape())

		# optimizer config
		(self.d_train_op, 
			self.d_learning_rate, 
				self.d_global_step) = get_optimizer_by_config(self.config['discriminator optimizer'],
																self.config['discriminator optimizer params'],
																self.d_loss, self.discriminator.vars, global_step_name='d_global_step')
		
		(self.g_train_op, 
			self.g_learning_rate, 
				self.g_global_step) = get_optimizer_by_config(self.config['generator optimizer'],
																self.config['generator optimizer params'],
																self.g_loss, self.generator.vars, global_step_name='g_global_step')

		# model saver
		self.saver = tf.train.Saver(self.vars + [self.d_global_step,self.g_global_step])


	def build_summary(self):
		if self.is_summary:
			# summary scalars are logged per step
			sum_list = []
			sum_list.append(tf.summary.scalar('discriminator/adversarial', self.d_loss_adv))
			sum_list.append(tf.summary.scalar('discriminator/gradient_penalty', self.d_loss_gp))
			sum_list.append(tf.summary.scalar('discriminator/loss', self.d_loss))
			sum_list.append(tf.summary.scalar('discriminator/lr', self.d_learning_rate))
			self.d_sum_scalar = tf.summary.merge(sum_list)

			sum_list = []
			sum_list.append(tf.summary.scalar('generator/loss', self.g_loss))
			sum_list.append(tf.summary.scalar('generator/lr', self.g_learning_rate))
			self.g_sum_scalar = tf.summary.merge(sum_list)

			# summary hists are logged by calling self.summary()
			sum_list = []
			sum_list += [tf.summary.histogram('discriminator/'+var.name, var) for var in self.discriminator.vars]
			sum_list += [tf.summary.histogram('generator/'+var.name, var) for var in self.generator.vars]
			self.sum_hist = tf.summary.merge(sum_list)
		else:
			self.d_sum_scalar = None
			self.g_sum_scalar = None
			self.sum_hist = None

	@property
	def vars(self):
		return self.discriminator.vars + self.generator.vars
	
	'''
		train operations
	'''
	def train_on_batch_supervised(self, sess, x_batch, y_batch):
		raise NotImplementedError


	def train_on_batch_unsupervised(self, sess, x_batch):

		g_step = sess.run(self.g_global_step)

		# if g_step < self.discriminator_warm_up_steps:
			# dis_train_step = self.discriminator_training_steps * 20
		# else:
		dis_train_step = self.discriminator_training_steps

		summary_list = []


		for i in range(dis_train_step):

			feed_dict = {
				self.x_real : x_batch,
				self.z : np.random.randn(x_batch.shape[0], self.z_dim),
				self.is_training : True
			}
			step_d, lr_d, loss_d, summary_d = self.train(sess, feed_dict, update_op=self.d_train_op,
															step=self.d_global_step,
															learning_rate=self.d_learning_rate,
															loss=self.d_loss,
															summary=self.d_sum_scalar)
			summary_list.append((step_d, summary_d))

		feed_dict = {
			self.z : np.random.randn(x_batch.shape[0], self.z_dim),
			self.is_training : True
		}

		step_g, lr_g, loss_g, summary_g = self.train(sess, feed_dict, update_op=self.g_train_op,
																step=self.g_global_step,
																learning_rate=self.g_learning_rate,
																loss=self.g_loss,
																summary=self.g_sum_scalar)
		summary_list.append((step_g, summary_g))

		return step_g, {'d':lr_d, 'g':lr_g}, {'d':loss_d,'g':loss_g}, summary_list, 

	'''
		test operation
	'''
	def predict(self, sess, x_batch):
		raise NotImplementedError

	def generate(self, sess, z_batch):
		feed_dict = {
			self.z : z_batch,
			self.is_training : False
		}
		x_batch = sess.run([self.x_fake], feed_dict = feed_dict)[0]
		return x_batch


	'''
		summary operation
	'''
	def summary(self, sess):
		if self.is_summary:
			sum = sess.run(self.sum_hist)
			return sum
		else:
			return None
