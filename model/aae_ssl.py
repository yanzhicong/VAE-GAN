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
from time import *


from utils.learning_rate import get_learning_rate
from utils.learning_rate import get_global_step
from utils.optimizer import get_optimizer
from utils.optimizer import get_optimizer_by_config
from utils.loss import get_loss

from math import sin, cos, sqrt

from .base_model import BaseModel


class AAESemiSupervised(BaseModel):

	"""
		Implementation of "Adversarial Autoencoders"
		Alireza Makhzani, Jonathon Shlens, Navdeep Jaitly, Ian Goodfellow, Brendan Frey

		@article{DBLP:journals/corr/MakhzaniSJG15,
			author    = {Alireza Makhzani and
							Jonathon Shlens and
							Navdeep Jaitly and
							Ian J. Goodfellow},
			title     = {Adversarial Autoencoders},
			journal   = {CoRR},
			volume    = {abs/1511.05644},
			year      = {2015},
			url       = {http://arxiv.org/abs/1511.05644},
			archivePrefix = {arXiv},
			eprint    = {1511.05644},
			timestamp = {Wed, 07 Jun 2017 14:42:14 +0200},
			biburl    = {https://dblp.org/rec/bib/journals/corr/MakhzaniSJG15},
			bibsource = {dblp computer science bibliography, https://dblp.org}
		}
	"""

	def __init__(self, config,
				 **kwargs
				 ):

		super(AAESemiSupervised, self).__init__(config, **kwargs)

		self.input_shape = config['input shape']
		self.z_dim = config['z_dim']
		self.nb_classes = config['nb classes']
		self.config = config

		assert('encoder' in self.config)
		assert('decoder' in self.config)
		assert('z discriminator' in self.config)
		assert('y discriminator' in self.config)

		self.discriminator_step = self.config.get('discriminator step', 1)
		self.generator_step =  self.config.get('generator step', 1)
		self.gan_type = self.config.get('gan type', 'wgan')

		assert(self.gan_type in ['wgan', 'dcgan'])

		self.prior_distribution = self.config.get(
			'prior distribution', 'normal')
		assert(self.prior_distribution in [
			'mixGaussian', 'swiss_roll', 'normal'])

		self.build_model()
		self.build_summary()

	def sample_prior(self, batch_size, prior):
		def to_categorical(y, num_classes):
			input_shape = y.shape
			y = y.ravel().astype(np.int32)
			n = y.shape[0]
			ret = np.zeros((n, num_classes), dtype=np.float32)
			indices = np.where(y >= 0)[0]
			ret[np.arange(n)[indices], y[indices]] = 1.0
			ret = ret.reshape(list(input_shape) + [num_classes, ])
			return ret

		if prior == 'mixGaussian':
			def sample(x, y, label, n_labels):
				shift = 3
				r = 2.0 * np.pi / n_labels * label
				new_x = x * np.cos(r) - y * np.sin(r)
				new_y = x * np.sin(r) + y * np.cos(r)
				new_x += shift * np.cos(r)
				new_y += shift * np.sin(r)
				return new_x, new_y
			x_var = 0.5
			y_var = 0.1
			x = np.random.normal(0, x_var, [batch_size, 1])
			y = np.random.normal(0, y_var, [batch_size, 1])
			label = np.random.randint(0, self.nb_classes, size=[
				batch_size, 1]).astype(np.float32)
			label_onehot = to_categorical(
				label[:, 0], self.nb_classes).astype(np.float32)
			x, y = sample(x, y, label, self.nb_classes)
			return np.concatenate([x, y], axis=1).astype(np.float32), label_onehot

		elif prior == 'normal':
			x = np.random.normal(0, 1.0, [batch_size, self.z_dim])
			return x

		elif prior == 'categorical':
			label = np.random.randint(0, self.nb_classes, size=[
				batch_size]).astype(np.float32)
			label_onehot = to_categorical(label, self.nb_classes)
			return label_onehot

		else:
			raise ValueError()

	def build_model(self):
		# network config
		self.z_discriminator = self.build_discriminator('z discriminator', params={
			'name' : 'Z_Discriminator'
		})
		self.y_discriminator = self.build_discriminator('y discriminator', params={
			'name' : 'Y_Discriminator'
		})
		self.encoder = self.build_encoder('encoder', params={
			'name' : 'Encoder',
			'output_dims' : self.z_dim + self.nb_classes
		})
		self.decoder = self.build_decoder('decoder', params={
			'name' : 'Decoder'
		})

		# build model
		self.img = tf.placeholder(
			tf.float32, shape=[None, ] + list(self.input_shape), name='img')
		self.label = tf.placeholder(
			tf.float32, shape=[None, self.nb_classes], name='label')

		self.real_z = tf.placeholder(
			tf.float32, shape=[None, self.z_dim], name='real_z')
		self.real_y = tf.placeholder(
			tf.float32, shape=[None, self.nb_classes], name='real_y')

		self.img_encode = self.encoder(self.img)

		self.img_z = self.img_encode[:, :self.z_dim]
		self.img_logits = self.img_encode[:, self.z_dim:]
		self.img_y = tf.nn.softmax(self.img_logits)

		self.img_recon = self.decoder(
			tf.concat([self.img_z, self.img_y], axis=1))

		self.dis_z_real = self.z_discriminator(self.real_z)
		self.dis_z_fake = self.z_discriminator(self.img_z)

		if self.gan_type == 'wgan':
			eplison = tf.random_uniform(
				shape=[tf.shape(self.real_z)[0], 1], minval=0.0, maxval=1.0)
			self.hat_z = (eplison * self.real_z) + ((1 - eplison) * self.img_z)
			self.dis_z_hat = self.z_discriminator(self.hat_z)

		self.dis_y_real = self.y_discriminator(self.real_y)
		self.dis_y_fake = self.y_discriminator(self.img_y)

		if self.gan_type == 'wgan':
			eplison2 = tf.random_uniform(
				shape=[tf.shape(self.real_y)[0], 1], minval=0.0, maxval=1.0)
			self.hat_y = (eplison2 * self.real_y) + ((1 - eplison2) * self.img_y)
			self.dis_y_hat = self.y_discriminator(self.hat_y)

		# generate image from z
		self.img_generate = self.decoder(
			tf.concat([self.real_z, self.real_y], axis=1))

		# loss config
		# reconstruction phase
		self.loss_recon = get_loss('reconstruction', 'l2', {
			'x': self.img, 'y': self.img_recon})

		# regulation phase

		if self.gan_type == 'wgan':
			self.loss_z_adv_down = get_loss('adversarial down', 'wassterstein', {
				'dis_real': self.dis_z_real, 'dis_fake': self.dis_z_fake})
			self.loss_z_gp = get_loss('gradient penalty', 'l2', {
				'x': self.hat_z, 'y': self.dis_z_hat})
			self.loss_z_adv_up = get_loss('adversarial up', 'wassterstein', {
				'dis_fake': self.dis_z_fake})

			self.loss_y_adv_down = get_loss('adversarial down', 'wassterstein', {
				'dis_real': self.dis_y_real, 'dis_fake': self.dis_y_fake})
			self.loss_y_gp = get_loss('gradient penalty', 'l2', {
				'x': self.hat_y,  'y': self.dis_y_hat})
			self.loss_y_adv_up = get_loss('adversarial up', 'wassterstein', {
				'dis_fake': self.dis_y_fake})

		elif self.gan_type == 'dcgan':
			self.loss_z_adv_down = get_loss('adversarial down', 'cross entropy', {
				'dis_real': self.dis_z_real, 'dis_fake': self.dis_z_fake})
			self.loss_z_adv_up = get_loss('adversarial up', 'cross entropy', {
				'dis_fake': self.dis_z_fake})

			self.loss_y_adv_down = get_loss('adversarial down', 'cross entropy', {
				'dis_real': self.dis_y_real, 'dis_fake': self.dis_y_fake})
			self.loss_y_adv_up = get_loss('adversarial up', 'cross entropy', {
				'dis_fake': self.dis_y_fake})

		# semi-supervised classification phase
		self.loss_cla = get_loss('classification', 'cross entropy', {
			'logits': self.img_logits, 'labels': self.label})

		self.ae_loss = self.loss_recon
		if self.gan_type == 'wgan':
			self.dz_loss = self.loss_z_adv_down + self.loss_z_gp
			self.dy_loss = self.loss_y_adv_down + self.loss_y_gp
		elif self.gan_type == 'dcgan':
			self.dz_loss = self.loss_z_adv_down
			self.dy_loss = self.loss_y_adv_down
		self.ez_loss = self.loss_z_adv_up
		self.ey_loss = self.loss_y_adv_up
		self.e_loss = self.loss_cla

		# optimizer config
		self.global_step, self.global_step_update = get_global_step()

		# reconstruction phase
		(self.ae_train_op,
		 self.ae_learning_rate,
		 self.ae_step) = self.build_optimizer('auto-encoder', self.loss_recon, self.encoder.vars + self.decoder.vars)

		# regulation phase
		(self.dz_train_op,
		 self.dz_learning_rate,
		 self.dz_step) = self.build_optimizer('discriminator', self.dz_loss, self.z_discriminator.vars)

		(self.dy_train_op,
		 self.dy_learning_rate,
		 self.dy_step) = self.build_optimizer('discriminator', self.dy_loss, self.y_discriminator.vars)

		(self.ez_train_op,
		 self.ez_learning_rate,
		 self.ez_step) = self.build_optimizer('encoder', self.ez_loss, self.encoder.vars)

		(self.ey_train_op,
		 self.ey_learning_rate,
		 self.ey_step) = self.build_optimizer('encoder', self.ey_loss, self.encoder.vars)

		# classification phase
		(self.e_train_op,
		 self.e_learning_rate,
		 self.e_step) = self.build_optimizer('classifier', self.e_loss, self.encoder.vars)

		# model saver
		self.saver = tf.train.Saver(self.z_discriminator.store_vars
									+ self.y_discriminator.store_vars
									+ self.encoder.store_vars
									+ self.decoder.store_vars
									+ [self.global_step])

	def build_summary(self):
		if self.has_summary:
			# summary scalars are logged per step
			sum_list = []
			sum_list.append(tf.summary.scalar('auto-encoder/loss', self.loss_recon))
			sum_list.append(tf.summary.scalar('auto-encoder/lr', self.ae_learning_rate))
			self.ae_sum_scalar = tf.summary.merge(sum_list)

			sum_list = []
			if self.gan_type == 'wgan':
				sum_list.append(tf.summary.scalar('z_discrimintor/adv_loss', self.loss_z_adv_down))
				sum_list.append(tf.summary.scalar('z_discrimintor/gp_loss', self.loss_z_gp))
			sum_list.append(tf.summary.scalar('z_discrimintor/loss', self.dz_loss))
			sum_list.append(tf.summary.scalar('z_discrimintor/lr', self.dz_learning_rate))
			self.dz_sum_scalar = tf.summary.merge(sum_list)

			sum_list = []
			if self.gan_type == 'wgan':
				sum_list.append(tf.summary.scalar('y_discrimintor/adv_loss', self.loss_y_adv_down))
				sum_list.append(tf.summary.scalar('y_discrimintor/gp_loss', self.loss_y_gp))
			sum_list.append(tf.summary.scalar('y_discrimintor/loss', self.dy_loss))
			sum_list.append(tf.summary.scalar('y_discrimintor/lr', self.dy_learning_rate))
			self.dy_sum_scalar = tf.summary.merge(sum_list)

			sum_list = []
			sum_list.append(tf.summary.scalar('encoder/z_loss', self.loss_z_adv_up))
			self.ez_sum_scalar = tf.summary.merge(sum_list)

			sum_list = []
			sum_list.append(tf.summary.scalar('encoder/y_loss', self.loss_y_adv_up))
			self.ey_sum_scalar = tf.summary.merge(sum_list)

			sum_list = []
			sum_list.append(tf.summary.scalar('encoder/supervised_loss', self.loss_cla))
			self.e_sum_scalar = tf.summary.merge(sum_list)

			# summary hists are logged by calling self.summary()
			sum_list = []
			sum_list += [tf.summary.histogram('z_discriminator/'+var.name, var) for var in self.z_discriminator.vars]
			sum_list += [tf.summary.histogram('y_discriminator/'+var.name, var) for var in self.y_discriminator.vars]
			sum_list += [tf.summary.histogram('encoder/'+var.name, var) for var in self.encoder.vars]
			sum_list += [tf.summary.histogram('decoder/'+var.name, var) for var in self.decoder.vars]
			self.sum_hist = tf.summary.merge(sum_list)
		else:
			self.d_sum_scalar = None
			self.g_sum_scalar = None
			self.sum_hist = None

	'''
		train operations
	'''

	def train_on_batch_supervised(self, sess, x_batch, y_batch):
		summary_list = []

		feed_dict = {
			self.img: x_batch,
			self.label: y_batch,
			self.is_training: True,
		}
		step_e, lr_e, loss_e, summary_e = self.train(sess, feed_dict, update_op=self.e_train_op,
													 step=self.e_step,
													 learning_rate=self.e_learning_rate,
													 loss=self.loss_cla,
													 summary=self.e_sum_scalar)
		summary_list.append((step_e, summary_e))
		step, _ = sess.run([self.global_step, self.global_step_update])
		return step, lr_e, loss_e, summary_list



	def train_on_batch_unsupervised(self, sess, x_batch):

		z_batch = self.sample_prior(x_batch.shape[0], prior='normal')
		y_batch = self.sample_prior(x_batch.shape[0], prior='categorical')
		summary_list = []

		feed_dict = {
			self.img: x_batch,
			self.is_training: True,
		}
		step_ae, lr_ae, loss_ae, summary_ae = self.train(sess, feed_dict, update_op=self.ae_train_op,
														 step=self.ae_step,
														 learning_rate=self.ae_learning_rate,
														 loss=self.ae_loss,
														 summary=self.ae_sum_scalar)
		summary_list.append((step_ae, summary_ae))

		for i in range(self.discriminator_step):
			feed_dict = {
				self.img: x_batch,
				self.is_training: True,
				self.real_z: z_batch,
			}
			step_dz, lr_dz, loss_dz, summary_dz = self.train(sess, feed_dict, update_op=self.dz_train_op,
															step=self.dz_step,
															learning_rate=self.dz_learning_rate,
															loss=self.dz_loss,
															summary=self.dz_sum_scalar)

			feed_dict = {
				self.img: x_batch,
				self.is_training: True,
				self.real_y: y_batch,
			}
			step_dy, lr_dy, loss_dy, summary_dy = self.train(sess, feed_dict, update_op=self.dy_train_op,
															step=self.dy_step,
															learning_rate=self.dy_learning_rate,
															loss=self.dy_loss,
															summary=self.dy_sum_scalar)

		summary_list.append((step_dz, summary_dz))
		summary_list.append((step_dy, summary_dy))

		for i in range(self.generator_step):
			feed_dict = {
				self.img: x_batch,
				self.is_training: True,
			}
			step_ez, lr_ez, loss_ez, summary_ez = self.train(sess, feed_dict, update_op=self.ez_train_op,
															step=self.ez_step,
															learning_rate=self.ez_learning_rate,
															loss=self.ez_loss,
															summary=self.ez_sum_scalar)
			feed_dict = {
				self.img: x_batch,
				self.is_training: True,
			}
			step_ey, lr_ey, loss_ey, summary_ey = self.train(sess, feed_dict, update_op=self.ey_train_op,
															step=self.ey_step,
															learning_rate=self.ey_learning_rate,
															loss=self.ey_loss,
															summary=self.ey_sum_scalar)
		summary_list.append((step_ez, summary_ez))
		summary_list.append((step_ey, summary_ey))
		
		step, _ = sess.run([self.global_step, self.global_step_update])

		return step, "[ae:%0.6f, gan:%0.6f]"%(lr_ae, lr_dz), "[ae:%0.4f, dz:%0.4f, dy:%0.4f, ez:%0.4f, ey:%0.4f]"%(loss_ae,loss_dz,loss_dy,loss_ez,loss_ey), summary_list,

	'''
		test operation
	'''

	def predict(self, sess, x_batch):
		feed_dict = {
			self.img: x_batch,
			self.is_training: False,
		}

		pred = sess.run([self.img_y], feed_dict=feed_dict)[0]
		return pred

	def hidden_variable_distribution(self, sess, x_batch):
		feed_dict = {
			self.img: x_batch,
			self.is_training: False
		}
		sample_z = sess.run([self.img_z], feed_dict=feed_dict)[0]
		return sample_z

	def generate(self, sess, z_batch, condition):
		if condition.ndim == 1:
			condition = np.tile(condition[np.newaxis, :], (z_batch.shape[0], 1))

		feed_dict = {
			self.real_z : z_batch, 
			self.real_y : condition,
			self.is_training : False,
		}
		sample_x = sess.run([self.img_generate], feed_dict=feed_dict)[0]
		return sample_x

	'''
		summary operation
	'''

	def summary(self, sess):
		if self.has_summary:
			summ = sess.run(self.sum_hist)
			return summ
		else:
			return None
