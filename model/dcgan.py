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

from utils.learning_rate import get_learning_rate
from utils.learning_rate import get_global_step
from utils.optimizer import get_optimizer
from utils.optimizer import get_optimizer_by_config
from utils.loss import get_loss

from .basemodel import BaseModel


class DCGAN(BaseModel):

	"""
		Implementation of "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks"
		Alec Radford, Luke Metz, Soumith Chintala
		
		@article{DBLP:journals/corr/RadfordMC15,
			author    = {Alec Radford and
						Luke Metz and
						Soumith Chintala},
			title     = {Unsupervised Representation Learning with Deep Convolutional Generative
						Adversarial Networks},
			journal   = {CoRR},
			volume    = {abs/1511.06434},
			year      = {2015},
			url       = {http://arxiv.org/abs/1511.06434},
			archivePrefix = {arXiv},
			eprint    = {1511.06434},
			timestamp = {Wed, 07 Jun 2017 14:40:30 +0200},
			biburl    = {https://dblp.org/rec/bib/journals/corr/RadfordMC15},
			bibsource = {dblp computer science bibliography, https://dblp.org}
		}
	"""

	def __init__(self, config,
		**kwargs
	):

		assert('discriminator' in config)
		assert('generator' in config)
		assert('input shape' in config)
		assert('z_dim' in config)

		super(DCGAN, self).__init__(config, **kwargs)

		self.input_shape = config['input shape']
		self.z_dim = config['z_dim']
		self.config = config

		self.discriminator_warm_up_steps = int(config.get('discriminator warm up steps', 40))
		self.discriminator_training_steps = int(config.get('discriminator training steps', 5))

		self.use_feature_matching_loss = config.get('use feature matching', False)
		self.feature_matching_end_points = config.get('feature matching end points', None)

		self.build_model()
		self.build_summary()

	def build_model(self):
		# network config
		self.config['discriminator params']['name'] = 'Discriminator'
		self.config['generator params']['name'] = 'Generator'
		self.discriminator = self.build_discriminator('discriminator')
		self.generator = self.build_generator('generator')

		# build model
		self.x_real = tf.placeholder(tf.float32, shape=[None, ] + list(self.input_shape), name='x_input')
		self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim], name='z')

		self.x_fake = self.generator(self.z)
		self.dis_real, self.dis_real_end_points = self.discriminator.features(self.x_real)
		self.dis_fake, self.dis_fake_end_points = self.discriminator.features(self.x_fake)


		# loss config
		self.d_loss_adv = get_loss('adversarial down', 'cross entropy', {'dis_real' : self.dis_real, 'dis_fake' : self.dis_fake})
		if self.use_feature_matching_loss:
			if self.feature_matching_end_points is None:
				self.feature_matching_end_points = [k for k in self.dis_real_end_points.keys() if 'conv' in k]
				print('feature matching end points : ', self.feature_matching_end_points)
			self.d_loss_fm = get_loss('feature matching', 'l2', 
									{'fx':self.dis_real_end_points, 'fy':self.dis_fake_end_points, 'fnames':self.feature_matching_end_points})
			self.d_loss_fm *= self.config.get('feature matching loss weight', 0.01)
			self.d_loss = self.d_loss_adv + self.d_loss_fm
		else:
			self.d_loss = self.d_loss_adv

		self.g_loss = get_loss('adversarial up', 'cross entropy', {'dis_fake' : self.dis_fake})

		# optimizer config
		self.global_step, self.global_step_update = get_global_step()

		# optimizer of discriminator configured without global step update
		# so we can keep the learning rate of discriminator the same as generator
		(self.d_train_op, 
			self.d_learning_rate, 
				self.d_global_step) = get_optimizer_by_config(self.config['discriminator optimizer'],
																self.config['discriminator optimizer params'],
																self.d_loss, self.discriminator.vars,
																self.global_step)
		(self.g_train_op, 
			self.g_learning_rate, 
				self.g_global_step) = get_optimizer_by_config(self.config['generator optimizer'],
																self.config['generator optimizer params'],
																self.g_loss, self.generator.vars,
																self.global_step, self.global_step_update)

		# model saver
		self.saver = tf.train.Saver(self.discriminator.store_vars 
									+ self.generator.store_vars
									+ [self.global_step])


	def build_summary(self):
		if self.is_summary:
			# summary scalars are logged per step
			sum_list = []
			if self.use_feature_matching_loss:
				sum_list.append(tf.summary.scalar('discriminator/loss', self.d_loss))
				sum_list.append(tf.summary.scalar('discriminator/adversaral_loss', self.d_loss_adv))
				sum_list.append(tf.summary.scalar('discriminator/feature_matching_loss', self.d_loss_fm))
			else:
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
	def generate(self, sess, z_batch):
		feed_dict = {
			self.z : z_batch,
			self.is_training : False
		}
		x_batch = sess.run([self.x_fake], feed_dict = feed_dict)[0]
		return x_batch

	def discriminate(self, sess, x_batch):
		feed_dict = {
			self.x_real : x_batch, 
			self.is_training : False
		}
		dis_x = sess.run([self.dis_real], feed_dict = feed_dict)[0][:, 0]
		return dis_x

	'''
		summary operation
	'''
	def summary(self, sess):
		if self.is_summary:
			summ = sess.run(self.sum_hist)
			return summ
		else:
			return None
