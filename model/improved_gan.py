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

from .base_model import BaseModel

class ImprovedGAN(BaseModel):

	"""
		Imple``mentation of "Improved Techniques for Training GANs"
		Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, Xi Chen``
		
		@article{DBLP:journals/corr/SalimansGZCRC16,
			author    = {Tim Salimans and
					   Ian J. Goodfellow and
					   Wojciech Zaremba and
					   Vicki Cheung and
					   Alec Radford and
					   Xi Chen},
			title     = {Improved Techniques for Training GANs},
			journal   = {CoRR},
			volume    = {abs/1606.03498},
			year      = {2016},
			url       = {http://arxiv.org/abs/1606.03498},
			archivePrefix = {arXiv},
			eprint    = {1606.03498},
			timestamp = {Wed, 07 Jun 2017 14:40:52 +0200},
			biburl    = {https://dblp.org/rec/bib/journals/corr/SalimansGZCRC16},
			bibsource = {dblp computer science bibliography, https://dblp.org}
		}
	"""

	def __init__(self, config):

		super(ImprovedGAN, self).__init__(config)

		self.input_shape = config['input shape']
		self.z_dim = config['z_dim']
		self.nb_classes = config['nb classes']
		self.config = config

		self.discriminator_warm_up_steps = int(config.get('discriminator warm up steps', 40))
		self.discriminator_training_steps = int(config.get('discriminator training steps', 5))

		self.feature_matching_end_points = config.get('feature matching end points', ['conv1_0', 'conv2_0', 'conv3_0', 'conv4_0'])

		self.build_model()
		self.build_summary()

	def build_model(self):
		# network config
		self.config['discriminator params']['name'] = 'Discriminator'
		self.config['discriminator params']["output dims"] = self.nb_classes + 1
		self.config['generator params']['name'] = 'Generator'
		self.discriminator = self._build_discriminator('discriminator')
		self.generator = self._build_generator('generator')

		# build model
		self.x_real = tf.placeholder(tf.float32, shape=[None, ] + list(self.input_shape), name='x_real')
		self.label_real = tf.placeholder(tf.float32, shape=[None, self.nb_classes], name='label_real')
		self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim], name='z')

		self.x_fake = self.generator(self.z)
		self.dis_real, self.dis_real_end_points = self.discriminator.features(self.x_real)
		self.dis_fake, self.dis_fake_end_points = self.discriminator.features(self.x_fake)

		self.prob_real = tf.nn.softmax(self.dis_real)

		# self.d_loss_feature_matching = get_loss('feature matching', 'l2', 
		# 										{'fx': self.dis_real_end_points, 'fy': self.dis_fake_end_points, 'fnames' : self.feature_matching_end_points})
		# self.d_loss_feature_matching *= self.config.get('feature matching loss weight', 0.01)

		# supervised loss config
		self.d_su_loss_adv = get_loss('adversarial down', 'supervised cross entropy', 
											{'dis_real' : self.dis_real, 'dis_fake' : self.dis_fake, 'label' : self.label_real})  
		self.d_su_loss_adv *= self.config.get('adversarial loss weight', 1.0)

		# self.d_su_loss = self.d_su_loss_adv + self.d_loss_feature_matching
		self.d_su_loss = self.d_su_loss_adv
		# self.g_su_loss = get_loss('adversarial up', 'supervised cross entropy', {'dis_fake' : self.dis_fake, 'label': self.label_real})

		# unsupervised loss config
		self.d_unsu_loss_adv = get_loss('adversarial down', 'unsupervised cross entropy', 
										{'dis_real' : self.dis_real, 'dis_fake' : self.dis_fake}) 
		self.d_unsu_loss_adv *= self.config.get('adversarial loss weight', 1.0)
		# self.d_unsu_loss = self.d_unsu_loss_adv + self.d_loss_feature_matching
		self.d_unsu_loss = self.d_unsu_loss_adv

		self.g_unsu_loss = get_loss('adversarial up', 'unsupervised cross entropy', {'dis_fake' : self.dis_fake})

		# optimizer config
		self.global_step, self.global_step_update = get_global_step()

		# optimizer of discriminator configured without global step update
		# so we can keep the learning rate of discriminator the same as generator
		(self.d_su_train_op, 
			self.d_su_learning_rate, 
				self.d_su_global_step) = get_optimizer_by_config(self.config['discriminator optimizer'],
																self.config['discriminator optimizer params'],
																self.d_su_loss, self.discriminator.vars,
																self.global_step, self.global_step_update)
		# (self.g_su_train_op, 
		# 	self.g_su_learning_rate, 
		# 		self.g_su_global_step) = get_optimizer_by_config(self.config['generator optimizer'],
		# 														self.config['generator optimizer params'],
		# 														self.g_su_loss, self.generator.vars,
		# 														self.global_step, self.global_step_update)

		(self.d_unsu_train_op, 
			self.d_unsu_learning_rate, 
				self.d_unsu_global_step) = get_optimizer_by_config(self.config['discriminator optimizer'],
																self.config['discriminator optimizer params'],
																self.d_unsu_loss, self.discriminator.vars,
																self.global_step)
		(self.g_unsu_train_op, 
			self.g_unsu_learning_rate, 
				self.g_unsu_global_step) = get_optimizer_by_config(self.config['generator optimizer'],
																self.config['generator optimizer params'],
																self.g_unsu_loss, self.generator.vars,
																self.global_step, self.global_step_update)

		# model saver
		self.saver = tf.train.Saver(self.discriminator.store_vars 
									+ self.generator.store_vars
									+ [self.global_step])


	def build_summary(self):
		if self.has_summary:
			# summary scalars are logged per step
			sum_list = []
			sum_list.append(tf.summary.scalar('supervised/discriminator/adervarial_loss', self.d_su_loss_adv))
			# sum_list.append(tf.summary.scalar('supervised/discriminator/feature_matching_loss', self.d_loss_feature_matching))
			sum_list.append(tf.summary.scalar('supervised/discriminator/loss', self.d_su_loss))
			sum_list.append(tf.summary.scalar('supervised/discriminator/lr', self.d_su_learning_rate))
			self.d_su_sum_scalar = tf.summary.merge(sum_list)

			# sum_list = []
			# sum_list.append(tf.summary.scalar('supervised/generator/loss', self.g_su_loss))
			# sum_list.append(tf.summary.scalar('supervised/generator/lr', self.g_su_learning_rate))
			# self.g_su_sum_scalar = tf.summary.merge(sum_list)

			sum_list = []
			sum_list.append(tf.summary.scalar('unsupervised/discriminator/adervarial_loss', self.d_unsu_loss_adv))
			# sum_list.append(tf.summary.scalar('unsupervised/discriminator/feature_matching_loss', self.d_loss_feature_matching))
			sum_list.append(tf.summary.scalar('unsupervised/discriminator/loss', self.d_unsu_loss))
			sum_list.append(tf.summary.scalar('unsupervised/discriminator/lr', self.d_unsu_learning_rate))
			self.d_unsu_sum_scalar = tf.summary.merge(sum_list)

			sum_list = []
			sum_list.append(tf.summary.scalar('unsupervised/generator/loss', self.g_unsu_loss))
			sum_list.append(tf.summary.scalar('unsupervised/generator/lr', self.g_unsu_learning_rate))
			self.g_unsu_sum_scalar = tf.summary.merge(sum_list)

			# summary hists are logged by calling self.summary()
			sum_list = []
			sum_list += [tf.summary.histogram('discriminator/'+var.name, var) for var in self.discriminator.vars]
			sum_list += [tf.summary.histogram('generator/'+var.name, var) for var in self.generator.vars]
			self.histogram_summary = tf.summary.merge(sum_list)
		else:
			self.d_sum_scalar = None
			self.g_sum_scalar = None
			self.histogram_summary = None

	@property
	def vars(self):
		return self.discriminator.vars + self.generator.vars
	
	'''
		train operations
	'''
	def train_on_batch_supervised(self, sess, x_batch, y_batch):
		dis_train_step = self.discriminator_training_steps
		# summary_list = []
		# for i in range(dis_train_step):
		feed_dict = {
			self.x_real : x_batch,
			self.label_real : y_batch,
			self.z : np.random.randn(x_batch.shape[0], self.z_dim),
			self.is_training : True
		}
		step_d, lr_d, loss_d, summary_d = self.train(sess, feed_dict, update_op=self.d_su_train_op,
														step=self.d_su_global_step,
														learning_rate=self.d_su_learning_rate,
														loss=self.d_su_loss,
														summary=self.d_su_sum_scalar)
		# summary_list.append((step_d, summary_d))
		# feed_dict = {
		# 	self.z : np.random.randn(x_batch.shape[0], self.z_dim),
		# 	self.label_real : y_batch,
		# 	self.is_training : True
		# }
		# step_g, lr_g, loss_g, summary_g = self.train(sess, feed_dict, update_op=self.g_su_train_op,
		# 														step=self.g_su_global_step,
		# 														learning_rate=self.g_su_learning_rate,
		# 														loss=self.g_su_loss,
		# 														summary=self.g_su_sum_scalar)
		# summary_list.append((step_g, summary_g))
		# return step_g, {'d':lr_d, 'g':lr_g}, {'d':loss_d,'g':loss_g}, summary_list, 
		return step_d, lr_d, loss_d, summary_d


	def train_on_batch_unsupervised(self, sess, x_batch):
		dis_train_step = self.discriminator_training_steps
		summary_list = []
		for i in range(dis_train_step):
			feed_dict = {
				self.x_real : x_batch,
				self.z : np.random.randn(x_batch.shape[0], self.z_dim),
				self.is_training : True
			}
			step_d, lr_d, loss_d, summary_d = self.train(sess, feed_dict, update_op=self.d_unsu_train_op,
															step=self.d_unsu_global_step,
															learning_rate=self.d_unsu_learning_rate,
															loss=self.d_unsu_loss,
															summary=self.d_unsu_sum_scalar)
		summary_list.append((step_d, summary_d))

		feed_dict = {
			self.z : np.random.randn(x_batch.shape[0], self.z_dim),
			self.is_training : True
		}
		step_g, lr_g, loss_g, summary_g = self.train(sess, feed_dict, update_op=self.g_unsu_train_op,
																step=self.g_unsu_global_step,
																learning_rate=self.g_unsu_learning_rate,
																loss=self.g_unsu_loss,
																summary=self.g_unsu_sum_scalar)
		summary_list.append((step_g, summary_g))
		return step_g, {'d':lr_d, 'g':lr_g}, {'d':loss_d,'g':loss_g}, summary_list, 

	'''
		test operation
	'''
	def predict(self, sess, x_batch):
		feed_dict = {
			self.x_real : x_batch,
			self.is_training : False
		}
		pred = sess.run([self.prob_real], feed_dict=feed_dict)[0]
		probs = pred[:, :-1]
		return probs
		
	def generate(self, sess, z_batch):
		feed_dict = {
			self.z : z_batch,
			self.is_training : False
		}
		x_batch = sess.run([self.x_fake], feed_dict=feed_dict)[0]
		return x_batch

	def discriminate(self, sess, x_batch):
		feed_dict = {
			self.x_real : x_batch, 
			self.is_training : False
		}
		dis_x = sess.run([self.dis_real], feed_dict=feed_dict)[0][:, 0]
		return dis_x

	'''
		summary operation
	'''
	def summary(self, sess):
		if self.has_summary:
			summ = sess.run(self.histogram_summary)
			return summ
		else:
			return None

		# feed_dict = {
		# 	self.x_real : x_batch,
		# 	self.is_training : False
		# }
		# pred = sess.run([self.prob_real], feed_dict=feed_dict)[0]
		# probs = pred[:, :-1]
		# return probs
		
	def generate(self, sess, z_batch):
		feed_dict = {
			self.z : z_batch,
			self.is_training : False
		}
		x_batch = sess.run([self.x_fake], feed_dict=feed_dict)[0]
		return x_batch

	def discriminate(self, sess, x_batch):
		feed_dict = {
			self.x_real : x_batch, 
			self.is_training : False
		}
		dis_x = sess.run([self.dis_real], feed_dict=feed_dict)[0][:, 0]
		return dis_x

	'''
		summary operation
	'''
	def summary(self, sess):
		if self.has_summary:
			summ = sess.run(self.histogram_summary)
			return summ
		else:
			return None
