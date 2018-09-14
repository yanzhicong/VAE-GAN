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
from utils.loss import get_loss

from .base_model import BaseModel


class SemiSupervisedSegmentationModel(BaseModel):

	def __init__(self, config,
		**kwargs
	):

		super(SemiSupervisedSegmentationModel, self).__init__(config, **kwargs)

		raise NotImplementedError

		# parameters must be configured
		self.input_shape = config['input shape']
		self.mask_size = config['mask size']
		self.nb_classes = config['nb_classes']
		self.config = config


		# optional params
		self.debug = config.get('debug', False)

		# build model
		self.is_training = tf.placeholder(tf.bool, name='is_training')

		self.build_model()
		self.build_summary()


	def build_model(self):

		self.img_u = tf.placeholder(tf.float32, shape=[None,] + self.input_shape, name='image_unlabelled_input')
		self.img_l = tf.placeholder(tf.float32, shape=[None,] + self.input_shape, name='image_labelled_input')
		self.mask_l = tf.placeholder(tf.float32, shape=[None,] + self.mask_size + [self.nb_classes], name='mask_input')

		###########################################################################
		# network define
		# 
		self.config['classifier params']['name'] = 'Segmentation'
		self.config['classifier params']['output_dims'] = self.hx_dim
		self.seg_classifier = get_classifier(self.config['classifier'], 
									self.config['classifier params'], self.is_training)


		self.config['discriminator params']['name'] = 'Segmentation'
		self.config['discriminator params']['output_dims'] = 1
		self.config['discriminator params']['output_activation'] = 'none'
		self.discriminator = get_discriminator(self.config['discriminator'], 
									self.config['discriminator params'], self.is_training)

		###########################################################################
		# for supervised training:
		self.mask_generated = self.seg_classifier(self.img_l)

		real_concated = tf.concatenate([self.img_l, self.mask_l], axis=-1)
		fake_concated = tf.concatenate([self.img_l, self.mask_generated], axis=-1)

		dis_real_concated = self.discriminator(real_concated)
		dis_fake_concated = self.discriminator(fake_concated)

		eplison = tf.random_uniform(shape=[tf.shape(self.img_l)[0], 1, 1, 1], minval=0.0, maxval=1.0)
		mask_hat = eplison * self.mask_l + (1 - eplison) * self.mask_generated
		concat_hat = tf.concatenate([self.img_l, mask_hat], axis=-1)

		dis_hat_concated = self.discriminator(concat_hat)


		self.d_su_loss_adv = (get_loss('adversarial down', 'wassterstein', {'dis_real' : dis_real_concated, 'dis_fake' : dis_fake_concated})
								* self.config.get('adversarial loss weight', 1.0))
		self.d_su_loss_gp = (get_loss('gradient penalty', 'l2', {'x' : concat_hat, 'y' : dis_hat_concated})
								* self.config.get('gradient penalty loss weight', 1.0))
		self.d_su_loss = self.d_su_loss_adv + self.d_su_loss_gp

		self.g_su_loss_adv = (get_loss('adversarial up', 'wassterstein', {'dis_fake' : dis_fake_concated})
								* self.config.get('adversarial loss weight', 1.0))

		self.g_su_loss_cls = (get_loss('segmentation', 'l2', {'predict' : self.mask_generated, 'mask':self.mask_l})
								* self.config.get('segmentation loss weight', 1.0))

		self.g_su_loss = self.g_su_loss_adv + self.g_su_loss_cls


		###########################################################################
		# optimizer configure
		(self.d_su_train_op,
			self.d_su_learning_rate,
				self.d_su_global_step) = get_optimizer_by_config(self.config['supervised optimizer'], self.config['supervised optimizer params'],
												self.d_su_loss, self.discriminator.vars, global_step_name='d_global_step_su')

		(self.g_su_train_op,
			self.g_su_learning_rate,
				self.g_su_global_step) = get_optimizer_by_config(self.config['supervised optimizer'], self.config['supervised optimizer params'],
												self.g_su_loss, self.generator.vars, global_step_name='g_global_step_su')

		###########################################################################
		# # for test models
		# # 
		# # xt --> mean_hxt, log_var_hxt
		# #               |
		# #             sample_hxt --> ytlogits --> ytprobs
		# # 			   |			    			 |
		# #		     [sample_hxt,    			  ytprobs] --> mean_hzt, log_var_hzt
		# #
		# mean_hxt, log_var_hxt = self.x_encoder(self.xt)
		# sample_hxt = self.draw_sample(mean_hxt, log_var_hxt)
		# ytlogits = self.hx_classifier(sample_hxt)
		# # test sample class probilities
		# self.ytprobs = tf.nn.softmax(ytlogits)
		# # test sample hidden variable distribution
		# self.mean_hzt, self.log_var_hzt = self.hx_y_encoder(tf.concat([sample_hxt, self.ytprobs], axis=1))


		###########################################################################
		# model saver
		self.saver = tf.train.Saver(self.store_vars + [self.d_su_global_step, self.g_su_global_step])

	def build_summary(self):

		if self.has_summary:

			# summary scalars are logged per step
			sum_list = []
			sum_list.append(tf.summary.scalar('m1/kl_z_loss', self.m1_loss_kl_z))
			sum_list.append(tf.summary.scalar('m1/reconstruction_loss', self.m1_loss_recon))
			sum_list.append(tf.summary.scalar('m1/loss', self.m1_loss))
			sum_list.append(tf.summary.scalar('m1/learning_rate', self.m1_learning_rate))

			self.m1_summary = tf.summary.merge(sum_list)

			sum_list = []
			sum_list.append(tf.summary.scalar('m2/supervised_kl_z_loss', self.m2_su_loss_kl_z))
			sum_list.append(tf.summary.scalar('m2/supervised_reconstruction_loss', self.m2_su_loss_recon))
			sum_list.append(tf.summary.scalar('m2/supervised_classification_loss', self.m2_su_loss_cls))
			sum_list.append(tf.summary.scalar('m2/supervised_regularization_loss', self.m2_su_loss_reg))
			sum_list.append(tf.summary.scalar('m2/supervised_loss', self.m2_su_loss))
			sum_list.append(tf.summary.scalar('m2/supervised_learning_rate', self.m2_supervised_learning_rate))
			self.m2_supervised_summary = tf.summary.merge(sum_list)

			sum_list = []
			sum_list.append(tf.summary.scalar('m2/unsupervised_kl_z_loss', self.m2_unsu_loss_kl_z))
			sum_list.append(tf.summary.scalar('m2/unsupervised_kl_y_loss', self.m2_unsu_loss_kl_y))
			sum_list.append(tf.summary.scalar('m2/unsupervised_reconstruction_loss', self.m2_unsu_loss_recon))
			sum_list.append(tf.summary.scalar('m2/unsupervised_regularization_loss', self.m2_unsu_loss_reg))
			sum_list.append(tf.summary.scalar('m2/unsupervised_loss', self.m2_unsu_loss))
			sum_list.append(tf.summary.scalar('m2/unsupervised_learning_rate', self.m2_unsupervised_learning_rate))
			self.m2_unsupervised_summary = tf.summary.merge(sum_list)
			
			# summary hists are logged by calling self.summary()
			sum_list = [tf.summary.histogram(var.name, var) for var in self.vars]
			self.histogram_summary = tf.summary.merge(sum_list)
		else:
			self.m1_summary = None
			self.m2_supervised_summary = None
			self.m2_unsupervised_summary = None
			self.histogram_summary = None

	'''
		network variables property
	'''
	# @property
	# def m1_vars(self):
	# 	return self.x_encoder.vars + self.hx_decoder.vars

	# @property
	# def m2_vars(self):
	# 	return self.hx_y_encoder.vars + self.hx_classifier.vars + self.hz_y_decoder.vars

	@property
	def vars(self):
		return self.seg_classifier.vars + self.discriminator.vars

	'''
		train operations
	'''
	def train_on_batch_supervised(self, sess, x_batch, y_batch):
		step = sess.run([self.m1_global_step])[0]

		if step < self.m1_train_steps:
			# for m1 can only be trained unsupervised
			feed_dict = {
				self.xu : x_batch,
				self.is_training : True
			}
			m1_step, lr, loss, summ = self.train(sess, feed_dict, update_op=self.m1_train_op,
																step=self.m1_global_step,
																learning_rate=self.m1_learning_rate,
																loss=self.m1_loss, summary=self.m1_summary)
			return m1_step, lr, loss, summ
		else:
			feed_dict = {
				self.xl : x_batch,
				self.yl : y_batch,
				self.is_training : True
			}
			m2_step, lr, loss, summ = self.train(sess, feed_dict, 
																update_op = self.m2_supervised_train_op,
																step=self.m2_global_step,
																learning_rate=self.m2_supervised_learning_rate,
																loss = self.m2_su_loss, summary=self.m2_supervised_summary)

			return m2_step + step, lr, loss, [(m2_step,summ)]

	def train_on_batch_unsupervised(self, sess, x_batch):
		raise NotImplementedError
		# step = sess.run([self.m1_global_step])[0]
		# feed_dict = {
		# 	self.xu : x_batch,
		# 	self.is_training : True
		# }

		# if step < self.m1_train_steps:
		# 	m1_step, lr, loss, summ = self.train(sess, feed_dict, update_op=self.m1_train_op,
		# 														step=self.m1_global_step,
		# 														learning_rate=self.m1_learning_rate,
		# 														loss=self.m1_loss, summary=self.m1_summary)
		# 	return m1_step, lr, loss, summ
		# else:
		# 	m2_step, lr, loss, summ = self.train(sess, feed_dict, 
		# 														update_op = self.m2_unsupervised_train_op,
		# 														step=self.m2_global_step,
		# 														learning_rate=self.m2_unsupervised_learning_rate,
		# 														loss = self.m2_unsu_loss, summary=self.m2_unsupervised_summary)

		# 	return m2_step + step, lr, loss, [(m2_step,summ)]


	'''
		test operations
	'''
	def predict(self, sess, x_batch):
		'''
			p(y | x)
		'''
		feed_dict = {
			self.xt : x_batch,
			self.is_training : False
		}
		y_pred = sess.run([self.ytprobs], feed_dict = feed_dict)[0]
		return y_pred

	def hidden_variable(self, sess, x_batch):
		'''
			p(z | x)
		'''
		feed_dict = {
			self.xt : x_batch,
			self.is_training : False
		}
		mean_hz, log_var_hz = sess.run([self.mean_hzt, self.log_var_hzt], feed_dict=feed_dict)
		return mean_hz, log_var_hz

	'''
		summary operations
	'''
	def summary(self, sess):
		if self.has_summary:
			sum = sess.run(self.histogram_summary)
			return sum
		else:
			return None


