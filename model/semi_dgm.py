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

from .basemodel import BaseModel


class SemiDeepGenerativeModel(BaseModel):
	"""
		Implementation of "Semi-Supervised Learning with Deep Generative Models"
		Diederik P. Kingma, Danilo J. Rezende, Shakir Mohamed, Max Welling

		@article{DBLP:journals/corr/KingmaRMW14,
			author    = {Diederik P. Kingma and
						Danilo Jimenez Rezende and
						Shakir Mohamed and
						Max Welling},
			title     = {Semi-Supervised Learning with Deep Generative Models},
			journal   = {CoRR},
			volume    = {abs/1406.5298},
			year      = {2014},
			url       = {http://arxiv.org/abs/1406.5298},
			archivePrefix = {arXiv},
			eprint    = {1406.5298},
			timestamp = {Wed, 07 Jun 2017 14:42:55 +0200},
			biburl    = {https://dblp.org/rec/bib/journals/corr/KingmaRMW14},
			bibsource = {dblp computer science bibliography, https://dblp.org}
		}
	"""

	def __init__(self, config,
		**kwargs
	):

		super(SemiDeepGenerativeModel, self).__init__(config, **kwargs)

		# parameters must be configured
		self.input_shape = config['input shape']
		self.hz_dim = config['hz_dim']
		self.hx_dim = config['hx_dim']
		self.nb_classes = config['nb_classes']
		self.config = config


		# optional params
		self.debug = config.get('debug', False)
		self.m1_train_steps = config.get('m1 train steps', 5000)
		self.m1_loss_weights = config.get('m1 loss weights', {})
		self.m2_loss_weights = config.get('m2 loss weights', {})

		# build model
		self.is_training = tf.placeholder(tf.bool, name='is_training')

		self.build_model_m1()
		self.build_model_m2()
		self.build_model()
		self.build_summary()



	def build_model_m1(self):

		self.xu = tf.placeholder(tf.float32, shape=[None,] + self.input_shape, name='xu_input')

		###########################################################################
		# network define
		# 
		# x_encoder : x -> hx
		self.config['x encoder params']['name'] = 'EncoderHX_X'
		self.config['x encoder params']['output_dims'] = self.hx_dim
		self.x_encoder = get_encoder(self.config['x encoder'], 
									self.config['x encoder params'], self.is_training)
		# decoder : hx -> x
		self.config['hx decoder params']['name'] = 'DecoderX_HX'
		# if self.config
		# self.config['hx decoder params']['output_dims'] = int(np.product(self.input_shape))
		self.hx_decoder = get_decoder(self.config['hx decoder'], self.config['hx decoder params'], self.is_training)

		###########################################################################
		# for unsupervised training:
		# 
		# xu --> mean_hxu, log_var_hxu ==> kl loss
		#					|
		# 			   sample_hxu --> xu_decode ==> reconstruction loss
		mean_hxu, log_var_hxu = self.x_encoder(self.xu)
		sample_hxu = self.draw_sample(mean_hxu, log_var_hxu)
		xu_decode = self.hx_decoder(sample_hxu)

		self.m1_loss_kl_z = (get_loss('kl', 'gaussian', {'mean' : mean_hxu, 'log_var' : log_var_hxu})
								* self.m1_loss_weights.get('kl z loss weight', 1.0))
		self.m1_loss_recon = (get_loss('reconstruction', 'mse', {'x' : self.xu, 'y' : xu_decode})
								* self.m1_loss_weights.get('reconstruction loss weight', 1.0))
		self.m1_loss = self.m1_loss_kl_z + self.m1_loss_recon


		###########################################################################
		# optimizer configure
		self.m1_global_step, m1_global_step_update = get_global_step('m1_step')

		(self.m1_train_op, 
			self.m1_learning_rate, 
				_) = get_optimizer_by_config(self.config['m1 optimizer'], self.config['m1 optimizer params'], 
													self.m1_loss, self.m1_vars, self.m1_global_step, m1_global_step_update)

	def build_model_m2(self):
		self.xl = tf.placeholder(tf.float32, shape=[None,] + self.input_shape, name='xl_input')
		self.yl = tf.placeholder(tf.float32, shape=[None, self.nb_classes], name='yl_input')

		###########################################################################
		# network define
		# 
		# hx_y_encoder : [hx, y] -> hz
		self.config['hx y encoder params']['name'] = 'EncoderHZ_HXY'
		self.config['hx y encoder params']['output_dims'] = self.hz_dim
		self.hx_y_encoder = get_encoder(self.config['hx y encoder'], 
									self.config['hx y encoder params'], self.is_training)
		# hz_y_decoder : [hz, y] -> hx
		self.config['hz y decoder params']['name'] = 'DecoderX_HZY'
		self.config['hz y decoder params']['output_dims'] = self.hx_dim
		self.hz_y_decoder = get_decoder(self.config['hz y decoder'], self.config['hz y decoder params'], self.is_training)
		# hx_classifier : hx -> ylogits
		self.config['hx classifier params']['name'] = 'ClassifierHX'
		self.config['hx classifier params']['output_dims'] = self.nb_classes
		self.hx_classifier = get_classifier(self.config['hx classifier'], self.config['hx classifier params'], self.is_training)

		###########################################################################
		# for supervised training:
		# 
		# xl --> mean_hxl, log_var_hxl
		#		  		  |
		#			 sample_hxl --> yllogits ==> classification loss
		#				  |
		# 			[sample_hxl, yl] --> mean_hzl, log_var_hzl ==> kl loss
		#				          |               |
		# 	  			        [yl,	   	   sample_hzl] --> xl_decode ==> reconstruction loss
		#
		mean_hxl, log_var_hxl = self.x_encoder(self.xl)
		sample_hxl = self.draw_sample(mean_hxl, log_var_hxl)
		yllogits = self.hx_classifier(sample_hxl)
		mean_hzl, log_var_hzl = self.hx_y_encoder(tf.concat([sample_hxl, self.yl], axis=1))
		sample_hzl = self.draw_sample(mean_hzl, log_var_hzl)
		decode_hxl = self.hz_y_decoder(tf.concat([sample_hzl, self.yl], axis=1))

		self.m2_su_loss_kl_z = (get_loss('kl', 'gaussian', {'mean' : mean_hzl, 
															'log_var' : log_var_hzl, })
								* self.m2_loss_weights.get('kl z loss weight', 1.0))
		self.m2_su_loss_recon = (get_loss('reconstruction', 'mse', {	'x' : sample_hxl, 
																		'y' : decode_hxl})
								* self.m2_loss_weights.get('reconstruction loss weight', 1.0))
		self.m2_su_loss_cls = (get_loss('classification', 'cross entropy', {'logits' : yllogits, 
																			'labels' : self.yl})
								* self.m2_loss_weights.get('classification loss weight', 1.0))
		self.m2_su_loss_reg = (get_loss('regularization', 'l2', { 'var_list' : self.hx_classifier.vars})
								* self.m2_loss_weights.get('regularization loss weight', 0.0001))
		self.m2_su_loss = ((self.m2_su_loss_kl_z + self.m2_su_loss_recon + self.m2_su_loss_cls + self.m2_su_loss_reg)
							* self.m2_loss_weights.get('supervised loss weight', 1.0))


		###########################################################################
		# for unsupervised training:
		#
		# xu --> mean_hxu, log_var_hxu
		#                |
		#             sample_hxu --> yulogits --> yuprobs
		# 				  |       
		#   		 [sample_hxu,    y0] --> mean_hzu0, log_var_hzu0 ==> kl_loss * yuprobs[0]
		# 				  |			  |					|
		#				  |			[y0,           sample_hzu0] --> decode_hxu0 ==> reconstruction loss * yuprobs[0]
		#				  |
		#   	     [sample_hxu,    y1] --> mean_hzu1, log_var_hzu1 ==> kl_loss * yuprobs[1]
		#				  |			  |			        |
		#				  |			[y1,           sample_hzu1] --> decode_hxu1 ==> reconstruction loss * yuprobs[1]
		#		.......
		#
		mean_hxu, log_var_hxu = self.x_encoder(self.xu)
		sample_hxu = self.draw_sample(mean_hxu, log_var_hxu)
		yulogits = self.hx_classifier(sample_hxu)
		yuprobs = tf.nn.softmax(yulogits)

		unsu_loss_kl_z_list = []
		unsu_loss_recon_list = []

		for i in range(self.nb_classes):
			yu_fake = tf.ones([tf.shape(self.xu)[0], ], dtype=tf.int32) * i
			yu_fake = tf.one_hot(yu_fake, depth=self.nb_classes)

			mean_hzu, log_var_hzu = self.hx_y_encoder(tf.concat([sample_hxu, yu_fake], axis=1))
			sample_hzu = self.draw_sample(mean_hzu, log_var_hzu)
			decode_hxu = self.hz_y_decoder(tf.concat([sample_hzu, yu_fake], axis=1))

			unsu_loss_kl_z_list.append(
				get_loss('kl', 'gaussian', {'mean' : mean_hzu, 
											'log_var' : log_var_hzu, 
											'instance_weight' : yuprobs[:, i] })
			)

			unsu_loss_recon_list.append(
				get_loss('reconstruction', 'mse', {	'x' : sample_hxu, 
													'y' : decode_hxu,
													'instance_weight' : yuprobs[:, i]})
			)

		self.m2_unsu_loss_kl_y = (get_loss('kl', 'bernoulli', { 'probs' : yuprobs}) * self.m2_loss_weights.get('kl y loss weight', 1.0))
		self.m2_unsu_loss_kl_z = (tf.reduce_sum(unsu_loss_kl_z_list) * self.m2_loss_weights.get('kl z loss weight', 1.0))
		self.m2_unsu_loss_recon = (tf.reduce_sum(unsu_loss_recon_list) * self.m2_loss_weights.get('reconstruction loss weight', 1.0))
		self.m2_unsu_loss_reg = (get_loss('regularization', 'l2', {'var_list' : self.hx_classifier.vars}) * self.m2_loss_weights.get('regularization', 0.0001))

		self.m2_unsu_loss = ((self.m2_unsu_loss_kl_z + self.m2_unsu_loss_recon + self.m2_unsu_loss_kl_y)
							* self.m2_loss_weights.get('unsupervised loss weight', 1.0))


		###########################################################################
		# optimizer configure
		self.m2_global_step, m2_global_step_update = get_global_step('m2_step')

		(self.m2_supervised_train_op,
			self.m2_supervised_learning_rate,
			 	_) = get_optimizer_by_config(self.config['m2 optimizer'], self.config['m2 optimizer params'], 
			 										self.m2_su_loss, self.m2_vars, self.m2_global_step, m2_global_step_update)


		(self.m2_unsupervised_train_op,
			self.m2_unsupervised_learning_rate,
			 	_) = get_optimizer_by_config(self.config['m2 optimizer'], self.config['m2 optimizer params'], 
			 										self.m2_unsu_loss, self.m2_vars, self.m2_global_step, m2_global_step_update)


	def build_model(self):
		# test input x
		self.xt = tf.placeholder(tf.float32, shape=[None,] + self.input_shape, name='xt_input')

		###########################################################################
		# for test models
		# 
		# xt --> mean_hxt, log_var_hxt
		#               |
		#             sample_hxt --> ytlogits --> ytprobs
		# 			   |			    			 |
		#		     [sample_hxt,    			  ytprobs] --> mean_hzt, log_var_hzt
		#
		mean_hxt, log_var_hxt = self.x_encoder(self.xt)
		sample_hxt = self.draw_sample(mean_hxt, log_var_hxt)
		ytlogits = self.hx_classifier(sample_hxt)
		# test sample class probilities
		self.ytprobs = tf.nn.softmax(ytlogits)
		# test sample hidden variable distribution
		self.mean_hzt, self.log_var_hzt = self.hx_y_encoder(tf.concat([sample_hxt, self.ytprobs], axis=1))


		###########################################################################
		# model saver
		self.saver = tf.train.Saver(self.vars + [self.m1_global_step, self.m2_global_step])

	def build_summary(self):

		if self.is_summary:

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
	@property
	def m1_vars(self):
		return self.x_encoder.vars + self.hx_decoder.vars

	@property
	def m2_vars(self):
		return self.hx_y_encoder.vars + self.hx_classifier.vars + self.hz_y_decoder.vars

	@property
	def vars(self):
		return self.x_encoder.vars + self.hx_decoder.vars + self.hx_y_encoder.vars + self.hx_classifier.vars + self.hz_y_decoder.vars


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
		step = sess.run([self.m1_global_step])[0]
		feed_dict = {
			self.xu : x_batch,
			self.is_training : True
		}

		if step < self.m1_train_steps:
			m1_step, lr, loss, summ = self.train(sess, feed_dict, update_op=self.m1_train_op,
																step=self.m1_global_step,
																learning_rate=self.m1_learning_rate,
																loss=self.m1_loss, summary=self.m1_summary)
			return m1_step, lr, loss, summ
		else:
			m2_step, lr, loss, summ = self.train(sess, feed_dict, 
																update_op = self.m2_unsupervised_train_op,
																step=self.m2_global_step,
																learning_rate=self.m2_unsupervised_learning_rate,
																loss = self.m2_unsu_loss, summary=self.m2_unsupervised_summary)

			return m2_step + step, lr, loss, [(m2_step,summ)]

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

	def hidden_variable_distribution(self, sess, x_batch):
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
		if self.is_summary:
			sum = sess.run(self.histogram_summary)
			return sum
		else:
			return None


