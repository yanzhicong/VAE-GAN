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


class SemiDeepGenerativeModel2(BaseModel):
	"""
		Implementation of "Semi-Supervised Learning with Deep Generative Models"
		Diederik P. Kingma, Danilo J. Rezende, Shakir Mohamed, Max Welling

		Experiment Model
		Including some modification to origin model/semi_dgm.py
		1. removing the m1/m2 stage from model/semi_dgm.py

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

	def __init__(self, config):

		super(SemiDeepGenerativeModel2, self).__init__(config)

		# parameters must be configured
		self.input_shape = config['input shape']
		self.hz_dim = config['hz_dim']
		self.hx_dim = config['hx_dim']
		self.nb_classes = config['nb_classes']
		self.config = config

		# optional params
		self.debug = config.get('debug', False)
		self.loss_weights = config.get('loss weights', {})

		# build model
		self.is_training = tf.placeholder(tf.bool, name='is_training')

		self.build_model()
		self.build_summary()

	def build_model(self):

		self.xu = tf.placeholder(tf.float32, shape=[None,] + self.input_shape, name='xu_input')
		self.xl = tf.placeholder(tf.float32, shape=[None,] + self.input_shape, name='xl_input')
		self.yl = tf.placeholder(tf.float32, shape=[None, self.nb_classes], name='yl_input')

		###########################################################################
		# network define
		# 
		# x_encoder : x -> hx
		self.config['x encoder params']['name'] = 'EncoderHX_X'
		self.config['x encoder params']["output dims"] = self.hx_dim
		self.x_encoder = get_encoder(self.config['x encoder'], self.config['x encoder params'], self.is_training)
		# hx_y_encoder : [hx, y] -> hz
		self.config['hx y encoder params']['name'] = 'EncoderHZ_HXY'
		self.config['hx y encoder params']["output dims"] = self.hz_dim
		self.hx_y_encoder = get_encoder(self.config['hx y encoder'], 
									self.config['hx y encoder params'], self.is_training)
		# hz_y_decoder : [hz, y] -> x_decode
		self.config['hz y decoder params']['name'] = 'DecoderX_HZY'
		self.config['hz y decoder params']["output dims"] = int(np.product(self.input_shape))
		self.hz_y_decoder = get_decoder(self.config['hz y decoder'], self.config['hz y decoder params'], self.is_training)
		# x_classifier : hx -> ylogits
		self.config['x classifier params']['name'] = 'ClassifierX'
		self.config['x classifier params']["output dims"] = self.nb_classes
		self.x_classifier = get_classifier(self.config['x classifier'], self.config['x classifier params'], self.is_training)


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

		hxl = self.x_encoder(self.xl)
		mean_hzl, log_var_hzl = self.hx_y_encoder(tf.concat([hxl, self.yl], axis=1))
		sample_hzl = self.draw_sample(mean_hzl, log_var_hzl)
		decode_xl = self.hz_y_decoder(tf.concat([sample_hzl, self.yl], axis=1))
		# decode_xl = self.hx_decoder(decode_hxl)

		yllogits = self.x_classifier(self.xl)


		self.su_loss_kl_z = (get_loss('kl', 'gaussian', {'mean' : mean_hzl, 
															'log_var' : log_var_hzl, })
								* self.loss_weights.get('kl z loss weight', 1.0))
		self.su_loss_recon = (get_loss('reconstruction', 'mse', {	'x' : self.xl, 
																		'y' : decode_xl})
								* self.loss_weights.get('reconstruction loss weight', 1.0))
		self.su_loss_cls = (get_loss('classification', 'cross entropy', {'logits' : yllogits, 
																			'labels' : self.yl})
								* self.loss_weights.get('classiciation loss weight', 1.0))

		self.su_loss_reg = (get_loss('regularization', 'l2', {'var_list' : self.x_classifier.vars })
								* self.loss_weights.get('regularization loss weight', 0.0001))

		self.su_loss = ((self.su_loss_kl_z + self.su_loss_recon + self.su_loss_cls + self.su_loss_reg)
							* self.loss_weights.get('supervised loss weight', 1.0))


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
		hxu = self.x_encoder(self.xu)
		yulogits = self.x_classifier(self.xu)
		yuprobs = tf.nn.softmax(yulogits)

		unsu_loss_kl_z_list = []
		unsu_loss_recon_list = []

		for i in range(self.nb_classes):
			yu_fake = tf.ones([tf.shape(self.xu)[0], ], dtype=tf.int32) * i
			yu_fake = tf.one_hot(yu_fake, depth=self.nb_classes)

			mean_hzu, log_var_hzu = self.hx_y_encoder(tf.concat([hxu, yu_fake], axis=1))
			sample_hzu = self.draw_sample(mean_hzu, log_var_hzu)
			decode_xu = self.hz_y_decoder(tf.concat([sample_hzu, yu_fake], axis=1))
			# decode_xu = self.hx_decoder(decode_hxu)

			unsu_loss_kl_z_list.append(
				get_loss('kl', 'gaussian', {'mean' : mean_hzu, 
											'log_var' : log_var_hzu, 
											'instance_weight' : yuprobs[:, i] })
			)

			unsu_loss_recon_list.append(
				get_loss('reconstruction', 'mse', {	'x' : self.xu, 
													'y' : decode_xu,
													'instance_weight' : yuprobs[:, i]})
			)

		self.unsu_loss_kl_y = (get_loss('kl', 'bernoulli', { 'probs' : yuprobs})
								* self.loss_weights.get('kl y loss weight', 1.0))
		self.unsu_loss_kl_z = (tf.reduce_sum(unsu_loss_kl_z_list)
								* self.loss_weights.get('kl z loss weight', 1.0))
		self.unsu_loss_recon = (tf.reduce_sum(unsu_loss_recon_list)
								* self.loss_weights.get('reconstruction loss weight', 1.0))

		self.unsu_loss_reg = (get_loss('regularization', 'l2', {'var_list' : self.x_classifier.vars })
								* self.loss_weights.get('regularization loss weight', 0.0001))

		self.unsu_loss = ((self.unsu_loss_kl_z + self.unsu_loss_recon + self.unsu_loss_kl_y + self.unsu_loss_reg)
							* self.loss_weights.get('unsupervised loss weight', 1.0))

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
		hxt = self.x_encoder(self.xt)
		ytlogits = self.x_classifier(self.xt)
		self.ytprobs = tf.nn.softmax(ytlogits)
		self.mean_hzt, self.log_var_hzt = self.hx_y_encoder(tf.concat([hxt, self.ytprobs], axis=1))

		###########################################################################
		# optimizer configure

		global_step, global_step_update = get_global_step()

		(self.supervised_train_op, 
			self.supervised_learning_rate, 
				_) = get_optimizer_by_config(self.config['optimizer'], 
											self.config['optimizer params'], self.su_loss, self.vars, 
											global_step, global_step_update)
		(self.unsupervised_train_op, 
			self.unsupervised_learning_rate,
				_) = get_optimizer_by_config(self.config['optimizer'], 
											self.config['optimizer parmas'], self.unsu_loss, self.vars,
											global_step, global_step_update)

		###########################################################################
		# model saver
		self.saver = tf.train.Saver(self.vars + [self.global_step,])

	def build_summary(self):

		if self.has_summary:
			common_sum_list = []
			common_sum_list.append(tf.summary.scalar('learning_rate', self.learning_rate))

			# summary scalars are logged per step
			sum_list = []
			sum_list.append(tf.summary.scalar('unsupervised/kl_z_loss', self.unsu_loss_kl_z))
			sum_list.append(tf.summary.scalar('unsupervised/kl_y_loss', self.unsu_loss_kl_y))
			sum_list.append(tf.summary.scalar('unsupervised/reconstruction_loss', self.unsu_loss_recon))
			sum_list.append(tf.summary.scalar('unsupervised/loss', self.unsu_loss))
			sum_list.append(tf.summary.scalar('unsupervised/learning_rate', self.unsupervised_learning_rate))

			self.unsupervised_summary = tf.summary.merge(sum_list + common_sum_list)
			
			sum_list = []
			sum_list.append(tf.summary.scalar('supervised/kl_z_loss', self.su_loss_kl_z))
			sum_list.append(tf.summary.scalar('supervised/reconstruction_loss', self.su_loss_recon))
			sum_list.append(tf.summary.scalar('supervised/clasification_loss', self.su_loss_cls))
			sum_list.append(tf.summary.scalar('supervised/loss', self.su_loss))
			sum_list.append(tf.summary.scalar('supervised/learning_rate', self.supervised_learning_rate))
			self.supervised_summary = tf.summary.merge(sum_list + common_sum_list)

			# summary hists are logged by calling self.summary()
			sum_list = [tf.summary.histogram(var.name, var) for var in self.vars]
			self.histogram_summary = tf.summary.merge(sum_list)
		else:
			# self.summary = None
			self.supervised_summary = None
			self.unsupervised_summary = None
			self.histogram_summary = None

	'''
		network variables property
	'''
	@property
	def vars(self):
		return self.x_encoder.vars + self.hx_y_encoder.vars + self.x_classifier.vars + self.hz_y_decoder.vars

	'''
		train operations
	'''
	def train_on_batch_supervised(self, sess, x_batch, y_batch):

		# step = sess.run([self.global_step])[0]

		feed_dict = {
			self.xl : x_batch,
			self.yl : y_batch,
			self.is_training : True
		}
		return self.train(sess, feed_dict, update_op = self.supervised_train_op,
									loss = self.su_loss, summary=self.supervised_summary)

	def train_on_batch_unsupervised(self, sess, x_batch):
		# step = sess.run([self.global_step])[0]
		feed_dict = {
			self.xu : x_batch,
			self.is_training : True
		}

		return self.train(sess, feed_dict, update_op = self.unsupervised_train_op,
									loss = self.unsu_loss, summary = self.unsupervised_summary)

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

	def generate(self, sess, z_batch):
		pass

	'''
		summary operations
	'''
	def summary(self, sess):
		if self.has_summary:
			sum = sess.run(self.histogram_summary)
			return sum
		else:
			return None
