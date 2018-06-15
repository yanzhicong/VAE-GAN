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

		self.input_shape = config['input_shape']
		self.z_dim = config['z_dim']
		self.nb_classes = config['nb_classes']
		self.config = config

		self.is_training = tf.placeholder(tf.bool, name='is_training')

		self.build_model()

		if self.is_summary:
			self.get_summary()


	def build_model(self):

		self.xl = tf.placeholder(tf.float32, shape=[None,] + self.input_shape, name='xl_input')
		self.yl = tf.placeholder(tf.float32, shape=[None, self.nb_classes], name='yl_input')		

		self.xu = tf.placeholder(tf.float32, shape=[None,] + self.input_shape, name='xu_input')
		self.eps = tf.placeholder(tf.float32, shape=[None, self.z_dim], name='eps')
		self.eps2 = tf.placeholder(tf.float32, shape=[None, self.z_dim, self.nb_classes], name='eps2')

		self.xtest = tf.placeholder(tf.float32, shape=[None,] + self.input_shape, name='x_test')


		self.x_encoder = get_encoder(self.config['x encoder'], self.config['x encoder params'], self.config, self.is_training,
					net_name='EncoderSimpleX')

		self.hx_y_encoder = get_encoder(self.config['hx_y encoder'], self.config['hx_y encoder params'], self.config, self.is_training,
					net_name='EncoderSimpleHXY')

		self.hx_classifier = get_classifier(self.config['hx classifier'], self.config['hx classifier params'], self.config, self.is_training)

		self.decoder = get_decoder(self.config['decoder'], self.config['decoder params'], self.config, self.is_training)

		###########################################################################
		# for supervised training:

		hxl = self.x_encoder(self.xl, reuse=False)

		yl_logits, end_points = self.hx_classifier(hxl, reuse=False)
		yl_probs = tf.nn.softmax(yl_logits)

		hx_yl = tf.concat([hxl, self.yl], axis=1)

		mean_z, log_var_z = self.hx_y_encoder(hx_yl, reuse=False)

		z_sample_l = mean_z + tf.exp(log_var_z / 2) * self.eps

		x_decode = self.decoder(z_sample_l, reuse=False)



		self.su_loss_kl_z = (get_loss('kl', 'gaussian', {'mean' : mean_z, 'log_var' : log_var_z})
								* self.config.get('loss kl z prod', 1.0))
		self.su_loss_recon = (get_loss('reconstruction', 'mse', {'x' : self.xl, 'y' : x_decode})
								* self.config.get('loss recon prod', 1.0))
		self.su_loss_cls = (get_loss('classification', 'cross entropy', {'logits' : yl_logits, 'labels' : self.yl})
								* self.config.get('loss cls prod', 1.0))
		self.su_loss = self.su_loss_kl_z + self.su_loss_recon + self.su_loss_cls

		# sel


		###########################################################################
		# for unsupervised training:

		hxu = self.x_encoder(self.xu, reuse=True)
		yu_logits, end_points = self.hx_classifier(hxu, reuse=True)
		yu_probs = tf.nn.softmax(yu_logits)
		yu_logprobs = tf.log(yu_probs)
		
		self.unsu_loss_kl_y = get_loss('kl', 'bernoulli', {'probs' : yu_probs})
		self.unsu_loss_kl_y *= self.config.get('loss kl y prod', 1.0)

		unsu_loss_kl_z_list = []
		unsu_loss_recon_list = []

		for i in range(self.nb_classes):
			y_fake = tf.ones([tf.shape(self.xu)[0], ], dtype=tf.int32) * i
			y_fake = tf.one_hot(y_fake, depth=self.nb_classes)
			hx_yf = tf.concat([hxu, y_fake], axis=1)

			mean_z, log_var_z = self.hx_y_encoder(hx_yf, reuse=True)

			z_sample_u = mean_z = tf.exp(log_var_z / 2) * self.eps2[:, :, i]

			x_decode_u = self.decoder(z_sample_u, reuse=True)

			unsu_loss_kl_z_list.append(
				get_loss('kl', 'gaussian', {'mean' : mean_z, 'log_var' : log_var_z}) * yu_probs[:, i]
			)
			unsu_loss_recon_list.append(
				get_loss('reconstruction', 'mse', {'x' : self.xu, 'y' : x_decode_u}) * yu_probs[:, i]
			)

		self.unsu_loss_kl_z = tf.reduce_sum(unsu_loss_kl_z_list)
		self.unsu_loss_kl_z *= self.config.get('loss kl z prod', 1.0)
		self.unsu_loss_recon = tf.reduce_sum(unsu_loss_recon_list)
		self.unsu_loss_recon *= self.config.get('loss recon unsupervised prod', 1.0)

		self.unsu_loss = self.unsu_loss_kl_y + self.unsu_loss_kl_z + self.unsu_loss_recon


		###########################################################################
		# for test models

		hxtest = self.x_encoder(self.xtest, reuse=True)
		ylogits, endpoints = self.hx_classifier(hxtest, reuse=True)
		self.ytest = tf.nn.softmax(ylogits)

		hxtest_ytest = tf.concat([hxtest, self.ytest], axis=1)
		self.mean_z_test, self.log_var_z_test = self.hx_y_encoder(hxtest_ytest, reuse=True)

		###########################################################################
		# optimizer configure
		self.global_step, self.global_step_update = get_global_step()
		if 'lr' in self.config:
			self.learning_rate = get_learning_rate(self.config['lr_scheme'], float(self.config['lr']), self.global_step, self.config['lr_params'])
			optimizer_params = {'learning_rate' : self.learning_rate}
		else:
			optimizer_params = {}


		self.supervised_optimizer = get_optimizer(self.config['optimizer'], optimizer_params, self.su_loss, 
						self.vars)
		self.unsupervised_optimizer = get_optimizer(self.config['optimizer'], optimizer_params, self.unsu_loss, 
						self.vars)

		self.supervised_train_update = tf.group([self.supervised_optimizer, self.global_step_update])
		self.unsupervised_train_update = tf.group([self.unsupervised_optimizer, self.global_step_update])

		###########################################################################
		# model saver
		self.saver = tf.train.Saver(self.vars + [self.global_step,])
		

	@property
	def vars(self):
		return self.x_encoder.vars + self.hx_y_encoder.vars + self.hx_classifier.vars + self.decoder.vars


	def train_on_batch_supervised(self, sess, x_batch, y_batch):
		feed_dict = {
			self.xl : x_batch,
			self.yl : y_batch,
			self.eps : np.random.randn(x_batch.shape[0], self.z_dim),
			self.is_training : True
		}
		return self.train(sess, feed_dict, 
								update_op = self.supervised_train_update,
								loss = self.su_loss,
								summary = self.su_sum_scalar)


	def train_on_batch_unsupervised(self, sess, x_batch):
		feed_dict = {
			self.xu : x_batch,
			self.eps2 : np.random.randn(x_batch.shape[0], self.z_dim, self.nb_classes),
			self.is_training : True
		}
		return self.train(sess, feed_dict, 
								update_op = self.unsupervised_train_update,
								loss = self.unsu_loss,
								summary = self.unsu_sum_scalar)


	def predict(self, sess, x_batch):
		'''
			p(y | x)s
		'''
		feed_dict = {
			self.xtest : x_batch,
			self.is_training : False
		}
		y_pred = sess.run([self.ytest], feed_dict = feed_dict)[0]
		return y_pred


	def hidden_variable_distribution(self, sess, x_batch):
		'''
			p(z | x)
		'''
		feed_dict = {
			self.xtest : x_batch,
			self.is_training : False
		}
		mean_z, log_var_z = sess.run([self.mean_z_test, self.log_var_z_test], feed_dict=feed_dict)
		return mean_z, log_var_z


	def summary(self, sess):
		if self.is_summary:
			sum = sess.run(self.sum_hist)
			return sum
		else:
			return None


	def get_summary(self):
		# summary scalars are logged per step

		sum_list = []
		sum_list.append(tf.summary.scalar('supervised/kl_z_loss', self.su_loss_kl_z))
		sum_list.append(tf.summary.scalar('supervised/reconstruction_loss', self.su_loss_recon))
		sum_list.append(tf.summary.scalar('supervised/classify loss', self.su_loss_cls))
		sum_list.append(tf.summary.scalar('supervised/loss', self.su_loss))
		self.su_sum_scalar = tf.summary.merge(sum_list)
		
		sum_list = []
		sum_list.append(tf.summary.scalar('unsupervised/kl_z_loss', self.unsu_loss_kl_z))
		sum_list.append(tf.summary.scalar('unsupervised/kl_y_loss', self.unsu_loss_kl_y))
		sum_list.append(tf.summary.scalar('unsupervised/reconstruction_loss', self.unsu_loss_recon))
		sum_list.append(tf.summary.scalar('unsupervised/loss', self.unsu_loss))
		self.unsu_sum_scalar = tf.summary.merge(sum_list)
		
		# summary hists are logged by calling self.summary()
		hist_sum_list = [tf.summary.histogram(var.name, var) for var in self.vars]
		self.sum_hist = tf.summary.merge(hist_sum_list)
