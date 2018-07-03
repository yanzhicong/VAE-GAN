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
import queue
import threading
import numpy as np

sys.path.append('.')
sys.path.append('../')

import tensorflow as tf

from validator.validator import get_validator

from .basetrainer import BaseTrainer


class SemiSupervisedTrainer(BaseTrainer):
	'''
		a semisupervised trainer class
		optional parameters including:
			'pretrain supervised step' : in the first, the model will be trained in supervised manner
			'pretrain unsupervised step' : then, the model will be trained in unsupervised manner
			'supervised step',
			'unsupervised step' : after above two train process, the model will be trained in
									both supervised manner and unsupervised manner in cycle,
									those two parameters control the ratio between the number of 
									supervised steps and unsupervised steps 
			'train steps' :  the total maximum train steps
			'continue train' : whether to load the checkpoint

		other parameters please refer to trainer/basetrainer.py class BaseTrainer,
	'''
	def __init__(self, config, model):
		self.config = config
		self.model = model

		super(SemiSupervisedTrainer, self).__init__(config, model)

		# 

		# optional parameters
		self.pretrain_steps = self.config.get('pretrain steps', [])
		self.supervised_step = self.config.get('supervised step', 1)
		self.unsupervised_step = self.config.get('unsupervised step', 1)

		# train data queue
		self.supervised_image_queue = queue.Queue(maxsize=100)
		self.supervised_image_inner_queue = queue.Queue(maxsize=self.batch_size * 300)

		self.unsupervised_image_queue = queue.Queue(maxsize=100)
		self.unsupervised_image_inner_queue = queue.Queue(maxsize=self.batch_size * 300)


		self.debug = self.config.get('debug', False)
		if self.debug:
			print('Semisupervised Trainer')
			print('\tsupervised step', self.config.get('supervised step', ''))
			print('\tunsupervised step', self.config.get('unsupervised step', ''))


		# others, for logging in screens
		self.su_epoch = 0
		self.su_step = 0
		self.su_lr = 0
		self.su_loss = 0
		self.unsu_epoch = 0
		self.unsu_step = 0
		self.unsu_lr = 0
		self.unsu_loss = 0


	def train(self, sess, dataset, model):


		if 'summary hyperparams string' in self.config:
			self.summary_writer = tf.summary.FileWriter(self.summary_dir + '/' + self.config['summary hyperparams string'], sess.graph)
		else:
			self.summary_writer = tf.summary.FileWriter(self.summary_dir, sess.graph)

		# start threads for queuing supervised train data and unsupervised train data
		# the supervised train data is stored in self.supervised_image_queue
		# the unsupervised train data is stored in self.unsupervised_image_queue
		self.coord = tf.train.Coordinator()
		threads = [threading.Thread(target=self.read_data_loop, 
										args=(self.coord, dataset, self.supervised_image_inner_queue, 'supervised')),
					threading.Thread(target=self.read_data_transport_loop, 
										args=(self.coord, self.supervised_image_inner_queue, self.supervised_image_queue, 'supervised'))]

		threads += [threading.Thread(target=self.read_data_loop, 
										args=(self.coord, dataset, self.unsupervised_image_inner_queue, 'unsupervised')),
					threading.Thread(target=self.read_data_transport_loop, 
										args=(self.coord, self.unsupervised_image_inner_queue, self.unsupervised_image_queue, 'unsupervised'))]
		for t in threads:
			t.start()


		self.train_initialize(sess, model)


		for train_manner, train_steps in self.pretrain_steps:
			if train_manner == 'supervised':
				for i in range(train_steps):
					epoch, batch_x, batch_y = self.supervised_image_queue.get()
					self.su_epoch = epoch
					step = self.train_inner_step(epoch, model, dataset, batch_x, batch_y, log_disp=False)
					self.log(step)
					if step >= int(self.config['train steps']):
							break
			elif train_manner == 'unsupervised':
				for i in range(train_steps):
					epoch, batch_x = self.unsupervised_image_queue.get()
					self.unsu_epoch = epoch
					step = self.train_inner_step(epoch, model, dataset, batch_x, log_disp=False)
					self.log(step)
					if step >= int(self.config['train steps']):
						break
			else :
				raise Exception('Wrong train manner ' + str(train_manner))


		# finally training in both supervised and unsupervised manner
		while True:
			for i in range(self.supervised_step):
				epoch, batch_x, batch_y = self.supervised_image_queue.get()
				self.su_epoch = epoch
				step = self.train_inner_step(epoch, model, dataset, batch_x, batch_y, log_disp=False)
				self.log(step)
				if step >= int(self.config['train steps']):
					break

			for i in range(self.unsupervised_step):
				epoch, batch_x = self.unsupervised_image_queue.get()
				self.unsu_epoch = epoch
				step = self.train_inner_step(epoch, model, dataset, batch_x, log_disp=False)
				self.log(step)
				if step >= int(self.config['train steps']):
					break

			if step >= int(self.config['train steps']):
				break

		# stop threads for queuing data
		self.coord.request_stop()
		while not self.supervised_image_queue.empty():
			epoch, batch_x, batch_y = self.supervised_image_queue.get()
		while not self.unsupervised_image_queue.empty():
			epoch, batch_x = self.unsupervised_image_queue.get()
		self.coord.join(threads)


	def log(self, step):
		if self.log_steps != 0 and step % self.log_steps == 0:
			print("supervised : [epoch : " + str(self.su_epoch) 
							+ ", step : " + str(self.su_step)
							+ ", lr : " + str(self.su_lr)
							+ ", loss : " + str(self.su_loss) + "] "
					"\\ unsupervised : [epoch : " + str(self.unsu_epoch) 
							+ ", step : " + str(self.unsu_step)
							+ ", lr : " + str(self.unsu_lr)
							+ ", loss : " +str(self.unsu_loss) + "]")
