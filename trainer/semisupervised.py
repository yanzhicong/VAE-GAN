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
	def __init__(self, config, model):
		self.config = config
		self.model = model

		super(SemiSupervisedTrainer, self).__init__(config, model)

		self.supervised_step = self.config.get('supervised step', 1)
		self.unsupervised_step = self.config.get('unsupervised step', 1)

		self.supervised_image_queue = queue.Queue(maxsize=5)
		self.supervised_image_inner_queue = queue.Queue(maxsize=self.batch_size * 3)

		self.unsupervised_image_queue = queue.Queue(maxsize=5)
		self.unsupervised_image_inner_queue = queue.Queue(maxsize=self.batch_size*3)


	def train(self, sess, dataset, model):

		self.summary_writer = tf.summary.FileWriter(self.summary_dir, sess.graph)
		sess.run(tf.global_variables_initializer())

		self.train_initialize(sess, model)

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


		while True:
			# print('supervised')
			for i in range(self.supervised_step):
				epoch, batch_x, batch_y = self.supervised_image_queue.get()
				step = self.train_inner_step(epoch, sess, model, dataset, batch_x, batch_y)
				if step > int(self.config['train steps']):
					break

			# print('unsupervised')
			for i in range(self.unsupervised_step):
				epoch, batch_x = self.unsupervised_image_queue.get()
				step = self.train_inner_step(epoch, sess, model, dataset, batch_x)

				if step > int(self.config['train steps']):
					break

			if step > int(self.config['train steps']):
				break

		self.coord.request_stop()
		while not self.supervised_image_queue.empty():
			epoch, batch_x, batch_y = self.supervised_image_queue.get()
		while not self.unsupervised_image_queue.empty():
			epoch, batch_x = self.unsupervised_image_queue.get()
		self.coord.join(threads)

