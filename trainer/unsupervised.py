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
sys.path.append('../')

import queue
import threading
import tensorflow as tf

from validator.validator import get_validator

from .base_trainer import BaseTrainer


class UnsupervisedTrainer(BaseTrainer):
	'''
	'''
	def __init__(self, config, model, sess):
		self.config = config
		self.model = model

		super(UnsupervisedTrainer, self).__init__(config, model, sess)
		self.multi_thread = self.config.get('multi thread', False)

		if self.multi_thread:
			self.buffer_depth = self.config.get('buffer depth', 50)
			self.train_data_queue = queue.Queue(maxsize=self.buffer_depth)
			self.train_data_inner_queue = queue.Queue(maxsize=self.batch_size*self.buffer_depth)

	def train(self, sess, dataset, model):

		# if 'summary hyperparams string' in self.config:
		# 	self.summary_writer = tf.summary.FileWriter(self.summary_dir + '/' + self.config['summary hyperparams string'], sess.graph)
		# else:
		# 	self.summary_writer = tf.summary.FileWriter(self.summary_dir, sess.graph)
		
		self.train_initialize(sess, model)

		if self.multi_thread:
			self.coord = tf.train.Coordinator()
			threads = [threading.Thread(target=self.read_data_loop, 
											args=(self.coord, dataset, self.train_data_inner_queue, 'train', 'unsupervised')),
						threading.Thread(target=self.read_data_transport_loop, 
											args=(self.coord, self.train_data_inner_queue, self.train_data_queue, 'train', 'unsupervised'))]
			for t in threads:
				t.start()

		if self.multi_thread : 
			# in multi thread model, the image data were read in by dataset.get_train_indices()
			# and dataset.read_train_image_by_index()
			while True:
				epoch, batch_x = self.train_data_queue.get()
				step = self.train_inner_step(epoch, model, dataset, batch_x)
				if self.train_data_queue.empty() and step % 10 == 0:
					print('info : train data buffer empty')
				if step > int(self.config['train steps']):
					break
		else:
			epoch = 0
			while True:
				# in single thread model, the image data were read in by dataset.iter_train_images()
				for ind, batch_x in dataset.iter_train_images(method='unsupervised'):
					step = self.train_inner_step(epoch, model, dataset, batch_x)
					if step > int(self.config['train steps']):
						return
				epoch += 1

		# join all thread when in multi thread model
		self.coord.request_stop()
		while not self.train_data_queue.empty():
			epoch, batch_x = self.train_data_queue.get()
		self.train_data_inner_queue.task_done()
		self.train_data_queue.task_done()
		self.coord.join(threads)


