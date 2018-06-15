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

class BaseTrainer(object):
	def __init__(self, config, model):
		self.config = config
		self.model = model

		self.summary_dir = os.path.join(self.config['assets dir'], self.config.get('summary dir', 'log'))
		self.checkpoint_dir = os.path.join(self.config['assets dir'], self.config.get('checkpoint dir', 'checkpoint'))

		if not os.path.exists(self.summary_dir):
			os.mkdir(self.summary_dir)
		if not os.path.exists(self.checkpoint_dir):
			os.mkdir(self.checkpoint_dir)

		self.summary_steps = int(self.config.get('summary steps', 0))
		self.log_steps = int(self.config.get('log steps', 0))
		self.save_steps = int(self.config.get('save checkpoint steps', 0))
<<<<<<< HEAD
		self.batch_size = int(self.config.get('batch_size', 16))

=======
>>>>>>> 2282cc26ef1f45414cb1b313771973457884af17

		self.validator_list = []
		for validator_config in self.config.get('validators', []):
			
			validator_params = validator_config.get('validator params', {})
			validator_params['assets dir'] = self.config['assets dir']

			validator = get_validator(validator_config['validator'], validator_params)

			if validator_config.get('has summary', False):
				validator.build_summary(self.model)

			validator_steps = int(validator_config['validate steps'])
			self.validator_list.append((validator_steps, validator))

<<<<<<< HEAD
		if self.config.get('print info', False):
			self.print_info()

	def print_info(self):
		print('Trainer Parameters :')
		print('\tsummary_dir : ', self.summary_dir)
		print('\tcheckpoint_dir : ', self.checkpoint_dir)
		print('\tsummary_steps : ', self.summary_steps)
		print('\tlog_steps : ', self.log_steps)
		print('\tsave_steps : ', self.save_steps)
		print('\tbatch_size : ', self.batch_size)


	def train_initialize(self, sess, model):
		sess.run(tf.global_variables_initializer())
		if self.config.get('continue train', False):
			if model.checkpoint_load(sess, self.checkpoint_dir):
				print("Continue Train...")
			else:
				print("Load Checkpoint Failed")

=======
>>>>>>> 2282cc26ef1f45414cb1b313771973457884af17

	def train_inner_step(self, epoch, sess, model, dataset, batch_x, batch_y=None):
		'''
			the inner function for train a batch of images,
			input :
				epoch, batch_x, batch_y : train batch,
				sess : tensorflow run time Session
				model : 
				dataset :
			return :
				the current train step
		'''
<<<<<<< HEAD
=======

>>>>>>> 2282cc26ef1f45414cb1b313771973457884af17
		if batch_y is None:
			step, lr, loss, summary = model.train_on_batch_unsupervised(sess, batch_x)
		else:
			step, lr, loss, summary = model.train_on_batch_supervised(sess, batch_x, batch_y)

		if summary:
			self.summary_writer.add_summary(summary, step-1)

		if self.log_steps != 0 and step % self.log_steps == 1:
			print("epoch : %d, step : %d, lr : %f, loss : %f"%(epoch, step-1, lr, loss))

		if self.summary_steps != 0 and step % self.summary_steps == 1:
			summary = model.summary(sess)
			if summary:
				self.summary_writer.add_summary(summary, step-1)

		if self.save_steps != 0 and step % self.save_steps == 1:
			model.checkpoint_save(sess, self.checkpoint_dir, step-1)

		for validator_steps, validator in self.validator_list:
			if validator_steps != 0 and step % validator_steps == 1:
				summary = validator.validate(model, dataset, sess, step-1)

				if summary is not None:
					self.summary_writer.add_summary(summary, step-1)

		return step


	'''
		multi thread util functions
		for example:

	'''
<<<<<<< HEAD
=======

>>>>>>> 2282cc26ef1f45414cb1b313771973457884af17
	def read_data_inner_loop(self, 
				coord, dataset, data_inner_queue, 
				indices, t_ind, nb_threads,
				epoch, method='supervised'):
		'''
			a inner read data loop thread, only be create or joined by read_data_loop.
<<<<<<< HEAD
			read data and put into @param.data_inner_queue in loop
=======
			read data and put into self.train_data_inner_queue in loop
>>>>>>> 2282cc26ef1f45414cb1b313771973457884af17
		'''
		if method == 'supervised':
			for i, ind in enumerate(indices):
				if not coord.should_stop():
					if i % nb_threads == t_ind:
<<<<<<< HEAD
=======

>>>>>>> 2282cc26ef1f45414cb1b313771973457884af17
						# read img and label by its index
						img, label = dataset.read_image_by_index_supervised(ind)
						if img is not None:
							data_inner_queue.put((epoch, img, label))
				else:
					break
		elif method == 'unsupervised':
			for i, ind in enumerate(indices):
				if not coord.should_stop():
					if i % nb_threads == t_ind:
<<<<<<< HEAD
						# read img by its index
=======
						# read img and label by its index
>>>>>>> 2282cc26ef1f45414cb1b313771973457884af17
						img = dataset.read_image_by_index_unsupervised(ind)
						if img is not None:
							data_inner_queue.put((epoch, img))
				else:
					break		
		else:
			raise Exception("wrong method of " + method)


	def read_data_loop(self, coord, dataset, data_inner_queue, method='supervised', nb_threads=4):
		'''
<<<<<<< HEAD
			create multiple threads to read data into @param.data_inner_queue
=======

>>>>>>> 2282cc26ef1f45414cb1b313771973457884af17
		'''
		epoch = 0
		while not coord.should_stop():

			# get train image indices of one epoch
			indices = dataset.get_image_indices(phase='train', method=method)

			# create multiple thread to read data
			threads = [threading.Thread(target=self.read_data_inner_loop, 
								args=(coord, dataset, data_inner_queue, 
										indices, t_ind, nb_threads, 
										epoch, method)) for t_ind in range(nb_threads)]
			for t in threads:
				t.start()

			coord.join(threads)
			epoch += 1


	def read_data_transport_loop(self, coord, data_inner_queue, data_queue, method='supervised'):
		'''
<<<<<<< HEAD
			transport data from @param.data_inner_queue to @param.data_queue in batch manner
=======
>>>>>>> 2282cc26ef1f45414cb1b313771973457884af17
		'''
		epoch_list = []
		batch_x = []
		batch_y = []

		if method == 'supervised':
			while not coord.should_stop():
				epoch, img, label = data_inner_queue.get()
				epoch_list.append(epoch)
				batch_x.append(img)
				batch_y.append(label)

				if len(batch_x) == self.batch_size:
					epoch = np.array(epoch_list).min()
					batch_x = np.array(batch_x)
					batch_y = np.array(batch_y)
					data_queue.put((epoch, batch_x, batch_y))
					epoch_list = []
					batch_x = []
					batch_y = []
<<<<<<< HEAD
					
			# clear the data inner queue to free the (read_data_inner_loop) thread
			while not data_inner_queue.empty():
				epoch, img, label = data_inner_queue.get()


		elif method == 'unsupervised':
			while not coord.should_stop():			
=======

		elif method == 'unsupervised':
>>>>>>> 2282cc26ef1f45414cb1b313771973457884af17
				epoch, img = data_inner_queue.get()
				epoch_list.append(epoch)
				batch_x.append(img)

				if len(batch_x) == self.batch_size:
					epoch = np.array(epoch_list).min()
					batch_x = np.array(batch_x)
					data_queue.put((epoch, batch_x))
					epoch_list = []
					batch_x = []
<<<<<<< HEAD

			# clear the data inner queue to free the (read_data_inner_loop) thread
			while not data_inner_queue.empty():
				epoch, img = data_inner_queue.get()
		else:
			raise Exception("wrong method of " + method)

=======
		else:
			raise Exception("wrong method of " + method)

		# clear the data inner queue to free the (read_data_inner_loop) thread
		while not data_inner_queue.empty():
			img, label = data_inner_queue.get()
>>>>>>> 2282cc26ef1f45414cb1b313771973457884af17
