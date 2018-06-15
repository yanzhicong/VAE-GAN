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

import tensorflow as tf

from validator.validator import get_validator


from .basetrainer import BaseTrainer



class UnsupervisedTrainer(BaseTrainer):
	def __init__(self, config, model):
		self.config = config
		self.model = model

		super(UnsupervisedTrainer, self).__init__(config, model)

		# self.summary_dir = os.path.join(self.config['assets dir'], self.config.get('summary dir', 'log'))
		# self.checkpoint_dir = os.path.join(self.config['assets dir'], self.config.get('checkpoint dir', 'checkpoint'))

		# if not os.path.exists(self.summary_dir):
		# 	os.mkdir(self.summary_dir)
		# if not os.path.exists(self.checkpoint_dir):
		# 	os.mkdir(self.checkpoint_dir)

		# self.summary_steps = int(self.config.get('summary steps', 0))
		# self.log_steps = int(self.config.get('log steps', 0))
		# self.save_steps = int(self.config.get('save checkpoint steps', 0))

		# self.validator_list = []
		# for validator_config in self.config.get('validators', []):
			
		# 	validator_params = validator_config.get('validator params', {})
		# 	validator_params['assets dir'] = self.config['assets dir']

		# 	validator = get_validator(validator_config['validator'], validator_params)
		# 	validator_steps = int(validator_config['validate steps'])
		# 	self.validator_list.append((validator_steps, validator))


	def train(self, sess, dataset, model):

		self.summary_writer = tf.summary.FileWriter(self.summary_dir, sess.graph)
		sess.run(tf.global_variables_initializer())

		if self.config.get('continue train', False):
			if model.checkpoint_load(sess, self.checkpoint_dir):
				print("Continue Train...")
			else:
				print("Load Checkpoint Failed")
			step = -1
		else:
			step = 0

		epoch = 0

		while True:
			for index, batch_x in dataset.iter_train_images_unsupervised():

				step = self.train_inner_step(epoch, sess, model, dataset, batch_x)

				# if self.summary_steps != 0 and step % self.summary_steps == 0:
				# 	summary = model.summary(sess)
				# 	if summary:
				# 		self.summary_writer.add_summary(summary, step)

				# step, lr, loss, summary = model.train_on_batch_unsupervised(sess, batch_x)

				# if summary:
				# 	self.summary_writer.add_summary(summary, step)

				# if self.log_steps != 0 and step % self.log_steps == 0:
				# 	print("epoch : %d, step : %d, lr : %f, loss : %f"%(epoch, step, lr, loss))

				# if self.save_steps != 0 and step % self.save_steps == 0:
				# 	model.checkpoint_save(sess, self.checkpoint_dir, step)

				# for validator_steps, validator in self.validator_list:
				# 	if validator_steps != 0 and step % validator_steps == 0:
				# 		validator.validate(model, dataset, sess, step)

				if step > int(self.config['train steps']):
					return
			epoch += 1
