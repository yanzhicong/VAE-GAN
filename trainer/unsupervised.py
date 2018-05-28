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
import tensorflow as tf


class UnsupervisedTrainer(object):
	def __init__(self, config):
		self.config = config

		self.summary_dir = os.path.join(self.config['assets dir'], self.config.get('summary dir', 'log'))
		self.checkpoint_dir = os.path.join(self.config['assets dir'], self.config.get('checkpoint dir', 'checkpoint'))

		if not os.path.exists(self.summary_dir):
			os.mkdir(self.summary_dir)
		if not os.path.exists(self.checkpoint_dir):
			os.mkdir(self.checkpoint_dir)

		self.summary_steps = int(self.config.get('summary steps', 0))
		self.log_steps = int(self.config.get('log steps', 0))
		self.save_steps = int(self.config.get('save checkpoint steps', 0))


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


		while True:
			for index, batch_x in dataset.iter_images():

				if self.summary_steps != 0 and step % self.summary_steps == 0:
					summary = model.summary(sess)
					self.summary_writer.add_summary(summary, step)

				if self.log_steps != 0 and step % self.log_steps == 0:
					print("step : %d, lr : %f, loss : %f"%(step, lr, loss))

				if self.save_steps != 0 and step % self.save_steps == 0:
					model.checkpoint_save(sess, self.checkpoint_dir, step)


				step, lr, loss, summary = model.train_on_batch_unsupervised(sess, batch_x)
				if summary:
					self.summary_writer.add_summary(summary, step)

				if step > int(self.config['train steps']):
					return
