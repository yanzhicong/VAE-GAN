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
import numpy as np

sys.path.append('.')
sys.path.append('../')

import tensorflow as tf
import tensorflow.contrib.layers as tcl


from tensorflow.contrib.tensorboard.plugins import projector


from .base_validator import BaseValidator

class TensorboardEmbedding(BaseValidator):
	""" Plot the model output or mediate features into tensorboard embedding panel.
	
	"""

	def __init__(self, config):
		super(TensorboardEmbedding, self).__init__(config)

		self.assets_dir = self.config['assets dir']
		self.log_dir = self.config.get('log dir', 'embedding')
		self.log_dir = os.path.join(self.assets_dir, self.log_dir)

		self.z_shape = list(self.config['z shape'])
		self.x_shape = list(self.config['x shape'])
		self.nb_samples = self.config.get('nb samples', 1000)
		self.batch_size = self.config.get('batch_size', 100)

		self.nb_samples = self.nb_samples // self.batch_size * self.batch_size


		if not os.path.exists(self.log_dir):
			os.mkdir(self.log_dir)

		with open(os.path.join(self.log_dir, 'metadata.tsv'), 'w') as f:
			f.write("Index\tLabel\n")
			for i in range(self.nb_samples):
				f.write("%d\t%d\n"%(i, 0))
			for i in range(self.nb_samples):
				f.write("%d\t%d\n"%(i+self.nb_samples, 1))

		summary_writer = tf.summary.FileWriter(self.log_dir)
		config = projector.ProjectorConfig()
		embedding = config.embeddings.add()
		embedding.tensor_name = "test"
		embedding.metadata_path = "metadata.tsv"
		projector.visualize_embeddings(summary_writer, config)

		self.plot_array_var = tf.get_variable('test', shape=[self.nb_samples*2, int(np.product(self.x_shape))])
		self.saver = tf.train.Saver([self.plot_array_var])

	def validate(self, model, dataset, sess, step):

		plot_array_list = []
		indices = dataset.get_image_indices(phase='train', method='unsupervised')
		indices = np.random.choice(indices, size=self.nb_samples)

		for i, ind in enumerate(indices):
			test_x = dataset.read_image_by_index(ind, phase='train', method='unsupervised')
			if isinstance(test_x, list):
				for x in test_x:
					x = x.reshape([-1,])
					plot_array_list.append(x)
					if len(plot_array_list) >= self.nb_samples:
						break
			elif test_x is not None:
				test_x = test_x.reshape([-1,])
				plot_array_list.append(test_x)
			if len(plot_array_list) >= self.nb_samples:
				break

		for i in range(self.nb_samples // self.batch_size):
			batch_z = np.random.randn(*([self.nb_samples,] + self.z_shape))
			batch_x = model.generate(sess, batch_z)
			for i in range(self.batch_size):
				plot_array_list.append(batch_x[i].reshape([-1]))

		plot_array_list = np.array(plot_array_list)

		sess.run(self.plot_array_var.assign(plot_array_list))

		self.saver.save(sess, os.path.join(self.log_dir, 'model.ckpt'), 
							global_step=step, 
							write_meta_graph=False,
							strip_default_attrs=True)

