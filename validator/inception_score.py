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

import os.path
import tarfile

import numpy as np
from six.moves import urllib
import glob
import scipy.misc
import math

import tensorflow as tf
import tensorflow.contrib.layers as tcl

from netutils.metric import get_metric
from .base_validator import BaseValidator


MODEL_DIR = './validator/inception_score/imagenet'
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'

g1=tf.Graph()

def _init_inception():
	# global softmax
	if not os.path.exists(MODEL_DIR):
		os.makedirs(MODEL_DIR)
	filename = DATA_URL.split('/')[-1]
	filepath = os.path.join(MODEL_DIR, filename)
	if not os.path.exists(filepath):
		def _progress(count, block_size, total_size):
			sys.stdout.write('\r>> Downloading %s %.1f%%' % (
				filename, float(count * block_size) / float(total_size) * 100.0))
			sys.stdout.flush()
		filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
		print()
		statinfo = os.stat(filepath)
		print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
	tarfile.open(filepath, 'r:gz').extractall(MODEL_DIR)

	with g1.as_default():

		with tf.gfile.FastGFile(os.path.join(
				MODEL_DIR, 'classify_image_graph_def.pb'), 'rb') as f:
			graph_def = tf.GraphDef()
			graph_def.ParseFromString(f.read())
			_ = tf.import_graph_def(graph_def, name='')

		with tf.Session(graph=g1) as sess:
			pool3 = sess.graph.get_tensor_by_name('pool_3:0')
			# Works with an arbitrary minibatch size.
			# ops = pool3.graph.get_operations()
			# for op_idx, op in enumerate(ops):
			# 	for o in op.outputs:
			# 		shape = o.get_shape()
			# 		shape = [s.value for s in shape]
			# 		new_shape = []
			# 		for j, s in enumerate(shape):
			# 			if s == 1 and j == 0:
			# 				new_shape.append(None)
			# 			else:
			# 				new_shape.append(s)
			# 		o.set_shape(tf.TensorShape(new_shape))
			# 		print(o.name, new_shape, o.get_shape())
			w = sess.graph.get_operation_by_name("softmax/logits/MatMul").inputs[1]
			logits = tf.matmul(tf.squeeze(pool3, [1, 2]), w)
			softmax = tf.nn.softmax(logits)
			return softmax

class InceptionScore(BaseValidator):
	"""	Test generative model with Inception Score measurement, and write inception score to tensorboard summary
	"""

	def __init__(self, config):

		super(InceptionScore, self).__init__(config)

		self.config = config
		self.nb_samples = config.get('num_samples', 100)
		self.z_shape = config['z shape']
		self.rounds = config.get('rounds', 10)
		self.scalar_range = config.get("output scalar range", [0.0, 1.0])		# the output range of generative model,
																		# for sigmoid activation model is [0.0, 1.0],
																		# and for tanh activation model is [-1.0, 1.0]
		self.softmax = _init_inception()
		self.has_summary = True

	def build_summary(self, model):
		self.inception_score = tf.placeholder(tf.float32,	name='inception_score')
		self.inception_score_std = tf.placeholder(tf.float32,	name='inception_score_std')

		self.summary_list = []
		self.summary_list.append(tf.summary.scalar('inception_score', self.inception_score))
		self.summary_list.append(tf.summary.scalar('inception_score_std', self.inception_score_std))

		self.summary = tf.summary.merge(self.summary_list)
		

	def validate(self, model, dataset, sess, step):

		all_samples = []

		for i in range(self.rounds):
			batch_z = np.random.randn(*([self.nb_samples, ] + self.z_shape))
			batch_x = model.generate(sess, batch_z)
			img = (((batch_x - self.scalar_range[0]) / (self.scalar_range[1] - self.scalar_range[0])) * 255.0).astype('int32')
			all_samples.append(img)

		all_samples = np.concatenate(all_samples, axis=0)
		inc_score, inc_score_std = self.get_inception_score(list(all_samples))

		print('inception score : ', inc_score)
		print('inception score std : ', inc_score_std)

		feed_dict = {
			self.inception_score : inc_score,
			self.inception_score_std : inc_score_std
		}

		summ = sess.run([self.summary], feed_dict=feed_dict)[0]
		return summ

	# Call this function with list of images. Each of elements should be a
	# numpy array with values ranging from 0 to 255.
	def get_inception_score(self, images, splits=10):
		assert(type(images) == list)
		assert(type(images[0]) == np.ndarray)
		assert(len(images[0].shape) == 3)
		assert(np.max(images[0]) > 10)
		assert(np.min(images[0]) >= 0.0)

		inps = []
		for img in images:
			img = img.astype(np.float32)
			inps.append(np.expand_dims(img, 0))

		bs = 1
		tfconfig = tf.ConfigProto()
		tfconfig.gpu_options.allow_growth = True
		with tf.Session(graph=g1, config=tfconfig) as sess:
			preds = []
			n_batches = int(math.ceil(float(len(inps)) / float(bs)))
			for i in range(n_batches):
				inp = inps[(i * bs):min((i + 1) * bs, len(inps))]
				inp = np.concatenate(inp, 0)
				pred = sess.run(self.softmax, {'ExpandDims:0': inp})
				preds.append(pred)
			preds = np.concatenate(preds, 0)

			scores = []
			for i in range(splits):
				part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
				kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
				kl = np.mean(np.sum(kl, 1))
				scores.append(np.exp(kl))
			return np.mean(scores), np.std(scores)

