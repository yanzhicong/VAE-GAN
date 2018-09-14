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

sys.path.append('.')
sys.path.append('../')


class BaseValidator(object):
	"""	The base of validator classes.
	validator is another import module in this project
	During training process, the validator.validate is called at intervals to view the
	model performence and detect bugs in model.
	"""
	def __init__(self, config):
		self.config = config
		self.assets_dir = config['assets dir']
		self.has_summary = False

	#
	# Please override the following functions in derived class
	# 
	def build_summary(self, model):
		pass

	def validate(self, model, dataset, sess, step):
		return NotImplementedError

	#
	# Util functions
	#
	def parallel_data_reading(self, dataset, indices, phase, method, buffer_depth, nb_threads=4):
		
		self.t_should_stop = False

		data_queue = queue.Queue(maxsize=buffer_depth)

		def read_data_inner_loop(dataset, data_queue, indices, t_ind, nb_threads):
			for i, ind in enumerate(indices):
				if i % nb_threads == t_ind:
					# read img and label by its index
					img, label = dataset.read_image_by_index(ind, 'val', 'supervised')
					if isinstance(img, list) and isinstance(label, list):
						for _img, _label in zip(img, label):
							data_queue.put((img, label))
					elif img is not None:
						data_queue.put((img, label))


		def read_data_loop(indices, dataset, data_queue, nb_threads):
			threads = [threading.Thread(target=read_data_inner_loop, 
				args=(dataset, data_queue, indices, t_ind, nb_threads)) for t_ind in range(nb_threads)]
			for t in threads:
				t.start()
			for t in threads:
				t.join()
			self.t_should_stop = True

		
		t = threading.Thread(target=read_data_loop, args=(indices, dataset, data_queue, nb_threads))
		t.start()

		return t, data_queue
