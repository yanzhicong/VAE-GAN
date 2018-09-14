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

from utils.learning_rate import get_global_step
from utils.loss import get_loss

from .base_model import BaseModel


def BaseDetectionModel(BaseModel):
	def __init__(self, config):
		BaseModel.__init__(self, config)
		self.config = config


	def build_proposal_layer(self, inputs, proposal_count, mns_threshold):
		"""
		"""
		post_nms_rois_training = self.config.get('nb post-nms rois in training', 2000)
		post_nms_rois_inference = self.config.get('nb post-nms rois in inference', 1000)

		rpn_nms_threshold = self.config.get('proposal nms threshold', 0.7)

		def train_proposal():

			scores = inputs[0][:, :, 1]
			deltas = inputs[1]
			deltas = deltas * np.reshape(self.config)

			pass
		def test_proposal():
			pass

		pass


	def detect(self, batch_x):
		pass



	# def get_anchors():
	# 	pass

