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


from utils.learning_rate import get_learning_rate
from utils.learning_rate import get_global_step
from utils.optimizer import get_optimizer
from utils.optimizer import get_optimizer_by_config
from utils.sample import get_sample
from utils.loss import get_loss

from .base_model import BaseModel
from network.base_network import BaseNetwork


class PyramidRpnNetwork(BaseNetwork):
	def __init__(self, config):
		super(PyramidRpnNetwork, self).__init__(config)
		self.config = config
		self.reuse = False

		self.anchors_per_location = int(self.config['anchors per location'])
		self.anchor_stride = int(self.config['anchor stride'])

	def __call__(self, features):
		assert len(features) == 5
		c1, c2, c3, c4, c5 = features

		with tf.variable_scope(self.name):
			if self.reuse:
				tf.get_variable_scope().reuse_variables()
			else:
				assert tf.get_variable_scope().reuse is False
				self.reuse = True

			p5 = self.conv2d('fpn_c5p5', c5, 256, 1)
			p4 = tf.add(self.conv2d('fpn_conv_c4p4', c4, 256, 1), self.upsample2d('fpn_p5up', p5, (2, 2)))
			p3 = tf.add(self.conv2d('fpn_conv_c3p3', c3, 256, 1), self.upsample2d('fpn_p4up', p4, (2, 2)))
			p2 = tf.add(self.conv2d('fpn_conv_c2p2', c2, 256, 1), self.upsample2d('fpn_p3up', p3, (2, 2)))
			
			p2 = self.conv2d('fpn_conv_p2', p2, 256, 3)
			p3 = self.conv2d('fpn_conv_p3', p3, 256, 3)
			p4 = self.conv2d('fpn_conv_p4', p4, 256, 3)
			p5 = self.conv2d('fpn_conv_p5', p5, 256, 3)
			p6 = self.maxpool2d('fpn_p6', p5, size=(1, 1), stride=2)

			inputs = {'p2':p2,'p3':p3,'p4':p4,'p5':p5,'p6':p6,}
			outputs = []
			for i in range(2, 7):
				inp = inputs['p%d'%i]
				shared = self.conv2d('fpn_conv_shared_p%d'%i, inp, 512, 3, act_fn='relu')

				cla = self.conv2d('fpn_conv_cls_p%d'%i, shared, 2 * self.anchors_per_location, 1)
				cla_logits = tf.reshape(cla, [tf.shape(cla)[0], -1, 2])
				cla_probs = tf.nn.softmax(cla_logits)

				bbox = self.conv2d('fpn_conv_bbox_p%d'%i, shared, 4 * self.anchors_per_location, 1)
				bbox = tf.reshape(bbox, [tf.shape(bbox)[0], -1, 4])

				outputs.append([cla_logits, cla_probs, bbox])

		return outputs


class Mask_RCNN(BaseModel):
	def __init__(self, config):
		BaseModel.__init__(self, config)
		self.config = config

		assert 'backbone classifier' in config
		assert 'rpn classifier' in config 

		self.image_meta_size = self.config.get('image meta size', 93)

		self.rpn_anchor_scales = self.config.get('rpn anchor scales', [32, 64, 128, 256, 512])
		self.rpn_anchor_ratios = self.config.get('rpn anchor ratios', [0.5, 1, 2])
		self.backbone_strides = self.config.get('backbone strides', [4, 8, 16, 32, 64])
		self.rpn_anchor_stride = self.config.get('anchor stride', 1)


	def build_model(self):

		self.backbone = self._build_classifier('backbone classifier', params={
			'name': 'BackBone'
		})
		self.fpn = PyramidRpnNetwork(self.config['fpn params'])
		self.rcnn = self._build_classifier('rcnn classifier', params={
			'name' : 'RCNN',
			# "output dims" :
		})
		self.mask_cnn = self._build_classifier('mask cnn classifier', params={
			'name' : 'MaskCNN'
		})
				
		self.input_img = tf.placeholder(shape=[None, None, 3], name="input_image")
		self.input_meta = tf.placeholder(shape=[self.image_meta_size], name="input_meta")

		y, features = self.backbone.features(self.input_img)
		backbone_feature_names = self.config['backbone feature names']
		backbone_feature = [features[f] for f in backbone_feature_names]

		fpn_output = self.fpn(backbone_feature)

		raise NotImplementedError

		# build anchors
		# anchors = self.build_anchors(anchors)

		# build proposal layer 
		# proposal_rois = self.build_proposal_layer(rpn_class, rpn_bbox, anchors)


		# for training 

			# build proposal target layer
			# target_rois, target_class_id, target_bbox, target_mask = self.build_proposal_target_layer(proposal_rois, gt)

			# build rcnn classifier
			# roi_features = self.extract_features(mvcnn_feature, target_rois)
			# roi_logits, roi_bbox = self.rcnn(roi_features)


			# build mask regresser
			# roi_mask = self.mask_rcnn(roi_features)


			# losses : 
			# 1. rpn classification loss
			#			
			# 2. rpn bbox regression loss
			#			
			# 3. rcnn classification loss
			# 			roi_logits --> roi_probs ==> target_class_id
			# 4. rcnn bbox regression loss
			#			roi_bbox ==> targete_bbox
			# 5. mask rcnn segmentation loss
			#			roi_mask ==> target_mask


		# for inference
			# build rcnn classifier
			# roi_features = self.extract_features(mvcnn_feature, proposal_rois)
			# roi_logits, roi_bbox = self.rcnn(roi_logits, roi_bbox)

			# detection
			# detections = self.detection_layer(roi_logits, roi_bbox)

			# create masks for detections
			# detection_boxes = detections.get_boxes()


	def build_rpn(self, features):
		pass


	def get_anchors(self, image_shape):
		"""Returns anchor pyramid for the given image size.
		"""

		def compute_backbone_shapes(image_shape):
			"""Computes the width and height of each stage of the backbone network.
			
			Returns:
				[N, (height, width)]. Where N is the number of stages
			"""
			return np.array(
				[[int(math.ceil(image_shape[0] / stride)),
					int(math.ceil(image_shape[1] / stride))]
					for stride in self.backbone_strides])

		def generate_pyramid_anchors(scales, ratios, feature_shapes, feature_strides,
								anchor_stride):
			"""Generate anchors at different levels of a feature pyramid. Each scale
			is associated with a level of the pyramid, but each ratio is used in
			all levels of the pyramid.

			Returns:
			anchors: [N, (y1, x1, y2, x2)]. All generated anchors in one array. Sorted
				with the same order of the given scales. So, anchors of scale[0] come
				first, then anchors of scale[1], and so on.
			"""
			# Anchors
			# [anchor_count, (y1, x1, y2, x2)]
			anchors = []
			for i in range(len(scales)):
				anchors.append(generate_anchors(scales[i], ratios, feature_shapes[i],
												feature_strides[i], anchor_stride))
			return np.concatenate(anchors, axis=0)

		def norm_boxes(boxes, shape):
			"""Converts boxes from pixel coordinates to normalized coordinates.
			boxes: [N, (y1, x1, y2, x2)] in pixel coordinates
			shape: [..., (height, width)] in pixels

			Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
			coordinates it's inside the box.

			Returns:
				[N, (y1, x1, y2, x2)] in normalized coordinates
			"""
			h, w = shape
			scale = np.array([h - 1, w - 1, h - 1, w - 1])
			shift = np.array([0, 0, 1, 1])
			return np.divide((boxes - shift), scale).astype(np.float32)

		backbone_shapes = compute_backbone_shapes(image_shape)

		# Cache anchors and reuse if image shape is the same
		if not hasattr(self, "_anchor_cache"):
			self._anchor_cache = {}

		if not tuple(image_shape) in self._anchor_cache:
			# Generate Anchors
			a = generate_pyramid_anchors(
				self.rpn_anchor_scales,
				self.rpn_anchor_ratios,
				backbone_shapes,
				self.backbone_strides,
				self.rpn_anchor_stride)
			# Keep a copy of the latest anchors in pixel coordinates because
			# it's used in inspect_model notebooks.
			# TODO: Remove this after the notebook are refactored to not use it
			self.anchors = a
			# Normalize coordinates
			self._anchor_cache[tuple(image_shape)] = norm_boxes(a, image_shape[:2])
		return self._anchor_cache[tuple(image_shape)]
