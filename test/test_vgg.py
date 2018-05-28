# from __future__ import absolute_import

import os
import sys

sys.path.append('.')
sys.path.append('../')

import tensorflow as tf

# from network import inception_v3
# from network.inception_v3 import InceptionV3

from network import vgg
from network.vgg import VGG16


if __name__ == '__main__':
	config = {
		'output_classes' : 10
	}
	model_config = {
		'is_training' : True
	}

	model = VGG16(config, model_config)

	x = tf.placeholder(tf.float32, shape=(None, 224, 224, 3), name='input')

	y, end_points = model(x)

	for name, value in end_points.items():
		print(name, '  --> ', value.get_shape())

