
import os
import sys

sys.path.append('.')
sys.path.append('../')

import tensorflow as tf

from network.inception_v3 import InceptionV3

if __name__ == '__main__':
	config = {
		'output_dims' : 1000,
		'output_activation' : 'softmax'
	}

	inception_model = InceptionV3(config, False)

	x = tf.placeholder(tf.float32, shape=(None, 299, 299, 3), name='input')

	y, end_points = inception_model(x)

	for name, value in end_points.items():
		print(name, '  --> ', value.get_shape())

