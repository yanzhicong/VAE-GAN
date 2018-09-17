
import os
import sys

sys.path.append('.')
sys.path.append('../')

import tensorflow as tf
from network.resnet import Resnet

if __name__ == '__main__':
	config = {
		'output_classes' : 10,
		'name' : 'Resnet50',
		'normalization' : 'fused_batch_norm',
		'output_dims' : 100,
		'debug' : True,
	}

	model = Resnet(config, True)
	x1 = tf.placeholder(tf.float32, shape=(None, 224, 224, 3), name='input')

	x2 = tf.placeholder(tf.float32, shape=(None, 224, 224, 3), name='input2')
	y1, end_points = model(x1)

	y2, end_points = model(x2)


	for name, value in end_points.items():
		print(name, '  --> ', value.get_shape())

	for var in model.vars:
		print(var.name, ' --> ', var.get_shape())

