
import os
import sys

sys.path.append('.')
sys.path.append('../')

import tensorflow as tf
from network.unet import UNet

if __name__ == '__main__':
	config = {
		"output dims" : 10,
		'name' : 'UNet'
	}

	is_training = tf.placeholder(tf.bool, name='is_training')

	model = UNet(config, is_training)
	x = tf.placeholder(tf.float32, shape=(None, 224, 224, 3), name='input')
	y, end_points = model(x)


	for name, value in end_points.items():
		print(name, '  --> ', value.get_shape())

	for var in model.vars:
		print(var.name, ' --> ', var.get_shape())

