
import os
import sys

sys.path.append('.')
sys.path.append('../')

import tensorflow as tf
from network.vgg import VGG

if __name__ == '__main__':
	config = {
		'output_classes' : 10
		'name' : 'VGG16'
	}

	model = VGG(config, True)
	x = tf.placeholder(tf.float32, shape=(None, 224, 224, 3), name='input')
	y, end_points = model(x)
	for name, value in end_points.items():
		print(name, '  --> ', value.get_shape())

