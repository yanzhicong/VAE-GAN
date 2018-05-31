# from __future__ import absolute_import

import os
import sys

sys.path.append('.')
sys.path.append('../')

import tensorflow as tf

# from network import inception_v3
# from network.inception_v3 import InceptionV3

# from network import vgg
# from network.vgg import VGG16

import matplotlib.pyplot as plt


from dataset.mnist import MNIST 

if __name__ == '__main__':
	config = {
		'batch_size' : 16,
		'input_shape' : [28, 28]
	}

	dataset = MNIST(config)

	# x = tf.placeholder(tf.float32, shape=(None, 224, 224, 3), name='input')

	# y, end_points = model(x)

	# for name, value in end_points.items():
		# print(name, '  --> ', value.get_shape())

	for ind, x_batch in dataset.iter_images():
		plt.figure(0)
		plt.imshow(x_batch[0, :, :])
		plt.pause(1)