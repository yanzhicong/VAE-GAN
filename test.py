
import os
import tensorflow as tf
from network import inception_v3
from network.inception_v3 import InceptionV3


if __name__ == '__main__':
	config = {
		'output_classes' : 10
	}
	model_config = {
		'is_training' : True
	}

	inception_model = InceptionV3(config, model_config)

	x = tf.placeholder(tf.float32, shape=(None, 299, 299, 3), name='input')

	y, end_points = inception_model(x)

	for name, value in end_points.items():
		print(name, '  --> ', value.get_shape())

