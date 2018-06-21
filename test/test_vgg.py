
import os
import sys

sys.path.append('.')
sys.path.append('../')

import tensorflow as tf
from network.vgg import VGG

if __name__ == '__main__':
	config = {

		"normalization" : "fused_batch_norm",

		"including conv" : True,
		"conv nb blocks" : 6,
		"conv nb layers" : [2, 2, 3, 3, 3, 0],
		"conv nb filters" : [64, 128, 256, 512, 512],
		"conv ksize" : [3, 3, 3, 3, 3],

		"including top" : True,
		"fc nb nodes" : [1024, 1024],

		"output dims" : 12,
		'name' : 'VGG16',


		'load pretrained weights' : 'config tianchi/guangdong3 classifier'
	}

	model = VGG(config, True)
	x = tf.placeholder(tf.float32, shape=(None, 256, 256, 3), name='input')
	y, end_points = model(x)


	for name, value in end_points.items():
		print(name, '  --> ', value.get_shape())

	for var in model.vars:
		print(var.name, ' --> ', var.get_shape())

	tfconfig = tf.ConfigProto()
	tfconfig.gpu_options.allow_growth = True

	with tf.Session(config=tfconfig) as sess:
		ret = model.load_pretrained_weights(sess)

		print(ret)


