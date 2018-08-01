
import os
import sys

sys.path.append('.')
sys.path.append('../')

import tensorflow as tf
import matplotlib.pyplot as plt

from dataset.pascal_voc import PASCAL_VOC

if __name__ == '__main__':

	# config = {
	# 	"output shape" : [64, 64, 3],
	# 	"scaling range" : [0.15, 0.25],
	# 	"crop range" : [0.3, 0.7],
	# 	"task" :  "classification",
	# 	"random mirroring" : False
	# }

	config = {
		"output shape" : [256, 256, 3],
		"scaling range" : [0.5, 1.5],
		"crop range" : [0.3, 0.7],
		"task" : "segmentation_class_aug",
		# "random mirroring" : False
	}


	dataset = PASCAL_VOC(config)
	indices = dataset.get_image_indices(phase='train', method='supervised')

	print(indices.shape)

	for ind in indices:

		img, mask = dataset.read_image_by_index(ind, phase='train', method='supervised')

		print(img.shape)
		print(mask.shape)
		print(mask.max(), mask.min())

		plt.figure(0)
		plt.imshow(img)
		plt.figure(1)
		plt.imshow(mask)
		plt.pause(4)


