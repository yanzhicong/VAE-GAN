import os
import sys

sys.path.append('.')
sys.path.append('../')

import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

from dataset.violence import Violence
from dataset.tianchi_guangdong_defect import TianChiGuangdongDefect


if __name__ == '__main__':
	config = {
		"output shape" : [224, 224, 3],
		"mil" : False,
		"use cache" : True,
        "one hot" : True,
        "show warning" : True
	}

	dataset = TianChiGuangdongDefect(config)
	indices = dataset.get_image_indices('trainval')

	print(len(indices))

	img_list = []

	for ind in indices:
		img, label = dataset.read_image_by_index(ind, phase='trainval', method='supervised')
		# print(label)

	dataset.time1 = 0.0
	dataset.count = 0
	
	print("")
	print("")
	print("round 2")
	print("")
	print("")
	for ind in indices:
		img, label = dataset.read_image_by_index(ind, phase='trainval', method='supervised')
		# print(label)
		# if img is not None:
		# 	plt.figure(0)
		# 	plt.clf()
		# 	plt.imshow(img)
		# 	plt.pause(1)

	config = {
		"output shape" : [224, 224, 3],
	}


	dataset = TianChiGuangdongDefect(config)
	indices = dataset.get_image_indices('trainval')


	# for ind in indices:
	# 	img_bag, label = dataset.read_image_by_index(ind, phase='trainval', method='supervised')
	# 	print(label)
	# 	if img_bag is not None:

	# 		plt.figure(0)
	# 		plt.clf()

	# 		row = 4
	# 		col = int(len(img_bag) / row)

	# 		print(len(img_bag), row, col)

	# 		for i in range(row):
	# 			for j in range(col):
	# 				plt.subplot(row, col, i * col+j+1)
	# 				plt.imshow(img_bag[i*col+j])
	# 		plt.pause(3)
