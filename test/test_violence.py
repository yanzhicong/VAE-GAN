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
		"output shape" : [224, 224, 3]
	}

	# dataset = Violence(config)
	# indices = dataset.get_image_indices('train')

	dataset = TianChiGuangdongDefect(config)
	indices = dataset.get_image_indices('trainval')

	print(len(indices))

	for ind in indices:
		img, label = dataset.read_image_by_index(ind, phase='trainval', method='supervised')
		print(label)
		if img is not None:

			result = img.copy()
			area = dataset.find_most_possible_metal_area(img)
			
			# cv2.line(img, area[0], area[1], 255, 2)
			# cv2.line(img, area[1], area[2], 255, 2)
			# cv2.line(img, area[2], area[3], 255, 2)
			# cv2.line(img, area[3], area[0], 255, 2)

			cv2.fillConvexPoly(result, np.array(area, np.int32), 255)

			result1 = dataset.crop_and_reshape_image_area(img, area)

			plt.figure(0)
			plt.clf()
			plt.imshow(img)
			plt.figure(1)
			plt.clf()
			plt.imshow(result)
			plt.figure(2)
			plt.clf()
			plt.imshow(result1)
			plt.pause(3)
