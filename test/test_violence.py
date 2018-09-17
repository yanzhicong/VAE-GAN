import os
import sys

sys.path.append('.')
sys.path.append('../')

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
			plt.figure(0)
			plt.clf()
			plt.imshow(img)
			plt.pause(3)

	