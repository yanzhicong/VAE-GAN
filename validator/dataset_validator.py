
import os
import sys
import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import norm


sys.path.append('.')
sys.path.append('../')

import tensorflow as tf
import tensorflow.contrib.layers as tcl


from utils.metric import get_metric

class DatasetValidator(object):
	def __init__(self, config):
		self.config = config
		self.nb_samples = config.get('num_samples', 30)
		self.metric = config.get('metric', 'accuracy')
		self.metric_type = config.get('metric type', 'top1')

		self.assets_dir = config['assets dir']


	def build_summary(self, model):
		if self.metric == 'accuracy':
			self.label = tf.placeholder(tf.float32, shape=[None, model.nb_classes],
							name='test_label')
			self.predict = tf.placeholder(tf.float32, shape=[None, model.nb_classes],
							name='test_predict')
			self.accuracy = get_metric(self.metric, self.metric_type, 
						{'probs' : self.predict, 'labels' : self.label, 'decay' : 1})			

			self.summary_list = []
			self.summary_list.append(tf.summary.scalar('test acc ' + self.metric_type, self.accuracy))

			self.log_filepath = os.path.join(self.assets_dir, 'test_dataset_' + self.metric + "_" + self.metric_type + '.csv')

			with open(self.log_filepath, 'w') as logfile:
				logfile.write('step,'+self.metric_type+'\n')

		self.summary = tf.summary.merge(self.summary_list)


	def validate(self, model, dataset, sess, step):
		label_list = []
		pred_list = []
		for ind, batch_x, batch_y in dataset.iter_test_images():
			pred_y = model.predict(sess, batch_x)
			label_list.append(batch_y)
			pred_list.append(pred_y)
		label_list = np.concatenate(label_list, axis=0)
		pred_list = np.concatenate(pred_list, axis=0)

		if self.metric == 'accuracy' : 
			feed_dict = {
				self.label : label_list,
				self.predict : pred_list,
			}
			acc, summary = sess.run([self.accuracy, self.summary], feed_dict=feed_dict)

			with open(self.log_filepath, 'a') as logfile:
				logfile.write('%d,%f\n'%(step, acc))

		return summary


