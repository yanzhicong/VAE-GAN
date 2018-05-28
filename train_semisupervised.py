# -*- coding: utf-8 -*-
# MIT License
# 
# Copyright (c) 2018 ZhicongYan
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================


import os
import sys
import tensorflow as tf


from shutil import copyfile

from cfgs.networkconfig import get_config
from dataset.dataset import get_dataset
from model.model import get_model
from trainer.trainer import get_trainer

import argparse


parser = argparse.ArgumentParser(description='')
parser.add_argument('--gpu_number',     type=str,   default='0')
parser.add_argument('--config_file',    type=str,   default='vae1')

args = parser.parse_args()

if __name__ == '__main__':
	os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
	os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_number
	tf.reset_default_graph()

	config = get_config(args.config_file)

	# make the assets directory and copy the config file to it
	if not os.path.exists(config['assets dir']):
		os.mkdir(config['assets dir'])
	copyfile(os.path.join('./cfgs', args.config_file + '.json'), os.path.join(config['assets dir'], args.config_file + '.json'))

	# prepare dataset
	dataset = get_dataset(config['dataset'], config['dataset params'])

	tfconfig = tf.ConfigProto()
	tfconfig.gpu_options.allow_growth = True


	with tf.Session(config=tfconfig) as sess:

		# build model
		config['ganmodel params']['assets dir'] = config['assets dir']
		model = get_model(config['ganmodel'], config['ganmodel params'])

		# start training
		config['trainer params']['assets dir'] = config['assets dir']
		trainer = get_trainer(config['trainer'], config['trainer params'])

		trainer.train(sess, dataset, model)
