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

# import datapipe.gan_trainset as gt
# import datapipe.stargan_trainset as st

import matplotlib.pyplot as plt
import tensorflow as tf

# from model.cvaegan import CVAEGAN

from cfgs.networkconfig import get_config




from model.model import get_model


# from ganmodel.dcgan import DCGAN
# from ganmodel.wgan import WGAN
# from ganmodel.stargan import StarGAN
# from ganmodel.generator import G_conv_pixel
# from ganmodel.generator import G_conv_pixel2
# from ganmodel.generator import G_conv_pixel3
# from ganmodel.generator import G_conv_unet
# from ganmodel.discriminator import D_conv_pixel
# from ganmodel.discriminator import D_conv_pixel2
# from ganmodel.discriminator import D_conv_pixel3

import argparse


parser = argparse.ArgumentParser(description='')
# parser.add_argument('--phase',          type=str,   default='train',    help='train or testsuspect or testtruth')
parser.add_argument('--gpu_number',     type=str,   default='0')
parser.add_argument('--config_file',    type=str,   default='cvaegan1')
# # parser.add_argument('')
# parser.add_argument('--assets_dir',     type=str,   default=os.path.join('.','assets','stargan10'))
# parser.add_argument('--log_dir',        type=str,   default='noise_log') # in assets/ directory
# parser.add_argument('--ckpt_dir',       type=str,   default='noise_checkpoint') # in assets/ directory
# parser.add_argument('--sample_dir',     type=str,   default='noise_sample') # in assets/ directory
# parser.add_argument('--test_dir',       type=str,   default='noise_test') # in assets/ directory
# parser.add_argument('--epoch',          type=int,   default=50)
# parser.add_argument('--batch_size',     type=int,   default=64)
# parser.add_argument('--image_size',     type=int,   default=64)
# parser.add_argument('--image_channel',  type=int,   default=2)
# parser.add_argument('--nf',             type=int,   default=16) # number of filters
# parser.add_argument('--n_label',        type=int,   default=1)
# parser.add_argument('--n_noise',        type=int,   default=1)

# parser.add_argument('--lambda_gp',      type=int,   default=10)
# parser.add_argument('--lambda_cls',     type=int,   default=1)
# parser.add_argument('--lambda_rec',     type=int,   default=50)
# parser.add_argument('--lr',             type=float, default=0.00001) # learning_rate
# parser.add_argument('--beta1',          type=float, default=0.5)
# parser.add_argument('--continue_train', type=bool,  default=True)
# parser.add_argument('--snapshot',       type=int,   default=300) # number of iterations to save files
# parser.add_argument('--adv_type',       type=str,   default='WGAN',     help='GAN or WGAN')
# parser.add_argument('--binary_attrs',   type=str,   default='0000000')

args = parser.parse_args()

# def data_iter(dataset, group, subapp):
#     epoch = 1
#     while True:
#         for batch_data in gt.iter_batch_images_in_dataset(dataset, group, subapp):
#             yield batch_data
#         epoch += 1

if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_number
    tf.reset_default_graph()
    
    # args.log_dir = os.path.join(args.assets_dir, args.log_dir)
    # args.ckpt_dir = os.path.join(args.assets_dir, args.ckpt_dir)
    # args.sample_dir = os.path.join(args.assets_dir, args.sample_dir)
    # args.test_dir = os.path.join(args.assets_dir, args.test_dir)

    config = get_config(args.config_file)

    # print(config)
    # model = CVAEGAN()

    model = get_model(config['ganmodel'], config['ganmodel params'])

    # # make directory if not exist
    # try: os.makedirs(args.log_dir)
    # except: pass
    # try: os.makedirs(args.ckpt_dir)
    # except: pass
    # try: os.makedirs(args.sample_dir)
    # except: pass
    # try: os.makedirs(args.test_dir)
    # except: pass

    # generator = G_conv_unet()
    # discriminator = D_conv_pixel()

    # lot_number = 'L637G458_1x'
    # # lot_number = 'L637G463_2x'
    # dataset = st.get_readable_dataset(lot_number)
    # dataset_info = st.get_dataset_info(lot_number)

    # groups = st.get_group_list(dataset_info)
    # print(groups)
    # group = groups[0]
    # print group

    # tfconfig = tf.ConfigProto()
    # tfconfig.gpu_options.allow_growth = True
    # with tf.Session(config=tfconfig) as sess:
    #     stargan = StarGAN(sess, discriminator, generator, args)
    #     stargan.train(st.iter_batch_images_in_dataset, dataset, group)

